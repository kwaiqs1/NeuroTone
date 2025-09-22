# audio/realtime.py
# Hi-Fi real-time: корректный STFT (50% overlap, Hann) + мягкий Wiener + сглаживание (attack/release)
# + вакуум вне речи + безопасные нотчи по триггерам. VAD/YAMNet работают на 16 kHz (внутренний ресемпл).

import math, threading, time, collections
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
from scipy.signal import resample_poly, iirnotch, sosfilt, sosfilt_zi, tf2sos
from scipy.signal.windows import hann
from scipy.fft import rfft, irfft

import tensorflow as tf
import tensorflow_hub as hub
import webrtcvad


@dataclass
class RTConfig:
    trigger_sensitivity: float = 0.12     # ниже -> агрессивнее ловит триггеры
    vacuum_strength: float = 0.8          # 0..1; 0.7–0.9 = мягкий "вакуум" без убийства качества
    protect_speech: bool = True
    enable_triggers: bool = True
    enable_vacuum: bool = True
    enable_denoise: bool = True


def _resample_to(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return y.astype(np.float32)
    g = math.gcd(sr_from, sr_to)
    up, down = sr_to // g, sr_from // g
    return resample_poly(y, up, down).astype(np.float32)


class RingBuffer:
    def __init__(self, seconds: float, sr: int):
        self.sr = sr
        self.maxlen = int(seconds * sr)
        self.buf = collections.deque(maxlen=self.maxlen)
    def push(self, x: np.ndarray):
        self.buf.extend(x.astype(np.float32).tolist())
    def tail_seconds(self, seconds: float) -> Optional[np.ndarray]:
        n = min(len(self.buf), int(seconds * self.sr))
        if n <= 0: return None
        return np.fromiter(list(self.buf)[-n:], dtype=np.float32)


class YAMNetDetector(threading.Thread):
    def __init__(self, ring: RingBuffer, sensitivity: float):
        super().__init__(daemon=True)
        self.ring = ring
        self.sens = float(sensitivity)
        self.flags = {"clock": False, "keyboard": False, "chewing": False}
        if not getattr(tf, "__version__", None):
            tf.__version__ = "2.15.0"
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = self.model.class_map_path().numpy().decode("utf-8")
        with tf.io.gfile.GFile(class_map_path, "r") as f:
            import csv
            self.class_map = [row["display_name"] for row in csv.DictReader(f)]
    def run(self):
        classes = {
            'chewing':  ['Chewing','Mastication','Mouth sounds'],
            'clock':    ['Tick-tock','Mechanical','Clock','Click'],
            'keyboard': ['Typing','Keyboard'],
        }
        while True:
            audio = self.ring.tail_seconds(1.0)
            if audio is not None and len(audio) >= self.ring.sr // 2:
                y16 = _resample_to(audio, self.ring.sr, 16000)
                scores, _, _ = self.model(y16)
                S = scores.numpy()
                for key, words in classes.items():
                    idxs = [i for i, name in enumerate(self.class_map)
                            if any(w.lower() in name.lower() for w in words)]
                    val = float(S[:, idxs].max()) if idxs else 0.0
                    self.flags[key] = (val > self.sens)
            time.sleep(0.25)


class OLAProcessor:
    """Streaming STFT/ISTFT c 50% overlap, Hann, perfect COLA и сглаживанием усиления."""
    def __init__(self, sr: int, frame: int = 1024):
        self.sr = sr
        self.N = int(frame)
        self.H = self.N // 2                    # hop = 50% overlap -> COLA для Hann
        self.win = hann(self.N, sym=False).astype(np.float32)

        # буферы анализа/синтеза
        self.inbuf = np.zeros(0, dtype=np.float32)
        self.tail = np.zeros(self.N - self.H, dtype=np.float32)  # для синтеза
        self.prev_gain = np.ones(self.N // 2 + 1, dtype=np.float32)

        # частоты бинoв
        self.freqs = np.fft.rfftfreq(self.N, 1.0 / self.sr)

        # шумовая оценка
        self.noise_psd = np.ones(self.N // 2 + 1, dtype=np.float32) * 1e-3

        # параметры сглаживания усиления (attack/release)
        self.attack = 0.6   # быстрее реагируем вниз (большее подавление)
        self.release = 0.9  # медленнее отпускаем вверх (избегаем «дробления»)

    def _gain_smooth(self, g: np.ndarray) -> np.ndarray:
        p = self.prev_gain
        go_down = g < p
        out = np.where(go_down, self.attack * p + (1 - self.attack) * g,
                       self.release * p + (1 - self.release) * g)
        self.prev_gain = out
        return out

    def _wiener_gain(self, Pxx: np.ndarray, speech: bool, vacuum: float) -> np.ndarray:
        # обновляем шум на НЕ-речевых кадрах
        if not speech:
            alpha = 0.95
            self.noise_psd = alpha * self.noise_psd + (1 - alpha) * Pxx
        snr = np.maximum(Pxx / (self.noise_psd + 1e-12) - 1.0, 0.0)
        g = snr / (snr + 1.0)

        # частотно-зависимый пол: в речи >= -12 dB, вне речи можно до -24..-30 dB
        floor_speech = 10 ** (-12 / 20)     # ~0.25
        floor_nons   = 10 ** (-24 / 20)     # ~0.06
        floor = floor_speech if speech else floor_nons

        # «вакуум» — мягкий аттенюатор вне речи в дальних полосах (низы<120, верхи>7k)
        if not speech and vacuum > 0.0:
            vac = max(0.0, min(1.0, vacuum))
            lows = self.freqs < 120
            highs = self.freqs > 7000
            g[lows | highs] *= (1.0 - 0.6 * vac)   # до ~ -4..-8 dB по краям

        g = np.clip(g, floor, 1.0)
        return self._gain_smooth(g)

    def process_push(self, x: np.ndarray, speech_flag: bool, vacuum_strength: float) -> np.ndarray:
        # накапливаем вход
        self.inbuf = np.concatenate([self.inbuf, x.astype(np.float32)])
        out = np.zeros(0, dtype=np.float32)

        while len(self.inbuf) >= self.H:
            # собираем окно: предыдущий «хвост» + H новых сэмплов
            need = self.N - self.H
            if len(self.inbuf) < self.N:
                # ждём ещё сэмплов
                break
            frame = self.inbuf[:self.N]
            self.inbuf = self.inbuf[self.H:]  # сдвиг на hop

            X = rfft(frame * self.win)
            Pxx = (np.abs(X) ** 2).astype(np.float32)
            G = self._wiener_gain(Pxx, speech_flag, vacuum_strength)
            Y = X * G
            y = irfft(Y).astype(np.float32)

            # Overlap-Add (с тем же окном): идеальная реконструкция без «радио»
            yw = y * self.win
            # первый кусок: хвост с прошлой итерации + начало текущего
            head = yw[:self.N - self.H] + self.tail
            tail = yw[self.N - self.H:]
            self.tail = tail

            out = np.concatenate([out, head])

        return out


class RealTimeProcessor:
    def __init__(self, cfg: RTConfig, stream_sr: int = 48000, blocksize: int = 480):
        self.cfg = cfg
        self.sr = int(stream_sr)
        # frame = 2*blocksize чтобы обеспечить 50% overlap на каждом callback
        self.frame = int(2 * blocksize)
        if self.frame % 2 == 1: self.frame += 1
        self.stft = OLAProcessor(self.sr, frame=self.frame)

        # VAD
        self.vad = webrtcvad.Vad(2)
        self.vad_hist = collections.deque(maxlen=3)

        # буфер для детектора
        self.ring = RingBuffer(2.0, self.sr)

        # нотч-банк (стабильный, через iirnotch + tf2sos)
        self.sos_bank, self.sos_state = self._build_notches()

        # YAMNet
        self.detector = None
        if cfg.enable_triggers:
            self.detector = YAMNetDetector(self.ring, cfg.trigger_sensitivity)
            self.detector.start()

    def _build_notches(self):
        layout = [
            (3000, 30), (6000, 30),     # clock
            (3500, 25),                 # keyboard
            (500, 15), (7000, 15),      # chewing
        ]
        sos_list, zi_list = [], []
        for f0, Q in layout:
            b, a = iirnotch(w0=f0, Q=Q, fs=self.sr)   # корректный дизайн под текущую sr
            sos = tf2sos(b, a)
            sos_list.append(sos)
            zi_list.append(sosfilt_zi(sos) * 0.0)
        return np.vstack(sos_list), zi_list

    def _vad_is_speech(self, block: np.ndarray) -> bool:
        y16 = _resample_to(block, self.sr, 16000)
        # аккуратно собираем 20 мс; если блока мало — дополним нулями
        need = 320
        fr = np.zeros(need, dtype=np.int16)
        n = min(len(y16), need)
        fr[:n] = np.clip(y16[:n] * 32767, -32768, 32767).astype(np.int16)
        try:
            return self.vad.is_speech(fr.tobytes(), 16000)
        except Exception:
            return False

    def _apply_notches(self, x: np.ndarray, flags: Dict[str, bool], speech: bool) -> np.ndarray:
        # в речи — ограничим глубину (оставим немного, чтобы не «рвать» тембр)
        depth_in_speech = 0.35   # ~ -9 dB
        depth_out       = 0.08   # ~ -22 dB

        use_idx = []
        if flags.get("clock", False):    use_idx += [0, 1]
        if flags.get("keyboard", False): use_idx += [2]
        if flags.get("chewing", False):  use_idx += [3, 4]

        y = x
        for i in use_idx:
            y, self.sos_state[i] = sosfilt(self.sos_bank[i:i+1], y, zi=self.sos_state[i])
            if speech:
                y = y * (1.0 - depth_in_speech) + x * depth_in_speech
            else:
                y = y * (1.0 - depth_out) + x * depth_out
        return y

    def process_block(self, in_block: np.ndarray) -> np.ndarray:
        self.ring.push(in_block.astype(np.float32))

        speech_now = self._vad_is_speech(in_block) if self.cfg.protect_speech else False
        self.vad_hist.append(speech_now)
        speech = (sum(self.vad_hist) >= 2)  # небольшое сглаживание по времени

        # STFT-обработка с overlap-add (Hi-Fi)
        out = in_block
        if self.cfg.enable_denoise:
            out = self.stft.process_push(in_block, speech_flag=speech,
                                         vacuum_strength=self.cfg.vacuum_strength if self.cfg.enable_vacuum else 0.0)
            # Если буфер ещё не «раскачался», возвращаем вход (чтобы не было тишины)
            if len(out) == 0:
                return in_block

        # триггеры (YAMNet)
        trig_flags = self.detector.flags if self.detector else {}
        out2 = self._apply_notches(out, trig_flags, speech=speech)

        # мягкий лимитинг (защита от клипа)
        peak = np.max(np.abs(out2)) + 1e-9
        if peak > 0.98:
            out2 = 0.98 * (out2 / peak)
        return out2.astype(np.float32)
