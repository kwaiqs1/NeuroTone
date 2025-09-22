# audio/realtime.py
# Ultra ANC + Ultra Trigger Kill (real-time, hi-fi):
# - STFT 50% overlap (Hann) + Wiener c полом и атакой/релизом
# - "Vacuum" вне речи (умный, без радио-побочек)
# - Trigger hold (300–500 мс) + транзиент-гейт (spectral flux 2–9 кГц)
# - Нотчи по триггерам с разной глубиной в речи/вне речи
# - VAD/YAMNet на 16 кГц (внутренний ресемпл), поток на sr устройств

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
    trigger_sensitivity: float = 0.10     # ниже -> агрессивнее ловит триггеры
    vacuum_strength: float = 0.9          # 0..1; вне речи "вакуум"
    protect_speech: bool = True
    enable_triggers: bool = True
    enable_vacuum: bool = True
    enable_denoise: bool = True
    ultra_anc: bool = True                # добав. глоб. аттенюатор вне речи
    trigger_hold_ms: int = 400            # сколько держать подавление после события
    transient_gate: bool = True           # глушить щелчки по spectral flux


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
        self.cls_words = {
            'chewing':  ['Chewing','Mastication','Mouth sounds'],
            'clock':    ['Tick-tock','Mechanical','Clock','Click'],
            'keyboard': ['Typing','Keyboard'],
        }
    def run(self):
        while True:
            audio = self.ring.tail_seconds(1.0)
            if audio is not None and len(audio) >= self.ring.sr // 2:
                y16 = _resample_to(audio, self.ring.sr, 16000)
                scores, _, _ = self.model(y16)
                S = scores.numpy()
                for key, words in self.cls_words.items():
                    idxs = [i for i, name in enumerate(self.class_map)
                            if any(w.lower() in name.lower() for w in words)]
                    val = float(S[:, idxs].max()) if idxs else 0.0
                    self.flags[key] = (val > self.sens)
            time.sleep(0.20)  # почаще обновляем


class OLAProcessor:
    """Streaming STFT/ISTFT c 50% overlap, Hann, COLA; сглаживание усиления; трекинг шума."""
    def __init__(self, sr: int, frame: int = 1024):
        self.sr = sr
        self.N = int(frame)
        self.H = self.N // 2
        self.win = hann(self.N, sym=False).astype(np.float32)
        self.inbuf = np.zeros(0, dtype=np.float32)
        self.tail = np.zeros(self.N - self.H, dtype=np.float32)

        self.freqs = np.fft.rfftfreq(self.N, 1.0 / self.sr)
        self.noise_psd = np.ones(self.N // 2 + 1, dtype=np.float32) * 1e-3

        # сглаживание усиления
        self.attack = 0.55
        self.release = 0.92
        self.prev_gain = np.ones(self.N // 2 + 1, dtype=np.float32)

        # для транзиентов
        self.prev_mag = np.zeros(self.N // 2 + 1, dtype=np.float32)

    def _gain_smooth(self, g: np.ndarray) -> np.ndarray:
        p = self.prev_gain
        go_down = g < p
        out = np.where(go_down, self.attack * p + (1 - self.attack) * g,
                       self.release * p + (1 - self.release) * g)
        self.prev_gain = out
        return out

    def _wiener_gain(self, Pxx: np.ndarray, speech: bool, vacuum: float, ultra_anc: bool) -> np.ndarray:
        # обновляем шум на не-речевых кадрах
        if not speech:
            alpha = 0.96
            self.noise_psd = alpha * self.noise_psd + (1 - alpha) * Pxx

        snr = np.maximum(Pxx / (self.noise_psd + 1e-12) - 1.0, 0.0)
        g = snr / (snr + 1.0)

        # пол по усилению
        floor_speech = 10 ** (-10 / 20)    # ~ -10 dB в речи (чуть мягче)
        floor_nons   = 10 ** (-26 / 20)    # ~ -26 dB вне речи
        floor = floor_speech if speech else floor_nons

        # "вакуум" вне речи: умный, по краям сильнее
        if not speech and vacuum > 0.0:
            vac = max(0.0, min(1.0, vacuum))
            lows  = self.freqs < 150
            highs = self.freqs > 7500
            g[lows | highs] *= (1.0 - 0.65 * vac)  # до ~ -4..-9 dB доп. прижатия
            if ultra_anc:
                # чуть шире прижмём средние, но без убийства речи
                mids = (self.freqs >= 800) & (self.freqs <= 2500)
                g[mids] *= (1.0 - 0.25 * vac)

        g = np.clip(g, floor, 1.0)
        return self._gain_smooth(g)

    def _transient_mask(self, mag: np.ndarray) -> float:
        """Возвращает силу транзиента 0..1 по spectral flux в 2–9 кГц."""
        band = (self.freqs >= 2000) & (self.freqs <= 9000)
        diff = np.maximum(mag - self.prev_mag, 0.0)
        flux = float(diff[band].sum() / (self.prev_mag[band].sum() + 1e-9))
        self.prev_mag = mag
        # пороги: >0.6 — явно, >0.35 — умеренно
        if flux > 0.6: return 1.0
        if flux > 0.35: return 0.5
        return 0.0

    def process_push(self, x: np.ndarray, speech_flag: bool, vacuum_strength: float, ultra_anc: bool) -> np.ndarray:
        self.inbuf = np.concatenate([self.inbuf, x.astype(np.float32)])
        out = np.zeros(0, dtype=np.float32)

        while len(self.inbuf) >= self.N:
            frame = self.inbuf[:self.N]
            self.inbuf = self.inbuf[self.H:]

            X = rfft(frame * self.win)
            mag = np.abs(X).astype(np.float32)
            Pxx = (mag ** 2)

            # транзиентная сила до расчёта гейна
            tr_power = self._transient_mask(mag)

            G = self._wiener_gain(Pxx, speech_flag, vacuum_strength, ultra_anc)

            # если транзиент и это не речь — ужмём ВЧ сильнее
            if tr_power > 0 and not speech_flag:
                hf = self.freqs > 2500
                G[hf] *= (0.25 if tr_power >= 1.0 else 0.5)

            Y = X * G
            y = irfft(Y).astype(np.float32)

            # идеально складываем
            yw = y * self.win
            head = yw[:self.N - self.H] + self.tail
            self.tail = yw[self.N - self.H:]
            out = np.concatenate([out, head])

        return out


class RealTimeProcessor:
    def __init__(self, cfg: RTConfig, stream_sr: int = 48000, blocksize: int = 480):
        self.cfg = cfg
        self.sr = int(stream_sr)
        self.frame = int(2 * blocksize)
        if self.frame % 2 == 1: self.frame += 1
        self.stft = OLAProcessor(self.sr, frame=self.frame)

        # VAD
        self.vad = webrtcvad.Vad(2)
        self.vad_hist = collections.deque(maxlen=3)

        # буфер в поточной частоте
        self.ring = RingBuffer(2.0, self.sr)

        # нотчи под триггеры
        self.sos_bank, self.sos_state = self._build_notches()

        # YAMNet
        self.detector = None
        if cfg.enable_triggers:
            self.detector = YAMNetDetector(self.ring, cfg.trigger_sensitivity)
            self.detector.start()

        # удержание подавления по триггерам
        self.hold_samples = int((cfg.trigger_hold_ms / 1000.0) * self.sr)
        self.active_until = {"clock": 0, "keyboard": 0, "chewing": 0}
        self.sample_counter = 0

    def _build_notches(self):
        layout = [
            (3000, 30), (6000, 30),     # clock
            (3500, 25),                 # keyboard
            (500, 15), (7000, 15),      # chewing
        ]
        sos_list, zi_list = [], []
        for f0, Q in layout:
            b, a = iirnotch(w0=f0, Q=Q, fs=self.sr)
            sos = tf2sos(b, a)
            sos_list.append(sos)
            zi_list.append(sosfilt_zi(sos) * 0.0)
        return np.vstack(sos_list), zi_list

    def _vad_is_speech(self, block: np.ndarray) -> bool:
        y16 = _resample_to(block, self.sr, 16000)
        need = 320
        fr = np.zeros(need, dtype=np.int16)
        n = min(len(y16), need)
        fr[:n] = np.clip(y16[:n] * 32767, -32768, 32767).astype(np.int16)
        try:
            return self.vad.is_speech(fr.tobytes(), 16000)
        except Exception:
            return False

    def _apply_notches(self, x: np.ndarray, flags: Dict[str, bool], speech: bool) -> np.ndarray:
        # глубина в речи/вне речи
        depth_in_speech = 0.30   # ~ -8.5 dB
        depth_out       = 0.02   # ~ -34 dB (почти в ноль)

        # долговременные флаги с hold
        now = self.sample_counter
        for k in self.active_until.keys():
            if flags.get(k, False):
                self.active_until[k] = now + self.hold_samples
        active = {k: (now < t) for k, t in self.active_until.items()}

        use_idx = []
        if active.get("clock", False):    use_idx += [0, 1]
        if active.get("keyboard", False): use_idx += [2]
        if active.get("chewing", False):  use_idx += [3, 4]

        y = x
        for i in use_idx:
            y, self.sos_state[i] = sosfilt(self.sos_bank[i:i+1], y, zi=self.sos_state[i])
            if speech:
                y = y * (1.0 - depth_in_speech) + x * depth_in_speech
            else:
                y = y * (1.0 - depth_out) + x * depth_out
        return y

    def process_block(self, in_block: np.ndarray) -> np.ndarray:
        self.sample_counter += len(in_block)
        self.ring.push(in_block.astype(np.float32))

        speech_now = self._vad_is_speech(in_block) if self.cfg.protect_speech else False
        self.vad_hist.append(speech_now)
        speech = (sum(self.vad_hist) >= 2)

        # STFT-обработка (hi-fi)
        out = in_block
        if self.cfg.enable_denoise:
            out = self.stft.process_push(
                in_block,
                speech_flag=speech,
                vacuum_strength=self.cfg.vacuum_strength if self.cfg.enable_vacuum else 0.0,
                ultra_anc=self.cfg.ultra_anc and (not speech)
            )
            if len(out) == 0:
                return in_block

        # триггеры: нотчи + hold
        trig_flags = self.detector.flags if self.detector else {}
        out2 = self._apply_notches(out, trig_flags, speech=speech)

        # мягкий лимитинг
        peak = np.max(np.abs(out2)) + 1e-9
        if peak > 0.98:
            out2 = 0.98 * (out2 / peak)
        return out2.astype(np.float32)
