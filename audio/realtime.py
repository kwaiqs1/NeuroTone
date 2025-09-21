# audio/realtime.py
# Реалтайм: микрофон -> шумодав/триггеры -> наушники.
# Поток идёт на stream_sr (на который согласны устройства), а для VAD/YAMNet
# локально ресемплим в 16 kHz.

import math, threading, time, collections
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import sounddevice as sd
from scipy.signal import iirnotch, sosfilt, sosfilt_zi, resample_poly
from scipy.signal.windows import get_window
from scipy.fft import rfft, irfft

import tensorflow as tf
import tensorflow_hub as hub
import webrtcvad


@dataclass
class RTConfig:
    trigger_sensitivity: float = 0.12
    vacuum_strength: float = 0.9
    protect_speech: bool = True
    enable_triggers: bool = True
    enable_vacuum: bool = True
    enable_denoise: bool = True


class RingBuffer:
    """Моно буфер последних N секунд при sr=stream_sr."""
    def __init__(self, seconds: float, sr: int):
        self.sr = sr
        self.maxlen = int(seconds * sr)
        self.buf = collections.deque(maxlen=self.maxlen)

    def push(self, x: np.ndarray):
        self.buf.extend(x.astype(np.float32).tolist())

    def tail_seconds(self, seconds: float) -> Optional[np.ndarray]:
        n = min(len(self.buf), int(seconds * self.sr))
        if n <= 0:
            return None
        return np.fromiter(list(self.buf)[-n:], dtype=np.float32)


def _resample_to(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return y.astype(np.float32)
    g = math.gcd(sr_from, sr_to)
    up, down = sr_to // g, sr_from // g
    return resample_poly(y, up, down).astype(np.float32)


class YAMNetDetector(threading.Thread):
    """Бэк-тред: раз ~0.25 c анализирует последнюю секунду и обновляет флаги триггеров."""
    def __init__(self, ring: RingBuffer, sensitivity: float):
        super().__init__(daemon=True)
        self.ring = ring
        self.sens = sensitivity
        self.flags = {"clock": False, "keyboard": False, "chewing": False}
        if not getattr(tf, "__version__", None):
            tf.__version__ = "2.15.0"
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = self.model.class_map_path().numpy().decode("utf-8")
        with tf.io.gfile.GFile(class_map_path, "r") as f:
            import csv
            self.class_map = [row["display_name"] for row in csv.DictReader(f)]

    def run(self):
        while True:
            audio = self.ring.tail_seconds(1.0)
            if audio is not None and len(audio) >= self.ring.sr // 2:
                y16 = _resample_to(audio, self.ring.sr, 16000)
                scores, _, _ = self.model(y16)  # [T, 521]
                S = scores.numpy()
                classes = {
                    'chewing':  ['Chewing','Mastication','Mouth sounds'],
                    'clock':    ['Tick-tock','Mechanical','Clock','Click'],
                    'keyboard': ['Typing','Keyboard'],
                }
                for key, words in classes.items():
                    idxs = [i for i, name in enumerate(self.class_map)
                            if any(w.lower() in name.lower() for w in words)]
                    val = float(S[:, idxs].max()) if idxs else 0.0
                    self.flags[key] = (val > self.sens)
            time.sleep(0.25)


class RealTimeProcessor:
    def __init__(self, cfg: RTConfig, stream_sr: int = 48000, blocksize: int = 256):
        self.cfg = cfg
        self.sr = int(stream_sr)
        self.blocksize = int(blocksize)

        # окно/FFT
        self.FFT_N = 1024  # ~21 ms @48k, ~23 ms @44.1k
        self.WINDOW = get_window("hann", self.FFT_N, fftbins=True).astype(np.float32)
        self.freqs = np.fft.rfftfreq(self.FFT_N, d=1.0 / self.sr)

        # VAD
        self.vad = webrtcvad.Vad(2)
        self.vad_cache = collections.deque(maxlen=1 + int(1000 / 20))

        # шумовая оценка для Винера
        self.noise_psd = np.ones(self.FFT_N // 2 + 1, dtype=np.float32) * 1e-3

        # нотчи под триггеры
        self.sos_bank, self.sos_state = self._build_notches()

        # кольцевой буфер и YAMNet
        self.ring = RingBuffer(2.0, self.sr)
        self.detector = None
        if cfg.enable_triggers:
            self.detector = YAMNetDetector(self.ring, cfg.trigger_sensitivity)
            self.detector.start()

    def _build_notches(self):
        # частоты нотчей указаны в Гц — пересчитываем под текущую sr
        layout = [(3000, 30), (6000, 30),    # clock
                  (3500, 25),               # keyboard
                  (500, 15), (7000, 15)]    # chewing
        sos_list = []
        zi_list = []
        for f0, q in layout:
            w0 = f0 / (self.sr / 2.0)
            b = np.array([1.0, -2.0*np.cos(np.pi*w0), 1.0], dtype=np.float32)
            a = np.array([1.0, -2.0*np.cos(np.pi*w0)/(1.0 + 1.0/(2*q*q)), (1.0 - 1.0/(2*q*q))], dtype=np.float32)
            # Формируем SOS вручную
            sos = np.array([b[0], b[1], b[2], 1.0, a[1], a[2]], dtype=np.float32).reshape(1, 6)
            sos_list.append(sos)
            zi_list.append(sosfilt_zi(sos) * 0.0)
        return np.vstack(sos_list), zi_list

    def _vad_is_speech(self, block: np.ndarray) -> bool:
        # ресемплим текущий блок в 16 kHz, собираем 20 ms
        y16 = _resample_to(block, self.sr, 16000)
        need = 320  # 20 ms @ 16k
        frame = np.zeros(need, dtype=np.int16)
        n = min(len(y16), need)
        frame[:n] = np.clip(y16[:n] * 32767, -32768, 32767).astype(np.int16)
        try:
            return self.vad.is_speech(frame.tobytes(), 16000)
        except Exception:
            return False

    def _wiener_gain(self, X_mag2: np.ndarray, is_speech: bool) -> np.ndarray:
        if not is_speech:
            alpha = 0.9
            self.noise_psd = alpha * self.noise_psd + (1 - alpha) * X_mag2
        snr = np.maximum(X_mag2 / (self.noise_psd + 1e-12) - 1.0, 0.0)
        H = snr / (snr + 1.0)
        floor = 10 ** (-30 / 20)
        H = np.clip(H, floor, 1.0)
        # немного сберечь < 250 Гц
        H[self.freqs < 250] = np.maximum(H[self.freqs < 250], 10 ** (-12 / 20))
        return H

    def _apply_notches(self, x: np.ndarray, flags: Dict[str, bool]) -> np.ndarray:
        # индексы: 0,1 -> clock; 2 -> keyboard; 3,4 -> chewing
        idxs = []
        if flags.get("clock", False):    idxs += [0, 1]
        if flags.get("keyboard", False): idxs += [2]
        if flags.get("chewing", False):  idxs += [3, 4]
        y = x
        for i in idxs:
            y, self.sos_state[i] = sosfilt(self.sos_bank[i:i+1], y, zi=self.sos_state[i])
        return y

    def process_block(self, in_block: np.ndarray) -> np.ndarray:
        self.ring.push(in_block.astype(np.float32))

        is_speech = self._vad_is_speech(in_block)
        self.vad_cache.append(is_speech)
        vad_flag = sum(self.vad_cache) > (len(self.vad_cache) // 2)

        out = in_block.copy()
        if self.cfg.enable_denoise:
            # формируем окно для FFT
            xw = np.zeros(self.FFT_N, dtype=np.float32)
            n = min(len(out), self.FFT_N)
            xw[:n] = out[:n]
            X = rfft(xw * self.WINDOW)
            H = self._wiener_gain(np.abs(X) ** 2, vad_flag)
            if self.cfg.enable_vacuum and (not vad_flag or not self.cfg.protect_speech):
                vac = max(0.03, 1.0 - 0.85 * self.cfg.vacuum_strength)
                H *= vac
            Y = X * H
            out = irfft(Y).astype(np.float32)[:len(out)]

        trig_flags = self.detector.flags if self.detector else {}
        out = self._apply_notches(out, trig_flags)
        return out
