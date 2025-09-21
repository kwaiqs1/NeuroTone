# audio/realtime.py
# Реалтайм: микрофон -> шумодав/триггеры -> наушники (full-duplex).
# Низкая задержка за счёт небольших блоков и лёгких операций в callback.

import threading, time, queue, collections
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import sounddevice as sd
from scipy.signal import iirnotch, sosfilt, sosfilt_zi, get_window, rfft, irfft

import tensorflow as tf
import tensorflow_hub as hub
import webrtcvad

# --------- Параметры по умолчанию ---------
SR = 16000             # работаем на 16 кГц (совместимо с VAD и легковесно)
BLOCK = 256            # 16 мс блок
FFT_N = 512            # окно для спектрального гейта
HOP = BLOCK            # без перекрытия в callback (простота)
WINDOW = get_window("hann", FFT_N, fftbins=True).astype(np.float32)

# Частотные диапазоны под триггеры
BANDS = {
    "clock":    [(2000, 9000)],
    "keyboard": [(2000, 6500)],
    "chewing":  [(180, 1200), (6000, 9500)],
}

@dataclass
class RTConfig:
    trigger_sensitivity: float = 0.12   # ниже -> агрессивнее
    vacuum_strength: float = 0.9        # 0..1 (сила «вакуума» вне речи)
    protect_speech: bool = True         # беречь речь
    enable_triggers: bool = True        # включить ловлю триггеров
    enable_vacuum: bool = True          # включить глобальное приглушение фона
    enable_denoise: bool = True         # включить спектральный денойз

class YAMNetDetector(threading.Thread):
    """Лёгкий бэк-тред: каждую ~0.25 с анализирует последний 1.0 с буфер и обновляет флаги триггеров."""
    def __init__(self, ring: "RingBuffer", sensitivity: float):
        super().__init__(daemon=True)
        self.ring = ring
        self.sens = sensitivity
        self.flags = {"clock": False, "keyboard": False, "chewing": False}
        # TF-Hub загрузка
        if not getattr(tf, "__version__", None):
            tf.__version__ = "2.15.0"
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")
        # класс-мап
        class_map_path = self.model.class_map_path().numpy().decode("utf-8")
        with tf.io.gfile.GFile(class_map_path, "r") as f:
            import csv
            self.class_map = [row["display_name"] for row in csv.DictReader(f)]

    def run(self):
        while True:
            audio = self.ring.tail_seconds(1.0)  # последний 1 сек сигнал
            if audio is not None and len(audio) >= SR // 2:
                y16 = audio.astype(np.float32)
                scores, _, _ = self.model(y16)  # [T, 521]
                S = scores.numpy()
                for key, words in {
                    'chewing':  ['Chewing','Mastication','Mouth sounds'],
                    'clock':    ['Tick-tock','Mechanical','Clock','Click'],
                    'keyboard': ['Typing','Keyboard']
                }.items():
                    idxs = [i for i, name in enumerate(self.class_map)
                            if any(w.lower() in name.lower() for w in words)]
                    val = float(S[:, idxs].max()) if idxs else 0.0
                    self.flags[key] = (val > self.sens)
            time.sleep(0.25)

class RingBuffer:
    """Простой моно буфер последних N секунд."""
    def __init__(self, seconds: float, sr: int = SR):
        self.maxlen = int(seconds * sr)
        self.buf = collections.deque(maxlen=self.maxlen)

    def push(self, x: np.ndarray):
        self.buf.extend(x.tolist())

    def tail_seconds(self, seconds: float) -> Optional[np.ndarray]:
        n = min(len(self.buf), int(seconds * SR))
        if n <= 0: return None
        arr = np.fromiter(list(self.buf)[-n:], dtype=np.float32)
        return arr

class RealTimeProcessor:
    def __init__(self, cfg: RTConfig):
        self.cfg = cfg
        # VAD
        self.vad = webrtcvad.Vad(2)
        self.vad_cache = collections.deque(maxlen=1 + int(1000/20))  # ~1s истории флагов
        # шумовая оценка для Винера
        self.noise_psd = np.ones(FFT_N//2+1, dtype=np.float32) * 1e-3
        # предрассчитанные бин-частоты
        self.freqs = np.fft.rfftfreq(FFT_N, d=1.0/SR)
        # нотч-фильтры с состояниями (sos для устойчивости)
        self.sos_bank, self.sos_state = self._build_notches()
        # поток безопасной передачи данных в callback
        self.ring = RingBuffer(2.0, SR)  # 2 секунды
        # YAMNet бэк-тред
        self.detector = None
        if cfg.enable_triggers:
            self.detector = YAMNetDetector(self.ring, cfg.trigger_sensitivity)
            self.detector.start()

    def _build_notches(self):
        """Готовим SOS для частот триггеров; применяются по флагам детектора."""
        sos_list = []
        for f0, q in [(3000,30), (6000,30),   # clock
                      (3500,25),             # keyboard
                      (500,15), (7000,15)]:  # chewing
            w0 = f0/(SR/2)
            b, a = iirnotch(w0=w0, Q=q)
            # перевод в SOS вручную
            sos = np.zeros((1,6), dtype=np.float32)
            sos[0,:] = [b[0], b[1], b[2], 1.0, a[1], a[2]]
            sos_list.append(sos)
        sos_arr = np.vstack(sos_list)
        zi = [sosfilt_zi(sos_arr[i:i+1]) * 0.0 for i in range(sos_arr.shape[0])]
        return sos_arr, zi

    def _vad_is_speech(self, block: np.ndarray) -> bool:
        # VAD требует int16 и ровно 20 мс; соберём 20 мс из 16 мс блока с паддингом
        frame = np.zeros(int(0.02*SR), dtype=np.int16)
        n = min(len(block), len(frame))
        frame[:n] = np.clip(block[:n]*32767, -32768, 32767).astype(np.int16)
        try:
            return self.vad.is_speech(frame.tobytes(), SR)
        except Exception:
            return False

    def _wiener_gain(self, X_mag2: np.ndarray, is_speech: bool) -> np.ndarray:
        # Оценка шума: обновляем на НЕ-речевых кадрах (скользящее среднее)
        if not is_speech:
            alpha = 0.9
            self.noise_psd = alpha*self.noise_psd + (1-alpha)*X_mag2
        snr = np.maximum(X_mag2 / (self.noise_psd + 1e-12) - 1.0, 0.0)
        H = snr / (snr + 1.0)
        # Пол гейна (−28..−32 dB)
        floor = 10 ** (-30/20)
        H = np.clip(H, floor, 1.0)
        # чуть бережнее <250 Гц
        H[self.freqs < 250] = np.maximum(H[self.freqs < 250], 10 ** (-12/20))
        return H

    def _apply_notches(self, x: np.ndarray, flags: Dict[str,bool]) -> np.ndarray:
        # применяем нужные НОТЧ-фильтры (clock/keyboard/chewing)
        # индексы: 0,1 -> clock; 2 -> keyboard; 3,4 -> chewing
        active_idx = []
        if flags.get("clock", False):    active_idx += [0,1]
        if flags.get("keyboard", False): active_idx += [2]
        if flags.get("chewing", False):  active_idx += [3,4]
        y = x
        for idx in active_idx:
            y, self.sos_state[idx] = sosfilt(self.sos_bank[idx:idx+1], y, zi=self.sos_state[idx])
        return y

    def process_block(self, in_block: np.ndarray) -> np.ndarray:
        # сохраняем в ринг для детектора
        self.ring.push(in_block.astype(np.float32))

        # VAD (прошлое + текущее решение сглаживаем)
        is_speech = self._vad_is_speech(in_block)
        self.vad_cache.append(is_speech)
        vad_flag = sum(self.vad_cache) > (len(self.vad_cache)//2)

        # Быстрый спектральный денойз (Винер)
        if self.cfg.enable_denoise:
            # соберём окно для FFT_N (с паддингом нулями)
            xw = np.zeros(FFT_N, dtype=np.float32)
            xw[:len(in_block)] = in_block
            X = rfft(xw * WINDOW)
            H = self._wiener_gain(np.abs(X)**2, vad_flag)
            # «вакуум» вне речи
            if self.cfg.enable_vacuum and (not vad_flag or not self.cfg.protect_speech):
                vac = max(0.03, 1.0 - 0.85*self.cfg.vacuum_strength)  # до ~−30 dB
                H *= vac
            Y = X * H
            out = irfft(Y).astype(np.float32)[:len(in_block)]
        else:
            out = in_block.copy()

        # Триггеры (динамические нотчи)
        trig_flags = self.detector.flags if self.detector else {}
        out = self._apply_notches(out, trig_flags)
        return out
