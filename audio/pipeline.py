# audio/pipeline.py
# Слышимый результат: покадровый детект YAMNet + мультибэнд-гейтинг + (опц.) плотный денойз.
import wave
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.signal import stft, istft, resample_poly

import tensorflow as tf
import tensorflow_hub as hub

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except Exception:
    HAS_NOISEREDUCE = False

# --- модель/классы ---
YAMNET_HANDLE = 'https://tfhub.dev/google/yamnet/1'
TRIGGER_KEYWORDS = {
    'chewing':  ['Chewing', 'Mastication', 'Mouth sounds'],
    'clock':    ['Tick-tock', 'Mechanical', 'Clock', 'Click'],
    'keyboard': ['Typing', 'Keyboard'],
}

@dataclass
class PipelineConfig:
    base_denoise: bool = True
    suppress_triggers: bool = True
    # ниже — агрессивнее. 0.15 = заметное подавление, 0.25 = умеренное
    trigger_sensitivity: float = 0.15

# ---------- WAV I/O ----------
def _read_wav_mono_float(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, 'rb') as wf:
        nch = wf.getnchannels()
        sw  = wf.getsampwidth()
        sr  = wf.getframerate()
        n   = wf.getnframes()
        raw = wf.readframes(n)
    if sw == 2:
        y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        y = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        y = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f'Unsupported WAV sample width: {sw} bytes')
    if nch > 1:
        y = y.reshape(-1, nch).mean(axis=1)
    return y, sr

def _write_wav_int16(path: str, y: np.ndarray, sr: int):
    y16 = np.clip(y, -1.0, 1.0)
    y16 = (y16 * 32767.0).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(y16.tobytes())

def _resample_to(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return y.astype(np.float32)
    g = np.gcd(sr, target_sr)
    up, down = target_sr // g, sr // g
    return resample_poly(y, up, down).astype(np.float32)

# ---------- pipeline ----------
class CalmCityPipeline:
    def __init__(self):
        # предохранитель для tensorflow_hub (редкий кейс)
        if not getattr(tf, "__version__", None):
            tf.__version__ = "2.15.0"
        self.model = hub.load(YAMNET_HANDLE)
        self.class_map = self._load_class_map()

    def _load_class_map(self):
        class_map_path = self.model.class_map_path().numpy().decode('utf-8')
        with tf.io.gfile.GFile(class_map_path, 'r') as f:
            import csv
            return [row['display_name'] for row in csv.DictReader(f)]

    # покадровые скоры из YAMNet
    def _frame_trigger_scores(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        y16 = _resample_to(y, sr, 16000)
        scores, _, _ = self.model(y16)             # [T, 521]
        S = scores.numpy()
        out = {}
        for key, words in TRIGGER_KEYWORDS.items():
            idxs = [i for i, name in enumerate(self.class_map)
                    if any(w.lower() in name.lower() for w in words)]
            out[key] = S[:, idxs].max(axis=1) if idxs else np.zeros((S.shape[0],), dtype=np.float32)
        return out  # {'clock': [T], ...}

    # плотный, но аккуратный денойз ДО гейтинга
    def _base_soft_denoise(self, y: np.ndarray, sr: int) -> np.ndarray:
        if HAS_NOISEREDUCE:
            # стационарный спектральный гейтинг, заметнее
            return nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.9)
        return y

    # мультибэнд-гейтинг по триггерам (сильно слышно)
    def _multiband_gate(self, y: np.ndarray, sr: int,
                        trig_frames: Dict[str, np.ndarray], sensitivity: float) -> np.ndarray:
        # STFT: окно/шаг под оффлайн-обработку
        nper, nover = 1024, 256
        f, t, Z = stft(y, fs=sr, nperseg=nper, noverlap=nover)  # Z: [freq, time] complex
        M = Z.shape[1]
        T_yam = next(iter(trig_frames.values())).shape[0] if trig_frames else 0
        if T_yam == 0:
            return y

        # маппинг индексов времени STFT -> YAMNet (приближённый, но устойчивый)
        map_idx = np.minimum((np.arange(M) * T_yam) // max(M, 1), T_yam - 1)

        thr = float(sensitivity)
        # базовая глубина ослабления; при низком thr становится глубже
        depth_base = 0.4 + 0.6 * (1.0 - thr)   # 0.15 => ~0.91 (≈ −19 dB)

        # частоты (Гц) для полос
        BANDS = {
            'clock':    [(2000, 9000)],            # тик-так, щелчки
            'keyboard': [(2000, 6000)],            # клавиатура
            'chewing':  [(200, 1200), (6000, 9500)]# чавканье + влажные транзиенты
        }

        for i in range(M):
            j = int(map_idx[i])
            frame_mask = np.ones_like(Z[:, i], dtype=np.float32)

            for key, bands in BANDS.items():
                s = float(trig_frames[key][j])
                if s > thr:
                    # чем выше над порогом — тем сильнее аттенюация
                    # ограничим минимум ~−24 dB (gain ≈ 0.06)
                    k = np.clip((s - thr) / max(1e-6, 1.0 - thr), 0.0, 1.0)
                    gain = np.clip(1.0 - depth_base * k, 0.06, 1.0)
                    for (lo, hi) in bands:
                        band = (f >= lo) & (f <= hi)
                        frame_mask[band] *= gain

            Z[:, i] *= frame_mask

        _, y_out = istft(Z, fs=sr, nperseg=nper, noverlap=nover)
        return y_out.astype(np.float32)

    def process_file(self, in_path: str, out_path: str, config: PipelineConfig) -> Dict:
        y, sr = _read_wav_mono_float(in_path)

        # 1) покадровые триггеры на оригинале
        trig_frames = self._frame_trigger_scores(y, sr) if config.suppress_triggers else {}

        # 2) базовый денойз
        if config.base_denoise:
            y = self._base_soft_denoise(y, sr)

        # 3) гейтинг по триггерам
        if config.suppress_triggers:
            y = self._multiband_gate(y, sr, trig_frames, config.trigger_sensitivity)

        # 4) нормализация
        peak = np.max(np.abs(y)) + 1e-9
        y = 0.95 * (y / peak)

        _write_wav_int16(out_path, y, sr)
        # для notes — усреднённые скоры (как раньше)
        mean_scores = {k: float(v.mean()) for k, v in trig_frames.items()} if trig_frames else {}
        return {'triggers': mean_scores, 'sr': sr}

# singleton
_pipeline_singleton = None
def get_pipeline() -> CalmCityPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = CalmCityPipeline()
    return _pipeline_singleton
