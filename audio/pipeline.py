# audio/pipeline.py
# Реально слышимая обработка: покадровые триггеры YAMNet + мультибэнд-гейтинг + (опционно) мягкий денойз.
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

# --- Модель и классы ---
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
    # ВАЖНО: чем НИЖЕ sensitivity, тем АГРЕССИВНЕЕ подавление (0.0 .. 1.0)
    trigger_sensitivity: float = 0.25

# ---------- Утилиты WAV без soundfile ----------
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

# ---------- Пайплайн ----------
class CalmCityPipeline:
    def __init__(self):
        # Предохранитель, если tf.__version__ вдруг «немой»
        if not getattr(tf, "__version__", None):
            tf.__version__ = "2.15.0"
        self.model = hub.load(YAMNET_HANDLE)     # скачает 1 раз
        self.class_map = self._load_class_map()

    def _load_class_map(self):
        class_map_path = self.model.class_map_path().numpy().decode('utf-8')
        with tf.io.gfile.GFile(class_map_path, 'r') as f:
            import csv
            return [row['display_name'] for row in csv.DictReader(f)]

    # ---- YAMNet: покадровые скоры триггеров ----
    def _frame_trigger_scores(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        y16 = _resample_to(y, sr, 16000)
        scores, _, _ = self.model(y16)                    # [T, 521]
        S = scores.numpy()
        out = {}
        for key, words in TRIGGER_KEYWORDS.items():
            idxs = [i for i, name in enumerate(self.class_map)
                    if any(w.lower() in name.lower() for w in words)]
            out[key] = S[:, idxs].max(axis=1) if idxs else np.zeros((S.shape[0],), dtype=np.float32)
        return out  # {'clock': [T], 'chewing':[T], 'keyboard':[T]}

    # ---- Мягкий денойз (опционально) до гейтинга ----
    def _base_soft_denoise(self, y: np.ndarray, sr: int) -> np.ndarray:
        if HAS_NOISEREDUCE:
            nref = min(len(y), int(sr*0.5))
            return nr.reduce_noise(y=y, y_noise=y[:nref], sr=sr, prop_decrease=0.8)  # пожёстче
        return y

    # ---- Мультибэнд-гейтинг (сильно заметно) ----
    def _multiband_gate(self, y: np.ndarray, sr: int, trig_frames: Dict[str, np.ndarray], sensitivity: float) -> np.ndarray:
        # STFT
        nper, nover = 512, 256                      # ~32 ms окно при 16 кГц; годится и для 44.1/48 кГц
        f, t, Z = stft(y, fs=sr, nperseg=nper, noverlap=nover)   # Z: [freq, time] complex
        M = Z.shape[1]                               # кол-во тайм-фреймов STFT

        # Сопоставим индексы временных фреймов YAMNet -> STFT
        # YAMNet даёт шаг ~0.48 c; просто масштабируем индексы
        T_yam = next(iter(trig_frames.values())).shape[0] if trig_frames else 0
        if T_yam == 0:
            return y  # на всякий случай
        map_idx = np.minimum((np.arange(M) * T_yam) // max(M,1), T_yam-1)

        # Порог = sensitivity; чем ниже sensitivity — тем агрессивней.
        thr = float(sensitivity)

        # Глубина подавления (0..1), растёт при меньшей чувствительности
        # (при sensitivity=0.25 → depth ≈ 0.7)
        depth_base = 0.3 + 0.6*(1.0 - thr)

        # Частотные полосы под триггеры (Гц)
        BANDS = {
            'clock':    [(2000, 8000)],             # щелчки, тик-так
            'keyboard': [(2000, 5000)],
            'chewing':  [(200, 1000), (6000, 9000)] # чавканье (низ-середина) + «мокрые» транзиенты повыше
        }

        # Применяем покадрово
        for i in range(M):
            j = int(map_idx[i])
            # итоговый множитель на этот фрейм по умолчанию 1.0 (не трогаем)
            frame_mask = np.ones_like(Z[:, i], dtype=np.float32)

            for key, bands in BANDS.items():
                score = float(trig_frames[key][j])
                if score > thr:
                    # чем выше score над порогом — тем сильнее ослабление
                    gain = 1.0 - depth_base * min((score - thr) / max(1e-6, 1.0 - thr), 1.0)
                    for (lo, hi) in bands:
                        band = (f >= lo) & (f <= hi)
                        frame_mask[band] *= gain

            Z[:, i] *= frame_mask

        # Обратное преобразование
        _, y_out = istft(Z, fs=sr, nperseg=nper, noverlap=nover)
        return y_out.astype(np.float32)

    def process_file(self, in_path: str, out_path: str, config: PipelineConfig) -> Dict:
        y, sr = _read_wav_mono_float(in_path)

        # 1) Покадровые триггеры (с оригинала — так надёжнее)
        trig_frames = self._frame_trigger_scores(y, sr) if config.suppress_triggers else {}

        # 2) Базовый мягкий денойз (по желанию)
        if config.base_denoise:
            y = self._base_soft_denoise(y, sr)

        # 3) Мультибэнд-гейтинг под триггеры
        if config.suppress_triggers:
            y = self._multiband_gate(y, sr, trig_frames, config.trigger_sensitivity)

        # 4) Нормализация уровня
        peak = np.max(np.abs(y)) + 1e-9
        y = 0.95 * (y / peak)

        _write_wav_int16(out_path, y, sr)
        # Для текущего шаблона notes выводим усреднённые скоры (как раньше)
        mean_scores = {k: float(v.mean()) for k, v in trig_frames.items()} if trig_frames else {}
        return {'triggers': mean_scores, 'sr': sr}

# Синглтон-пайплайн
_pipeline_singleton = None
def get_pipeline() -> CalmCityPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = CalmCityPipeline()
    return _pipeline_singleton
