# audio/pipeline.py
import wave
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.signal import stft, istft, resample_poly

import tensorflow as tf
import tensorflow_hub as hub
import webrtcvad

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except Exception:
    HAS_NOISEREDUCE = False

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
    trigger_sensitivity: float = 0.15
    preserve_speech: bool = True
    vacuum_mode: bool = True
    vacuum_strength: float = 0.8  # 0..1

# -------- WAV I/O (без soundfile) --------
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

# -------- Пайплайн --------
class CalmCityPipeline:
    def __init__(self):
        if not getattr(tf, "__version__", None):
            tf.__version__ = "2.15.0"
        self.model = hub.load(YAMNET_HANDLE)
        self.class_map = self._load_class_map()
        # WebRTC VAD (0..3) — 2: баланс, 3: максимально агрессивный
        self.vad = webrtcvad.Vad(2)

    def _load_class_map(self):
        class_map_path = self.model.class_map_path().numpy().decode('utf-8')
        with tf.io.gfile.GFile(class_map_path, 'r') as f:
            import csv
            return [row['display_name'] for row in csv.DictReader(f)]

    # --- покадровые триггеры (YAMNet) ---
    def _frame_trigger_scores(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        y16 = _resample_to(y, sr, 16000)
        scores, _, _ = self.model(y16)  # [T, 521]
        S = scores.numpy()
        out = {}
        for key, words in TRIGGER_KEYWORDS.items():
            idxs = [i for i, name in enumerate(self.class_map)
                    if any(w.lower() in name.lower() for w in words)]
            out[key] = S[:, idxs].max(axis=1) if idxs else np.zeros((S.shape[0],), dtype=np.float32)
        return out

    # --- речь/не речь (WebRTC VAD) ---
    def _vad_frames(self, y: np.ndarray, sr: int, aggressiveness: int = 2) -> np.ndarray:
        # работаем на 16 кГц, шаг 20 мс (320 сэмплов)
        y16 = _resample_to(y, sr, 16000)
        y16_i16 = np.clip(y16 * 32767, -32768, 32767).astype(np.int16)

        frame_len = 320  # 20 ms @16k
        n_frames = len(y16_i16) // frame_len
        vad = webrtcvad.Vad(aggressiveness)
        flags = np.zeros(n_frames, dtype=np.bool_)

        for i in range(n_frames):
            frame = y16_i16[i*frame_len:(i+1)*frame_len].tobytes()
            try:
                flags[i] = vad.is_speech(frame, 16000)
            except Exception:
                flags[i] = False
        return flags  # длиной T_vad

    # --- плотный денойз ДО гейтинга ---
    def _base_soft_denoise(self, y: np.ndarray, sr: int) -> np.ndarray:
        if HAS_NOISEREDUCE:
            # статционный режим; ощутимо убирает «шум комнаты/улицы»
            return nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.95)
        return y

    # --- «вакуум» + мультибэнд-гейтинг по триггерам ---
    def _gate_with_vacuum(self, y: np.ndarray, sr: int,
                          trig_frames: Dict[str, np.ndarray],
                          vad_flags: np.ndarray,
                          sensitivity: float,
                          vacuum_on: bool,
                          vacuum_strength: float,
                          protect_speech: bool) -> np.ndarray:

        sensitivity = 0.15 if sensitivity in (None, '') else float(sensitivity)
        vacuum_strength = 0.8 if vacuum_strength in (None, '') else float(vacuum_strength)

        # STFT
        nper, nover = 1024, 256
        f, t, Z = stft(y, fs=sr, nperseg=nper, noverlap=nover)
        M = Z.shape[1]

        # маппинг временных индексов: STFT -> YAMNet/VAD
        T_yam = next(iter(trig_frames.values())).shape[0] if trig_frames else 0
        T_vad = len(vad_flags)
        map_yam = np.minimum((np.arange(M) * max(T_yam,1)) // max(M,1), max(T_yam-1,0))
        map_vad = np.minimum((np.arange(M) * max(T_vad,1)) // max(M,1), max(T_vad-1,0))

        thr = float(sensitivity)

        # частотные полосы:
        BANDS = {
            'clock':    [(2000, 9000)],
            'keyboard': [(2000, 6000)],
            'chewing':  [(200, 1200), (6000, 9500)]
        }

        # глобальная «вакуумная» база (чем больше vacuum_strength, тем сильнее глушим фон вне речи)
        # вне речи target_gain ~ 1 - (0.75 * strength)  (например, 0.8 → ~0.4 = −8 dB)
        # в полосах триггеров — ещё сильнее (см. ниже)
        base_vacuum = max(0.05, 1.0 - 0.75 * float(vacuum_strength))  # не ниже −26 dB вне-сеточно

        for i in range(M):
            j_y = int(map_yam[i]) if T_yam > 0 else 0
            j_v = int(map_vad[i]) if T_vad > 0 else 0
            is_speech = bool(vad_flags[j_v]) if T_vad > 0 else False

            # стартовая маска кадра
            frame_mask = np.ones_like(Z[:, i], dtype=np.float32)

            # 1) «вакуум» (вне речи)
            if vacuum_on and (not is_speech or not protect_speech):
                frame_mask *= base_vacuum

            # 2) триггеры — динамическая глубина, до полного «mute» вне речи
            for key, bands in BANDS.items():
                s = float(trig_frames.get(key, np.zeros((1,), np.float32))[j_y]) if T_yam > 0 else 0.0
                if s > thr:
                    # уровень «давления» от 0..1
                    k = np.clip((s - thr) / max(1e-6, 1.0 - thr), 0.0, 1.0)

                    # в речи: оставляем минимум ~−18 dB (0.12), чтобы не ломать артикуляцию
                    # вне речи: можно почти в "0" (почти полное удаление)
                    min_gain_in_speech = 0.12
                    min_gain_out_speech = 0.02

                    depth = 0.5 + 0.5 * k  # 0.5..1.0
                    if is_speech and protect_speech:
                        gain = np.clip(1.0 - depth, min_gain_in_speech, 1.0)
                    else:
                        gain = np.clip(1.0 - 1.2 * depth, min_gain_out_speech, 1.0)

                    for (lo, hi) in bands:
                        band = (f >= lo) & (f <= hi)
                        frame_mask[band] *= gain

            Z[:, i] *= frame_mask

        _, y_out = istft(Z, fs=sr, nperseg=nper, noverlap=nover)
        return y_out.astype(np.float32)

    def process_file(self, in_path: str, out_path: str, config: PipelineConfig) -> Dict:
        y, sr = _read_wav_mono_float(in_path)

        # 1) покадровые триггеры
        trig_frames = self._frame_trigger_scores(y, sr) if config.suppress_triggers else {}

        # 2) денойз до гейтинга
        if config.base_denoise:
            y = self._base_soft_denoise(y, sr)

        # 3) VAD (речь)
        vad_flags = self._vad_frames(y, sr, aggressiveness=2) if config.preserve_speech else np.zeros((0,), dtype=np.bool_)

        # 4) вакуум + триггеры
        y = self._gate_with_vacuum(
            y, sr,
            trig_frames=trig_frames,
            vad_flags=vad_flags,
            sensitivity=config.trigger_sensitivity,
            vacuum_on=config.vacuum_mode,
            vacuum_strength=config.vacuum_strength,
            protect_speech=config.preserve_speech
        )

        # 5) нормализация
        peak = np.max(np.abs(y)) + 1e-9
        y = 0.95 * (y / peak)

        _write_wav_int16(out_path, y, sr)
        mean_scores = {k: float(v.mean()) for k, v in trig_frames.items()} if trig_frames else {}
        return {'triggers': mean_scores, 'sr': sr}

# singleton
_pipeline_singleton = None
def get_pipeline() -> CalmCityPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = CalmCityPipeline()
    return _pipeline_singleton
