# audio/pipeline.py
# Жёсткая оффлайн-обработка: агрессивный спектральный денойз + "вакуум" вне речи (VAD)
# + покадровые триггеры YAMNet + подавление транзиентов (spectral flux).

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
    trigger_sensitivity: float = 0.12   # ниже -> агрессивнее
    preserve_speech: bool = True
    vacuum_mode: bool = True
    vacuum_strength: float = 0.9        # 0..1; вне речи глушим фон очень сильно

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

# ---------- основа ----------
class CalmCityPipeline:
    def __init__(self):
        if not getattr(tf, "__version__", None):
            tf.__version__ = "2.15.0"  # предохранитель для TF-Hub
        self.yamnet = hub.load(YAMNET_HANDLE)
        self.class_map = self._load_class_map()
        self.vad = webrtcvad.Vad(2)  # 0..3 (2 — баланс, 3 — максимум)

    def _load_class_map(self):
        class_map_path = self.yamnet.class_map_path().numpy().decode('utf-8')
        with tf.io.gfile.GFile(class_map_path, 'r') as f:
            import csv
            return [row['display_name'] for row in csv.DictReader(f)]

    # --- YAMNet покадрово ---
    def _frame_trigger_scores(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        y16 = _resample_to(y, sr, 16000)
        scores, _, _ = self.yamnet(y16)     # [T, 521]
        S = scores.numpy()
        out = {}
        for key, words in TRIGGER_KEYWORDS.items():
            idxs = [i for i, name in enumerate(self.class_map)
                    if any(w.lower() in name.lower() for w in words)]
            out[key] = S[:, idxs].max(axis=1) if idxs else np.zeros((S.shape[0],), dtype=np.float32)
        return out

    # --- VAD (20 мс @ 16 кГц) ---
    def _vad_flags(self, y: np.ndarray, sr: int, aggressiveness=2) -> np.ndarray:
        y16 = _resample_to(y, sr, 16000)
        s = np.clip(y16 * 32767, -32768, 32767).astype(np.int16)
        step = 320  # 20 ms
        n = len(s) // step
        vad = webrtcvad.Vad(aggressiveness)
        flags = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            frame = s[i*step:(i+1)*step].tobytes()
            try:
                flags[i] = vad.is_speech(frame, 16000)
            except Exception:
                flags[i] = False
        return flags

    # --- агрессивный базовый денойз (минимум-статистика + гейтинг/Винер) ---
    def _aggressive_denoise_mask(self, Z: np.ndarray, fs: int, speech_mask_t: np.ndarray) -> np.ndarray:
        # Z: [freq, time] complex
        mag = np.abs(Z)
        power = mag**2 + 1e-12

        # оценка шума по кадрам, где нет речи (если таких нет — возьмём 20-й перцентиль по времени)
        T = power.shape[1]
        if speech_mask_t.size > 0:
            map_vad = np.minimum((np.arange(T) * speech_mask_t.size) // max(T, 1), speech_mask_t.size - 1)
            non_speech = ~speech_mask_t[map_vad]
        else:
            non_speech = np.zeros(T, dtype=bool)

        if non_speech.any():
            noise_psd = np.median(power[:, non_speech], axis=1, keepdims=True)
        else:
            noise_psd = np.percentile(power, 20, axis=1, keepdims=True)

        # Винер: H = SNR/(SNR+1), SNR ~ max(P/N - 1, 0)
        snr = np.maximum(power / (noise_psd + 1e-12) - 1.0, 0.0)
        H = snr / (snr + 1.0)

        # Пол по усилению (жёсткий): −28..−32 dB
        floor = 10 ** (-30 / 20)
        H = np.clip(H, floor, 1.0)

        # Чуть бережнее < 250 Гц (чтобы не "дышало")
        # и в узкой области 1–4 кГц при речи (согласные/гласные)
        f_bins = np.linspace(0, fs/2, Z.shape[0])
        low = f_bins < 250
        H[low, :] = np.maximum(H[low, :], 10 ** (-12/20))

        return H

    # --- детектор транзиентов (spectral flux) ---
    def _transient_mask_t(self, Z: np.ndarray, fs: int) -> np.ndarray:
        mag = np.abs(Z)
        # позитивный спектральный поток в 2–9 кГц
        f_bins = np.linspace(0, fs/2, Z.shape[0])
        band = (f_bins >= 2000) & (f_bins <= 9000)
        diff = np.maximum(mag[:, 1:] - mag[:, :-1], 0.0)
        flux = diff[band, :].sum(axis=0)
        # порог: медиана + 3*MAD
        med = np.median(flux)
        mad = np.median(np.abs(flux - med)) + 1e-9
        thr = med + 3.0 * mad
        mask_t = np.zeros(Z.shape[1], dtype=np.bool_)
        mask_t[1:] = flux > thr
        # расширим на соседние кадры
        for i in range(1, len(mask_t)-1):
            if mask_t[i]:
                mask_t[i-1] = True
                mask_t[i+1] = True
        return mask_t

    def _process_masking(self, y: np.ndarray, sr: int,
                         trig_frames: Dict[str, np.ndarray],
                         speech_flags: np.ndarray,
                         sensitivity: float,
                         vacuum_on: bool,
                         vacuum_strength: float,
                         protect_speech: bool) -> np.ndarray:

        # STFT
        nper, nover = 1024, 256
        f, t, Z = stft(y, fs=sr, nperseg=nper, noverlap=nover)   # [F, T]
        F, T = Z.shape

        # базовая маска денойза
        H_denoise = self._aggressive_denoise_mask(Z, sr, speech_flags)

        # временная маска VAD в координаты STFT
        if speech_flags.size > 0:
            map_vad = np.minimum((np.arange(T) * speech_flags.size) // max(T, 1), speech_flags.size - 1)
            vad_t = speech_flags[map_vad]
        else:
            vad_t = np.zeros(T, dtype=np.bool_)

        # покадровые триггеры YAMNet в координаты STFT
        yam_t = {}
        for key, arr in trig_frames.items():
            map_y = np.minimum((np.arange(T) * arr.size) // max(T, 1), max(arr.size - 1, 0))
            yam_t[key] = arr[map_y] if arr.size > 0 else np.zeros(T, dtype=np.float32)

        # транзиенты (щелчки) как дополнительный супрессор
        trans_t = self._transient_mask_t(Z, sr)

        # начальная маска усиления
        G = H_denoise.copy()

        # "вакуум" вне речи: глобально прижмём фон
        if vacuum_on:
            base_vacuum = np.clip(1.0 - 0.85 * float(vacuum_strength), 0.03, 1.0)  # до ~−30 dB
            for i in range(T):
                if (not vad_t[i]) or (not protect_speech):
                    G[:, i] *= base_vacuum

        # частотные полосы для триггеров
        BANDS = {
            'clock':    [(2000, 9000)],
            'keyboard': [(2000, 6500)],
            'chewing':  [(180, 1200), (6000, 9500)],
        }
        f_bins = np.linspace(0, sr/2, F)

        thr = float(0.12 if sensitivity in (None, '') else sensitivity)

        # максимальное подавление (вне речи/триггер): почти в ноль
        MIN_GAIN_OUT = 0.01   # ~−40 dB
        # минимальное в речи: бережнее
        MIN_GAIN_SPEECH = 0.12  # ~−18 dB

        for i in range(T):
            is_speech = bool(vad_t[i])

            # доп. гашение на транзиентах (2–9 кГц)
            if trans_t[i]:
                band = (f_bins >= 2000) & (f_bins <= 9000)
                G[band, i] *= 0.15 if not is_speech else 0.35

            # триггеры YAMNet
            for key, bands in BANDS.items():
                s = float(yam_t.get(key, np.zeros((), np.float32))[i]) if key in yam_t else 0.0
                if s > thr:
                    k = np.clip((s - thr) / max(1e-6, 1.0 - thr), 0.0, 1.0)  # 0..1
                    # в речи гасятся мягче, вне — максимально
                    target_min = MIN_GAIN_SPEECH if (is_speech and protect_speech) else MIN_GAIN_OUT
                    # глубина 0.6..1.0 (очень сильно)
                    depth = 0.6 + 0.4 * k
                    gain = np.clip(1.0 - depth, target_min, 1.0)
                    for (lo, hi) in bands:
                        band = (f_bins >= lo) & (f_bins <= hi)
                        G[band, i] *= gain

        # применяем маску и обратно в сигнал
        Z_proc = Z * G
        _, y_out = istft(Z_proc, fs=sr, nperseg=nper, noverlap=nover)
        return y_out.astype(np.float32)

    def process_file(self, in_path: str, out_path: str, config: PipelineConfig) -> Dict:
        # безопасность для чисел
        sens = 0.12 if config.trigger_sensitivity in (None, '') else float(config.trigger_sensitivity)
        vacs = 0.9  if config.vacuum_strength    in (None, '') else float(config.vacuum_strength)

        y, sr = _read_wav_mono_float(in_path)

        # 1) покадровые триггеры
        trig_frames = self._frame_trigger_scores(y, sr) if config.suppress_triggers else {}

        # 2) базовый денойз (жёсткий)
        if config.base_denoise:
            # если есть noisereduce — слегка предварительно «подсушим»,
            # основное всё равно сделает наш маскер
            if HAS_NOISEREDUCE:
                y = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.95)

        # 3) VAD (речь)
        speech_flags = self._vad_flags(y, sr, aggressiveness=2) if config.preserve_speech else np.zeros((0,), dtype=np.bool_)

        # 4) маскирование (вакуум + триггеры + транзиенты + денойз)
        y = self._process_masking(
            y, sr,
            trig_frames=trig_frames,
            speech_flags=speech_flags,
            sensitivity=sens,
            vacuum_on=config.vacuum_mode,
            vacuum_strength=vacs,
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
