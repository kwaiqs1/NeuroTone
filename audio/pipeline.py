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
    from audio.learn.trigger_infer import LocalTriggerInfer
    HAS_LOCAL = True
except Exception:
    HAS_LOCAL = False


try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except Exception:
    HAS_NOISEREDUCE = False

YAMNET_HANDLE = 'https://tfhub.dev/google/yamnet/1'
# Используем более точные имена классов YAMNet + подстроки
TRIGGER_KEYWORDS = {
    'chewing':  ['Chewing, mastication', 'Lip smacking', 'Mouth'],
    'clock':    ['Tick-tock', 'Knock', 'Tap'],
    'keyboard': ['Typing', 'Computer keyboard', 'Mouse click', 'Keys jangling'],
}

@dataclass
class PipelineConfig:
    base_denoise: bool = True
    suppress_triggers: bool = True
    trigger_sensitivity: float = 0.12
    preserve_speech: bool = True
    vacuum_mode: bool = True
    vacuum_strength: float = 0.9

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
        # Исправление масштаба для 32-бит PCM
        y = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483647.0
    else:
        raise ValueError(f'Unsupported WAV sample width: {sw} bytes')
    if nch > 1:
        y = y.reshape(-1, nch).mean(axis=1)
    return y.astype(np.float32), sr

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
    g = int(np.gcd(int(sr), int(target_sr)))
    up, down = target_sr // g, sr // g
    return resample_poly(y, up, down).astype(np.float32)

class CalmCityPipeline:
    def __init__(self):
        # В rare случаях tf.__version__ отсутствует — подстраховка не нужна, но пусть останется безопасно.
        if not getattr(tf, "__version__", None):
            tf.__version__ = "2.15.0"
        self.yamnet = hub.load(YAMNET_HANDLE)
        self.class_map = self._load_class_map()
        self.vad = webrtcvad.Vad(2)
        self.local = LocalTriggerInfer(sr_stream=16000) if HAS_LOCAL else None



    def _local_frame_scores(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Возвращает покадровые (шаг ~0.5с) вероятности локальных классов:
        chewing, ticktock, keyboard, mouseclick, beep
        """
        if self.local is None or not self.local.enabled:
            return {}

        y16 = _resample_to(y, sr, 16000)
        win = self.local.target_sr                      # 1.0 c
        hop = int(0.5 * self.local.target_sr)           # 0.5 c

        keys = ["chewing", "ticktock", "keyboard", "mouseclick", "beep"]
        acc = {k: [] for k in keys}

        if y16.size < win:
            seg = np.pad(y16, (0, win - y16.size))
            p = self.local._probs(seg)
            for k in keys:
                acc[k].append(float(p.get(k, 0.0)))
        else:
            for i in range(0, y16.size - win + 1, hop):
                seg = y16[i:i+win]
                p = self.local._probs(seg)
                for k in keys:
                    acc[k].append(float(p.get(k, 0.0)))

        return {k: np.array(v, dtype=np.float32) if len(v) else np.zeros((0,), np.float32) for k, v in acc.items()}



    def _load_class_map(self):
        class_map_path = self.yamnet.class_map_path().numpy().decode('utf-8')
        with tf.io.gfile.GFile(class_map_path, 'r') as f:
            import csv
            return [row['display_name'] for row in csv.DictReader(f)]

    def _frame_trigger_scores(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        y16 = _resample_to(y, sr, 16000)
        scores, _, _ = self.yamnet(y16)  # [frames, classes]
        S = scores.numpy()
        out = {}
        for key, words in TRIGGER_KEYWORDS.items():
            idxs = [i for i, name in enumerate(self.class_map)
                    if any(w.lower() in name.lower() for w in words)]
            out[key] = S[:, idxs].max(axis=1) if idxs else np.zeros((S.shape[0],), dtype=np.float32)
        return out

    def _vad_flags(self, y: np.ndarray, sr: int, aggressiveness=2) -> np.ndarray:
        y16 = _resample_to(y, sr, 16000)
        s = np.clip(y16 * 32767, -32768, 32767).astype(np.int16)
        step = 320  # 20ms @16k
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

    def _aggressive_denoise_mask(self, Z: np.ndarray, fs: int, speech_mask_t: np.ndarray) -> np.ndarray:
        mag = np.abs(Z)
        power = mag**2 + 1e-12

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

        snr = np.maximum(power / (noise_psd + 1e-12) - 1.0, 0.0)
        H = snr / (snr + 1.0)

        floor = 10 ** (-30 / 20)
        H = np.clip(H, floor, 1.0)

        f_bins = np.linspace(0, fs/2, Z.shape[0])
        low = f_bins < 250
        H[low, :] = np.maximum(H[low, :], 10 ** (-12/20))

        return H

    def _transient_mask_t(self, Z: np.ndarray, fs: int) -> np.ndarray:
        mag = np.abs(Z)
        f_bins = np.linspace(0, fs/2, Z.shape[0])
        band = (f_bins >= 2000) & (f_bins <= 9000)
        diff = np.maximum(mag[:, 1:] - mag[:, :-1], 0.0)
        flux = diff[band, :].sum(axis=0)
        med = np.median(flux)
        mad = np.median(np.abs(flux - med)) + 1e-9
        thr = med + 3.0 * mad
        mask_t = np.zeros(Z.shape[1], dtype=np.bool_)
        mask_t[1:] = flux > thr

        # Расширяем окрестности всплесков (−1/+1) — устойчивее к быстрым щелчкам
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

        nper, nover = 1024, 256
        f, t, Z = stft(y, fs=sr, nperseg=nper, noverlap=nover)
        F, T = Z.shape

        H_denoise = self._aggressive_denoise_mask(Z, sr, speech_flags)

        if speech_flags.size > 0:
            map_vad = np.minimum((np.arange(T) * speech_flags.size) // max(T, 1), speech_flags.size - 1)
            vad_t = speech_flags[map_vad]
        else:
            vad_t = np.zeros(T, dtype=np.bool_)

        yam_t = {}
        for key, arr in trig_frames.items():
            map_y = np.minimum((np.arange(T) * arr.size) // max(T, 1), max(arr.size - 1, 0))
            yam_t[key] = arr[map_y] if arr.size > 0 else np.zeros(T, dtype=np.float32)

        trans_t = self._transient_mask_t(Z, sr)

        G = H_denoise.copy()

        if vacuum_on:
            # Чуть мягче при речи
            base_vacuum = np.clip(1.0 - 0.85 * float(vacuum_strength), 0.03, 1.0)
            for i in range(T):
                if (not vad_t[i]) or (not protect_speech):
                    G[:, i] *= base_vacuum

        BANDS = {
            'clock':     [(2000, 9000)],
            'ticktock':  [(2000, 9000)],
            'keyboard':  [(2000, 6500)],
            'mouseclick':[(2000, 9000)],
            'chewing':   [(180, 1200), (6000, 9500)],
        }
        f_bins = np.linspace(0, sr/2, F)

        thr = float(0.12 if sensitivity in (None, '') else sensitivity)

        MIN_GAIN_OUT = 0.01
        MIN_GAIN_SPEECH = 0.12

        for i in range(T):
            is_speech = bool(vad_t[i])

            if trans_t[i]:
                band = (f_bins >= 2000) & (f_bins <= 9000)
                G[band, i] *= 0.15 if not is_speech else 0.35

            for key, bands in BANDS.items():
                s = float(yam_t.get(key, np.zeros((), np.float32))[i]) if key in yam_t else 0.0
                if s > thr:
                    k = np.clip((s - thr) / max(1e-6, 1.0 - thr), 0.0, 1.0)  # 0..1
                    target_min = MIN_GAIN_SPEECH if (is_speech and protect_speech) else MIN_GAIN_OUT
                    depth = 0.6 + 0.4 * k
                    gain = np.clip(1.0 - depth, target_min, 1.0)
                    for (lo, hi) in bands:
                        band = (f_bins >= lo) & (f_bins <= hi)
                        G[band, i] *= gain

        Z_proc = Z * G
        _, y_out = istft(Z_proc, fs=sr, nperseg=nper, noverlap=nover)

        # Нормализация пика
        y_out = y_out.astype(np.float32)
        peak = float(np.max(np.abs(y_out)) + 1e-9)
        if peak > 0:
            y_out = 0.95 * (y_out / peak)
        return y_out



    def process_file(self, in_path: str, out_path: str, config: PipelineConfig) -> Dict:
        sens = 0.12 if config.trigger_sensitivity in (None, '') else float(config.trigger_sensitivity)
        vacs = 0.9  if config.vacuum_strength    in (None, '') else float(config.vacuum_strength)

        y, sr = _read_wav_mono_float(in_path)

        trig_frames = self._frame_trigger_scores(y, sr) if config.suppress_triggers else {}





        if config.suppress_triggers and self.local and self.local.enabled:
            loc = self._local_frame_scores(y, sr)

            def _stretch(v: np.ndarray, L: int) -> np.ndarray:
                if v.size == 0:
                    return np.zeros(L, dtype=np.float32)
                idx = (np.arange(L) * v.size) // max(v.size, 1)
                idx = np.clip(idx, 0, v.size - 1)
                return v[idx]

            # Сливаем по ключам: если ключ совпадает, берём поэлементный максимум (после выравнивания длины)
            for k, arr in loc.items():
                if k in trig_frames:
                    a, b = trig_frames[k], arr
                    if a.size and b.size:
                        L = max(a.size, b.size)
                        trig_frames[k] = np.maximum(_stretch(a, L), _stretch(b, L))
                    else:
                        trig_frames[k] = a if a.size else b
                else:
                    trig_frames[k] = arr





        if config.base_denoise and HAS_NOISEREDUCE:
            # Лёгкий стационарный гейт до нашей маски — меньше «фона» на входе
            y = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.95)

        speech_flags = self._vad_flags(y, sr, aggressiveness=2) if config.preserve_speech else np.zeros((0,), dtype=np.bool_)

        y = self._process_masking(
            y, sr,
            trig_frames=trig_frames,
            speech_flags=speech_flags,
            sensitivity=sens,
            vacuum_on=config.vacuum_mode,
            vacuum_strength=vacs,
            protect_speech=config.preserve_speech
        )

        _write_wav_int16(out_path, y, sr)
        mean_scores = {k: float(v.mean()) for k, v in trig_frames.items()} if trig_frames else {}
        return {'triggers': mean_scores, 'sr': sr}

_pipeline_singleton = None
def get_pipeline() -> CalmCityPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = CalmCityPipeline()
    return _pipeline_singleton
