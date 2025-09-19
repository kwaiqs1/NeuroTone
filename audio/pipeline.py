import io
import json
from dataclasses import dataclass
from typing import Dict, Tuple


import numpy as np
import librosa
import soundfile as sf
from scipy.signal import iirnotch, lfilter

import tensorflow as tf
import tensorflow_hub as hub


try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except Exception:
    HAS_NOISEREDUCE = False

YAMNET_HANDLE = 'https://tfhub.dev/google/yamnet/1'



TRIGGER_KEYWORDS = {
    'chewing': ['Chewing', 'Mastication', 'Mouth sounds'],
    'clock': ['Tick-tock', 'Mechanical', 'Clock', 'Click'],
    'keyboard': ['Typing', 'Keyboard'],
}


@dataclass
class PipelineConfig:
    base_denoise: bool = True
    suppress_triggers: bool = True
    trigger_sensitivity: float = 0.25


class CalmCityPipeline:
    def __init__(self):
        self.model = hub.load(YAMNET_HANDLE) # downloads on first run
        self.class_map = self._load_class_map()


    def _load_class_map(self):
        # Pull class map from the model assets
        # model.class_map_path().numpy() is supported by yamnet SavedModel
        class_map_path = self.model.class_map_path().numpy().decode('utf-8')
        with tf.io.gfile.GFile(class_map_path, 'r') as f:
            import csv
            reader = csv.DictReader(f)
            names = [row['display_name'] for row in reader]
        return names


    def _resample_mono(self, y: np.ndarray, sr: int, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        if y.ndim > 1:
            y = librosa.to_mono(y)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        return y.astype(np.float32), target_sr


    def detect_triggers(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        # YAMNet expects mono 16 kHz float32
        y16, _ = self._resample_mono(y, sr, 16000)
        scores, embeddings, spectrogram = self.model(y16)
        scores_np = scores.numpy()
        mean_scores = scores_np.mean(axis=0)  # average over time frames

        # map keywords → max score among matching classes
        result = {}
        for key, words in TRIGGER_KEYWORDS.items():
            idxs = [i for i, name in enumerate(self.class_map) if any(w.lower() in name.lower() for w in words)]
            val = float(np.max(mean_scores[idxs])) if idxs else 0.0
            result[key] = val
        return result


    def base_soft_denoise(self, y: np.ndarray, sr: int) -> np.ndarray:
        # Gentle denoise: either spectral gating via noisereduce, or light lowpass shelf if unavailable
        if HAS_NOISEREDUCE:
            # estimate noise from the first 0.5s (or shorter if clip is short)
            nref = min(len(y), int(sr * 0.5))
            noise_profile = y[:nref]
            return nr.reduce_noise(y=y, y_noise=noise_profile, sr=sr, prop_decrease=0.5)
        # Fallback: do nothing major; return original
        return y


    def dynamic_trigger_suppression(self, y: np.ndarray, sr: int, trig: Dict[str, float], sensitivity: float) -> np.ndarray:
        out = y.copy()
        # Simple rules: if a trigger score exceeds sensitivity, apply a gentle notch or de-esser band
        def apply_notch(signal, f0, q=20.0, gain=0.8):
            b, a = iirnotch(w0=f0 / (sr / 2), Q=q)
            filtered = lfilter(b, a, signal)
            return (gain * filtered + (1 - gain) * signal)

        if trig.get('clock', 0) > sensitivity:
            # Clock ticks around 2–8 kHz; notch at 3 kHz and 6 kHz
            out = apply_notch(out, 3000, q=30, gain=0.7)
            out = apply_notch(out, 6000, q=30, gain=0.7)
        if trig.get('keyboard', 0) > sensitivity:
            # Typing clicks ~2–4 kHz
            out = apply_notch(out, 3500, q=25, gain=0.75)
        if trig.get('chewing', 0) > sensitivity:
            # Chewing / mouth sounds are low-mid and transient; notch ~500 Hz + mild high-shelf de-essing around 5–8k
            out = apply_notch(out, 500, q=15, gain=0.8)
            out = apply_notch(out, 7000, q=15, gain=0.75)
        return out


    def process_file(self, in_path: str, out_path: str, config: PipelineConfig) -> Dict:
        y, sr = librosa.load(in_path, sr=None, mono=False)
        if y.ndim > 1:
            y = librosa.to_mono(y)
        # YAMNet trigger detection
        trig = self.detect_triggers(y, sr) if config.suppress_triggers else {}

        # Base denoise first
        if config.base_denoise:
            y = self.base_soft_denoise(y, sr)

        # Dynamic trigger suppression
        if config.suppress_triggers:
            y = self.dynamic_trigger_suppression(y, sr, trig, sensitivity=config.trigger_sensitivity)

        # normalize to -1..1 (soft)
        peak = np.max(np.abs(y)) + 1e-9
        y = 0.95 * y / peak

        sf.write(out_path, y, sr)
        return {
            'triggers': trig,
            'sr': sr,
        }


# Convenience function
_pipeline_singleton = None


def get_pipeline() -> CalmCityPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = CalmCityPipeline()
    return _pipeline_singleton