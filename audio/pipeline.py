# audio/pipeline.py
import json
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import iirnotch, lfilter, resample_poly

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
    trigger_sensitivity: float = 0.25  # lower = more aggressive

class CalmCityPipeline:
    def __init__(self):
        self.model = hub.load(YAMNET_HANDLE)
        self.class_map = self._load_class_map()

    def _load_class_map(self):
        class_map_path = self.model.class_map_path().numpy().decode('utf-8')
        with tf.io.gfile.GFile(class_map_path, 'r') as f:
            import csv
            reader = csv.DictReader(f)
            names = [row['display_name'] for row in reader]
        return names

    def _to_mono(self, y: np.ndarray) -> np.ndarray:
        if y.ndim == 2:
            return y.mean(axis=1)
        return y

    def _resample(self, y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
        if sr == target_sr:
            return y.astype(np.float32)
        g = np.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        return resample_poly(y, up, down).astype(np.float32)

    def detect_triggers(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        y16 = self._resample(self._to_mono(y), sr, 16000)
        scores, _, _ = self.model(y16)
        mean_scores = scores.numpy().mean(axis=0)

        result = {}
        for key, words in TRIGGER_KEYWORDS.items():
            idxs = [i for i, name in enumerate(self.class_map)
                    if any(w.lower() in name.lower() for w in words)]
            result[key] = float(np.max(mean_scores[idxs])) if idxs else 0.0
        return result

    def base_soft_denoise(self, y: np.ndarray, sr: int) -> np.ndarray:
        if HAS_NOISEREDUCE:
            nref = min(len(y), int(sr*0.5))
            return nr.reduce_noise(y=y, y_noise=y[:nref], sr=sr, prop_decrease=0.5)
        return y

    def dynamic_trigger_suppression(self, y: np.ndarray, sr: int,
                                   trig: Dict[str, float], sensitivity: float) -> np.ndarray:
        out = y.copy()

        def apply_notch(signal, f0, q=20.0, gain=0.8):
            b, a = iirnotch(w0=f0/(sr/2), Q=q)
            filtered = lfilter(b, a, signal)
            return (gain * filtered + (1-gain) * signal)

        if trig.get('clock', 0) > sensitivity:
            out = apply_notch(out, 3000, q=30, gain=0.7)
            out = apply_notch(out, 6000, q=30, gain=0.7)
        if trig.get('keyboard', 0) > sensitivity:
            out = apply_notch(out, 3500, q=25, gain=0.75)
        if trig.get('chewing', 0) > sensitivity:
            out = apply_notch(out, 500, q=15, gain=0.8)
            out = apply_notch(out, 7000, q=15, gain=0.75)
        return out

    def process_file(self, in_path: str, out_path: str, config: PipelineConfig) -> Dict:
        y, sr = sf.read(in_path, always_2d=False)
        y = self._to_mono(y).astype(np.float32)

        trig = self.detect_triggers(y, sr) if config.suppress_triggers else {}

        if config.base_denoise:
            y = self.base_soft_denoise(y, sr)

        if config.suppress_triggers:
            y = self.dynamic_trigger_suppression(y, sr, trig, sensitivity=config.trigger_sensitivity)

        peak = np.max(np.abs(y)) + 1e-9
        y = 0.95 * y / peak

        sf.write(out_path, y, sr)
        return {'triggers': trig, 'sr': sr}

_pipeline_singleton = None
def get_pipeline() -> CalmCityPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = CalmCityPipeline()
    return _pipeline_singleton
