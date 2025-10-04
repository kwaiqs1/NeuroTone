import os, json
from typing import Dict
import numpy as np
import tensorflow as tf

class LocalTriggerInfer:
    """
    Keras-модель + TF-мелы, без librosa/soxr.
    Совместима с реалтаймом (push/get_probs/boost).
    """
    def __init__(self, model_dir: str = "models", sr_stream: int = 48000):
        self.model_path   = os.path.join(model_dir, "trigger_cls_keras.h5")
        self.classes_path = os.path.join(model_dir, "trigger_classes.json")
        self.enabled = False

        self.sr = int(sr_stream)
        self.target_sr = 16000
        self.decim = max(1, self.sr // self.target_sr)

        self.buf = np.zeros(self.target_sr, dtype=np.float32)  # 1 сек окно @16k
        self.pos = 0
        self.acc = 0.0
        self.period = 0.5
        self.last_probs: Dict[str, float] = {}

        self.n_mels = 64
        self.hop    = 160
        self.n_fft  = 1024
        self.fmin, self.fmax = 40.0, 8000.0
        self.fix_T  = 120

        if os.path.exists(self.model_path) and os.path.exists(self.classes_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                with open(self.classes_path, "r", encoding="utf-8") as f:
                    self.classes = json.load(f)
                self.enabled = True
            except Exception:
                self.enabled = False

    @staticmethod
    def _power_to_db(S: np.ndarray, ref: float | np.ndarray = 1.0, amin: float = 1e-10, top_db: float = 80.0) -> np.ndarray:
        S = np.maximum(S, amin)
        log_spec = 10.0 * np.log10(S)
        if np.isscalar(ref):
            ref = max(float(ref), amin)  # <-- защита
            log_spec -= 10.0 * np.log10(ref)
        else:
            log_spec -= 10.0 * np.log10(np.maximum(ref, amin))
        if top_db is not None:
            log_spec = np.maximum(log_spec, log_spec.max() - float(top_db))
        return log_spec


    def _mel_from_wav(self, y16: np.ndarray) -> np.ndarray:
        pad = self.n_fft // 2
        ypad = np.pad(y16, (pad, pad), mode="reflect").astype(np.float32)

        frames = tf.signal.stft(
            tf.convert_to_tensor(ypad, dtype=tf.float32),
            frame_length=self.n_fft, frame_step=self.hop,
            window_fn=tf.signal.hann_window, pad_end=False
        )  # [T, F]
        power = tf.math.square(tf.abs(frames))  # [T, F]

        mel_w = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=self.n_fft // 2 + 1,
            sample_rate=self.target_sr,
            lower_edge_hertz=self.fmin,
            upper_edge_hertz=self.fmax
        )  # [F, M]
        mel = tf.matmul(power, mel_w)  # [T, M]
        mel = mel.numpy().astype(np.float32).T  # [M, T]

        S_db = self._power_to_db(mel, ref=np.max(mel))
        mu, std = S_db.mean(), S_db.std() + 1e-6
        S_n = (S_db - mu) / std

        if S_n.shape[1] < self.fix_T:
            pad_t = self.fix_T - S_n.shape[1]
            S_n = np.pad(S_n, ((0,0),(0,pad_t)), mode="constant")
        elif S_n.shape[1] > self.fix_T:
            i0 = (S_n.shape[1] - self.fix_T)//2
            S_n = S_n[:, i0:i0+self.fix_T]
        return S_n.astype(np.float32)

    def _probs(self, wav16: np.ndarray) -> Dict[str, float]:
        if not self.enabled:
            return {}
        S = self._mel_from_wav(wav16)
        x = S[None, ..., None]  # [1, M, T, 1]
        p = self.model.predict(x, verbose=0)[0]  # softmax
        return { self.classes[i]: float(p[i]) for i in range(len(self.classes)) }

    def push(self, x: np.ndarray, sr: int, dt: float):
        if not self.enabled:
            return
        # децимация до 16 кГц (простая, но быстрая)
        x16 = x[::self.decim] if self.decim > 1 else x
        if x16.size == 0:
            return
        n = x16.size
        end = self.pos + n
        if end <= self.buf.size:
            self.buf[self.pos:end] = x16
        else:
            k = self.buf.size - self.pos
            self.buf[self.pos:] = x16[:k]
            self.buf[:n-k] = x16[k:]
        self.pos = (self.pos + n) % self.buf.size
        self.acc += dt
        if self.acc >= self.period:
            self.acc = 0.0
            try:
                wav = np.copy(self.buf)
                self.last_probs = self._probs(wav)
            except Exception:
                pass

    def get_probs(self) -> Dict[str, float]:
        return dict(self.last_probs)

    def boost(self, keys=("chewing","ticktock","keyboard","mouseclick")) -> float:
        if not self.enabled or not self.last_probs:
            return 0.0
        m = max([self.last_probs.get(k, 0.0) for k in keys] + [0.0])
        # картируем 0.3..0.9 -> 0..1
        return float(np.clip((m - 0.3) / 0.6, 0.0, 1.0))
