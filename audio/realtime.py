# audio/realtime.py
# Ultra ANC + Ultra Trigger Kill (YAMNet + PANNs), hi-fi real-time.

import math, threading, time, collections
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
from scipy.signal import resample_poly, iirnotch, sosfilt, sosfilt_zi, tf2sos
from scipy.signal.windows import hann
from scipy.fft import rfft, irfft

import webrtcvad
import tensorflow as tf
import tensorflow_hub as hub

# --- PyTorch/PANNs ---
try:
    import torch
    from panns_inference import AudioTagging, labels
    TORCH_OK = True
except Exception:
    TORCH_OK = False


# ===== Config =================================================================

@dataclass
class RTConfig:
    trigger_sensitivity: float = 0.08    # ниже => агрессивнее (0.06–0.12)
    vacuum_strength: float = 1.0         # 0..1
    protect_speech: bool = True
    enable_triggers: bool = True
    enable_vacuum: bool = True
    enable_denoise: bool = True
    ultra_anc: bool = True
    trigger_hold_ms: int = 500
    transient_gate: bool = True


# ===== Utils ==================================================================

def _resample_to(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return y.astype(np.float32)
    g = math.gcd(sr_from, sr_to)
    up, down = sr_to // g, sr_from // g
    return resample_poly(y, up, down).astype(np.float32)


class RingBuffer:
    def __init__(self, seconds: float, sr: int):
        self.sr = sr
        self.maxlen = int(seconds * sr)
        self.buf = collections.deque(maxlen=self.maxlen)
    def push(self, x: np.ndarray):
        self.buf.extend(x.astype(np.float32).tolist())
    def tail_seconds(self, seconds: float) -> Optional[np.ndarray]:
        n = min(len(self.buf), int(seconds * self.sr))
        if n <= 0: return None
        return np.fromiter(list(self.buf)[-n:], dtype=np.float32)


# ===== AI Detectors ============================================================

class YAMNetDetector(threading.Thread):
    """TF-Hub YAMNet — быстрый теггер; обновляет флаги self.flags раз в 200 мс."""
    def __init__(self, ring: RingBuffer, sensitivity: float):
        super().__init__(daemon=True)
        self.ring = ring
        self.sens = float(sensitivity)
        self.flags = {"clock": False, "keyboard": False, "chewing": False}
        if not getattr(tf, "__version__", None):
            tf.__version__ = "2.15.0"
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = self.model.class_map_path().numpy().decode("utf-8")
        with tf.io.gfile.GFile(class_map_path, "r") as f:
            import csv
            self.class_map = [row["display_name"] for row in csv.DictReader(f)]
        self.cls_words = {
            'chewing':  ['Chewing','Mastication','Mouth sounds'],
            'clock':    ['Tick-tock','Mechanical','Clock','Click'],
            'keyboard': ['Typing','Keyboard'],
        }
    def run(self):
        while True:
            audio = self.ring.tail_seconds(1.0)
            if audio is not None and len(audio) >= self.ring.sr // 2:
                y16 = _resample_to(audio, self.ring.sr, 16000)
                scores, _, _ = self.model(y16)
                S = scores.numpy()
                for key, words in self.cls_words.items():
                    idxs = [i for i, name in enumerate(self.class_map)
                            if any(w.lower() in name.lower() for w in words)]
                    val = float(S[:, idxs].max()) if idxs else 0.0
                    self.flags[key] = (val > self.sens)
            time.sleep(0.20)


class PANNsDetector(threading.Thread):
    """PANNs CNN14 (PyTorch) — усиливаем надёжность теггера. Работает, если torch установлен."""
    def __init__(self, ring: RingBuffer, sensitivity: float):
        super().__init__(daemon=True)
        self.enabled = TORCH_OK
        self.ring = ring
        self.sens = float(sensitivity) * 0.8  # чуть чувствительнее YAMNet
        self.flags = {"clock": False, "keyboard": False, "chewing": False}
        if self.enabled:
            self.model = AudioTagging(checkpoint_path=None, device='cpu')  # качает веса с интернетов 1й раз
            # маппинг классов
            self.map = {
                'chewing':  ['Chewing','Mastication','Mouth sounds'],
                'clock':    ['Tick-tock','Clock','Clicking'],
                'keyboard': ['Typing','Keyboard'],
            }
    def run(self):
        if not self.enabled:
            return
        while True:
            audio = self.ring.tail_seconds(1.0)
            if audio is not None and len(audio) >= self.ring.sr // 2:
                y16 = _resample_to(audio, self.ring.sr, 32000)  # PANNs любит 32k
                clipwise_output, embedding = self.model.inference(y16)
                probs = clipwise_output[0]
                names = labels
                for key, words in self.map.items():
                    p = 0.0
                    for w in words:
                        # найдём ближайшее имя
                        idxs = [i for i,n in enumerate(names) if w.lower() in n.lower()]
                        if idxs:
                            p = max(p, float(probs[idxs].max()))
                    self.flags[key] = (p > self.sens)
            time.sleep(0.25)


# ===== STFT Processor ==========================================================

class OLAProcessor:
    """Streaming STFT/ISTFT c 50% overlap, Hann, COLA; сглаживание усиления; трекинг шума; transient-гейт."""
    def __init__(self, sr: int, frame: int = 1024):
        self.sr = sr
        self.N = int(frame)
        self.H = self.N // 2
        self.win = hann(self.N, sym=False).astype(np.float32)
        self.inbuf = np.zeros(0, dtype=np.float32)
        self.tail = np.zeros(self.N - self.H, dtype=np.float32)

        self.freqs = np.fft.rfftfreq(self.N, 1.0 / self.sr)
        self.noise_psd = np.ones(self.N // 2 + 1, dtype=np.float32) * 1e-3

        # сглаживание усиления
        self.attack = 0.55
        self.release = 0.92
        self.prev_gain = np.ones(self.N // 2 + 1, dtype=np.float32)

        # транзиенты
        self.prev_mag = np.zeros(self.N // 2 + 1, dtype=np.float32)

    def _gain_smooth(self, g: np.ndarray) -> np.ndarray:
        p = self.prev_gain
        go_down = g < p
        out = np.where(go_down, self.attack * p + (1 - self.attack) * g,
                       self.release * p + (1 - self.release) * g)
        self.prev_gain = out
        return out

    def _wiener_gain(self, Pxx: np.ndarray, speech: bool, vacuum: float, ultra_anc: bool) -> np.ndarray:
        if not speech:
            alpha = 0.96
            self.noise_psd = alpha * self.noise_psd + (1 - alpha) * Pxx

        snr = np.maximum(Pxx / (self.noise_psd + 1e-12) - 1.0, 0.0)
        g = snr / (snr + 1.0)

        floor_speech = 10 ** (-10 / 20)   # ~ -10 dB в речи
        floor_nons   = 10 ** (-28 / 20)   # ~ -28 dB вне речи
        floor = floor_speech if speech else floor_nons

        if not speech and vacuum > 0.0:
            vac = max(0.0, min(1.0, vacuum))
            lows  = self.freqs < 150
            highs = self.freqs > 7500
            g[lows | highs] *= (1.0 - 0.7 * vac)   # сильнее прижимаем края
            if ultra_anc:
                mids = (self.freqs >= 600) & (self.freqs <= 3000)
                g[mids] *= (1.0 - 0.35 * vac)      # мягко садим середину

        g = np.clip(g, floor, 1.0)
        return self._gain_smooth(g)

    def _transient_mask(self, mag: np.ndarray) -> float:
        band = (self.freqs >= 2000) & (self.freqs <= 9000)
        diff = np.maximum(mag - self.prev_mag, 0.0)
        flux = float(diff[band].sum() / (self.prev_mag[band].sum() + 1e-9))
        self.prev_mag = mag
        if flux > 0.6: return 1.0
        if flux > 0.35: return 0.5
        return 0.0

    def process_push(self, x: np.ndarray, speech_flag: bool, vacuum_strength: float, ultra_anc: bool) -> np.ndarray:
        self.inbuf = np.concatenate([self.inbuf, x.astype(np.float32)])
        out = np.zeros(0, dtype=np.float32)

        while len(self.inbuf) >= self.N:
            frame = self.inbuf[:self.N]
            self.inbuf = self.inbuf[self.H:]

            X = rfft(frame * self.win)
            mag = np.abs(X).astype(np.float32)
            Pxx = (mag ** 2)

            tr_power = self._transient_mask(mag)
            G = self._wiener_gain(Pxx, speech_flag, vacuum_strength, ultra_anc)

            if tr_power > 0 and not speech_flag:
                hf = self.freqs > 2500
                G[hf] *= (0.25 if tr_power >= 1.0 else 0.5)

            Y = X * G
            y = irfft(Y).astype(np.float32)

            yw = y * self.win
            head = yw[:self.N - self.H] + self.tail
            self.tail = yw[self.N - self.H:]
            out = np.concatenate([out, head])

        return out


# ===== Real-time Processor =====================================================

BANDS = {
    "clock":    [(2000, 9000)],
    "keyboard": [(2000, 6500)],
    "chewing":  [(180, 1200), (6000, 9500)],
}

class RealTimeProcessor:
    def __init__(self, cfg: RTConfig, stream_sr: int = 48000, blocksize: int = 480, hard_mute_triggers: bool = False):
        self.cfg = cfg
        self.sr = int(stream_sr)
        self.frame = int(2 * blocksize)
        if self.frame % 2 == 1: self.frame += 1
        self.stft = OLAProcessor(self.sr, frame=self.frame)

        # VAD
        self.vad = webrtcvad.Vad(2)
        self.vad_hist = collections.deque(maxlen=3)

        # буферы
        self.ring = RingBuffer(2.0, self.sr)

        # нотчи
        self.sos_bank, self.sos_state = self._build_notches()

        # AI детекторы
        self.det_yam = None
        self.det_pann = None
        if cfg.enable_triggers:
            self.det_yam = YAMNetDetector(self.ring, cfg.trigger_sensitivity)
            self.det_yam.start()
            if TORCH_OK:
                self.det_pann = PANNsDetector(self.ring, cfg.trigger_sensitivity)
                self.det_pann.start()

        # hold по триггерам
        self.hold_samples = int((cfg.trigger_hold_ms / 1000.0) * self.sr)
        self.active_until = {"clock": 0, "keyboard": 0, "chewing": 0}
        self.sample_counter = 0

        self.hard_mute = bool(hard_mute_triggers)

        # подготовим индексы полос
        self._band_bins = {}
        freqs = np.fft.rfftfreq(self.stft.N, 1.0 / self.sr)
        for k, ranges in BANDS.items():
            idxs = []
            for lo, hi in ranges:
                idxs.append(np.where((freqs >= lo) & (freqs <= hi))[0])
            self._band_bins[k] = np.unique(np.concatenate(idxs) if idxs else np.array([], dtype=int))

    def _build_notches(self):
        layout = [
            (3000, 30), (6000, 30),     # clock
            (3500, 25),                 # keyboard
            (500, 15), (7000, 15),      # chewing
        ]
        sos_list, zi_list = [], []
        for f0, Q in layout:
            b, a = iirnotch(w0=f0, Q=Q, fs=self.sr)
            sos = tf2sos(b, a)
            sos_list.append(sos)
            zi_list.append(sosfilt_zi(sos) * 0.0)
        return np.vstack(sos_list), zi_list

    def _vad_is_speech(self, block: np.ndarray) -> bool:
        y16 = _resample_to(block, self.sr, 16000)
        need = 320
        fr = np.zeros(need, dtype=np.int16)
        n = min(len(y16), need)
        fr[:n] = np.clip(y16[:n] * 32767, -32768, 32767).astype(np.int16)
        try:
            return self.vad.is_speech(fr.tobytes(), 16000)
        except Exception:
            return False

    def _merged_trigger_flags(self) -> Dict[str, bool]:
        flags = {"clock": False, "keyboard": False, "chewing": False}
        if self.det_yam:
            for k in flags: flags[k] = flags[k] or bool(self.det_yam.flags.get(k, False))
        if self.det_pann and TORCH_OK:
            for k in flags: flags[k] = flags[k] or bool(self.det_pann.flags.get(k, False))
        return flags

    def _apply_notches_and_hardmute(self, x_time: np.ndarray, speech: bool, use_flags: Dict[str,bool]) -> np.ndarray:
        y = x_time
        # i) IIR нотчи (мягко в речи, глубже вне речи)
        depth_in_speech = 0.30   # ~ -8.5 dB
        depth_out       = 0.02   # ~ -34 dB
        use_idx = []
        if use_flags.get("clock", False):    use_idx += [0, 1]
        if use_flags.get("keyboard", False): use_idx += [2]
        if use_flags.get("chewing", False):  use_idx += [3, 4]
        for i in use_idx:
            y2, self.sos_state[i] = sosfilt(self.sos_bank[i:i+1], y, zi=self.sos_state[i])
            mix = depth_in_speech if speech else depth_out
            y = y2*(1.0 - mix) + y*mix

        # ii) hard-mute в частотных полосах (вне речи = ноль; в речи = сильное ослабление)
        if any(use_flags.values()):
            # STFT одного окна для точечного вмешательства
            N = self.stft.N
            H = self.stft.H
            win = self.stft.win
            if len(y) < N:
                pad = np.zeros(N, dtype=np.float32)
                pad[:len(y)] = y
                frame = pad
            else:
                frame = y[:N]
            X = rfft(frame * win)
            # построим маску
            G = np.ones_like(X, dtype=np.float32)
            for k, active in use_flags.items():
                if not active: continue
                bins = self._band_bins.get(k, [])
                if bins.size == 0: continue
                if speech:
                    G[bins] *= 10 ** (-22/20)   # ~ -22 dB в речи
                else:
                    if self.hard_mute:
                        G[bins] = 0.0           # В НОЛЬ
                    else:
                        G[bins] *= 10 ** (-35/20)
            Y = X * G
            y_f = irfft(Y).astype(np.float32)
            y[:min(len(y), N)] = y_f[:min(len(y), N)]
        return y

    def process_block(self, in_block: np.ndarray) -> np.ndarray:
        self.sample_counter += len(in_block)
        self.ring.push(in_block.astype(np.float32))

        speech_now = self._vad_is_speech(in_block) if self.cfg.protect_speech else False
        # небольшое сглаживание голоса
        if not hasattr(self, "vad_hist"): self.vad_hist = collections.deque(maxlen=3)
        self.vad_hist.append(speech_now)
        speech = (sum(self.vad_hist) >= 2)

        # STFT-ANC/денойз
        out = in_block
        if self.cfg.enable_denoise:
            out = self.stft.process_push(
                in_block,
                speech_flag=speech,
                vacuum_strength=self.cfg.vacuum_strength if self.cfg.enable_vacuum else 0.0,
                ultra_anc=self.cfg.ultra_anc and (not speech)
            )
            if len(out) == 0:
                return in_block

        # триггеры: объединённые флаги + hold
        trig_now = self._merged_trigger_flags() if self.cfg.enable_triggers else {"clock":False,"keyboard":False,"chewing":False}
        now = self.sample_counter
        for k, v in trig_now.items():
            if v:
                # продлеваем подавление
                self.active_until[k] = now + int((self.cfg.trigger_hold_ms / 1000.0) * self.sr)

        active = {k: (now < t) for k, t in self.active_until.items()}
        out2 = self._apply_notches_and_hardmute(out, speech=speech, use_flags=active)

        # мягкий лимитер
        peak = float(np.max(np.abs(out2)) + 1e-9)
        if peak > 0.98:
            out2 = 0.98 * (out2 / peak)
        return out2.astype(np.float32)
