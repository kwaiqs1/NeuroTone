# audio/realtime.py

from __future__ import annotations
import math
import threading
from dataclasses import dataclass
from typing import Optional, Dict
import csv

import numpy as np
from scipy.signal import butter, sosfilt

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    _YAMNET_OK = True
except Exception:
    _YAMNET_OK = False


try:
    from audio.learn.trigger_infer import LocalTriggerInfer
    _LOCAL_OK = True
except Exception:
    _LOCAL_OK = False

SR_SAFE_MIN = 16000
SR_SAFE_MAX = 96000

@dataclass
class RTConfig:
    trigger_sensitivity: float = 0.08
    vacuum_strength: float = 1.0
    protect_speech: bool = True
    enable_triggers: bool = True
    enable_vacuum: bool = True
    enable_denoise: bool = True
    ultra_anc: bool = True
    hard_mute_triggers: bool = True
    hard_mute_db: float = 70.0
    noise_floor_db: float = 80.0


def hz_to_bin(hz: float, nfft: int, sr: int) -> int:
    return int(np.clip(round(hz * nfft / sr), 0, nfft // 2))

def undb(x_db: float) -> float:
    return 10.0 ** (x_db / 20.0)

def sqrt_hann(n: int) -> np.ndarray:
    h = np.hanning(n).astype(np.float32)
    h = np.sqrt(np.maximum(h, 1e-8))
    return h

class YamnetTrigger:
    def __init__(self, sr_stream: int):
        self.enabled = _YAMNET_OK
        self.lock = threading.Lock()
        self.last_probs: Dict[str, float] = {}
        self.targets_exact = {
            "Chewing, mastication", "Lip smacking",
            "Typing", "Computer keyboard", "Mouse click",
            "Tick-tock", "Knock", "Tap", "Clapping", "Keys jangling",
        }
        self.targets_sub = {"chewing","smacking","typing","keyboard","mouse click","tick-tock","knock","tap","clapping","keys"}
        self._acc_t = 0.0
        self._period = 0.5
        self._buf = np.zeros(16000, dtype=np.float32)
        self._pos = 0
        self._decim = max(1, int(sr_stream // 16000))
        if self.enabled:
            try:
                self._yam = hub.load("https://tfhub.dev/google/yamnet/1")
                path = self._yam.class_map_path().numpy().decode("utf-8")
                with tf.io.gfile.GFile(path, "r") as f:
                    reader = csv.DictReader(f)
                    self._labels = [(row.get("display_name") or row.get("name") or "").strip() for row in reader]
            except Exception:
                self.enabled = False



    def _label_is_target(self, name: str) -> bool:
        if not name: return False
        if name in self.targets_exact: return True
        low = name.lower()
        return any(s in low for s in self.targets_sub)

    def push(self, x: np.ndarray, sr: int, dt: float):
        if not self.enabled: return
        x16 = x[::self._decim] if self._decim > 1 else x
        if not x16.size: return
        n = x16.size
        end = self._pos + n
        if end <= self._buf.size:
            self._buf[self._pos:end] = x16
        else:
            k = self._buf.size - self._pos
            self._buf[self._pos:] = x16[:k]
            self._buf[:n-k] = x16[k:]
        self._pos = (self._pos + n) % self._buf.size
        self._acc_t += dt
        if self._acc_t < self._period: return
        self._acc_t = 0.0
        try:
            wav = np.copy(self._buf).astype(np.float32)
            scores, _, _ = self._yam(wav)
            p = np.mean(scores.numpy(), axis=0)
            top = {}
            for i, pr in enumerate(p):
                name = self._labels[i] if i < len(self._labels) else str(i)
                if self._label_is_target(name):
                    top[name] = float(pr)
            with self.lock:
                self.last_probs = top
        except Exception:
            pass

    def boost(self) -> float:
        if not self.enabled: return 0.0
        with self.lock:
            if not self.last_probs: return 0.0
            m = max(self.last_probs.values())
        return float(np.clip((m - 0.1) / 0.4, 0.0, 1.0))

class RealTimeProcessor:
    def __init__(self, cfg: RTConfig, stream_sr: int = 48000, blocksize: int = 480):
        self.cfg = cfg
        self.sr = int(np.clip(stream_sr, SR_SAFE_MIN, SR_SAFE_MAX))
        target_win_ms = 40.0
        nfft = int(2 ** math.ceil(math.log2(self.sr * target_win_ms / 1000.0)))
        nfft = int(np.clip(nfft, 1024, 4096))
        self.nfft = nfft
        self.hop = nfft // 2
        self.win = sqrt_hann(nfft)
        self.eps = 1e-8

        self.hpf = butter(2, 60 / (self.sr / 2), btype='highpass', output='sos')
        lpf_cut = min(16000, int(self.sr * 0.45))
        self.lpf = butter(4, lpf_cut / (self.sr / 2), btype='lowpass', output='sos')

        self._inbuf = np.zeros(0, dtype=np.float32)
        self._prev_tail = np.zeros(self.nfft - self.hop, dtype=np.float32)
        self._outbuf = np.zeros(0, dtype=np.float32)

        self.noise_psd = np.ones(self.nfft // 2 + 1, dtype=np.float32) * 1e-6
        self.alpha_noise = 0.92
        self.prev_gain = np.ones_like(self.noise_psd)

        self.vad_hist = 0.0

        self.yam = YamnetTrigger(self.sr)
        self.local = LocalTriggerInfer(sr_stream=self.sr) if _LOCAL_OK else None
        self.last_mag = np.zeros(self.nfft // 2 + 1, dtype=np.float32)

    def _energy_band(self, x: np.ndarray, lo=200, hi=4000) -> float:
        X = np.fft.rfft(x * self.win[:x.size], n=self.nfft)
        mag = np.abs(X)
        i0, i1 = hz_to_bin(lo, self.nfft, self.sr), hz_to_bin(hi, self.nfft, self.sr)
        return float(np.mean(mag[i0:i1] ** 2))

    def _vad(self, x: np.ndarray) -> float:
        e = self._energy_band(x)
        noise_lvl = float(np.median(self.noise_psd))
        snr = e / (noise_lvl + 1e-12)
        p = 1.0 if snr > 20.0 else (snr / 20.0)
        self.vad_hist = 0.9 * self.vad_hist + 0.1 * p
        return float(np.clip(self.vad_hist, 0.0, 1.0))

    def _process_frame(self, frame: np.ndarray, p_speech: float,
                       yam_boost: float, local_probs: Dict[str,float]) -> np.ndarray:
        X = np.fft.rfft(frame * self.win)
        mag = np.abs(X).astype(np.float32)
        phase = np.angle(X).astype(np.float32)

        upd = self.alpha_noise if p_speech > 0.4 else 0.80
        self.noise_psd = upd * self.noise_psd + (1.0 - upd) * (mag ** 2)

        y = mag.copy()

        if self.cfg.enable_denoise:
            loc_boost = 0.0
            if local_probs:
                m = max(local_probs.get(k,0.0) for k in ["chewing","ticktock","keyboard","mouseclick"] + ["beep"])
                loc_boost = float(np.clip((m - 0.3) / 0.6, 0.0, 1.0))
            trig_boost = max(yam_boost, loc_boost)

            base_alpha = 2.2 + (1.6 if self.cfg.ultra_anc else 0.0) + 1.6 * trig_boost
            noise = np.maximum(self.noise_psd, 1e-12)
            snr = np.maximum((mag ** 2) / noise - 1.0, 0.0)
            wiener = snr / (snr + 1.0 + 1e-9)

            over = base_alpha * (1.0 - 0.55 * p_speech)
            gain = np.clip(wiener - over * np.sqrt(noise) / (mag + 1e-9), 0.0, 1.0)


            hf_lo = hz_to_bin(1800, self.nfft, self.sr)
            crest = np.maximum(0.0, (mag - (0.88 * self.last_mag + 1e-6)) / (self.last_mag + 1e-6))
            tk = np.zeros_like(mag)
            tk[hf_lo:] = np.clip(crest[hf_lo:] - (0.55 - self.cfg.trigger_sensitivity), 0.0, 1.0)
            self.last_mag = 0.9 * self.last_mag + 0.1 * mag
            gain *= (1.0 - 0.88 * tk)




            hard_min = undb(-self.cfg.hard_mute_db)
            def cut_band(lo, hi, depth=0.6, hard=False):
                i0, i1 = hz_to_bin(lo, self.nfft, self.sr), hz_to_bin(hi, self.nfft, self.sr)
                i1 = min(i1, gain.size - 1)
                if i1 <= i0:
                    return
                if hard:
                    gain[i0:i1+1] = np.minimum(gain[i0:i1+1], hard_min)
                else:
                    d = np.clip(depth, 0.0, 0.95)
                    gain[i0:i1+1] *= (1.0 - d)

            p_chew  = local_probs.get("chewing", 0.0) if local_probs else 0.0
            p_click = max(local_probs.get("ticktock",0.0),
                          local_probs.get("keyboard",0.0),
                          local_probs.get("mouseclick",0.0)) if local_probs else 0.0
            p_beep  = local_probs.get("beep", 0.0) if local_probs else 0.0

            thr_soft = 0.45
            thr_hard = 0.55
            allow_hard_now = self.cfg.hard_mute_triggers and not (self.cfg.protect_speech and p_speech > 0.15)


            if p_chew > thr_soft:
                hard = allow_hard_now and (p_chew >= thr_hard)
                cut_band(180, 1200, depth=0.55 * (1.0 + trig_boost), hard=hard)
                cut_band(6000, 9000, depth=0.50 * (1.0 + trig_boost), hard=hard)

            if p_click > thr_soft:
                hard = allow_hard_now and (p_click >= thr_hard)
                cut_band(2000, 9000, depth=0.60 * (1.0 + trig_boost), hard=hard)

            important = (p_beep >= 0.5)




            if local_probs:
                if local_probs.get("chewing", 0.0) >= 0.5:
                    cut_band(180, 1200, depth=0.55 * (1.0 + trig_boost))
                    cut_band(6000, 9000, depth=0.50 * (1.0 + trig_boost))
                if max(local_probs.get("ticktock",0.0), local_probs.get("keyboard",0.0), local_probs.get("mouseclick",0.0)) >= 0.45:
                    cut_band(2000, 9000, depth=0.60 * (1.0 + trig_boost))

                important = local_probs.get("beep", 0.0) >= 0.5
            else:
                important = False


            if self.cfg.protect_speech and p_speech > 0.15:
                lo, hi = hz_to_bin(250, self.nfft, self.sr), hz_to_bin(3800, self.nfft, self.sr)
                protect = 0.3 + 0.7 * (1.0 - self.cfg.trigger_sensitivity)
                gain[lo:hi] = np.maximum(gain[lo:hi], protect * 0.4)

            self.prev_gain = 0.75 * self.prev_gain + 0.25 * gain
            gain = self.prev_gain

            min_gain = undb(-38.0 - 10.0 * float(self.cfg.vacuum_strength))
            gain = np.clip(gain, min_gain, 1.0)
            y *= gain

        if self.cfg.enable_vacuum:
            vac = np.clip(self.cfg.vacuum_strength, 0.0, 1.0)
            att = (16.0 + 16.0 * vac) * (0.6 if (local_probs and local_probs.get("beep",0.0)>=0.5) else 1.0)
            if local_probs:
                b = max(local_probs.get(k,0.0) for k in ["chewing","ticktock","keyboard","mouseclick"])
                att += 8.0 * float(np.clip((b - 0.3)/0.6, 0.0, 1.0))
            post = undb(-att) ** (1.0 - p_speech)
            y *= post

        floor = undb(-self.cfg.noise_floor_db)
        y = np.maximum(y, floor * np.sqrt(self.noise_psd))

        Y = y * np.exp(1j * phase)
        y_time = np.fft.irfft(Y).real.astype(np.float32)
        y_time = sosfilt(self.lpf, y_time)
        return y_time

    def process_block(self, mono_in: np.ndarray) -> np.ndarray:
        x = mono_in.astype(np.float32)
        x = sosfilt(self.hpf, x)
        x = np.clip(x, -1.0, 1.0)

        dt = len(x) / float(self.sr)
        self.yam.push(x, self.sr, dt)
        yam_boost = self.yam.boost() if self.cfg.enable_triggers else 0.0

        local_probs = {}
        if self.cfg.enable_triggers and self.local and self.local.enabled:
            self.local.push(x, self.sr, dt)
            local_probs = self.local.get_probs()

        p_speech_now = self._vad(x) if self.cfg.protect_speech else 0.0

        self._inbuf = np.concatenate([self._inbuf, x])
        out_chunks = []

        while self._inbuf.size >= self.hop:
            frame = np.empty(self.nfft, dtype=np.float32)
            frame[:self.nfft - self.hop] = self._prev_tail
            new = self._inbuf[:self.hop]
            frame[self.nfft - self.hop:] = new
            self._prev_tail = frame[self.hop:]
            self._inbuf = self._inbuf[self.hop:]

            p_speech = 0.7 * self.vad_hist + 0.3 * p_speech_now

            y_frame = self._process_frame(frame, p_speech, yam_boost, local_probs)

            a = y_frame[:self.hop] * self.win[:self.hop]
            b = y_frame[self.hop:] * self.win[self.hop:]

            if self._outbuf.size < self.hop:
                pad = np.zeros(self.hop - self._outbuf.size, dtype=np.float32)
                self._outbuf = np.concatenate([self._outbuf, pad])

            mixed = (self._outbuf[:self.hop] + a)
            out_chunks.append(mixed.astype(np.float32))
            self._outbuf = b.copy()

        total_len = len(x)
        if out_chunks:
            y = np.concatenate(out_chunks)
        else:
            y = np.zeros(0, dtype=np.float32)

        if y.size < total_len:
            need = total_len - y.size
            tail = self._outbuf[:need] if self._outbuf.size >= need else np.pad(self._outbuf, (0, need - self._outbuf.size))
            y = np.concatenate([y, tail])
            if self._outbuf.size >= need:
                self._outbuf = self._outbuf[need:]
            else:
                self._outbuf = np.zeros(0, dtype=np.float32)
        elif y.size > total_len:
            y = y[:total_len]

        peak = float(np.max(np.abs(y)) + 1e-9)
        if peak > 1.0:
            y = y / peak
        return np.clip(y, -1.0, 1.0)
