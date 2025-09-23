# audio/realtime.py
# Реал-тайм обработчик c гибридным шумодавом и гибкой выборкой триггеров (YAMNet).

from __future__ import annotations
import math
import threading
from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional, Set

import numpy as np
from scipy.signal import butter, sosfilt

# --- опциональная YAMNet ---
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    _YAMNET_OK = True
except Exception:
    _YAMNET_OK = False


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


# ----------------- утилиты -----------------
def hz_to_bin(hz: float, nfft: int, sr: int) -> int:
    return int(np.clip(round(hz * nfft / sr), 0, nfft // 2))


def undb(x_db: float) -> float:
    return 10.0 ** (x_db / 20.0)


def sqrt_hann(n: int) -> np.ndarray:
    h = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    return np.sqrt(np.maximum(h, 1e-8)).astype(np.float32)


# ----------------- YAMNet wrapper -----------------
_COMMON_TRIGGERS = [
    "Chewing, mastication", "Lip smacking", "Crunch",
    "Typing", "Computer keyboard", "Mouse click", "Mouse", "Keys jangling",
    "Knock", "Tap", "Clapping", "Tick-tock", "Gulp", "Cough", "Sneeze",
    "Sniff", "Breathing", "Telephone bell ringing", "Alarm clock",
    "Siren", "Buzzer"
]

def _ru_label(en: str) -> str:
    # Очень лёгкий «словари + эвристики», чтобы всё показывалось по-русски.
    # Для неизвестных меток — просто возвращаем исходную строку.
    m = {
        "Chewing, mastication": "Чавканье / пережёвывание",
        "Lip smacking": "Чмокание / лип-шум",
        "Crunch": "Хруст",
        "Typing": "Печать (набор текста)",
        "Computer keyboard": "Клавиатура",
        "Mouse click": "Щелчок мыши",
        "Mouse": "Мышь (звук)",
        "Keys jangling": "Звенящие ключи",
        "Knock": "Стук",
        "Tap": "Постукивание",
        "Clapping": "Хлопок",
        "Tick-tock": "Тик-так",
        "Gulp": "Глоток / сглатывание",
        "Cough": "Кашель",
        "Sneeze": "Чих",
        "Sniff": "Шмыганье",
        "Breathing": "Дыхание",
        "Telephone bell ringing": "Звонок телефона",
        "Alarm clock": "Будильник",
        "Siren": "Сирена",
        "Buzzer": "Зуммер / противный писк",
        "Speech": "Речь",
        "Whispering": "Шёпот",
        "Chatter": "Гул голосов",
        "Footsteps": "Шаги",
        "Door": "Дверь",
        "Doorbell": "Дверной звонок",
    }
    return m.get(en, en)

class YamnetTrigger:
    """
    Лёгкая RT-обёртка:
    - Буфер ~1 c при 16 кГц, раз в 0.5 c обновляет вероятности.
    - Список таргетов можно менять на лету (из UI).
    """
    def __init__(self, sr_stream: int, selected: Optional[Iterable[str]] = None):
        self.enabled = _YAMNET_OK
        self.lock = threading.Lock()
        self.last_probs: Dict[str, float] = {}
        self.targets: Set[str] = set(selected) if selected else set(_COMMON_TRIGGERS)
        self._acc_t = 0.0
        self._period = 0.5
        self._buf = np.zeros(16000, dtype=np.float32)
        self._pos = 0
        self._decim = max(1, int(sr_stream // 16000))
        if self.enabled:
            try:
                self._yam = hub.load("https://tfhub.dev/google/yamnet/1")
                path = self._yam.class_map_path().numpy().decode("utf-8")
                labels_txt = tf.io.read_file(path).numpy().decode("utf-8")
                self._labels = [ln.strip() for ln in labels_txt.splitlines() if ln.strip()]
            except Exception:
                self.enabled = False

    def set_targets(self, selected: Iterable[str]):
        with self.lock:
            self.targets = set(selected)

    def available_labels(self) -> List[str]:
        # Все классы YAMNet (если доступен), иначе — минимальный набор
        if self.enabled:
            return list(self._labels)
        return list(_COMMON_TRIGGERS)

    def push(self, x: np.ndarray, sr: int, dt: float):
        if not self.enabled:
            return
        x16 = x[::self._decim] if self._decim > 1 else x
        if not x16.size:
            return
        n = x16.size
        end = self._pos + n
        if end <= self._buf.size:
            self._buf[self._pos:end] = x16
        else:
            k = self._buf.size - self._pos
            self._buf[self._pos:] = x16[:k]
            self._buf[:n - k] = x16[k:]
        self._pos = (self._pos + n) % self._buf.size
        self._acc_t += dt
        if self._acc_t < self._period:
            return
        self._acc_t = 0.0
        try:
            wav = np.copy(self._buf).astype(np.float32)
            scores, _, _ = self._yam(wav)
            p = np.mean(scores.numpy(), axis=0)
            top: Dict[str, float] = {}
            for i, pr in enumerate(p):
                name = self._labels[i] if i < len(self._labels) else str(i)
                if name in self.targets:
                    top[name] = float(pr)
            with self.lock:
                self.last_probs = top
        except Exception:
            pass

    def boost(self) -> float:
        if not self.enabled:
            return 0.0
        with self.lock:
            if not self.last_probs:
                return 0.0
            m = max(self.last_probs.values())
        return float(np.clip((m - 0.1) / 0.4, 0.0, 1.0))


# ----------------- основной DSP -----------------
class RealTimeProcessor:
    def __init__(self, cfg: RTConfig, stream_sr: int = 48000, blocksize: int = 480,
                 selected_triggers: Optional[Iterable[str]] = None):
        self.cfg = cfg
        self.sr = int(np.clip(stream_sr, SR_SAFE_MIN, SR_SAFE_MAX))

        # STFT: ~40 мс окно, 50% overlap
        target_win_ms = 40.0
        nfft = int(2 ** math.ceil(math.log2(self.sr * target_win_ms / 1000.0)))
        nfft = int(np.clip(nfft, 1024, 4096))
        self.nfft = nfft
        self.hop = nfft // 2
        self.win = sqrt_hann(nfft)
        self.eps = 1e-8

        # Пред/пост фильтры
        self.hpf = butter(2, 60 / (self.sr / 2), btype='highpass', output='sos')
        lpf_cut = min(16000, int(self.sr * 0.45))
        self.lpf = butter(4, lpf_cut / (self.sr / 2), btype='lowpass', output='sos')

        # Буферы STFT-OLA
        self._inbuf = np.zeros(0, dtype=np.float32)
        self._prev_tail = np.zeros(self.nfft - self.hop, dtype=np.float32)
        self._outbuf = np.zeros(0, dtype=np.float32)

        # Шум/маска
        self.noise_psd = np.ones(self.nfft // 2 + 1, dtype=np.float32) * 1e-6
        self.alpha_noise = 0.92
        self.prev_gain = np.ones_like(self.noise_psd)

        # VAD/Transient
        self.vad_hist = 0.0
        self.last_mag = np.zeros(self.nfft // 2 + 1, dtype=np.float32)

        # Комфорт-шум
        rng = np.random.default_rng(123)
        w = rng.standard_normal(self.nfft).astype(np.float32)
        self.comfort = (w / (np.max(np.abs(w)) + self.eps) * undb(-62)).astype(np.float32)

        # YAMNet
        self.yam = YamnetTrigger(self.sr, selected_triggers)

    # --- helpers ---
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

    # --- один STFT кадр ---
    def _process_frame(self, frame: np.ndarray, p_speech: float, trig_boost: float) -> np.ndarray:
        X = np.fft.rfft(frame * self.win)
        mag = np.abs(X).astype(np.float32)
        phase = np.angle(X).astype(np.float32)

        upd = self.alpha_noise if p_speech > 0.4 else 0.80
        self.noise_psd = upd * self.noise_psd + (1.0 - upd) * (mag ** 2)

        y = mag.copy()

        if self.cfg.enable_denoise:
            base_alpha = 2.2 + (1.6 if self.cfg.ultra_anc else 0.0) + 1.6 * trig_boost
            noise = np.maximum(self.noise_psd, 1e-12)
            snr = np.maximum((mag ** 2) / noise - 1.0, 0.0)
            wiener = snr / (snr + 1.0 + 1e-9)

            over = base_alpha * (1.0 - 0.55 * p_speech)
            gain = np.clip(wiener - over * np.sqrt(noise) / (mag + 1e-9), 0.0, 1.0)

            # transient killer (HF)
            hf_lo = hz_to_bin(1800, self.nfft, self.sr)
            crest = np.maximum(0.0, (mag - (0.88 * self.last_mag + 1e-6)) / (self.last_mag + 1e-6))
            tk = np.zeros_like(mag)
            tk[hf_lo:] = np.clip(crest[hf_lo:] - (0.6 - self.cfg.trigger_sensitivity), 0.0, 1.0)
            self.last_mag = 0.9 * self.last_mag + 0.1 * mag
            gain *= (1.0 - 0.85 * tk)

            # защита речи
            if self.cfg.protect_speech and p_speech > 0.15:
                lo, hi = hz_to_bin(250, self.nfft, self.sr), hz_to_bin(3800, self.nfft, self.sr)
                protect = 0.3 + 0.7 * (1.0 - self.cfg.trigger_sensitivity)
                gain[lo:hi] = np.maximum(gain[lo:hi], protect * 0.4)

            # сглаживание маски
            self.prev_gain = 0.75 * self.prev_gain + 0.25 * gain
            gain = self.prev_gain

            # ограничение глубины
            min_gain = undb(-38.0 - 10.0 * float(self.cfg.vacuum_strength))
            gain = np.clip(gain, min_gain, 1.0)

            y *= gain

        # вакуум
        if self.cfg.enable_vacuum:
            vac = np.clip(self.cfg.vacuum_strength, 0.0, 1.0)
            att = 16.0 + 16.0 * vac + 8.0 * trig_boost
            post = undb(-att) ** (1.0 - p_speech)
            y *= post

        # пол (анти-дззз)
        floor = undb(-62.0)
        y = np.maximum(y, floor * np.sqrt(self.noise_psd))

        Y = y * np.exp(1j * phase)
        y_time = np.fft.irfft(Y).real.astype(np.float32)
        y_time = sosfilt(self.lpf, y_time)
        return y_time

    # --- публичный блок ---
    def process_block(self, mono_in: np.ndarray) -> np.ndarray:
        x = mono_in.astype(np.float32)
        x = sosfilt(self.hpf, x)
        x = np.clip(x, -1.0, 1.0)

        dt = len(x) / float(self.sr)
        self.yam.push(x, self.sr, dt)
        trig_boost = self.yam.boost() if self.cfg.enable_triggers else 0.0
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
            y_frame = self._process_frame(frame, p_speech, trig_boost)

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
            self._outbuf = self._outbuf[need:] if self._outbuf.size >= need else np.zeros(0, dtype=np.float32)
        elif y.size > total_len:
            y = y[:total_len]

        return np.clip(y, -1.0, 1.0)

    # ---- сервис для UI ----
    def available_trigger_labels(self) -> List[Dict[str, str]]:
        labels = self.yam.available_labels()
        return [{"id": lab, "label": lab, "ru": _ru_label(lab)} for lab in labels]

    def set_selected_triggers(self, labels: Iterable[str]):
        self.yam.set_targets(labels)
