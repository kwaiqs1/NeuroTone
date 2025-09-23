# audio/realtime.py
# Реал-тайм обработка: гибридный шумодав + удаление триггеров с защитой речи.
from __future__ import annotations
import math
import threading
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
from scipy.signal import butter, sosfilt

# YAMNet (опционально). Если хаб не доступен – работаем без него.
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
    trigger_sensitivity: float = 0.08  # 0.02..0.30, ниже — агрессивнее
    vacuum_strength: float = 1.0       # 0..1, мощность приглушения вне речи
    protect_speech: bool = True
    enable_triggers: bool = True
    enable_vacuum: bool = True
    enable_denoise: bool = True
    ultra_anc: bool = True             # +10..15 dB к оверсабтракшн


# --------- утилиты ---------
def hz_to_bin(hz: float, nfft: int, sr: int) -> int:
    return int(np.clip(round(hz * nfft / sr), 0, nfft // 2))


def db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(1e-12, x))


def undb(x_db: float) -> float:
    return 10.0 ** (x_db / 20.0)


def hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


# --------- опциональная YAMNet ---------
class YamnetTrigger:
    """
    Низкобюджетная RT-обёртка над YAMNet:
    - буферит ~1.0 сек 16кГц моно и раз в 0.5 сек обновляет вероятности.
    - если TF/Hub недоступны — работает в no-op режиме.
    """
    def __init__(self, sr_stream: int):
        self.enabled = _YAMNET_OK
        self.lock = threading.Lock()
        self.last_probs: Dict[str, float] = {}
        self.target_labels = {
            # ключевые триггеры
            "Chewing, mastication",
            "Crunch",
            "Lip smacking",
            "Typing",
            "Computer keyboard",
            "Mouse click",
            "Tick-tock",
            "Knock",
            "Clapping",
            "Tap",
            "Keys jangling",
        }
        self._t = 0
        self._every = 0.5  # сек, как часто обновляем
        self._buf16 = np.zeros(16000, dtype=np.float32)
        self._pos = 0
        self._decim = max(1, int(sr_stream // 16000))

        if self.enabled:
            try:
                self._yam = hub.load("https://tfhub.dev/google/yamnet/1")
                self._class_names = self._yam.class_map_path().numpy().decode("utf-8")
                # загрузим список имён
                txt = tf.io.read_file(self._class_names).numpy().decode("utf-8")
                self._labels = [line.strip() for line in txt.splitlines() if line.strip()]
            except Exception:
                self.enabled = False

    def push(self, x: np.ndarray, sr_stream: int, dt_sec: float):
        if not self.enabled:
            return
        # даунсэмплим до 16кГц дешёвым способом (pick decim)
        x16 = x[::self._decim] if self._decim > 1 else x
        if x16.size == 0:
            return
        # кольцевой буфер 1 сек
        n = x16.size
        end = self._pos + n
        if end <= self._buf16.size:
            self._buf16[self._pos:end] = x16
        else:
            k = self._buf16.size - self._pos
            self._buf16[self._pos:] = x16[:k]
            self._buf16[:n - k] = x16[k:]
        self._pos = (self._pos + n) % self._buf16.size

        self._t += dt_sec
        if self._t < self._every:
            return
        self._t = 0.0

        # YAMNet ждёт float32 16кГц
        try:
            wav = np.copy(self._buf16).astype(np.float32)
            scores, _, _ = self._yam(wav)
            p = np.mean(scores.numpy(), axis=0)  # усредним по фреймам
            top = {}
            for idx, prob in enumerate(p):
                name = self._labels[idx] if idx < len(self._labels) else str(idx)
                if name in self.target_labels:
                    top[name] = float(prob)
            with self.lock:
                self.last_probs = top
        except Exception:
            # если что-то пошло не так — просто игнор
            pass

    def trigger_boost(self) -> float:
        """0..1 — насколько сильно давим из-за YAMNet."""
        if not self.enabled:
            return 0.0
        with self.lock:
            if not self.last_probs:
                return 0.0
            # берём максимум по таргетам
            m = max(self.last_probs.values())
        # плавное усиление: >0.1 уже заметно, >0.3 сильно
        return float(np.clip((m - 0.1) / 0.4, 0.0, 1.0))


# --------- основной процессор ---------
class RealTimeProcessor:
    def __init__(self, cfg: RTConfig, stream_sr: int = 48000, blocksize: int = 480):
        self.cfg = cfg
        self.sr = int(np.clip(stream_sr, SR_SAFE_MIN, SR_SAFE_MAX))
        self.bs = int(blocksize)

        # FFT окно (2х блок + округление до степени 2)
        nwin = max(256, int(2 * self.bs))
        self.nfft = int(2 ** math.ceil(math.log2(nwin)))
        self.hop = self.bs
        self.win = hann(self.nfft).astype(np.float32)
        self.olap = self.nfft - self.hop
        self._ola = np.zeros(self.olap, dtype=np.float32)  # overlap buffer

        # фильтры:
        # HPF ~ 60 Гц, чтобы убрать инфраниз
        self.hpf = butter(2, 60 / (self.sr / 2), btype='highpass', output='sos')
        # LPF 9 кГц (для агрессивного ANC и Bluetooth каналов)
        self.lpf = butter(4, 9000 / (self.sr / 2), btype='lowpass', output='sos')

        # Оценка шума (спектральная) и сглаживание
        self.noise_psd = np.ones(self.nfft // 2 + 1, dtype=np.float32) * 1e-6
        self.alpha_noise = 0.92  # скорость следования шуму

        # VAD простейший (энергетический + диапазон 200..4к)
        self.vad_hist = 0.0

        # Комфорт-шум (pink-ish): генерим при инициализации
        rng = np.random.default_rng(123)
        white = rng.standard_normal(self.nfft).astype(np.float32)
        # 1/f shaping
        freqs = np.fft.rfftfreq(self.nfft, 1 / self.sr)
        shape = 1.0 / np.maximum(1.0, np.sqrt(freqs + 1.0))
        cn_spec = np.fft.rfft(white) * shape
        self.comfort = (np.fft.irfft(cn_spec).real / 50.0).astype(np.float32)  # ≈ −34 dBfs

        # YAMNet (опционально)
        self.yam = YamnetTrigger(self.sr)

        # сглаживание для transient killer
        self.last_mag = np.zeros(self.nfft // 2 + 1, dtype=np.float32)

    # --------- вспомогательные оценки ---------
    def _energy_band(self, x: np.ndarray, lo_hz=200, hi_hz=4000) -> float:
        # энергия полосы (для VAD)
        X = np.fft.rfft(x * self.win[:x.size], n=self.nfft)
        mag = np.abs(X)
        i0 = hz_to_bin(lo_hz, self.nfft, self.sr)
        i1 = hz_to_bin(hi_hz, self.nfft, self.sr)
        e = float(np.mean(mag[i0:i1] ** 2))
        return e

    def _vad(self, x: np.ndarray) -> float:
        # простая энергия + сглаживание
        e = self._energy_band(x)
        # нормируем на медиану шума
        noise_level = float(np.median(self.noise_psd))
        snr_est = e / (noise_level + 1e-12)
        p = 1.0 if snr_est > 20.0 else (snr_est / 20.0)  # 0..1
        # сгладим
        self.vad_hist = 0.85 * self.vad_hist + 0.15 * p
        return float(np.clip(self.vad_hist, 0.0, 1.0))

    # --------- основной шаг ---------
    def process_block(self, mono_in: np.ndarray) -> np.ndarray:
        x = mono_in.astype(np.float32)

        # базовая предобработка
        x = sosfilt(self.hpf, x)
        x = np.clip(x, -1.0, 1.0)

        # подадим во вспомогательные детекторы
        dt = len(x) / float(self.sr)
        self.yam.push(x, self.sr, dt)
        p_trig_yam = self.yam.trigger_boost() if self.cfg.enable_triggers else 0.0

        # VAD
        p_speech = self._vad(x) if self.cfg.protect_speech else 0.0

        # STFT
        frame = np.zeros(self.nfft, dtype=np.float32)
        frame[:len(x)] = x
        frame *= self.win
        X = np.fft.rfft(frame)
        mag = np.abs(X).astype(np.float32)
        phase = np.angle(X).astype(np.float32)

        # оценка шума (обновляем сильнее, когда речи нет)
        update_rate = self.alpha_noise if p_speech > 0.4 else 0.80
        self.noise_psd = update_rate * self.noise_psd + (1.0 - update_rate) * (mag ** 2)

        y = mag.copy()

        if self.cfg.enable_denoise:
            # Винер с оверсабтракшн
            alpha = 3.5 if self.cfg.ultra_anc else 2.2
            # усиливаем при обнаруженных триггерах
            alpha += 2.0 * p_trig_yam

            noise = np.maximum(self.noise_psd, 1e-12)
            snr = np.maximum((mag ** 2) / noise - 1.0, 0.0)
            wiener = snr / (snr + 1.0 + 1e-9)

            # оверсабтракшн (чуть сильнее вне речи)
            over = alpha * (1.0 - 0.6 * p_speech)
            gain = np.clip(wiener - over * np.sqrt(noise) / (mag + 1e-9), 0.0, 1.0)

            # transient killer — ловим резкие всплески в ВЧ
            hf_lo = hz_to_bin(1800, self.nfft, self.sr)
            crest = np.maximum(0.0, (mag - (0.85 * self.last_mag + 1e-6)) / (self.last_mag + 1e-6))
            tk = np.zeros_like(mag)
            tk[hf_lo:] = np.clip(crest[hf_lo:] - (0.6 - self.cfg.trigger_sensitivity), 0.0, 1.0)
            self.last_mag = 0.85 * self.last_mag + 0.15 * mag

            # усилим глушение там, где tk высок
            gain *= (1.0 - 0.9 * tk)

            # защита речи: ослабляем подавление в 250..3800 Гц
            if self.cfg.protect_speech and p_speech > 0.1:
                lo = hz_to_bin(250, self.nfft, self.sr)
                hi = hz_to_bin(3800, self.nfft, self.sr)
                protect = 0.25 + 0.75 * (1.0 - self.cfg.trigger_sensitivity)  # 0.25..1
                gain[lo:hi] = np.maximum(gain[lo:hi], protect * 0.35)

            # применим
            y *= gain

        # Вакуум вне речи (после денойза)
        if self.cfg.enable_vacuum:
            vac = np.clip(self.cfg.vacuum_strength, 0.0, 1.0)
            # вне речи — −(18..32) дБ, но плавно
            att_db = 18.0 + 14.0 * vac + 10.0 * p_trig_yam
            post_gain = undb(-att_db) ** (1.0 - p_speech)
            y *= post_gain

        # чтобы избежать «дззз», инжектим очень тихий комфорт-шум в подавленных частотах
        floor = undb(-60.0)  # −60 dB
        y = np.maximum(y, floor * np.sqrt(self.noise_psd))

        # обратно в временную область
        Y = y * np.exp(1j * phase)
        out_frame = np.fft.irfft(Y).real
        # LPF для BT-артефактов
        out_frame = sosfilt(self.lpf, out_frame).astype(np.float32)

        # overlap-add: выдаём ровно len(x) с учётом накопленного хвоста
        out = out_frame[:len(x)]
        if self._ola.size:
            k = min(len(out), self._ola.size)
            out[:k] += self._ola[:k]
            # обновим ola буфер
            tail = out_frame[len(x):len(x) + self._ola.size]
            if tail.size < self._ola.size:
                buf = np.zeros_like(self._ola)
                buf[:tail.size] = tail
                self._ola = buf
            else:
                self._ola = tail.astype(np.float32)

        # нормализация на всякий
        out = np.clip(out, -1.0, 1.0)

        return out
