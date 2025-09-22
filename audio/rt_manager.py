# audio/rt_manager.py
# Управление одной real-time сессией (start/stop/status) в фоне.
# Использует RealTimeProcessor из audio.realtime и sounddevice Stream (duplex).

import threading
import time
from typing import Optional, Dict, Any

import numpy as np
import sounddevice as sd

from .realtime import RealTimeProcessor, RTConfig


class RTSessionManager:
    """Singleton-менеджер одной RT-сессии."""
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.Stream] = None
        self._running: bool = False
        self._status: Dict[str, Any] = {"running": False}
        self._last_error: Optional[str] = None
        self._proc: Optional[RealTimeProcessor] = None
        self._params: Dict[str, Any] = {}

    @classmethod
    def instance(cls) -> "RTSessionManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = RTSessionManager()
        return cls._instance

    # ---------- Devices ----------
    @staticmethod
    def list_devices() -> Dict[str, Any]:
        devs = sd.query_devices()
        hostapis = sd.query_hostapis()
        def host_name(idx):
            try:
                return hostapis[idx]["name"]
            except Exception:
                return "Unknown"

        out = []
        for i, d in enumerate(devs):
            out.append({
                "id": i,
                "name": d.get("name"),
                "hostapi": host_name(d.get("hostapi", 0)),
                "max_input_channels": d.get("max_input_channels", 0),
                "max_output_channels": d.get("max_output_channels", 0),
                "default_samplerate": d.get("default_samplerate", None),
            })
        return {"devices": out, "default_input": sd.default.device[0], "default_output": sd.default.device[1]}

    # ---------- Start/Stop ----------
    def start(self,
              input_id: int,
              output_id: int,
              sensitivity: float = 0.08,
              vacuum: float = 1.0,
              block: int = 480,
              protect_speech: bool = True,
              enable_triggers: bool = True,
              enable_vacuum: bool = True,
              ultra_anc: bool = True,
              samplerate: Optional[int] = None) -> Dict[str, Any]:

        self.stop()  # Остановим, если что-то уже работало

        # Подбор каналов: пробуем mono (1), иначе 2
        indev = sd.query_devices(input_id)
        outdev = sd.query_devices(output_id)
        in_max = int(indev.get("max_input_channels", 0))
        out_max = int(outdev.get("max_output_channels", 0))

        if in_max >= 1 and out_max >= 1:
            channels = 1
        elif in_max >= 2 and out_max >= 2:
            channels = 2
        else:
            raise RuntimeError("Выбранные устройства не совместимы по числу каналов.")

        # Частота дискретизации
        sr = int(samplerate or (outdev.get("default_samplerate") or 48000))
        # Если устройство не тянет sr — попробуем fallback на 48000 → 44100
        for try_sr in [sr, 48000, 44100]:
            try:
                sd.check_input_settings(device=input_id, channels=channels, samplerate=try_sr)
                sd.check_output_settings(device=output_id, channels=channels, samplerate=try_sr)
                sr = try_sr
                break
            except Exception:
                sr = None
        if sr is None:
            raise RuntimeError("Не удалось подобрать допустимую частоту дискретизации для пары устройств.")

        cfg = RTConfig(
            trigger_sensitivity=sensitivity,
            vacuum_strength=vacuum,
            protect_speech=bool(protect_speech),
            enable_triggers=bool(enable_triggers),
            enable_vacuum=bool(enable_vacuum),
            enable_denoise=True,
            ultra_anc=bool(ultra_anc),
        )

        self._params = {
            "input_id": input_id, "output_id": output_id,
            "channels": channels, "samplerate": sr, "block": int(block),
            "cfg": cfg.__dict__,
        }

        proc = RealTimeProcessor(cfg, stream_sr=sr, blocksize=int(block))
        self._proc = proc
        self._running = True
        self._last_error = None
        self._status = {"running": True, "params": self._params, "frames": 0}

        def _cb(indata, outdata, frames, t, status):
            try:
                if status:
                    self._status["callback_status"] = str(status)
                # Вход → mono
                if self._params["channels"] == 2 and indata.shape[1] >= 2:
                    x = indata[:, :2].mean(axis=1).astype(np.float32)
                else:
                    x = indata[:, 0].astype(np.float32) if indata.ndim == 2 else indata.astype(np.float32)

                y = proc.process_block(x)
                # Выравниваем длину под frames
                if len(y) < frames:
                    # паддинг нулями (задержка на старте)
                    buf = np.zeros(frames, dtype=np.float32)
                    buf[:len(y)] = y
                    y = buf
                elif len(y) > frames:
                    y = y[-frames:]

                if self._params["channels"] == 2 and outdata.shape[1] >= 2:
                    outdata[:, 0] = y
                    outdata[:, 1] = y
                else:
                    outdata[:, 0] = y

                self._status["frames"] = self._status.get("frames", 0) + frames
            except Exception as e:
                self._last_error = str(e)
                raise

        def _run():
            try:
                with sd.Stream(
                    device=(input_id, output_id),
                    samplerate=self._params["samplerate"],
                    blocksize=self._params["block"],
                    dtype="float32",
                    channels=self._params["channels"],
                    callback=_cb,
                    latency="low",
                    clip_off=True,
                    dither_off=True,
                    never_drop_input=True,
                ) as st:
                    self._stream = st
                    while self._running:
                        time.sleep(0.1)
            except Exception as e:
                self._last_error = str(e)
            finally:
                self._running = False
                self._stream = None

        th = threading.Thread(target=_run, daemon=True)
        self._thread = th
        th.start()
        return self.status()

    def stop(self) -> Dict[str, Any]:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._proc = None
        self._status = {"running": False}
        return self.status()

    def status(self) -> Dict[str, Any]:
        s = dict(self._status)
        s["running"] = bool(self._running)
        s["last_error"] = self._last_error
        return s
