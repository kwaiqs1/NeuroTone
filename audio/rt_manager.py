# audio/rt_manager.py
# Надёжный запуск real-time аудио под Windows:
# WASAPI shared, перебор форматов, совместимость как в CLI,
# и безопасный внутренний blocksize для DSP (без деления на ноль).

import threading
import time
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import sounddevice as sd

from .realtime import RealTimeProcessor, RTConfig


class RTSessionManager:
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

    # ---------- devices ----------
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

    # ---------- helpers ----------
    @staticmethod
    def _host_name_of(device_id: int) -> str:
        api_idx = sd.query_devices(device_id).get("hostapi", 0)
        return sd.query_hostapis()[api_idx].get("name", "Unknown")

    @staticmethod
    def _mk_wasapi_shared(device_id: int):
        name = RTSessionManager._host_name_of(device_id).lower()
        if "wasapi" in name:
            return sd.WasapiSettings(exclusive=False)  # только shared
        return None

    def _candidate_samplerates(self, indev, outdev, user_sr: Optional[int]) -> List[int]:
        s: List[int] = []
        if user_sr:
            s.append(int(user_sr))
        for v in [outdev.get("default_samplerate"), indev.get("default_samplerate"), 48000, 44100]:
            if v and int(v) not in s:
                s.append(int(v))
        return s

    def _candidate_channel_pairs(self, in_max: int, out_max: int) -> List[Tuple[int, int]]:
        # приоритет — максимально совместимый моно/моно (как в CLI), затем мионо→стерео и т.д.
        pairs: List[Tuple[int, int]] = []
        if in_max >= 1 and out_max >= 1:
            pairs.append((1, 1))
        if in_max >= 1 and out_max >= 2:
            pairs.append((1, 2))
        if in_max >= 2 and out_max >= 2:
            pairs.append((2, 2))
        if in_max >= 2 and out_max >= 1:
            pairs.append((2, 1))
        seen = set(); out: List[Tuple[int, int]] = []
        for p in pairs:
            if p not in seen:
                out.append(p); seen.add(p)
        if not out:
            raise RuntimeError("Выбранные устройства несовместимы по числу каналов.")
        return out

    def _pick_working_combo(
        self,
        input_id: int,
        output_id: int,
        want_block: int,
        user_sr: Optional[int]
    ) -> Tuple[int, Tuple[int, int], int, Tuple[str, str], Optional[tuple]]:
        """Подбираем (sr, (in_ch,out_ch), block, (in_dtype,out_dtype), extra_settings)."""
        indev = sd.query_devices(input_id)
        outdev = sd.query_devices(output_id)

        ch_pairs = self._candidate_channel_pairs(
            int(indev.get("max_input_channels", 0)),
            int(outdev.get("max_output_channels", 0)),
        )
        sr_list = self._candidate_samplerates(indev, outdev, user_sr)
        # 0 первым — безопасно для драйвера (ОС выберет размер)
        blocks = [0, int(want_block), 960, 1024, 2048, 512, 256]
        dtype_pairs = [("float32", "float32"), ("int16", "int16")]

        # Только WASAPI shared (если доступен), иначе без extra_settings
        ex_in = self._mk_wasapi_shared(input_id)
        ex_out = self._mk_wasapi_shared(output_id)
        extra_variants = [((ex_in, ex_out) if (ex_in or ex_out) else (None, None))]

        last_err = None

        def _dummy_cb(indata, outdata, frames, t, status):
            if outdata is not None:
                outdata[:] = 0

        for (in_ch, out_ch) in ch_pairs:
            for sr in sr_list:
                for bs in blocks:
                    for (dt_in, dt_out) in dtype_pairs:
                        for ex_in_v, ex_out_v in extra_variants:
                            kwargs = dict(
                                device=(input_id, output_id),
                                samplerate=int(sr),
                                blocksize=int(bs),
                                callback=_dummy_cb,
                                channels=(int(in_ch), int(out_ch)),
                                dtype=(dt_in, dt_out),
                            )
                            if ex_in_v is not None or ex_out_v is not None:
                                kwargs["extra_settings"] = (ex_in_v, ex_out_v)
                            try:
                                test = sd.Stream(**kwargs)
                                test.close()
                                return (int(sr), (int(in_ch), int(out_ch)), int(bs), (dt_in, dt_out),
                                        ((ex_in_v, ex_out_v) if (ex_in_v or ex_out_v) else None))
                            except Exception as e:
                                last_err = e
                                continue

        # --- Резерв: как в CLI (channels=1, float32, blocksize=0, без extra_settings)
        try:
            def _cb_compat(indata, outdata, frames, t, status):
                if outdata is not None:
                    outdata[:] = 0
            sr_cli = int(user_sr or outdev.get("default_samplerate") or 48000)
            test = sd.Stream(device=(input_id, output_id),
                             samplerate=sr_cli, blocksize=0,
                             channels=1, dtype="float32",
                             callback=_cb_compat)
            test.close()
            return sr_cli, (1, 1), 0, ("float32", "float32"), None
        except Exception as e:
            last_err = e

        raise RuntimeError(f"Не удалось открыть аудиопоток: {last_err}")

    # ---------- control ----------
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

        self.stop()

        sr, (in_ch, out_ch), bs, (dt_in, dt_out), extra = self._pick_working_combo(
            input_id=input_id, output_id=output_id,
            want_block=block, user_sr=samplerate
        )

        # === ВАЖНО: внутренний размер блока для DSP (исправление "float division by zero") ===
        # Если драйвер оставил blocksize=0, берём безопасный внутренний размер ~20 мс.
        proc_block = int(sr * 0.02) if bs == 0 else int(bs)
        if proc_block < 240:  # не даём слишком маленький блок
            proc_block = 240

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
            "samplerate": sr, "block": bs,
            "proc_block": proc_block,
            "in_channels": in_ch, "out_channels": out_ch,
            "in_dtype": dt_in, "out_dtype": dt_out,
            "cfg": cfg.__dict__,
            "hostapi_in": self._host_name_of(input_id),
            "hostapi_out": self._host_name_of(output_id),
            "extra_mode": ("wasapi-shared" if extra else "none"),
        }

        proc = RealTimeProcessor(cfg, stream_sr=sr, blocksize=proc_block)
        self._proc = proc
        self._running = True
        self._last_error = None
        self._status = {"running": True, "params": self._params, "frames": 0}

        def _cb(indata, outdata, frames, t, status):
            try:
                if status:
                    self._status["callback_status"] = str(status)

                # вход -> моно на обработку
                if indata.ndim == 2 and indata.shape[1] > 1:
                    x = indata[:, :2].mean(axis=1).astype(np.float32)
                else:
                    x = indata[:, 0].astype(np.float32) if indata.ndim == 2 else indata.astype(np.float32)

                y = proc.process_block(x)

                # выравниваем длину
                if len(y) < frames:
                    buf = np.zeros(frames, dtype=np.float32); buf[:len(y)] = y; y = buf
                elif len(y) > frames:
                    y = y[-frames:]

                if outdata is not None:
                    if outdata.ndim == 2 and outdata.shape[1] >= 2:
                        outdata[:, 0] = y; outdata[:, 1] = y
                    else:
                        outdata[:, 0] = y

                self._status["frames"] = self._status.get("frames", 0) + frames
            except Exception as e:
                self._last_error = str(e)
                raise

        def _run():
            try:
                kwargs = dict(
                    device=(input_id, output_id),
                    samplerate=sr,
                    blocksize=bs,  # для драйвера можно оставить 0 — это ок
                    channels=(in_ch, out_ch),
                    dtype=(dt_in, dt_out),
                    callback=_cb,
                )
                if extra:
                    kwargs["extra_settings"] = extra  # только wasapi-shared
                with sd.Stream(**kwargs) as st:
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
                self._stream.stop(); self._stream.close()
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
