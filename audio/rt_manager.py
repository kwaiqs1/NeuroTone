# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import sounddevice as sd

from .realtime import RealTimeProcessor, RTConfig

# Безопасные значения по умолчанию
DEFAULT_SR_CANDIDATES = [48000, 44100]
DEFAULT_BLOCK = 480
DEFAULT_CHANNELS = 1  # см. комментарий в start(): один канал для duplex Stream


def _hostapis() -> List[str]:
    try:
        ha = sd.query_hostapis()
        return [h.get("name", f"hostapi#{i}") for i, h in enumerate(ha)]
    except Exception:
        return []


def _fmt_label(idx: int, dev: Dict[str, Any], hostapis_names: List[str]) -> str:
    h = hostapis_names[dev.get("hostapi", 0)] if hostapis_names else ""
    ins, outs = int(dev.get("max_input_channels", 0)), int(dev.get("max_output_channels", 0))
    name = dev.get("name", f"device#{idx}")
    tail = f"(in:{ins}, out:{outs})"
    hstr = f" — {h}" if h else ""
    return f"[{idx}] {name}{hstr} {tail}"


@dataclass
class _State:
    stream: Optional[sd.Stream] = None
    proc: Optional[RealTimeProcessor] = None
    frames: int = 0
    running: bool = False
    sr: Optional[int] = None
    block: Optional[int] = None
    channels: Optional[int] = None
    last_error: Optional[str] = None
    started_at: Optional[float] = None


class RTManager:
    def __init__(self) -> None:
        self.s = _State()

    # ---------- devices ----------
    def list_devices(self) -> Dict[str, Any]:
        try:
            devs = sd.query_devices()
            hostapis_names = _hostapis()

            inputs, outputs = [], []
            for i, d in enumerate(devs):
                lab = _fmt_label(i, d, hostapis_names)
                if int(d.get("max_input_channels", 0)) > 0:
                    inputs.append({"index": i, "label": lab})
                if int(d.get("max_output_channels", 0)) > 0:
                    outputs.append({"index": i, "label": lab})

            di, do = sd.default.device
            return {
                "ok": True,
                "inputs": inputs,
                "outputs": outputs,
                "default_in": int(di) if di is not None else (inputs[0]["index"] if inputs else None),
                "default_out": int(do) if do is not None else (outputs[0]["index"] if outputs else None),
            }
        except Exception as e:
            return {"ok": False, "error": f"list_devices: {e}"}

    # ---------- start/stop ----------
    def start(
        self,
        in_index: int,
        out_index: int,
        samplerate: Optional[int] = None,
        blocksize: Optional[int] = None,
        channels: Optional[int] = None,
        cfg_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # аккуратно останавливаем, если уже запущено
        if self.s.stream is not None:
            self.stop()

        # резолвим параметры
        sr = int(samplerate) if samplerate else None
        if sr is None:
            # пробуем взять дефолт устройства, если не подходит — fallback
            try:
                idef = sd.query_devices(in_index).get("default_samplerate")
                odef = sd.query_devices(out_index).get("default_samplerate")
                cand = [int(idef) if idef else None, int(odef) if odef else None]
                sr = next((c for c in cand if c), None)
            except Exception:
                sr = None
        if sr is None:
            for c in DEFAULT_SR_CANDIDATES:
                sr = c
                break

        block = int(blocksize) if blocksize else DEFAULT_BLOCK
        ch = int(channels) if channels else DEFAULT_CHANNELS  # duplex Stream требует одинаковые in/out

        # конфиг realtime процессора
        cfg = RTConfig(
            trigger_threshold=float(cfg_dict.get("trigger_threshold", 0.10)) if cfg_dict else 0.10,
            vacuum_strength=float(cfg_dict.get("vacuum_strength", 1.0)) if cfg_dict else 1.0,
            protect_speech=bool(cfg_dict.get("protect_speech", True)) if cfg_dict else True,
            trigger_kill=bool(cfg_dict.get("trigger_kill", True)) if cfg_dict else True,
            ultra_anc=bool(cfg_dict.get("ultra_anc", False)) if cfg_dict else False,
            selected_triggers=cfg_dict.get("selected_triggers", []) if cfg_dict else [],
        )
        proc = RealTimeProcessor(cfg)

        def _cb(indata, outdata, frames, time_info, status):
            if status:
                # просто пишем статус в лог статистики, не валим поток
                self.s.last_error = str(status)
            try:
                # indata/outdata shape: (frames, ch)
                mono_in = indata[:, 0] if indata.ndim == 2 else indata
                mono_out = proc.process_block(mono_in.astype(np.float32, copy=False))
                if mono_out.ndim == 1:
                    mono_out = mono_out.reshape(-1, 1)
                # дублируем на все каналы (их ch одинаково для in/out)
                if mono_out.shape[1] < ch:
                    mono_out = np.repeat(mono_out, ch, axis=1)
                outdata[:] = mono_out[:frames, :ch]
                self.s.frames += int(frames)
            except Exception as e:
                self.s.last_error = f"callback: {e}"

        # создаём duplex stream (одинаковые channels для in/out)
        try:
            stream = sd.Stream(
                samplerate=sr,
                blocksize=block,
                device=(in_index, out_index),
                dtype="float32",
                channels=ch,
                callback=_cb,
            )
            stream.start()

            self.s = _State(
                stream=stream,
                proc=proc,
                frames=0,
                running=True,
                sr=sr,
                block=block,
                channels=ch,
                last_error=None,
                started_at=time.time(),
            )
            return {"ok": True, "sr": sr, "block": block, "channels": ch}
        except Exception as e:
            self.s.last_error = str(e)
            return {"ok": False, "error": f"start: {e}"}

    def stop(self) -> Dict[str, Any]:
        try:
            if self.s.stream is not None:
                try:
                    self.s.stream.stop()
                finally:
                    self.s.stream.close()
            self.s = _State()
            return {"ok": True}
        except Exception as e:
            self.s.last_error = str(e)
            return {"ok": False, "error": f"stop: {e}"}

    def stats(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "running": self.s.running,
            "frames": self.s.frames,
            "samplerate": self.s.sr,
            "block": self.s.block,
            "channels": self.s.channels,
            "last_error": self.s.last_error,
            "uptime_sec": (time.time() - self.s.started_at) if (self.s.running and self.s.started_at) else 0.0,
        }


# Синглтон на модуль
rt_manager = RTManager()
