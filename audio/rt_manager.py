# audio/rt_manager.py
# Менеджер реального времени: поток sounddevice + наш DSP, управление из вьюх.

import json
import threading
from typing import Optional, Iterable, Dict, Any

import numpy as np
import sounddevice as sd

from .realtime import RealTimeProcessor, RTConfig


class _RTState:
    def __init__(self):
        self.lock = threading.Lock()
        self.stream: Optional[sd.Stream] = None
        self.proc: Optional[RealTimeProcessor] = None
        self.frames = 0
        self.samplerate = None
        self.channels = None
        self.block = None
        self.running = False

state = _RTState()


def _make_processor(cfg_dict: Dict[str, Any], samplerate: int, blocksize: int,
                    triggers: Optional[Iterable[str]]) -> RealTimeProcessor:
    cfg = RTConfig(
        trigger_sensitivity=float(cfg_dict.get("trigger_sensitivity", 0.08)),
        vacuum_strength=float(cfg_dict.get("vacuum_strength", 1.0)),
        protect_speech=bool(cfg_dict.get("protect_speech", True)),
        enable_triggers=bool(cfg_dict.get("enable_triggers", True)),
        enable_vacuum=bool(cfg_dict.get("enable_vacuum", True)),
        enable_denoise=bool(cfg_dict.get("enable_denoise", True)),
        ultra_anc=bool(cfg_dict.get("ultra_anc", True)),
    )
    return RealTimeProcessor(cfg, stream_sr=int(samplerate),
                             blocksize=int(blocksize),
                             selected_triggers=list(triggers) if triggers else None)


def start(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    params: {
        in_index, out_index, samplerate?, block, channels,
        cfg: {...},
        triggers: [label1, label2, ...]
    }
    """
    with state.lock:
        if state.stream is not None:
            stop()

        in_index = int(params["in_index"])
        out_index = int(params["out_index"])
        samplerate = int(params.get("samplerate") or 0) or None
        block = int(params.get("block") or 480)
        channels = int(params.get("channels") or 1)
        cfg = params.get("cfg", {})
        triggers = params.get("triggers") or []

        # если samplerate не задан — спросим у устройства
        if samplerate is None:
            dev = sd.query_devices(in_index)
            samplerate = int(dev["default_samplerate"]) if dev.get("default_samplerate") else 48000

        proc = _make_processor(cfg, samplerate, block, triggers)

        def callback(indata, outdata, frames, time, status):
            if status:
                # просто пишем в stderr и продолжим
                print("SD status:", status)
            # берём 1 канал (моно)
            mono = indata[:, 0].copy()
            y = proc.process_block(mono)
            # моно → стерео при необходимости
            if outdata.shape[1] == 1:
                outdata[:, 0] = y
            else:
                outdata[:, 0] = y
                outdata[:, 1] = y
            state.frames += frames

        stream = sd.Stream(
            device=(in_index, out_index),
            samplerate=samplerate,
            blocksize=block,
            dtype="float32",
            channels=(channels, 2 if channels < 2 else channels)  # in, out
        )

        stream.start(callback=callback)

        state.stream = stream
        state.proc = proc
        state.frames = 0
        state.samplerate = samplerate
        state.channels = channels
        state.block = block
        state.running = True

        return {
            "ok": True,
            "samplerate": samplerate,
            "channels": channels,
            "block": block,
        }


def stop() -> Dict[str, Any]:
    with state.lock:
        if state.stream is not None:
            try:
                state.stream.stop()
                state.stream.close()
            except Exception:
                pass
            state.stream = None
            state.proc = None
        state.running = False
        return {"ok": True}


def stats() -> Dict[str, Any]:
    with state.lock:
        return {
            "running": state.running,
            "frames": state.frames,
            "samplerate": state.samplerate,
            "channels": state.channels,
            "block": state.block,
        }


def available_triggers() -> Dict[str, Any]:
    """
    Возвращает список {id,label,ru} для UI.
    Если DSP ещё не создан, поднимаем временный процессор на дефолтных значениях.
    """
    with state.lock:
        proc = state.proc

    tmp_created = False
    if proc is None:
        # мини-временный процессор только чтобы прочитать список меток
        proc = _make_processor({}, 48000, 480, [])
        tmp_created = True

    try:
        items = proc.available_trigger_labels()
        return {"ok": True, "items": items}
    finally:
        if tmp_created:
            # просто даём GC убрать — ничего не стартовали
            pass
