# -*- coding: utf-8 -*-
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import sounddevice as sd

# Реальный DSP/ИИ-процессор и его конфиг
try:
    from .realtime import RealTimeProcessor, RTConfig as _RTConfig
except Exception as e:  # fall back, чтобы страница не падала при импорт-ошибках
    RealTimeProcessor = None  # type: ignore
    _RTConfig = None          # type: ignore
    _import_error = e
else:
    _import_error = None


# Безопасная обёртка вокруг RTConfig: принимаем «алиасы» полей
def _make_rt_config(cfg: Dict[str, Any]) -> Any:
    """
    Приводим входные параметры к тем, что реально есть у RTConfig в realtime.py.
    Поддерживаем алиасы: 'trigger_threshold' -> 'sensitivity' и т.п.
    """
    if _RTConfig is None:
        raise RuntimeError(f"Realtime engine is not available: {_import_error}")

    # алиасы/значения по умолчанию
    in_index = int(cfg.get("in"))
    out_index = int(cfg.get("out"))
    samplerate = cfg.get("sr")
    samplerate = int(samplerate) if samplerate not in (None, "", "null") else None
    block = int(cfg.get("block", 480))

    # ключ чувствительности — фронт шлёт 'sensitivity'; старые версии могли слать 'trigger_threshold'
    sensitivity = cfg.get("sensitivity", cfg.get("trigger_threshold", 0.08))
    sensitivity = float(sensitivity)

    vacuum_strength = float(cfg.get("vacuum_strength", 1.0))
    protect_speech = bool(cfg.get("protect_speech", True))
    trigger_kill = bool(cfg.get("trigger_kill", True))
    ultra_anc = bool(cfg.get("ultra_anc", False))
    triggers = cfg.get("triggers") or []
    if not isinstance(triggers, list):
        triggers = [triggers]

    # Конструируем тот же RTConfig, что ожидает движок
    return _RTConfig(
        in_index=in_index,
        out_index=out_index,
        samplerate=samplerate,
        block=block,
        sensitivity=sensitivity,
        vacuum_strength=vacuum_strength,
        protect_speech=protect_speech,
        trigger_kill=trigger_kill,
        ultra_anc=ultra_anc,
        triggers=triggers,
    )


# Глобальное состояние одного RT-потока
_lock = threading.Lock()
_rt = None        # type: Optional[RealTimeProcessor]
_last_error = ""
_stats: Dict[str, Any] = {
    "running": False,
    "frames": 0,
    "samplerate": None,
    "channels": None,
    "block": None,
    "last_error": "",
}


def _set_error(msg: str) -> None:
    global _last_error, _stats
    _last_error = msg
    _stats["last_error"] = msg


def list_devices() -> Dict[str, Any]:
    """Вернёт входы/выходы и дефолтные индексы для UI."""
    try:
        devices = sd.query_devices()
        default_in, default_out = sd.default.device
    except Exception as e:
        return {"ok": False, "error": str(e), "inputs": [], "outputs": []}

    inputs, outputs = [], []
    for i, d in enumerate(devices):
        label = f"[{i}] {d['name']} — {d['hostapi']}" if "hostapi" in d else f"[{i}] {d['name']}"
        if int(d.get("max_input_channels", 0)) > 0:
            inputs.append({"index": i, "label": label})
        if int(d.get("max_output_channels", 0)) > 0:
            outputs.append({"index": i, "label": label})

    return {
        "ok": True,
        "inputs": inputs,
        "outputs": outputs,
        "default_in": default_in if isinstance(default_in, int) else None,
        "default_out": default_out if isinstance(default_out, int) else None,
    }


# Небольшая библиотека «триггеров» (рус)
try:
    # если в realtime.py уже есть словарь с русским списком — используем его
    from .realtime import TRIGGER_LIBRARY_RU as _TRIGS  # type: ignore
    TRIGGERS_RU = list(_TRIGS)
except Exception:
    TRIGGERS_RU = [
        "чавканье", "тик-так", "стук", "крик", "печать на клавиатуре",
        "скрип", "звон посуды", "шёпот", "кашель", "чих", "свист", "хлопок",
        "шорох пакета", "лай собаки", "смех",
    ]


def trigger_list() -> Dict[str, Any]:
    return {"ok": True, "items": TRIGGERS_RU}


def start(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Стартуем один RT-процессор (если есть старый — останавливаем)."""
    global _rt, _stats
    with _lock:
        _set_error("")
        # останавливаем прежний
        if _rt is not None:
            try:
                _rt.stop()
            except Exception:
                pass
            _rt = None
        # создаём новый
        try:
            rt_cfg = _make_rt_config(cfg)
            if RealTimeProcessor is None:
                raise RuntimeError(f"Realtime engine is not available: {_import_error}")
            _rt = RealTimeProcessor(rt_cfg)
            _rt.start()
        except Exception as e:
            _set_error(str(e))
            return {"ok": False, "error": str(e)}

        # обновляем статусы
        try:
            _stats.update({
                "running": True,
                "frames": 0,
                "samplerate": getattr(_rt, "sr", getattr(_rt, "samplerate", None)),
                "channels": getattr(_rt, "channels", 1),
                "block": getattr(_rt, "block", cfg.get("block", 480)),
                "last_error": "",
            })
        except Exception:
            pass

        return {
            "ok": True,
            "sr": _stats["samplerate"],
            "channels": _stats["channels"],
            "block": _stats["block"],
        }


def stop() -> Dict[str, Any]:
    global _rt, _stats
    with _lock:
        if _rt is not None:
            try:
                _rt.stop()
            except Exception as e:
                _set_error(str(e))
            _rt = None
        _stats["running"] = False
    return {"ok": True}


def stats() -> Dict[str, Any]:
    """Немного статистики для UI."""
    global _rt, _stats
    with _lock:
        # аккуратно читаем счётчик кадров, если движок его ведёт
        try:
            if _rt is not None and hasattr(_rt, "frames_processed"):
                _stats["frames"] = int(getattr(_rt, "frames_processed"))
        except Exception:
            pass
        return dict(_stats)
