# -*- coding: utf-8 -*-
import json
from typing import Any, Iterable, Optional, List

from django.http import JsonResponse, HttpRequest
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt

from .rt_manager import rt_manager


def _first_key(d: dict, names: Iterable[str]):
    for k in names:
        if k in d:
            return d[k]
    return None

def _as_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    if v in (None, "", "null", "None"):
        return default
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default

def _as_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v in (None, "", "null", "None"):
        return default
    try:
        return float(v)
    except Exception:
        return default

def _as_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v in (None, "", "null", "None"):
        return default
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")

def _as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return [v]


@require_GET
def rt_page(request: HttpRequest):
    return render(request, "audio/realtime.html")


@require_GET
def list_devices(request: HttpRequest):
    res = rt_manager.list_devices()
    status = 200 if res.get("ok") else 500
    return JsonResponse(res, status=status)


@require_GET
def list_triggers(request: HttpRequest):
    try:
        from .realtime import TRIGGER_LABELS_RU  # если есть
        items = TRIGGER_LABELS_RU
    except Exception:
        items = ["чавканье", "тик-так", "клавиатура", "стук", "сирена", "лай собаки", "крик"]
    return JsonResponse({"ok": True, "items": items})


@csrf_exempt
@require_POST
def start_rt(request: HttpRequest):
    try:
        data = json.loads(request.body.decode("utf-8")) if (
            request.content_type and "application/json" in request.content_type
        ) else request.POST.dict()

        in_idx = _as_int(_first_key(data, ["in", "in_dev", "input", "input_device", "inputDevice", "inIndex"]))
        out_idx = _as_int(_first_key(data, ["out", "out_dev", "output", "output_device", "outputDevice", "outIndex"]))
        if in_idx is None or out_idx is None:
            return JsonResponse({"ok": False, "error": "input/output device not provided"}, status=400)

        sr = _as_int(_first_key(data, ["sr", "samplerate", "sample_rate", "fs"]))
        block = _as_int(_first_key(data, ["block", "blocksize", "frames", "frame_size"]))
        channels = _as_int(_first_key(data, ["channels", "ch"]))

        cfg = {
            "trigger_threshold": _as_float(_first_key(data, ["trig_thresh", "trigger", "sensitivity"]), 0.10),
            "vacuum_strength": _as_float(_first_key(data, ["vacuum", "vacuum_strength", "anc_strength"]), 1.0),
            "protect_speech": _as_bool(_first_key(data, ["protect", "protect_speech"]), True),
            "trigger_kill": _as_bool(_first_key(data, ["trigger_kill", "kill_triggers"]), True),
            "ultra_anc": _as_bool(_first_key(data, ["ultra_anc", "anc_ultra"]), False),
            "selected_triggers": _as_list(_first_key(data, ["triggers", "selected_triggers"])),
        }

        res = rt_manager.start(
            in_index=in_idx,
            out_index=out_idx,
            samplerate=sr,
            blocksize=block,
            channels=channels,
            cfg_dict=cfg,
        )
        status = 200 if res.get("ok") else 500
        return JsonResponse(res, status=status)
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)


@csrf_exempt
@require_POST
def stop_rt(request: HttpRequest):
    res = rt_manager.stop()
    status = 200 if res.get("ok") else 500
    return JsonResponse(res, status=status)


@require_GET
def rt_stats(request: HttpRequest):
    return JsonResponse(rt_manager.stats())
