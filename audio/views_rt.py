import json
from typing import Any, Iterable, Optional, List

from django.http import JsonResponse, HttpRequest
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt

# ВАЖНО: импортируем именно объект-синглтон менеджера
from .rt_manager import rt_manager


# ---------- helpers ----------

def _first_key(d: dict, names: Iterable[str]) -> Optional[Any]:
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
        # "a,b,c" -> ["a","b","c"]
        return [x.strip() for x in v.split(",") if x.strip()]
    return [v]


# ---------- PAGE ----------
@require_GET
def rt_page(request: HttpRequest):
    return render(request, "audio/realtime.html")


# ---------- API: devices ----------
@require_GET
def list_devices(request: HttpRequest):
    try:
        return JsonResponse(rt_manager.list_devices())
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)


# ---------- API: triggers (каталог) ----------
@require_GET
def list_triggers(request: HttpRequest):
    try:
        # если в realtime.py есть словарь с переводами — используем
        try:
            from .realtime import TRIGGER_LABELS_RU  # type: ignore
            items = TRIGGER_LABELS_RU
        except Exception:
            items = [
                "чавканье", "тик-так часов", "клавиатура",
                "стук", "сирена", "лай собаки", "крик"
            ]
        return JsonResponse({"ok": True, "items": items})
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)


# ---------- API: start ----------
@csrf_exempt
@require_POST
def start_rt(request: HttpRequest):
    try:
        # поддерживаем JSON и form-data
        if request.content_type and "application/json" in request.content_type:
            data = json.loads(request.body.decode("utf-8"))
        else:
            data = request.POST.dict()

        # поддерживаем множество вариантов имён ключей
        in_idx = _as_int(_first_key(data, ["in", "in_dev", "input", "input_device", "inputDevice", "inIndex"]))
        out_idx = _as_int(_first_key(data, ["out", "out_dev", "output", "output_device", "outputDevice", "outIndex"]))

        if in_idx is None or out_idx is None:
            return JsonResponse({
                "ok": False,
                "error": "input/output device not provided (ожидаю ключи: in/out или input_device/output_device)"
            }, status=400)

        sr = _as_int(_first_key(data, ["sr", "samplerate", "sample_rate", "fs"]))
        block = _as_int(_first_key(data, ["block", "blocksize", "frames", "frame_size"]))
        channels = _as_int(_first_key(data, ["channels", "ch"]))

        trig_thresh = _as_float(_first_key(data, ["trig_thresh", "trigger", "trigger_threshold", "sensitivity"]), 0.10)
        vacuum = _as_float(_first_key(data, ["vacuum", "vacuum_strength", "anc_strength"]), 1.0)
        protect = _as_bool(_first_key(data, ["protect", "protect_speech"]), True)
        trigger_kill = _as_bool(_first_key(data, ["trigger_kill", "kill", "kill_triggers"]), True)
        ultra_anc = _as_bool(_first_key(data, ["ultra_anc", "ultra", "anc_ultra"]), False)

        triggers = _first_key(data, ["triggers", "selected_triggers"])
        triggers = _as_list(triggers)

        cfg = {
            "trigger_threshold": trig_thresh,
            "vacuum_strength": vacuum,
            "protect_speech": protect,
            "trigger_kill": trigger_kill,
            "ultra_anc": ultra_anc,
            "selected_triggers": triggers,
        }

        res = rt_manager.start(
            in_index=in_idx,
            out_index=out_idx,
            samplerate=sr,
            blocksize=block,
            channels=channels,
            cfg_dict=cfg,
        )
        return JsonResponse(res)
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)


# ---------- API: stop ----------
@csrf_exempt
@require_POST
def stop_rt(request: HttpRequest):
    try:
        return JsonResponse(rt_manager.stop())
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)


# ---------- API: stats ----------
@require_GET
def rt_stats(request: HttpRequest):
    try:
        return JsonResponse(rt_manager.stats())
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)
