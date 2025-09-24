import json
from django.http import JsonResponse, HttpRequest
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt

# ВАЖНО: импортируем ИМЕННО объект-синглтон, а не модуль
from .rt_manager import rt_manager


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
    """
    Возвращаем список доступных триггеров для UI.
    Если в realtime.py есть локализованный список — используем его,
    иначе выдаём безопасный короткий набор по умолчанию.
    """
    try:
        try:
            from .realtime import TRIGGER_LABELS_RU  # если есть
            items = TRIGGER_LABELS_RU
        except Exception:
            items = [
                "чавканье", "тик-так часов", "клавиатурные клики",
                "стук", "сирена", "собака", "крик"
            ]
        return JsonResponse({"ok": True, "items": items})
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)


# ---------- API: start ----------
@csrf_exempt
@require_POST
def start_rt(request: HttpRequest):
    try:
        data = json.loads(request.body.decode("utf-8"))

        in_idx = int(data["in"])
        out_idx = int(data["out"])

        # optional
        sr = data.get("sr")
        sr = int(sr) if sr not in (None, "", 0, "0") else None

        block = data.get("block")
        block = int(block) if block not in (None, "", 0, "0") else None

        channels = data.get("channels")
        channels = int(channels) if channels not in (None, "", 0, "0") else None

        # Конфиг для RT-процессора (имена ключей совпадают с RTConfig)
        cfg = {
            "trigger_threshold": float(data.get("trig_thresh", 0.10)),
            "vacuum_strength": float(data.get("vacuum", 1.0)),
            "protect_speech": bool(data.get("protect", True)),
            "trigger_kill": bool(data.get("trigger_kill", True)),
            "ultra_anc": bool(data.get("ultra_anc", False)),
            "selected_triggers": data.get("triggers", []),
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
