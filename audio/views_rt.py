# audio/views_rt.py
from django.http import JsonResponse, HttpRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json

from .rt_manager import RTSessionManager


def realtime_page(request: HttpRequest):
    """Страница с UI."""
    return render(request, "audio/realtime.html")


def api_devices(request: HttpRequest):
    mgr = RTSessionManager.instance()
    return JsonResponse(mgr.list_devices())


@csrf_exempt
def api_rt_start(request: HttpRequest):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        payload = request.POST.dict()

    try:
        mgr = RTSessionManager.instance()
        res = mgr.start(
            input_id=int(payload.get("input_id")),
            output_id=int(payload.get("output_id")),
            sensitivity=float(payload.get("sensitivity", 0.08)),
            vacuum=float(payload.get("vacuum", 1.0)),
            block=int(payload.get("block", 480)),
            protect_speech=bool(payload.get("protect_speech", True)),
            enable_triggers=bool(payload.get("enable_triggers", True)),
            enable_vacuum=bool(payload.get("enable_vacuum", True)),
            ultra_anc=bool(payload.get("ultra_anc", True)),
            samplerate=int(payload["samplerate"]) if payload.get("samplerate") else None,
        )
        return JsonResponse(res)
    except Exception as e:
        return JsonResponse({"running": False, "error": str(e)}, status=400)


@csrf_exempt
def api_rt_stop(request: HttpRequest):
    mgr = RTSessionManager.instance()
    res = mgr.stop()
    return JsonResponse(res)


def api_rt_status(request: HttpRequest):
    mgr = RTSessionManager.instance()
    return JsonResponse(mgr.status())
