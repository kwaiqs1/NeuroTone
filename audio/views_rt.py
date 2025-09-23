# audio/views_rt.py
# Веб-ручки для страницы /rt/: устройства, старт/стоп, список триггеров.

import json
from typing import Any, Dict, List

from django.http import JsonResponse, HttpRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

import sounddevice as sd

from . import rt_manager


def rt_page(request: HttpRequest):
    return render(request, "audio/realtime.html")


def list_devices(request: HttpRequest):
    devs = sd.query_devices()
    # строим красивый список
    items = []
    for idx, d in enumerate(devs):
        in_ch = int(d.get("max_input_channels", 0))
        out_ch = int(d.get("max_output_channels", 0))
        host = d.get("hostapi", 0)
        host_name = sd.query_hostapis()[host]["name"] if isinstance(host, int) else str(host)
        name = f"[{idx}] {d['name']} — {host_name} (in:{in_ch}, out:{out_ch})"
        items.append({
            "index": idx,
            "name": name,
            "in": in_ch,
            "out": out_ch,
            "default_samplerate": d.get("default_samplerate"),
        })
    return JsonResponse({"ok": True, "items": items})


def list_triggers(request: HttpRequest):
    data = rt_manager.available_triggers()
    # добавим плоский список только названий (ru) для быстрого поиска
    simple = [it.get("ru") or it.get("label") for it in data.get("items", [])]
    data["names"] = simple
    return JsonResponse(data)


@csrf_exempt
def start_rt(request: HttpRequest):
    try:
        if request.method != "POST":
            return JsonResponse({"ok": False, "error": "POST only"}, status=405)
        body = request.body.decode("utf-8") or "{}"
        params: Dict[str, Any] = json.loads(body)

        res = rt_manager.start(params)
        return JsonResponse({"ok": True, **res})
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)})


@csrf_exempt
def stop_rt(request: HttpRequest):
    try:
        res = rt_manager.stop()
        return JsonResponse({"ok": True, **res})
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)})


def rt_stats(request: HttpRequest):
    return JsonResponse({"ok": True, **rt_manager.stats()})
