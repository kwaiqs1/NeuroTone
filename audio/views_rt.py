# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from typing import Any, Dict

from django.http import JsonResponse, HttpRequest, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt  # <-- важно

from . import rt_manager


def rt_page(request: HttpRequest) -> HttpResponse:
    return render(request, "audio/realtime.html")


def _json_ok(data: Dict[str, Any], status: int = 200) -> JsonResponse:
    return JsonResponse(data, status=status, json_dumps_params={"ensure_ascii": False})


def _json_err(msg: str, status: int = 400) -> JsonResponse:
    return JsonResponse({"ok": False, "error": msg}, status=status, json_dumps_params={"ensure_ascii": False})


def rt_devices(request: HttpRequest) -> JsonResponse:
    data = rt_manager.list_devices()
    if not data.get("ok"):
        return _json_err(data.get("error", "devices error"))
    return _json_ok(data)


def rt_triggers(request: HttpRequest) -> JsonResponse:
    return _json_ok(rt_manager.trigger_list())


@csrf_exempt  # <-- снимаем CSRF, чтобы fetch POST из UI не получал HTML-403
def rt_start(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return _json_err("POST required", 405)

    try:
        payload = json.loads(request.body or "{}")
    except Exception:
        payload = {}

    result = rt_manager.start(payload)
    return _json_ok(result) if result.get("ok") else _json_err(result.get("error", "start error"))


@csrf_exempt  # <-- тоже важно
def rt_stop(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return _json_err("POST required", 405)
    return _json_ok(rt_manager.stop())


def rt_stats(request: HttpRequest) -> JsonResponse:
    return _json_ok(rt_manager.stats())
