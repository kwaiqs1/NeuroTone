# audio/rt_urls.py
from django.urls import path
from . import views_rt

# ВАЖНО: здесь больше НЕТ префикса 'rt/' — он уже задан в calmcity/urls.py

urlpatterns = [
    path("", views_rt.rt_page, name="rt_page"),                 # /rt/
    path("devices", views_rt.list_devices, name="rt_devices"),  # /rt/devices
    path("triggers", views_rt.list_triggers, name="rt_triggers"),  # /rt/triggers
    path("start", views_rt.start_rt, name="rt_start"),          # /rt/start
    path("stop", views_rt.stop_rt, name="rt_stop"),             # /rt/stop
    path("stats", views_rt.rt_stats, name="rt_stats"),          # /rt/stats
]
