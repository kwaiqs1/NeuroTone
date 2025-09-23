# audio/rt_urls.py
from django.urls import path

from . import views_rt

urlpatterns = [
    path("rt/", views_rt.rt_page, name="rt_page"),
    path("rt/devices", views_rt.list_devices, name="rt_devices"),
    path("rt/triggers", views_rt.list_triggers, name="rt_triggers"),
    path("rt/start", views_rt.start_rt, name="rt_start"),
    path("rt/stop", views_rt.stop_rt, name="rt_stop"),
    path("rt/stats", views_rt.rt_stats, name="rt_stats"),
]
