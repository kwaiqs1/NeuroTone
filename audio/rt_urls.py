# audio/rt_urls.py
from django.urls import path
from . import views_rt

urlpatterns = [
    path("", views_rt.realtime_page, name="realtime"),
    path("api/devices/", views_rt.api_devices, name="rt_devices"),
    path("api/start/", views_rt.api_rt_start, name="rt_start"),
    path("api/stop/", views_rt.api_rt_stop, name="rt_stop"),
    path("api/status/", views_rt.api_rt_status, name="rt_status"),
]
