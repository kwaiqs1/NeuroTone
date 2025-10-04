# audio/urls.py
from django.urls import path, include
from . import views


app_name = 'audio'


urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_and_process, name='upload'),
    path('detail/<int:pk>/', views.detail, name='detail'),
    path("rt/", include("audio.rt_urls"))
]