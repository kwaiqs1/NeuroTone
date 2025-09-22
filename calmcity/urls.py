# calmcity/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),

    # Старые роуты приложения (страницы загрузки/просмотра файлов и т.п.)
    path('', include('audio.urls')),

    # Новый раздел Real-Time UI:
    path('rt/', include('audio.rt_urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
