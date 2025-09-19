from django.contrib import admin
from .models import AudioUpload


@admin.register(AudioUpload)
class AudioUploadAdmin(admin.ModelAdmin):
    list_display = ('id', 'original_file', 'processed_file', 'created_at')
    readonly_fields = ('created_at',)