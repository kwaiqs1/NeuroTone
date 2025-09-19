from django.db import models


class AudioUpload(models.Model):
    original_file = models.FileField(upload_to='originals/')
    processed_file = models.FileField(upload_to='processed/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True, default='')


    def __str__(self):
        return f"AudioUpload #{self.pk} ({self.original_file.name})"