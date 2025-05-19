from django.db import models
from django.contrib.auth.models import User
import os
import uuid

def model_upload_path(instance, filename):
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('models', filename)

def video_upload_path(instance, filename):
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('videos', filename)

class TrainedModel(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    file = models.FileField(upload_to=model_upload_path)
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    accuracy = models.FloatField(default=0.0)
    
    def __str__(self):
        return self.name

class SignVideo(models.Model):
    word = models.CharField(max_length=100)
    video = models.FileField(upload_to=video_upload_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def __str__(self):
        return self.word

class TranslationSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    model = models.ForeignKey(TrainedModel, on_delete=models.CASCADE)
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    translation_text = models.TextField(blank=True)
    
    def __str__(self):
        return f"Session by {self.user.username} at {self.start_time}"
