from django.contrib import admin
from .models import TrainedModel, SignVideo, TranslationSession

@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'created_by', 'created_at', 'accuracy')
    search_fields = ('name', 'description')
    list_filter = ('created_at', 'created_by')

@admin.register(SignVideo)
class SignVideoAdmin(admin.ModelAdmin):
    list_display = ('word', 'uploaded_by', 'uploaded_at')
    search_fields = ('word',)
    list_filter = ('uploaded_at', 'uploaded_by')

@admin.register(TranslationSession)
class TranslationSessionAdmin(admin.ModelAdmin):
    list_display = ('user', 'model', 'start_time', 'end_time')
    search_fields = ('user__username', 'translation_text')
    list_filter = ('start_time', 'user')
