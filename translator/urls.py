from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('data-processor/', views.data_processor, name='data_processor'),
    path('model-trainer/', views.model_trainer, name='model_trainer'),
    path('realtime-translator/', views.realtime_translator, name='realtime_translator'),
    path('upload-video/', views.upload_video, name='upload_video'),
    path('upload-model/', views.upload_model, name='upload_model'),
    path('process-data/', views.process_data, name='process_data'),
    path('train-model/', views.train_model, name='train_model'),
    path('translate-video/', views.translate_video, name='translate_video'),
    path('api/translate-frame/', views.translate_frame, name='translate_frame'),
]
