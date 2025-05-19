from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import SignVideo, TrainedModel

class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField()
    
    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = SignVideo
        fields = ['word', 'video']

class ModelUploadForm(forms.ModelForm):
    class Meta:
        model = TrainedModel
        fields = ['name', 'description', 'file', 'accuracy']

class DataProcessorForm(forms.Form):
    data_file = forms.FileField(
        label="Data File",
        help_text="Upload a processed data file (.pickle format)"
    )

class ModelTrainerForm(forms.Form):
    pickle_file = forms.FileField()
