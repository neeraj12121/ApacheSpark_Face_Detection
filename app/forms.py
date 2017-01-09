from django import forms
from .models import image

class ImageForm(forms.ModelForm):
    class Meta:
        model = image
        fields={'img'}