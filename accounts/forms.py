from django import forms
from .models import User  # Import your user model

class RegistrationForm(forms.ModelForm):
    class Meta:
        model = User  # Use your user model
        fields = ['email', 'name', 'phone', 'age', 'gender', 'password']
