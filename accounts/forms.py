from django import forms
from .models import User  # Import your user model

class RegistrationForm(forms.ModelForm):
    class Meta:
        model = User  # Use your user model
        fields = ['email', 'name', 'phone', 'age', 'gender', 'password']

        error_messages = {
            'name': {
                'required': 'Please fill in your name.',
            },
            'email': {
                'required': 'Please fill in your email address.',
            },
            'phone': {
                'required': 'Please fill in your phone number.',
            },
            'age': {
                'required': 'Please fill in your age.',
            },
            'gender': {
                'required': 'Please select your gender.',
            },
            'password': {
                'required': 'Please fill in your password.',
            },
        }


class UserProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['name', 'phone', 'age', 'gender']


class UserProfileFormEdit(forms.ModelForm):
    class Meta:
        model = User
        fields = ['email', 'name', 'phone', 'age', 'gender']
