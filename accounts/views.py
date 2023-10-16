from django.http import HttpResponse
from django.shortcuts import render, redirect
from accounts.forms import RegistrationForm
from django.contrib import messages, auth
from django.contrib.auth.decorators import login_required, user_passes_test
from django.core.exceptions import PermissionDenied
from . models import User
from django.db import IntegrityError  # Import IntegrityError
from django.contrib.auth import authenticate, login
from django.http import JsonResponse

from django.http import JsonResponse, HttpResponseRedirect
from django.urls import reverse
from .forms import RegistrationForm  # Import your RegistrationForm
from .models import User  # Import your User model
from django.shortcuts import render

from django.http import JsonResponse
from django.urls import reverse
from django.http import HttpResponseRedirect
from .forms import RegistrationForm
from .models import User

def register(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            # Check if a user with the provided email already exists
            if User.objects.filter(email=email).exists():
                return JsonResponse({'success': False, 'error': 'A user with this email address already exists. Please choose a different email.'})
            else:
                try:
                    user = User.objects.create_user(
                        email=email,
                        password=form.cleaned_data['password'],
                        name=form.cleaned_data['name'],
                        phone=form.cleaned_data['phone'],
                        age=form.cleaned_data['age'],
                        gender=form.cleaned_data['gender']
                    )
                    return JsonResponse({'success': True})
                except IntegrityError as e:
                    return JsonResponse({'success': False, 'error': 'An error occurred while creating your account. Please try again.'})
        else:
            print(form.errors)
            errors = dict(form.errors.items())
            return JsonResponse({'success': False, 'error': 'Please correct the errors in the form', 'form_errors': errors})
    else:
        form = RegistrationForm()

    context = {
        'form': form,
    }
    
    return render(request, 'home.html', context)

def register_success(request):
    return render(request, 'register.html')

def forgot_password(request):
    return render(request,'forgot_password.html')

@login_required(login_url = 'home')
def dashboard(request):
    return render(request,'inner-page.html')

def user_login(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        if email and password:
            user = authenticate(request, email=email, password=password)

            if user is not None:
                login(request, user)
                return JsonResponse({'success': True, 'message': 'Login successful'})
            else:
                return JsonResponse({'success': False, 'message': 'Invalid email or password'})
        else:
            return JsonResponse({'success': False, 'message': 'Please enter both email and password'})
    
    return redirect('dashboard')

def logout(request):
    auth.logout(request)
    messages.info(request,'You are logged out.')
    return redirect('home')


