from django.forms import ValidationError
from django.http import HttpResponse, HttpResponseForbidden
from django.shortcuts import render, redirect
from accounts.forms import RegistrationForm
from django.contrib import messages, auth
from django.contrib.auth.decorators import login_required, user_passes_test
from django.core.exceptions import PermissionDenied
from accounts.utils import send_password_reset_email
from . models import User
from django.db import IntegrityError  # Import IntegrityError
from django.contrib.auth import authenticate, login
from django.http import JsonResponse
from django.urls import reverse
from .forms import RegistrationForm  # Import your RegistrationForm
from django.utils.http import urlsafe_base64_decode
from django.contrib.auth.tokens import default_token_generator
from django.utils.encoding import force_str

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

def forgot_password(request):
    if request.method == 'POST':
        email = request.POST['email']
        if User.objects.filter(email=email).exists():
            user = User.objects.get(email__exact=email)
            # Send Reset Password Email (You can implement this part)
            send_password_reset_email(request, user)
            messages.success(request, 'Password reset link has been sent to your email address.')
            return JsonResponse({'success': True, 'message': 'Password reset link has been sent to your email address.'})
        else:
            messages.error(request, 'Account does not exist')
            return JsonResponse({'success': False, 'message': 'Account does not exist'})
    
    return render(request, 'forgot_password.html')


@login_required(login_url = 'user_login')
def dashboard(request):
    return render(request,'inner-page.html')

from django.contrib.auth import get_user_model

User = get_user_model()

def reset_password_validate(request, uidb64, token):
    try:
        uid = urlsafe_base64_decode(uidb64).decode()
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist, ValidationError):
        user = None

    if user is not None and default_token_generator.check_token(user, token):
        request.session['uid'] = uid  # Store the user's ID in the session
        messages.info(request, 'Please reset your password')
        return redirect('reset_password')
    else:
        messages.error(request, 'Invalid reset link or token')
        return redirect('/')


def reset_password(request):
    uid = request.session.get('uid', None)

    if uid:
        if request.method == 'POST':
            password = request.POST['password']
            confirm_password = request.POST['confirm_password']

            if password == confirm_password:
                user = User.objects.get(pk=uid)
                user.set_password(password)
                user.is_active = True
                user.save()
                messages.success(request, 'Password reset successfully')
                return redirect('/')  # Redirect to the login page or any other page
            else:
                messages.error(request, 'Passwords do not match!')
                return redirect('reset_password')

        return render(request, 'reset_password.html')
    else:
        messages.error(request, 'Invalid reset link or token')
        return redirect('/')
