from django.conf import settings
from django.forms import ValidationError
from django.shortcuts import render, redirect
from accounts.forms import RegistrationForm
from django.contrib import messages, auth
from django.contrib.auth.decorators import login_required
from accounts.utils import send_password_reset_email
from . models import User
from django.db import IntegrityError  # Import IntegrityError
from django.contrib.auth import authenticate, login
from django.utils.http import urlsafe_base64_decode
from django.contrib.auth.tokens import default_token_generator
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

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


@login_required(login_url='/')
def dashboard(request):
    return render(request,'inner-page.html')

from django.contrib.auth import get_user_model

User = get_user_model()

from django.urls import reverse

from django.shortcuts import redirect

from django.http import HttpResponseRedirect

def reset_password_validate(request, uidb64, token):
    try:
        uid = urlsafe_base64_decode(uidb64).decode()
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist, ValidationError):
        user = None

    if user is not None and default_token_generator.check_token(user, token):
        request.session['uid'] = uid  # Store the user's ID in the session
        messages.info(request, 'Please reset your password')

        # Pass uidb64 and token as query parameters in the redirect
        reset_password_url = reverse('reset_password')
        reset_password_url += f'?uidb64={uidb64}&token={token}'
        context ={
            'uidb64': uidb64,
            'token' : token,
        }
        return HttpResponseRedirect(reset_password_url, context)
    else:
        messages.error(request, 'Invalid reset link or token')
        return redirect('/')

def reset_password(request):
    uidb64 = request.GET.get('uidb64')
    token = request.GET.get('token')
    
    if not uidb64 or not token:
        # Handle the case when query parameters are missing
        messages.error(request, 'Invalid reset link or token')
        return redirect('/')

    uid = urlsafe_base64_decode(uidb64).decode()
    user = User.objects.get(pk=uid)
    
    if not default_token_generator.check_token(user, token):
        # Handle the case when the token is invalid
        messages.error(request, 'Invalid reset link or token')
        return redirect('/')

    if request.method == 'POST':
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']

        if password == confirm_password:
            user.set_password(password)
            user.is_active = True
            user.save()
            messages.success(request, 'Password reset successfully')
            return JsonResponse({'success': True})  # Successful password reset
        else:
            messages.error(request, 'Passwords do not match!')
            return JsonResponse({'success': False, 'message': 'Passwords do not match'})

    return render(request, 'reset_password.html')


from django.conf import settings
from django.core.mail import EmailMessage

@csrf_exempt
def contact_form(request):
    if request.method == 'POST':
        name = request.POST.get('name', '')
        email = request.POST.get('email', '')
        subject = request.POST.get('subject', '')
        message = request.POST.get('message', '')
     
        # You can add your email sending logic here
        # Replace the following with your own email sending code
        # Example using Django's EmailMessage class:
        email_to_be_send = EmailMessage(
            subject=subject,
            body=message,
            from_email=email,  # Use the user's email as the "from" address
            to=[settings.DEFAULT_FROM_EMAIL],  # Replace with your recipient's email address
            reply_to=[email],  # Use the user's email for reply-to
        )
        email_to_be_send.send()

        return JsonResponse({'success': True})
    return JsonResponse({'success': False})


