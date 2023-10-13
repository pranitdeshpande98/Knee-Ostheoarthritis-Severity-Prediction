from django.http import HttpResponse
from django.shortcuts import render, redirect
from . forms import RegistrationForm
from django.contrib import messages, auth
from django.contrib.auth.decorators import login_required, user_passes_test
from django.core.exceptions import PermissionDenied
from . models import User

def dashboard(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your account has been registered successfully!')
    else:
        form = RegistrationForm()
    context = {
        'form': form,
    }
    return render(request, 'inner-page.html', context)


def forgot_password(request):
    pass

def login(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']

        user = auth.authenticate(email = email, password = password)
        if user is not None:
            auth.login(request,user)
            messages.success(request,'You are now logged in')
            return redirect('inner-page.html')
        else:
            messages.error(request,'Invalid Login Credentials')
        
    return render(request,'inner-page.html')