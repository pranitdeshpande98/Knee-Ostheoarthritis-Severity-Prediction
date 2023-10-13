from django.shortcuts import render, redirect
from .forms import RegistrationForm
from django.contrib import messages, auth
from django.contrib.auth.decorators import login_required, user_passes_test
from django.core.exceptions import PermissionDenied
from . models import User



def dashboard(request):
    if request.user.is_authenticated:
        messages.warning(request,'You are already logged in!')
#        return redirect('myAccount')

    elif request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            form.save()  # Save the user to the database
            return redirect('registration_success')  # Redirect to a success page
        else:
            print(form.errors)
    else:
        form = RegistrationForm()
        context = {
            'form' : form,
        }

    return render(request, 'inner-page.html', context)