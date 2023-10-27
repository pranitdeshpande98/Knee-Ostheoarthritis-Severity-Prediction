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
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from .models import PredictionRun
from datetime import datetime
import os
from io import BytesIO
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.conf import settings
from django.core.mail import EmailMessage
import matplotlib
from django.db.models import Q
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
matplotlib.use("Agg")

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
        return redirect('reset_password')
    else:
        messages.error(request, 'Invalid reset link or token')
        return redirect('/')

def reset_password(request):
    if request.method == 'POST':
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']
        if password == confirm_password:
            pk = request.session.get('uid')
            user = User.objects.get(pk = pk)
            user.set_password(password)
            user.is_active = True
            user.save()
            messages.success(request, 'Password reset successfully')
            return redirect('/')
        else:
            messages.error(request, 'Passwords do not match!')
            return redirect('reset_password')

    return render(request, 'reset_password.html')

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

class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]

def make_gradcam_heatmap(grad_model, img_array, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    image_width, image_height = img.size
    jet_heatmap = jet_heatmap.resize((image_width, image_height))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(
        superimposed_img
    )

    return superimposed_img

def generate_run_id():
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")
    return timestamp

def generate_bar_chart(y_pred):
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.barh(class_names, y_pred, height=0.55, align="center")
    for i, (c, p) in enumerate(zip(class_names, y_pred)):
        ax.text(p + 2, i - 0.2, f"{p:.2f}%")
    ax.grid(axis="x")
    ax.set_xlim([0, 120])
    ax.set_xticks(range(0, 101, 20))
    return fig

def prediction(request):
    y_pred = None
    if request.method == "POST" and request.FILES.get("image"):
        # Load the model (replace with the path to your model)
        model = tf.keras.models.load_model("model_Xception_ft.hdf5")
        target_size = (224, 224)

# Process the uploaded image
        uploaded_image = request.FILES["image"]
        img = Image.open(uploaded_image)
        img = img.resize(target_size)
        img = img.convert("RGB")  # Ensure the image is in RGB format

# Convert the image to a NumPy array
        img_array = np.array(img)

# Check the shape of img_array
        print("Image shape:", img_array.shape)

# Perform the prediction
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.xception.preprocess_input(img_array)
        y_pred = model.predict(img_array)

        y_pred = 100 * y_pred[0]  # Only access elements if y_pred is not None
        print(y_pred)
        number = np.where(y_pred == np.amax(y_pred))
        probability = np.amax(y_pred)
        grade = str(class_names[np.amax(number)])
        value=f"{class_names[np.amax(number)]} - {probability:.2f}%",
        print(value)

        # Generate Grad-CAM heatmap
        grad_model = tf.keras.models.clone_model(model)
        grad_model.set_weights(model.get_weights())
        grad_model.layers[-1].activation = None
        grad_model = tf.keras.models.Model(
            inputs=[grad_model.inputs],
            outputs=[
                grad_model.get_layer("global_average_pooling2d_1").input,
                grad_model.output,
            ],
        )
        heatmap = make_gradcam_heatmap(grad_model, img_array)
        heatmap_img = save_and_display_gradcam(img, heatmap)

        # Create a unique run_id based on the date and timestamp
        run_id = generate_run_id()

        # Save prediction results in the database
        prediction_run = PredictionRun(
            user=request.user,
            run_id=run_id,
            severity_grade=grade,
        )
        # Convert the input image to JPEG format
        input_byte_array = BytesIO()
        img.save(input_byte_array, format='JPEG')
        input_byte_array.seek(0)

# Create an InMemoryUploadedFile for the input image
        input_file = InMemoryUploadedFile(
            input_byte_array, None, f"{run_id}_input.jpg", 'image/jpeg',
            input_byte_array.tell(), None
            )

# Save the input image to your model
        # Save the input image to the model
        prediction_run.input_image.save(f"{run_id}_input.jpg", input_file)
        # Convert the heatmap image to bytes
        heatmap_byte_array = BytesIO()
        heatmap_img.save(heatmap_byte_array, format='JPEG')
        heatmap_byte_array.seek(0)

# Create an InMemoryUploadedFile
        heatmap_file = InMemoryUploadedFile(
            heatmap_byte_array, None, f"{run_id}_heatmap.jpg", 'image/jpeg',
            heatmap_byte_array.tell(), None
        )

# Save the heatmap to your model
        prediction_run.gradcam_heatmap.save(f"{run_id}_heatmap.jpg", heatmap_file)
# Save Grad-CAM heatmap
#        heatmap_img_path = os.path.join(settings.MEDIA_ROOT, f"{run_id}_heatmap.jpg")
#        heatmap_img.save(heatmap_img_path)

        # Generate and save the bar chart image
        bar_chart = generate_bar_chart(y_pred)
        bar_chart_img_path = os.path.join(settings.MEDIA_ROOT, f"analysis_plots/{run_id}_bar_chart.jpg")
        bar_chart.savefig(bar_chart_img_path, bbox_inches='tight', pad_inches=0)

        prediction_run.bar_chart_analysis.name = f"analysis_plots/{run_id}_bar_chart.jpg"
        prediction_run.save()
        print(prediction_run.input_image.url)

# Prepare JSON response for displaying results
        response_data = {
            "severity_grade": grade,
            "probability": value,
            "run_id": run_id,
            "heatmap_url": prediction_run.gradcam_heatmap.url,
            "input_url":  prediction_run.input_image.url,
            "bar_chart_url": prediction_run.bar_chart_analysis.url,
        }

        return JsonResponse(response_data)

    return render(request, 'inner-page.html')

def dashboard_popup(request):
    # Get all prediction runs for the current user
    prediction_runs = PredictionRun.objects.filter(user=request.user)

    # Sort by severity grade (default sorting)
    sort_by = request.GET.get('sort', 'severity_grade')
    if sort_by == 'run_id':
        prediction_runs = prediction_runs.order_by('run_id')

    # Filter by grade and run_id
    filter_grade = request.GET.get('filter_grade')
    filter_run_id = request.GET.get('filter_run_id')

    if filter_grade:
        prediction_runs = prediction_runs.filter(severity_grade=filter_grade)

    if filter_run_id:
        prediction_runs = prediction_runs.filter(run_id=filter_run_id)

    # Implement searching
    search_query = request.GET.get('search')
    if search_query:
        prediction_runs = prediction_runs.filter(
            Q(severity_grade__icontains=search_query) |
            Q(run_id__icontains=search_query)
        )

    # Paginate the results (show 5 items per page)
    paginator = Paginator(prediction_runs, 5)
    page = request.GET.get('page')
    try:
        prediction_runs = paginator.page(page)
    except PageNotAnInteger:
        prediction_runs = paginator.page(1)
    except EmptyPage:
        prediction_runs = paginator.page(paginator.num_pages)

    context = {
        'prediction_runs': prediction_runs,
    }

    return render(request, 'dashboard_popup.html', context)

def profile(request):
    return HttpResponse("Hi")

def generate_pdf(request,user_name,pk):
    return render(request, 'generate_pdf.html')