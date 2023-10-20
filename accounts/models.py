from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _

class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')

        email = self.normalize_email(email)

        # Check if a user with the provided email already exists
        if self.filter(email=email).exists():
            raise ValueError('A user with this email address already exists. Please choose a different email.')

        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user


    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError(_('Superuser must have is_staff=True.'))
        if extra_fields.get('is_superuser') is not True:
            raise ValueError(_('Superuser must have is_superuser=True.'))

        return self.create_user(email, password, **extra_fields)

class User(AbstractBaseUser, PermissionsMixin):
    MALE = 'Male'
    FEMALE = 'Female'
    OTHERS = 'Others'

    GENDER_CHOICES = [
        (MALE, 'Male'),
        (FEMALE, 'Female'),
        (OTHERS, 'Others'),
    ]

    email = models.EmailField(unique=True)
    name = models.CharField(max_length=100)
    phone = models.CharField(max_length=12)
    age = models.PositiveIntegerField()
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)


    date_joined = models.DateTimeField(default=timezone.now)
    last_login = models.DateTimeField(default=timezone.now)
    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)

    objects = CustomUserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['name', 'phone', 'age', 'gender']

    def __str__(self):
        return self.email



class PredictionRun(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Link to your User model
    run_id = models.CharField(max_length=255, unique=True)  # Combination of date and timestamp
    severity_grade = models.CharField(max_length=20)
    input_image = models.ImageField(upload_to='uploaded_images/')
    gradcam_heatmap = models.ImageField(upload_to='heatmaps/')
    bar_chart_analysis = models.ImageField(upload_to='analysis_plots/')

    def __str__(self):
        return f"{self.user.email} - {self.run_id}"
