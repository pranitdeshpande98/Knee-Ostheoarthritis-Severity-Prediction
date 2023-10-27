from django.urls import path,include
from . import views

urlpatterns = [
    path('register/',views.register,name='register'),
    path('register_success/',views.register_success,name="register_success"),
    path('user_login/',views.user_login,name = 'user_login'),
    path('dashboard/',views.dashboard,name="dashboard"),
    path('logout/', views.logout, name = 'logout'),
    path('forgot_password/',views.forgot_password,name = 'forgot_password'),
    path('reset_password_validate/<uidb64>/<token>',views.reset_password_validate,name = 'reset_password_validate'),
    path('reset_password/', views.reset_password, name='reset_password'),
    path('contact/', views.contact_form, name='contact_form'),
    path('prediction/',views.prediction,name="prediction"),
    path('dashboard_popup/', views.dashboard_popup, name='dashboard_popup'),
    path('generate_pdf/<str:user_name>/<int:pk>', views.generate_pdf, name='generate_pdf'),
    path('profile/update/',views.profile,name="profile"),

]