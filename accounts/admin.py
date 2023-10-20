from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import PredictionRun, User

class CustomUserAdmin(UserAdmin):
    list_display = ('email', 'name', 'phone', 'age', 'gender', 'is_active', 'is_staff')
    list_filter = ('is_active', 'is_staff', 'gender')
    search_fields = ('email', 'name', 'phone', 'age')
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal Info', {'fields': ('name', 'phone', 'age', 'gender')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'groups', 'user_permissions')}),
        ('Important Dates', {'fields': ('last_login', 'date_joined')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'password1', 'password2', 'name', 'phone', 'age', 'gender', 'is_active', 'is_staff')}
        ),
    )
    ordering = ('email',)

admin.site.register(User, CustomUserAdmin)
admin.site.register(PredictionRun)