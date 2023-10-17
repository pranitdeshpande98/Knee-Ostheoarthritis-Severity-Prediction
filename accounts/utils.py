from base64 import urlsafe_b64encode
from django.core.mail import EmailMessage
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes
from django.contrib.auth.tokens import default_token_generator
from django.conf import settings
from django.template.loader import render_to_string
from django.core.mail import EmailMessage
from django.utils.safestring import mark_safe  # Import mark_safe

def send_password_reset_email(request, user):
    from_email = settings.DEFAULT_FROM_EMAIL
    current_site = get_current_site(request)
    mail_subject = 'Reset your Password'
    
    # Use mark_safe to mark the HTML content as safe
    message = mark_safe(render_to_string('emails/reset_password_email.html', {
        'user': user,
        'domain' : current_site.domain,
        'uid' : urlsafe_base64_encode(force_bytes(user.pk)),
        'token': default_token_generator.make_token(user),
    }))
    
    to_email = user.email
    mail = EmailMessage(mail_subject, message, from_email, to=[to_email])
    mail.content_subtype = "html"  # Set content subtype to HTML
    mail.send()
