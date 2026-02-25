"""application URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
"""
# Django imports
from django.urls import re_path
from django.contrib import admin
from django.conf.urls.static import static
from django.contrib.auth.views import PasswordResetView
from django.urls import path, include
from django.conf import settings


urlpatterns = [
    # admin urls
    path('', include('apps.administrator.urls')),
    # enable the admin interface
    re_path(r'^administration/', admin.site.urls),
    # auth urls
    path('password_reset/', PasswordResetView.as_view(
        html_email_template_name='registration/password_reset_email.html'
    ), name='password_reset'),
    # path('', include('django.contrib.auth.urls')),
    # CKEditor 5 urls
    path('ckeditor5/', include('django_ckeditor_5.urls')),
    # user urls
    path('', include('apps.user.urls')),
    # ai chatbot urls
    path('', include('ai_chatbot.urls')),

]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL,
                          document_root=settings.STATIC_ROOT)
