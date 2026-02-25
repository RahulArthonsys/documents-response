"""application admin URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from .views import (
    AdminLoginView, AdminDashboardView, AdminLogoutView, AdminChangePasswordView,
    SubscriptionPlanListView, SubscriptionPlanCreateView,
    SubscriptionPlanUpdateView, SubscriptionPlanDeleteView,
)
from django.urls import path, include


urlpatterns = [
    # Auth Urls
    path('admin/login/', AdminLoginView.as_view(), name="admin-login"),
    path('admin/dashboard/', AdminDashboardView.as_view(), name="admin-dashboard"),
    path('admin/logout/', AdminLogoutView.as_view(), name="admin-logout"),
    path('admin/change-password/', AdminChangePasswordView.as_view(), name="admin-change-password"),

    # Subscription Plans
    path('admin/subscription-plans/', SubscriptionPlanListView.as_view(), name="admin-subscription-plan-list"),
    path('admin/subscription-plans/add/', SubscriptionPlanCreateView.as_view(), name="admin-subscription-plan-add"),
    path('admin/subscription-plans/edit/<int:pk>/', SubscriptionPlanUpdateView.as_view(), name="admin-subscription-plan-edit"),
    path('admin/subscription-plans/delete/<int:pk>/', SubscriptionPlanDeleteView.as_view(), name="admin-subscription-plan-delete"),
]

