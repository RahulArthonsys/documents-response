import datetime
import json
from datetime import date
from random import randint

from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views import View
from django.utils.decorators import method_decorator
from django.shortcuts import render, redirect
from django.urls import reverse, reverse_lazy
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib import messages
from django.views.decorators.cache import never_cache
from django.utils import timezone
from django.db.models import Count
from django.db.models.functions import TruncMonth
from django.contrib.auth import get_user_model
from django.views.generic import ListView, CreateView, UpdateView, DeleteView
from django.contrib.messages.views import SuccessMessageMixin
from application.custom_classes import AdminRequiredMixin
from administrator.models import SubscriptionPlan, UserSubscription

User = get_user_model()
from django.http import HttpResponseRedirect


@method_decorator(never_cache, name='dispatch')
class AdminLoginView(View):
    template_name = 'administrator/login.html'
    success_url = 'admin-dashboard'
    login_url = 'admin-login'
    success_message = 'You have successfully logged in.'
    failure_message = 'Please check credentials.'

    def get(self, request):
        if request.user.is_authenticated and (request.user.is_superuser or request.user.is_staff):
            return HttpResponseRedirect(reverse(self.success_url))
        return render(request, self.template_name)

    def post(self, request):
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username,
                            password=password)
        if user and (user.is_superuser or user.is_staff):
            login(request, user)
            messages.success(request, self.success_message)
            return HttpResponseRedirect(reverse(self.success_url))
        else:
            messages.error(request, self.failure_message)
            return HttpResponseRedirect(reverse(self.login_url))


class AdminLogoutView(AdminRequiredMixin, LoginRequiredMixin, View):
    def get(self, request):
        logout(request)
        messages.success(request, 'You have successfully logged out.')
        return redirect('admin-login')


class AdminChangePasswordView(AdminRequiredMixin, LoginRequiredMixin, View):
    template_name = 'administrator/change_password.html'

    def get(self, request):
        form = PasswordChangeForm(request.user)
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  # Important!
            messages.success(request, 'Your password has been successfully updated!')
        else:
            messages.error(request, 'Error occured while changing password, please enter a proper password.')
            return render(request, self.template_name, {'form': form})
        return redirect('admin-dashboard')


class AdminDashboardView(AdminRequiredMixin, LoginRequiredMixin, View):

    def get(self, request):
        now = timezone.now()
        last_30_days = now - datetime.timedelta(days=30)
        # Core counts
        users_count = User.objects.filter(is_staff=False, is_superuser=False).count()
        active_users_count = User.objects.filter(is_staff=False, is_superuser=False, is_active=True).count()
        new_users_count = User.objects.filter(is_staff=False, is_superuser=False, date_joined__gte=last_30_days).count()

        inactive_users_count = users_count - active_users_count

        # Subscription stats
        plans_count = SubscriptionPlan.objects.count()
        active_plans_count = SubscriptionPlan.objects.filter(is_active=True).count()
        active_subs_count = UserSubscription.objects.filter(
            is_active=True, end_date__gte=timezone.now().date()
        ).count()
        recent_plans = SubscriptionPlan.objects.order_by('-created_at')[:5]

        # Recent users
        recent_users = User.objects.filter(is_staff=False, is_superuser=False).order_by('-date_joined')[:10]

        # Monthly registrations for the past 12 months
        twelve_months_ago = now - datetime.timedelta(days=365)
        monthly_data = (
            User.objects
            .filter(is_staff=False, is_superuser=False, date_joined__gte=twelve_months_ago)
            .annotate(month=TruncMonth('date_joined'))
            .values('month')
            .annotate(count=Count('id'))
            .order_by('month')
        )

        chart_labels = []
        chart_values = []
        for entry in monthly_data:
            chart_labels.append(entry['month'].strftime('%b %Y'))
            chart_values.append(entry['count'])

        context = {
            'users_count': users_count,
            'active_users_count': active_users_count,
            'inactive_users_count': inactive_users_count,
            'new_users_count': new_users_count,
            'recent_users': recent_users,
            'chart_labels': json.dumps(chart_labels),
            'chart_values': json.dumps(chart_values),
            'plans_count': plans_count,
            'active_plans_count': active_plans_count,
            'active_subs_count': active_subs_count,
            'recent_plans': recent_plans,
        }
        return render(request, 'administrator/dashboard.html', context)


# ── Subscription Plan CRUD ──────────────────────────────────────────────────

class SubscriptionPlanListView(AdminRequiredMixin, LoginRequiredMixin, ListView):
    model = SubscriptionPlan
    template_name = 'administrator/subscription_plan_list.html'
    context_object_name = 'plans'


class SubscriptionPlanCreateView(AdminRequiredMixin, LoginRequiredMixin, SuccessMessageMixin, CreateView):
    model = SubscriptionPlan
    template_name = 'administrator/subscription_plan_form.html'
    fields = ['name', 'description', 'price', 'duration_days', 'is_active']
    success_message = 'Subscription plan created successfully.'
    success_url = reverse_lazy('admin-subscription-plan-list')


class SubscriptionPlanUpdateView(AdminRequiredMixin, LoginRequiredMixin, SuccessMessageMixin, UpdateView):
    model = SubscriptionPlan
    template_name = 'administrator/subscription_plan_form.html'
    fields = ['name', 'description', 'price', 'duration_days', 'is_active']
    success_message = 'Subscription plan updated successfully.'
    success_url = reverse_lazy('admin-subscription-plan-list')


class SubscriptionPlanDeleteView(AdminRequiredMixin, LoginRequiredMixin, DeleteView):
    model = SubscriptionPlan
    success_url = reverse_lazy('admin-subscription-plan-list')

    def delete(self, request, *args, **kwargs):
        self.get_object().delete()
        from django.http import JsonResponse
        return JsonResponse({'delete': 'ok'})

