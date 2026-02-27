import datetime
from random import randint
import pandas as pd
from django.contrib import messages
from django.contrib.auth import get_user_model, authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.forms import SetPasswordForm, PasswordChangeForm
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.messages.views import SuccessMessageMixin
from django.db.models import Q
from django.http import JsonResponse, HttpResponseRedirect
from django.shortcuts import render, get_object_or_404, redirect

from django.urls import reverse_lazy, reverse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.cache import never_cache
from django.views.generic import CreateView, DeleteView, UpdateView, TemplateView
from application.custom_classes import AjayDatatableView, AdminRequiredMixin, UserRequiredMixin

from .forms import CreateUserForm, EditUserForm, UserSignupForm, EditUserProfileForm
User = get_user_model()


class CreateUserView(AdminRequiredMixin, LoginRequiredMixin, SuccessMessageMixin, CreateView):
    model = User
    form_class = CreateUserForm
    template_name = 'admin/user/form.html'
    success_message = "User created successfully"
    success_url = reverse_lazy('admin-user-list')


class UpdateUserView(AdminRequiredMixin, LoginRequiredMixin, SuccessMessageMixin, UpdateView):
    model = User
    form_class = EditUserForm
    template_name = 'admin/user/form.html'
    success_message = "User updated successfully"
    success_url = reverse_lazy('admin-user-list')


class ListUserView(AdminRequiredMixin, LoginRequiredMixin, TemplateView):
    template_name = 'admin/user/list.html'


class ListUserViewJson(AdminRequiredMixin, AjayDatatableView):
    model = User
    column_defs = [
        {'name': 'first_name', 'title': 'First Name', 'visible': True, 'searchable': True, 'orderable': True},
        {'name': 'last_name',  'title': 'Last Name',  'visible': True, 'searchable': True, 'orderable': True},
        {'name': 'email',      'title': 'Email',      'visible': True, 'searchable': True, 'orderable': True},
        {'name': 'is_active',  'title': 'Status',     'visible': True, 'searchable': False, 'orderable': True},
        {'name': 'actions',    'title': 'Actions',    'visible': True, 'searchable': False, 'orderable': False, 'placeholder': True, 'className': 'text-right'},
    ]
    # extra_search_columns = ['']

    def get_initial_queryset(self):
        return self.model.objects.filter(Q(is_staff=False) | Q(is_superuser=False))

    def render_column(self, row, column):
        if column == 'is_active':
            if row.is_active:
                return '<span class="badge" style="background:rgba(34,197,94,.12);color:#16a34a;font-weight:600;padding:4px 10px;border-radius:6px;font-size:.75rem;">Active</span>'
            else:
                return '<span class="badge" style="background:rgba(239,68,68,.1);color:#dc2626;font-weight:600;padding:4px 10px;border-radius:6px;font-size:.75rem;">Inactive</span>'

        if column == 'actions':
            edit_action = (
                '<a href="{}" class="btn btn-sm mr-1" '
                'style="background:rgba(79,142,247,.1);color:#4f8ef7;border-radius:8px;font-size:.76rem;font-weight:600;padding:4px 12px;border:none;">'
                '<i class="fas fa-pen" style="font-size:.68rem;margin-right:3px;"></i>Edit</a>'
            ).format(reverse('admin-user-edit', kwargs={'pk': row.pk}))
            delete_action = (
                '<a href="javascript:;" class="remove_record btn btn-sm" '
                'data-url="{}" '
                'style="background:rgba(239,68,68,.1);color:#ef4444;border-radius:8px;font-size:.76rem;font-weight:600;padding:4px 12px;border:none;">'
                '<i class="fas fa-trash-alt" style="font-size:.68rem;margin-right:3px;"></i>Delete</a>'
            ).format(reverse('admin-user-delete', kwargs={'pk': row.pk}))
            return edit_action + delete_action
        else:
            return super(ListUserViewJson, self).render_column(row, column)


class DetailUserView(AdminRequiredMixin, LoginRequiredMixin, TemplateView):
    template_name = 'admin/user/detail.html'

    def get_context_data(self, **kwargs):
        context = super(DetailUserView, self).get_context_data(**kwargs)
        context['user'] = User.objects.get(id=kwargs['pk'])
        return context


class DeleteUserView(AdminRequiredMixin, LoginRequiredMixin, DeleteView):
    model = User

    def delete(self, request, *args, **kwargs):
        self.get_object().delete()
        payload = {'delete': 'ok'}
        return JsonResponse(payload)


class ChangeUserPasswordView(AdminRequiredMixin, LoginRequiredMixin, View):
    form_class = SetPasswordForm
    template_name = 'admin/user/change_password.html'
    success_message = 'Password Updated Successfully!'

    def get(self, request, user_id, *args, **kwargs):
        form = self.form_class(get_object_or_404(User, pk=user_id))
        return render(request, self.template_name, {'form': form})

    def post(self, request, user_id, *args, **kwargs):
        form = self.form_class(get_object_or_404(User, pk=user_id), request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, self.success_message)
        else:
            return render(request, self.template_name, {'form': form})

        return HttpResponseRedirect(reverse('admin-user-list'))


class ListEmporiaView(AdminRequiredMixin, LoginRequiredMixin, TemplateView):
    template_name = 'admin/user/emporia_list.html'


@method_decorator(never_cache, name='dispatch')
class LandingView(View):
    admin_home_url = 'admin-dashboard'
    user_home_url = 'user-home'
    user_login_url = 'user-login'

    def get(self, request):
        if request.user.is_authenticated and not (request.user.is_superuser or request.user.is_staff):
            return HttpResponseRedirect(reverse(self.user_home_url))
        if request.user.is_authenticated and (request.user.is_superuser or request.user.is_staff):
            return HttpResponseRedirect(reverse(self.admin_home_url))
        return HttpResponseRedirect(reverse(self.user_login_url))


# Frontend user views

@method_decorator(never_cache, name='dispatch')
class LoginView(View):
    template_name = 'user/user/login.html'
    success_url = 'user-home'
    login_url = 'user-login'
    success_message = 'You have successfully logged in.'
    failure_message = 'Please check credentials.'

    def get(self, request):
        if request.user.is_authenticated and not (request.user.is_superuser or request.user.is_staff):
            return HttpResponseRedirect(reverse(self.success_url))
        return render(request, self.template_name)

    def post(self, request):
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username,
                            password=password)
        if user and not (user.is_superuser or user.is_staff):
            login(request, user)
            messages.success(request, self.success_message)
            return HttpResponseRedirect(reverse(self.success_url))
        else:
            messages.error(request, self.failure_message)
            return HttpResponseRedirect(reverse(self.login_url))


class RegisterView(SuccessMessageMixin, CreateView):
    model = User
    form_class = UserSignupForm
    template_name = 'user/user/register.html'
    success_message = "You have registered successfully"
    success_url = reverse_lazy('user-login')


class LogoutView(UserRequiredMixin, LoginRequiredMixin, View):

    def get(self, request):
        logout(request)
        messages.success(request, 'You have successfully logged out.')
        return redirect('user-login')


class UpdateUserProfileView(UserRequiredMixin, LoginRequiredMixin, SuccessMessageMixin, UpdateView):
    model = User
    form_class = EditUserProfileForm
    template_name = 'user/user/profile.html'
    success_message = "Profile updated successfully"
    success_url = reverse_lazy('user-home')

    def get_object(self, queryset=None):
        return self.request.user


class ChangeUserSelfPasswordView(LoginRequiredMixin, View):
    template_name = 'user/user/change_password.html'

    def get(self, request):
        form = PasswordChangeForm(request.user)
        return render(request, self.template_name, {'form': form})

    def post(self, request, *args, **kwargs):
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  # Important!
            messages.success(request, 'Your password has been successfully updated!')
        else:
            messages.error(request, 'Error occurred while changing password, please enter a proper password.')
            return render(request, self.template_name, {'form': form})
        return redirect('user-profile')


class HomePageView(UserRequiredMixin, TemplateView):
    template_name = 'user/user/home.html'

    def get_context_data(self, **kwargs):
        kwargs = super(HomePageView, self).get_context_data(**kwargs)
        return kwargs


