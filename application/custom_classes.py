from django import http
from django.contrib.auth.mixins import AccessMixin
from django.db.models import Q
from django.urls import reverse_lazy
from ajax_datatable.views import AjaxDatatableView
from django.db import models
from django.http import Http404

from django.contrib.auth.views import redirect_to_login


class AjayDatatableView(AjaxDatatableView):
    extra_search_columns = []
    exclude_from_search_columns = []


class UserRequiredMixin(AccessMixin):
    """Verify that the current user is authenticated."""
    login_url = 'user-login'

    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect_to_login(self.request.get_full_path(), self.get_login_url(), self.get_redirect_field_name())
        elif not (request.user.is_superuser or request.user.is_staff):
            return super().dispatch(request, *args, **kwargs)
        else:
            raise Http404


class AdminRequiredMixin(AccessMixin):
    login_url = 'admin-login'

    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect_to_login(self.request.get_full_path(), self.get_login_url(), self.get_redirect_field_name())
        elif request.user.is_superuser or request.user.is_staff:
            return super().dispatch(request, *args, **kwargs)
        raise Http404


class CorsMiddleware(object):
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        if (request.method == "OPTIONS"  and "HTTP_ACCESS_CONTROL_REQUEST_METHOD" in request.META):
            response = http.HttpResponse()
            response["Content-Length"] = "0"
            response["Access-Control-Max-Age"] = 86400
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "DELETE, GET, OPTIONS, PATCH, POST, PUT"
        response["Access-Control-Allow-Headers"] = "accept, accept-encoding, authorization, content-type, dnt, origin, user-agent, x-csrftoken, x-requested-with"
        return response