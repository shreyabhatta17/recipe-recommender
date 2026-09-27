from django.contrib import admin
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView, LogoutView
from django.views.generic import TemplateView
from django.urls import include, path

from .views import KitchenView, RecipeDetailView


class RecommendationsPageView(TemplateView):
    template_name = "recommendations.html"


recommendations_page = login_required(RecommendationsPageView.as_view())
kitchen_page = login_required(KitchenView.as_view())
recipe_detail_page = login_required(RecipeDetailView.as_view())

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("api.urls")),
    path("", recommendations_page, name="recommendations"),
    path("kitchen/", kitchen_page, name="kitchen"),
    path("recipe/<int:recipe_id>/", recipe_detail_page, name="recipe-detail"),
    path("accounts/login/", LoginView.as_view(template_name="registration/login.html"), name="login"),
    path("accounts/logout/", LogoutView.as_view(), name="logout"),
]
