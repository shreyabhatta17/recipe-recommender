from django.urls import path

from . import views

urlpatterns = [
    path("recommendations/today/", views.RecommendationsTodayView.as_view(), name="recommendations-today"),
    path("interactions/", views.InteractionCreateView.as_view(), name="interactions-create"),
    path("ratings/", views.RatingCreateView.as_view(), name="ratings-create"),
]
