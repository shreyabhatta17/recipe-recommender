from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from . import services
from .models import Rating, Recipe
from .serializers import InteractionSerializer, RatingSerializer, RecommendationSerializer


class RecommendationsTodayView(APIView):
    """
    GET /api/recommendations/today/
    All recommendation logic lives in services.get_recommendations_for_user()
    and the untouched recommender package it calls -- this view is just the
    HTTP wrapper: auth, call, serialize, respond.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            raw = services.get_recommendations_for_user(request.user.id, k=10)
        except services.RecommenderNotReady:
            return Response({"detail": "Recommender not ready"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        recipes_by_id = Recipe.objects.in_bulk([r["recipe_id"] for r in raw])
        ratings_by_recipe_id = {
            rating.recipe_id: rating
            for rating in Rating.objects.filter(
                user=request.user,
                recipe_id__in=recipes_by_id,
            )
        }
        results = []
        for r in raw:
            recipe = recipes_by_id.get(r["recipe_id"])
            if recipe is None:
                continue  # shouldn't happen once import_recipes has run fully, but don't 500 on a gap
            results.append({
                "recipe_id": r["recipe_id"],
                "score": r["score"],
                "title": recipe.title,
                "minutes": recipe.minutes,
                "image_url": recipe.image_url,
                "rating": ratings_by_recipe_id.get(recipe.recipe_id).score
                if recipe.recipe_id in ratings_by_recipe_id else None,
            })

        serializer = RecommendationSerializer(results, many=True)
        return Response(serializer.data)


class InteractionCreateView(APIView):
    """POST /api/interactions/ -- records a view or a cook, feeding the re-ranker's recency filter."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = InteractionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save(user=request.user)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class RatingCreateView(APIView):
    """POST /api/ratings/ -- create or update the current user's recipe rating."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = RatingSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        recipe = serializer.validated_data["recipe"]
        score = serializer.validated_data["score"]
        rating, created = Rating.objects.update_or_create(
            user=request.user,
            recipe=recipe,
            defaults={"score": score},
        )
        return Response(
            {"recipe_id": rating.recipe_id, "score": rating.score},
            status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
        )
