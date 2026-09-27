from rest_framework import serializers

from .models import Interaction, Recipe


class RecipeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recipe
        fields = ["recipe_id", "title", "minutes", "n_steps", "image_url"]


class RecommendationSerializer(serializers.Serializer):
    """
    Shapes engine.get_recommendations()'s raw output (recipe_id + score)
    into a response that also carries the recipe's display fields, without
    engine.py needing to know anything about Django models.
    """
    recipe_id = serializers.IntegerField()
    score = serializers.FloatField()
    title = serializers.CharField()
    minutes = serializers.IntegerField(allow_null=True)
    image_url = serializers.CharField(allow_blank=True)
    rating = serializers.IntegerField(allow_null=True, required=False)


class InteractionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Interaction
        fields = ["recipe", "action"]


class RatingSerializer(serializers.Serializer):
    recipe_id = serializers.PrimaryKeyRelatedField(source="recipe", queryset=Recipe.objects.all())
    score = serializers.IntegerField(min_value=0, max_value=5)
