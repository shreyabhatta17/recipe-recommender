import ast

from django.shortcuts import get_object_or_404
from django.views.generic import TemplateView

from api.models import Interaction, Rating, Recipe


def parse_recipe_list(value):
    """Return readable list data from JSON lists or Food.com-style strings."""
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return [raw]
        if isinstance(parsed, (list, tuple)):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [raw]
    return []


class KitchenView(TemplateView):
    template_name = "kitchen.html"
    placeholder_palettes = [
        ("#dfe9d8", "#35644d"),
        ("#f4e0ae", "#8d6824"),
        ("#ead8d7", "#8b4e4a"),
        ("#d8e6e2", "#2f6654"),
        ("#e6dfca", "#71603b"),
    ]

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        cooked = []
        seen_recipe_ids = set()

        interactions = (
            Interaction.objects
            .filter(user=self.request.user, action=Interaction.COOKED)
            .select_related("recipe")
            .order_by("-created_at", "-pk")
        )
        ratings_by_recipe_id = {
            rating.recipe_id: rating
            for rating in Rating.objects.filter(
                user=self.request.user,
                recipe_id__in=interactions.values("recipe_id"),
            )
        }
        for interaction in interactions:
            if interaction.recipe_id in seen_recipe_ids:
                continue
            seen_recipe_ids.add(interaction.recipe_id)
            title = interaction.recipe.title or "Recipe"
            title_hash = sum((index + 1) * ord(character) for index, character in enumerate(title))
            background, foreground = self.placeholder_palettes[title_hash % len(self.placeholder_palettes)]
            rating = ratings_by_recipe_id.get(interaction.recipe_id)
            score = rating.score if rating else 0
            cooked.append({
                "interaction": interaction,
                "recipe": interaction.recipe,
                "placeholder_background": background,
                "placeholder_foreground": foreground,
                "initial": title.strip()[:1].upper() or "R",
                "rating": score,
                "rating_stars": [score >= value for value in range(1, 6)],
            })

        context["cooked_recipes"] = cooked
        return context


class RecipeDetailView(TemplateView):
    template_name = "recipe_detail.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        recipe = get_object_or_404(Recipe, recipe_id=kwargs["recipe_id"])
        saved_rating = Rating.objects.filter(user=self.request.user, recipe=recipe).first()
        context.update({
            "recipe": recipe,
            "description": recipe.description.strip() or "Not available",
            "ingredients": parse_recipe_list(recipe.ingredients),
            "steps": parse_recipe_list(getattr(recipe, "steps", None)),
            "rating": saved_rating.score if saved_rating else 0,
        })
        return context
