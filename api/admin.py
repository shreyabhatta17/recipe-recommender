from django.contrib import admin

from .models import Interaction, Rating, Recipe


@admin.register(Recipe)
class RecipeAdmin(admin.ModelAdmin):
    list_display = ("recipe_id", "title", "minutes", "n_steps")
    search_fields = ("title", "recipe_id")
    list_filter = ("n_steps",)
    ordering = ("recipe_id",)


@admin.register(Rating)
class RatingAdmin(admin.ModelAdmin):
    list_display = ("user", "recipe", "score", "created_at")
    list_filter = ("score", "created_at")
    search_fields = ("user__username", "recipe__title", "recipe__recipe_id")
    autocomplete_fields = ("recipe",)
    date_hierarchy = "created_at"


@admin.register(Interaction)
class InteractionAdmin(admin.ModelAdmin):
    list_display = ("user", "recipe", "action", "created_at")
    list_filter = ("action", "created_at")
    search_fields = ("user__username", "recipe__title", "recipe__recipe_id")
    autocomplete_fields = ("recipe",)
    date_hierarchy = "created_at"
