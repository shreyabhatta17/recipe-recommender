from django.conf import settings
from django.db import models


class Recipe(models.Model):
    """
    Mirrors the recipe_id from the Colab pipeline exactly -- recipe_id here
    IS the recipe_id the pickled artefacts (recipe_id_to_row, etc.) index by.
    Kept as a plain IntegerField primary key (not Django's default
    auto-incrementing id) so imported data lines up with the notebook
    without any id-remapping step.
    """
    recipe_id = models.IntegerField(primary_key=True)
    title = models.CharField(max_length=500)
    ingredients = models.JSONField(default=list)
    steps = models.TextField(blank=True, default="")
    description = models.TextField(blank=True, default="")
    tags = models.JSONField(default=list)
    minutes = models.PositiveIntegerField(null=True, blank=True)
    n_steps = models.PositiveIntegerField(null=True, blank=True)
    image_url = models.URLField(blank=True, default="")

    def __str__(self):
        return self.title


class Rating(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="ratings")
    recipe = models.ForeignKey(Recipe, on_delete=models.CASCADE, related_name="ratings")
    score = models.PositiveSmallIntegerField()  # 0-5, matches the dataset's rating scale
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "recipe")

    def __str__(self):
        return f"{self.user_id} -> {self.recipe_id}: {self.score}"


class Interaction(models.Model):
    """
    Feeds the re-ranker's recency filter (see recommender.reranker). Distinct
    from Rating: a view or a cook doesn't require a star rating, and this is
    what lets the re-ranker suppress something the user looked at yesterday
    even if they never rated it.
    """
    VIEWED = "viewed"
    COOKED = "cooked"
    ACTION_CHOICES = [(VIEWED, "Viewed"), (COOKED, "Cooked")]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="interactions")
    recipe = models.ForeignKey(Recipe, on_delete=models.CASCADE, related_name="interactions")
    action = models.CharField(max_length=10, choices=ACTION_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["user", "created_at"])]

    def __str__(self):
        return f"{self.user_id} {self.action} {self.recipe_id}"
