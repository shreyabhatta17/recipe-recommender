"""
Loads the processed recipes.parquet from Day 1's Colab pipeline into the
live Recipe table. Run once after `migrate`, and again any time the
underlying recipe catalogue changes.

Usage:
    python manage.py import_recipes --path data/recipes.parquet
"""
import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from api.models import Recipe


class Command(BaseCommand):
    help = "Import recipes.parquet (from the Colab pipeline) into the Recipe table."

    def add_arguments(self, parser):
        parser.add_argument("--path", default="data/recipes.parquet", help="Path to recipes.parquet")
        parser.add_argument("--batch-size", type=int, default=1000)

    def handle(self, *args, **options):
        path = options["path"]
        try:
            df = pd.read_parquet(path)
        except FileNotFoundError:
            raise CommandError(
                f"Couldn't find {path}. Copy recipes.parquet from your Colab "
                f"Drive folder (recipe_recommender/processed/recipes.parquet) into this project first."
            )

        self.stdout.write(f"Loaded {len(df):,} recipes from {path}")

        has_tags = "tags" in df.columns
        has_minutes = "minutes" in df.columns
        has_n_steps = "n_steps" in df.columns
        has_steps = "steps" in df.columns

        recipes = []
        for row in df.itertuples(index=False):
            raw_ingredients = row.ingredients.tolist() if hasattr(row.ingredients, "tolist") else row.ingredients
            raw_steps = row.steps.tolist() if has_steps and hasattr(row.steps, "tolist") else row.steps if has_steps else []
            recipes.append(Recipe(
                recipe_id=int(row.recipe_id),
                title=str(row.name)[:500] if pd.notna(row.name) else "",
                ingredients=raw_ingredients if isinstance(raw_ingredients, list) else [],
                steps=repr(list(raw_steps)) if isinstance(raw_steps, (list, tuple)) else "",
                description=str(row.description) if pd.notna(row.description) else "",
                tags=list(row.tags) if has_tags and isinstance(row.tags, list) else [],
                minutes=int(row.minutes) if has_minutes and pd.notna(row.minutes) else None,
                n_steps=int(row.n_steps) if has_n_steps and pd.notna(row.n_steps) else None,
            ))

        batch_size = options["batch_size"]
        Recipe.objects.bulk_create(recipes, batch_size=batch_size, ignore_conflicts=True)
        self.stdout.write(self.style.SUCCESS(f"Imported {len(recipes):,} recipes."))
