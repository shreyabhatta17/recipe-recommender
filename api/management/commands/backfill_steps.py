import ast
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from api.models import Recipe


class Command(BaseCommand):
    help = "Backfill Recipe.steps from RAW_recipes.csv or the processed recipes parquet."

    def add_arguments(self, parser):
        parser.add_argument(
            "--path",
            default=None,
            help="Path to RAW_recipes.csv or a processed recipes parquet file.",
        )
        parser.add_argument("--batch-size", type=int, default=1000)

    @staticmethod
    def serialize_steps(value):
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return repr([value]) if value.strip() else ""
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return repr([str(step).strip() for step in value if str(step).strip()])
        return ""

    def handle(self, *args, **options):
        project_root = Path(__file__).resolve().parents[3]
        requested_path = Path(options["path"]) if options["path"] else None
        if requested_path:
            source_path = requested_path
        else:
            csv_path = project_root / "data" / "RAW_recipes.csv"
            parquet_path = project_root / "data" / "recipes.parquet"
            source_path = csv_path if csv_path.exists() else parquet_path
            if not csv_path.exists() and parquet_path.exists():
                self.stdout.write("RAW_recipes.csv not found; using data/recipes.parquet fallback.")

        if not source_path.exists():
            raise CommandError(f"Could not find source data at {source_path}")

        if source_path.suffix.lower() == ".csv":
            dataframe = pd.read_csv(source_path)
        elif source_path.suffix.lower() == ".parquet":
            dataframe = pd.read_parquet(source_path)
        else:
            raise CommandError("Source must be a .csv or .parquet file")

        id_column = "recipe_id" if "recipe_id" in dataframe.columns else "id" if "id" in dataframe.columns else None
        if not id_column or "steps" not in dataframe.columns:
            raise CommandError("Source must contain an id/recipe_id column and a steps column")

        recipes_by_id = Recipe.objects.in_bulk()
        updates = []
        matched = 0
        unmatched = []
        for row in dataframe[[id_column, "steps"]].itertuples(index=False, name=None):
            try:
                recipe_id = int(row[0])
            except (TypeError, ValueError):
                unmatched.append(row[0])
                continue
            recipe = recipes_by_id.get(recipe_id)
            if recipe is None:
                unmatched.append(recipe_id)
                continue
            recipe.steps = self.serialize_steps(row[1])
            updates.append(recipe)
            matched += 1
            if len(updates) >= options["batch_size"]:
                Recipe.objects.bulk_update(updates, ["steps"], batch_size=options["batch_size"])
                updates.clear()
                self.stdout.write(f"Processed {matched:,} matched recipes...")

        if updates:
            Recipe.objects.bulk_update(updates, ["steps"], batch_size=options["batch_size"])

        self.stdout.write(self.style.SUCCESS(
            f"Backfilled steps for {matched:,} recipes; {len(unmatched):,} source recipe_ids could not be matched."
        ))
        if unmatched:
            self.stdout.write("Unmatched recipe_ids: " + ", ".join(str(recipe_id) for recipe_id in unmatched))
