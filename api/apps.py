import json
import pickle
from pathlib import Path

from django.apps import AppConfig
from django.conf import settings


class ApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "api"

    def ready(self):
        """
        Load the Colab-trained artefacts ONCE when the server process starts,
        not per-request. Populates module-level globals in api.recommender_state
        that views.py reads from. If artefacts are missing (e.g. first-time
        setup before running the import), logs a warning instead of crashing
        the whole server -- lets `manage.py migrate` etc. still work.
        """
        from api import recommender_state as state

        artifact_dir = Path(settings.BASE_DIR) / "artifacts"

        try:
            state.vectorizer = pickle.load(open(artifact_dir / "tfidf_vectorizer.pkl", "rb"))
            state.tfidf_matrix = pickle.load(open(artifact_dir / "tfidf_matrix.pkl", "rb"))
            state.valid_mask = pickle.load(open(artifact_dir / "valid_mask.pkl", "rb"))
            state.coll_model = pickle.load(open(artifact_dir / "coll_model.pkl", "rb"))
            index_maps = pickle.load(open(artifact_dir / "index_maps.pkl", "rb"))
            state.user_index = index_maps["user_index"]
            state.item_index = index_maps["item_index"]
            state.index_item = index_maps["index_item"]
            state.recipe_id_to_row = index_maps["recipe_id_to_row"]
            state.row_to_recipe_id = index_maps["row_to_recipe_id"]

            config = json.load(open(artifact_dir / "config.json"))
            state.best_alpha = config["best_alpha"]
            state.best_lambda = config["best_lambda"]
            state.cold_start_threshold = config["cold_start_threshold"]

            state.cb_kwargs = dict(
                tfidf_matrix=state.tfidf_matrix,
                recipe_id_to_row=state.recipe_id_to_row,
                row_to_recipe_id=state.row_to_recipe_id,
                valid_mask=state.valid_mask,
            )
            state.coll_kwargs = dict(
                user_index=state.user_index,
                item_index=state.item_index,
                index_item=state.index_item,
            )
            state.loaded = True
            print(f"[api] recommender artefacts loaded from {artifact_dir}")
        except FileNotFoundError as e:
            state.loaded = False
            print(f"[api] WARNING: recommender artefacts not found ({e}). "
                  f"Run `python manage.py import_recipes` after placing artifacts/ "
                  f"in the project root. API will return 503 until then.")
