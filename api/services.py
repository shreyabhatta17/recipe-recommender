"""
Bridges the Django ORM (live database) and the recommender package (expects
a pandas DataFrame + plain dicts, exactly as it did in Colab). This is the
"service layer" the proposal describes -- all recommendation logic funnels
through here, so views.py stays a thin HTTP wrapper with zero ML logic.
"""
import pandas as pd
from django.db.models import Count
from django.utils import timezone

from recommender import engine
from recommender.reranker import recent_recipe_ids

from . import recommender_state as state
from .models import Interaction, Rating


class RecommenderNotReady(Exception):
    """Raised when api.apps.ApiConfig.ready() hasn't successfully loaded artefacts yet."""


def _user_ratings_df(user_id) -> pd.DataFrame:
    """
    A tiny DataFrame of just this user's ratings, shaped exactly like the
    `train` DataFrame from Colab. content_based.liked_items() only ever
    filters by this one user_id, so we don't need the full training set
    live -- just this user's rows.
    """
    rows = Rating.objects.filter(user_id=user_id).values("user_id", "recipe_id", "score")
    df = pd.DataFrame(list(rows))
    if df.empty:
        return pd.DataFrame(columns=["user_id", "recipe_id", "rating"])
    return df.rename(columns={"score": "rating"})


def _recent_recipe_ids(user_id, recency_days: int) -> set:
    """Bridges Interaction (Django model) -> recent_recipe_ids (expects a DataFrame)."""
    rows = Interaction.objects.filter(user_id=user_id).values("user_id", "recipe_id", "created_at")
    df = pd.DataFrame(list(rows))
    if df.empty:
        return set()
    df = df.rename(columns={"created_at": "date"})
    return recent_recipe_ids(user_id, df, reference_date=timezone.now(), recency_days=recency_days)


def get_recommendations_for_user(user_id, k: int = 10, recency_days: int = 14) -> list[dict]:
    """
    The one function views.py calls. Mirrors engine.get_recommendations()'s
    Colab signature, but sources `train`, `train_counts`, `seen`, and
    `recently_seen` from the live database instead of Colab's pickled frames.
    """
    if not state.loaded:
        raise RecommenderNotReady("Recommender artefacts not loaded -- run import_recipes / check artifacts/ directory")

    train = _user_ratings_df(user_id)
    seen = set(train["recipe_id"].tolist())
    train_counts = {user_id: len(seen)}

    recently_seen = _recent_recipe_ids(user_id, recency_days)

    recs = engine.get_recommendations(
        user_id,
        train,
        train_counts,
        state.cb_kwargs,
        state.coll_model,
        state.coll_kwargs,
        seen,
        k=k,
        alpha=state.best_alpha,
        lambda_div=state.best_lambda,
        recently_seen=recently_seen,
    )

    if not recs:
        # Zero-history users: content-based has no liked items to build a
        # profile from, and collaborative has no row for them either -- both
        # legitimately return nothing. Fall back to popularity so a brand
        # new visitor still sees something instead of an empty page.
        recs = _popularity_fallback(k, exclude=seen)
    return recs


def _popularity_fallback(k: int, exclude: set) -> list[dict]:
    """Most-rated recipes in the live DB, computed on demand (not pickled --
    ratings keep accumulating after Colab, so a live count stays accurate)."""
    top = (
        Rating.objects.exclude(recipe_id__in=exclude)
        .values("recipe_id")
        .annotate(n=Count("id"))
        .order_by("-n")[:k]
    )
    return [{"recipe_id": row["recipe_id"], "score": float(row["n"])} for row in top]
