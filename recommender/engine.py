"""
Orchestration layer: blends content-based and collaborative scores, with a
cold-start switch for users with too little rating history for CF to help.

Day 2 scope: candidate generation + blending only. The re-ranker (recency +
diversity) plugs into get_recommendations() on Day 3 — see the TODO below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import content_based as cb
from . import collaborative as coll
from . import reranker

COLD_START_THRESHOLD = 5  # fewer than this many training ratings -> content-only


def _normalize(scores: dict) -> dict:
    """Min-max scale a {id: score} dict to [0, 1]. Empty/flat input returns as-is."""
    if not scores:
        return scores
    values = np.array(list(scores.values()))
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def blend(
    user_id,
    train: pd.DataFrame,
    train_counts: dict,
    cb_kwargs: dict,
    coll_model,
    coll_kwargs: dict,
    seen: set,
    alpha: float = 0.5,
    n_candidates: int = 100,
) -> list[tuple]:
    """
    Returns [(recipe_id, blended_score), ...] sorted descending.

    cb_kwargs:   dict of the fixed args content_based.recommend needs besides
                 user_id/train/seen (tfidf_matrix, recipe_id_to_row, row_to_recipe_id, valid_mask)
    coll_kwargs: dict of the fixed args collaborative.recommend needs besides
                 model/user_id/seen_recipe_ids (user_index, item_index, index_item)
    """
    n_train = train_counts.get(user_id, 0)

    content_candidates = cb.recommend(user_id, train, seen=seen, n=n_candidates, **cb_kwargs)
    content_scores = _normalize(dict(content_candidates))

    if n_train < COLD_START_THRESHOLD:
        # cold start: content-only, CF has nothing reliable to say yet
        ranked = sorted(content_scores.items(), key=lambda kv: -kv[1])
        return ranked

    coll_candidates = coll.recommend(coll_model, user_id, seen_recipe_ids=seen, n=n_candidates, **coll_kwargs)
    coll_scores = _normalize(dict(coll_candidates))

    all_ids = set(content_scores) | set(coll_scores)
    blended = {
        rid: alpha * coll_scores.get(rid, 0.0) + (1 - alpha) * content_scores.get(rid, 0.0)
        for rid in all_ids
    }
    return sorted(blended.items(), key=lambda kv: -kv[1])


def get_recommendations(
    user_id,
    train: pd.DataFrame,
    train_counts: dict,
    cb_kwargs: dict,
    coll_model,
    coll_kwargs: dict,
    seen: set,
    k: int = 10,
    alpha: float = 0.5,
    lambda_div: float = 0.5,
    recently_seen: set | None = None,
    n_candidates: int = 100,
) -> list[dict]:
    """
    Public entry point. Returns top-k recommendations as dicts.

    lambda_div: MMR trade-off passed to reranker.rerank -- 1.0 disables
    diversity re-ranking (pure relevance, same as Day 2 behaviour).
    recently_seen: recipe_ids to exclude regardless of score (e.g. viewed in
    the last N days -- see reranker.recent_recipe_ids). None/empty = no-op,
    matching offline evaluation where `seen` already excludes everything.
    """
    candidates = blend(user_id, train, train_counts, cb_kwargs, coll_model, coll_kwargs, seen, alpha, n_candidates)
    if recently_seen:
        candidates = reranker.filter_recent(candidates, recently_seen)
    final = reranker.rerank(candidates, cb_kwargs["tfidf_matrix"], cb_kwargs["recipe_id_to_row"], k=k, lambda_div=lambda_div)
    return [{"recipe_id": rid, "score": score} for rid, score in final]
