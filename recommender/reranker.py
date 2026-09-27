"""
Re-ranking layer: recency filtering + diversity via greedy Maximal Marginal
Relevance (MMR). Sits on top of engine.blend()'s candidate list, before
truncating to the final top-k shown to the user.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def recent_recipe_ids(
    user_id,
    interactions: pd.DataFrame,
    reference_date,
    recency_days: int = 14,
) -> set:
    """
    Recipe ids the user interacted with in the `recency_days` window ending
    at `reference_date`. In offline evaluation, reference_date is usually the
    max date in the dataset (a stand-in for "today"); in production it's the
    real current timestamp. Requires `interactions` to have user_id,
    recipe_id, and date columns (ratings or, in production, a dedicated
    view/cooked interaction log).
    """
    window_start = pd.Timestamp(reference_date) - pd.Timedelta(days=recency_days)
    recent = interactions[
        (interactions["user_id"] == user_id)
        & (interactions["date"] >= window_start)
        & (interactions["date"] <= reference_date)
    ]
    return set(recent["recipe_id"])


def filter_recent(candidates: list[tuple], recently_seen: set) -> list[tuple]:
    """Drop candidates the user has interacted with inside the recency window."""
    if not recently_seen:
        return candidates
    return [(rid, score) for rid, score in candidates if rid not in recently_seen]


def rerank(
    candidates: list[tuple],
    tfidf_matrix: csr_matrix,
    recipe_id_to_row: dict,
    k: int = 10,
    lambda_div: float = 0.5,
) -> list[tuple]:
    """
    Greedy MMR: iteratively picks the candidate maximising
        lambda_div * relevance_score  -  (1 - lambda_div) * max_similarity_to_already_picked
    lambda_div=1.0 reduces to plain relevance ranking (no diversity pressure).
    Lower values trade top-line relevance for a less repetitive list.

    Candidates whose recipe_id has no row in the TF-IDF catalogue (shouldn't
    normally happen) are treated as having zero similarity to everything --
    they're never penalised for being "too similar", just ranked by score.
    """
    if not candidates:
        return []
    if len(candidates) <= k:
        # nothing to trade off -- everything makes the list regardless of order
        pass

    scores = np.array([s for _, s in candidates], dtype=np.float32)
    lo, hi = scores.min(), scores.max()
    norm_scores = (scores - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(scores)

    ids = [rid for rid, _ in candidates]
    rows = [recipe_id_to_row.get(rid) for rid in ids]
    valid_idx = [i for i, r in enumerate(rows) if r is not None]

    if not valid_idx:
        order = np.argsort(-norm_scores)
        return [candidates[i] for i in order[:k]]

    vectors = tfidf_matrix[[rows[i] for i in valid_idx]]
    sim_matrix = cosine_similarity(vectors)
    valid_pos = {orig_i: pos for pos, orig_i in enumerate(valid_idx)}

    remaining = list(range(len(candidates)))
    selected: list[int] = []

    while remaining and len(selected) < k:
        best_i, best_val = None, -np.inf
        selected_valid_pos = [valid_pos[j] for j in selected if j in valid_pos]
        for i in remaining:
            if i in valid_pos and selected_valid_pos:
                sim_to_selected = sim_matrix[valid_pos[i], selected_valid_pos].max()
            else:
                sim_to_selected = 0.0
            mmr_val = lambda_div * norm_scores[i] - (1 - lambda_div) * sim_to_selected
            if mmr_val > best_val:
                best_val, best_i = mmr_val, i
        selected.append(best_i)
        remaining.remove(best_i)

    return [candidates[i] for i in selected]


def intra_list_diversity(recipe_ids: list, tfidf_matrix: csr_matrix, recipe_id_to_row: dict) -> float:
    """
    Mean pairwise dissimilarity (1 - cosine similarity) within a recommended
    list. Higher = more diverse. Used to plot the precision/diversity
    trade-off across lambda_div values -- not used at inference time.
    """
    rows = [recipe_id_to_row[r] for r in recipe_ids if r in recipe_id_to_row]
    if len(rows) < 2:
        return 0.0
    sim_matrix = cosine_similarity(tfidf_matrix[rows])
    n = len(rows)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    return float(1.0 - upper_tri.mean())
