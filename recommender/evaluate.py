"""
Evaluation harness: Precision@K / Recall@K under the per-user temporal split,
plus a most-popular baseline that every subsequent model must beat.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

RELEVANCE_THRESHOLD = 4  # a held-out rating >= this counts as "relevant"


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def precision_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    if not recommended:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def catalogue_coverage(recommendation_lists, catalogue) -> float:
    """Fraction of the catalogue appearing in at least one recommendation list."""
    catalogue = set(catalogue)
    if not catalogue:
        return 0.0
    recommended = {rid for recs in recommendation_lists for rid in recs}
    return len(recommended & catalogue) / len(catalogue)


def intra_list_diversity(
    recommendation_lists,
    tfidf_matrix,
    recipe_id_to_row: dict,
) -> float:
    """Mean within-list pairwise dissimilarity, using ``1 - cosine``."""
    diversities = []
    for recipe_ids in recommendation_lists:
        rows = [recipe_id_to_row[rid] for rid in recipe_ids if rid in recipe_id_to_row]
        if len(rows) < 2:
            continue
        similarities = cosine_similarity(tfidf_matrix[rows])
        upper = similarities[np.triu_indices(len(rows), k=1)]
        diversities.append(float(1.0 - upper.mean()))
    return float(np.mean(diversities)) if diversities else 0.0


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------

def build_relevant_sets(test: pd.DataFrame, threshold: int = RELEVANCE_THRESHOLD) -> dict:
    """user_id -> set of recipe_ids the user rated >= threshold in the test period."""
    relevant = test[test["rating"] >= threshold]
    out = defaultdict(set)
    for uid, rid in zip(relevant["user_id"], relevant["recipe_id"]):
        out[uid].add(rid)
    return out


def train_interaction_counts(train: pd.DataFrame) -> dict:
    """user_id -> number of training interactions (used to slice cold-start users)."""
    return train.groupby("user_id").size().to_dict()


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------

def popularity_baseline(train: pd.DataFrame):
    """
    Returns a model_fn(user_id, seen, k) -> list[recipe_id] that always
    recommends the globally most-rated recipes, excluding ones the user has
    already interacted with in training.
    """
    ranked = (
        train.groupby("recipe_id")
        .agg(n=("rating", "size"), mean_rating=("rating", "mean"))
        .sort_values(["n", "mean_rating"], ascending=False)
        .index.tolist()
    )

    def model_fn(user_id, seen: set, k: int = 10) -> list:
        out = []
        for rid in ranked:
            if rid in seen:
                continue
            out.append(rid)
            if len(out) >= k:
                break
        return out

    return model_fn


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------

def evaluate(
    model_fn,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
    cold_start_max_interactions: int = 5,
) -> dict:
    """
    Run model_fn over every test user, report overall P@K/R@K and the
    cold-start slice (users with < cold_start_max_interactions training
    interactions) separately.

    model_fn signature: model_fn(user_id, seen: set, k: int) -> list[recipe_id]
    """
    relevant_sets = build_relevant_sets(test)
    seen_by_user = train.groupby("user_id")["recipe_id"].apply(set).to_dict()
    train_counts = train_interaction_counts(train)

    rows = []
    for user_id, relevant in relevant_sets.items():
        seen = seen_by_user.get(user_id, set())
        recommended = model_fn(user_id, seen, k)
        p = precision_at_k(recommended, relevant, k)
        r = recall_at_k(recommended, relevant, k)
        n_train = train_counts.get(user_id, 0)
        rows.append((user_id, p, r, n_train))

    df = pd.DataFrame(rows, columns=["user_id", "precision", "recall", "n_train"])
    cold = df[df["n_train"] < cold_start_max_interactions]

    return {
        "n_users": len(df),
        "precision@k": df["precision"].mean() if len(df) else float("nan"),
        "recall@k": df["recall"].mean() if len(df) else float("nan"),
        "n_cold_users": len(cold),
        "precision@k_cold": cold["precision"].mean() if len(cold) else float("nan"),
        "recall@k_cold": cold["recall"].mean() if len(cold) else float("nan"),
    }


def sampled_evaluate(
    score_fn,
    train: pd.DataFrame,
    test: pd.DataFrame,
    all_item_ids,
    n_negatives: int = 99,
    k: int = 10,
    seed: int = 42,
    cold_start_max_interactions: int = 5,
) -> dict:
    """
    Standard 'sampled ranking' protocol (Cremonesi et al., 2010): rank each
    user's held-out relevant item(s) against a small pool of random unseen
    negatives, rather than the entire catalogue. Ranking against the full
    catalogue structurally favours popularity -- this protocol is the
    standard way to check whether personalised models actually beat it once
    that structural bias is removed.

    score_fn(user_id, candidate_ids) -> {recipe_id: score} must score ONLY
    the given candidate_ids (cheap -- see content_based.score_candidates and
    collaborative *Model.score_rows for the per-model implementations).
    """
    rng = np.random.default_rng(seed)
    relevant_sets = build_relevant_sets(test)
    seen_by_user = train.groupby("user_id")["recipe_id"].apply(set).to_dict()
    train_counts = train_interaction_counts(train)
    all_items_arr = np.asarray(list(all_item_ids))

    rows = []
    for user_id, relevant in relevant_sets.items():
        seen = seen_by_user.get(user_id, set())
        excluded = seen | relevant

        negatives, tries = [], 0
        while len(negatives) < n_negatives and tries < n_negatives * 20:
            cand = all_items_arr[rng.integers(0, len(all_items_arr))]
            tries += 1
            if cand not in excluded and cand not in negatives:
                negatives.append(cand)

        candidate_ids = list(relevant) + negatives
        scores = score_fn(user_id, candidate_ids)
        ranked = sorted(candidate_ids, key=lambda x: -scores.get(x, 0.0))

        p = precision_at_k(ranked, relevant, k)
        r = recall_at_k(ranked, relevant, k)
        rows.append((user_id, p, r, train_counts.get(user_id, 0)))

    df = pd.DataFrame(rows, columns=["user_id", "precision", "recall", "n_train"])
    cold = df[df["n_train"] < cold_start_max_interactions]

    return {
        "n_users": len(df),
        "precision@k": df["precision"].mean() if len(df) else float("nan"),
        "recall@k": df["recall"].mean() if len(df) else float("nan"),
        "n_cold_users": len(cold),
        "precision@k_cold": cold["precision"].mean() if len(cold) else float("nan"),
        "recall@k_cold": cold["recall"].mean() if len(cold) else float("nan"),
    }


def sampled_evaluate_ranked(
    rank_fn,
    train: pd.DataFrame,
    test: pd.DataFrame,
    all_item_ids,
    n_negatives: int = 99,
    k: int = 10,
    seed: int = 42,
    cold_start_max_interactions: int = 5,
    tfidf_matrix=None,
    recipe_id_to_row: dict | None = None,
    catalogue=None,
) -> dict:
    """
    Same sampled-pool protocol as sampled_evaluate, but for functions that
    return an already-ORDERED list of ids (e.g. engine.get_recommendations
    after MMR re-ranking) rather than a {id: score} dict to sort ourselves.
    Needed because MMR re-ranking is a sequential selection process, not a
    single scalar score per candidate.

    rank_fn(user_id, candidate_ids) -> ordered list of recipe_ids (top-k or
    a full ranking over candidate_ids -- only the first k are used).
    """
    rng = np.random.default_rng(seed)
    relevant_sets = build_relevant_sets(test)
    seen_by_user = train.groupby("user_id")["recipe_id"].apply(set).to_dict()
    train_counts = train_interaction_counts(train)
    all_items_arr = np.asarray(list(all_item_ids))

    rows = []
    recommendation_lists = []
    for user_id, relevant in relevant_sets.items():
        seen = seen_by_user.get(user_id, set())
        excluded = seen | relevant

        negatives, tries = [], 0
        while len(negatives) < n_negatives and tries < n_negatives * 20:
            cand = all_items_arr[rng.integers(0, len(all_items_arr))]
            tries += 1
            if cand not in excluded and cand not in negatives:
                negatives.append(cand)

        candidate_ids = list(relevant) + negatives
        ranked = rank_fn(user_id, candidate_ids)

        p = precision_at_k(ranked, relevant, k)
        r = recall_at_k(ranked, relevant, k)
        rows.append((user_id, p, r, train_counts.get(user_id, 0)))
        recommendation_lists.append(ranked[:k])

    df = pd.DataFrame(rows, columns=["user_id", "precision", "recall", "n_train"])
    cold = df[df["n_train"] < cold_start_max_interactions]

    results = {
        "n_users": len(df),
        "precision@k": df["precision"].mean() if len(df) else float("nan"),
        "recall@k": df["recall"].mean() if len(df) else float("nan"),
        "n_cold_users": len(cold),
        "precision@k_cold": cold["precision"].mean() if len(cold) else float("nan"),
        "recall@k_cold": cold["recall"].mean() if len(cold) else float("nan"),
    }
    if tfidf_matrix is not None and recipe_id_to_row is not None and catalogue is not None:
        results["coverage"] = catalogue_coverage(recommendation_lists, catalogue)
        results["intra_list_diversity"] = intra_list_diversity(
            recommendation_lists, tfidf_matrix, recipe_id_to_row
        )
    return results


def print_report(name: str, results: dict) -> None:
    print(
        f"{name:<24} "
        f"P@10={results['precision@k']:.4f}  R@10={results['recall@k']:.4f}  "
        f"(n={results['n_users']})   |   cold: "
        f"P@10={results['precision@k_cold']:.4f}  R@10={results['recall@k_cold']:.4f}  "
        f"(n={results['n_cold_users']})"
    )
