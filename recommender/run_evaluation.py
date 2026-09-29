"""Standalone offline evaluation for the already-trained recommender artifacts.

Example:
    python -m recommender.run_evaluation --interactions data/interactions.csv

The interaction export is intentionally an input rather than an artifact: the
repository contains the trained models and recipe catalogue, but not the
historical interaction table needed to construct a temporal test split.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import collaborative, content_based, evaluate, preprocessing, reranker


def _normalise(scores: dict) -> dict:
    if not scores:
        return {}
    values = np.asarray(list(scores.values()), dtype=float)
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-12:
        return {key: 0.0 for key in scores}
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


def _load_interactions(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        interactions = pd.read_parquet(path)
    else:
        interactions = pd.read_csv(path, parse_dates=["date"])
    required = {"user_id", "recipe_id", "rating", "date"}
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"interaction file is missing columns: {sorted(missing)}")
    interactions = interactions.copy()
    interactions["date"] = pd.to_datetime(interactions["date"])
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
    return interactions.dropna(subset=sorted(required)).reset_index(drop=True)


def _sample_users(train: pd.DataFrame, test: pd.DataFrame, max_users: int | None, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not max_users:
        return train, test
    users = test["user_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    users = rng.choice(users, size=min(max_users, len(users)), replace=False)
    keep = set(users.tolist())
    return train[train.user_id.isin(keep)].copy(), test[test.user_id.isin(keep)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--negatives", type=int, default=99)
    parser.add_argument("--max-users", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with (args.artifacts / "index_maps.pkl").open("rb") as fh:
        maps = pickle.load(fh)
    with (args.artifacts / "coll_model.pkl").open("rb") as fh:
        coll_model = pickle.load(fh)
    with (args.artifacts / "tfidf_matrix.pkl").open("rb") as fh:
        tfidf_matrix = pickle.load(fh)
    with (args.artifacts / "valid_mask.pkl").open("rb") as fh:
        valid_mask = pickle.load(fh)
    config = json.loads((args.artifacts / "config.json").read_text())

    interactions = _load_interactions(args.interactions)
    train, test = preprocessing.temporal_split(interactions)
    train, test = _sample_users(train, test, args.max_users, args.seed)
    train_counts = evaluate.train_interaction_counts(train)
    seen_by_user = train.groupby("user_id")["recipe_id"].apply(set).to_dict()

    cb_kwargs = {
        "tfidf_matrix": tfidf_matrix,
        "recipe_id_to_row": maps["recipe_id_to_row"],
        "row_to_recipe_id": maps["row_to_recipe_id"],
        "valid_mask": valid_mask,
    }
    coll_kwargs = {
        "user_index": maps["user_index"],
        "item_index": maps["item_index"],
        "index_item": maps["index_item"],
    }
    catalogue = set(maps["recipe_id_to_row"])

    def blend_scores(user_id, candidate_ids, alpha):
        seen = seen_by_user.get(user_id, set())
        content_scores = _normalise(content_based.score_candidates(
            user_id, train, candidate_ids=candidate_ids,
            tfidf_matrix=tfidf_matrix, recipe_id_to_row=maps["recipe_id_to_row"],
        ))
        user_row = maps["user_index"].get(user_id)
        coll_scores = {}
        if user_row is not None:
            item_rows = [maps["item_index"][rid] for rid in candidate_ids if rid in maps["item_index"]]
            raw = coll_model.score_rows(user_row, item_rows)
            coll_scores = {rid: float(score) for rid, score in zip(
                (rid for rid in candidate_ids if rid in maps["item_index"]), raw
            )}
            coll_scores = _normalise(coll_scores)
        # Match engine.blend's cold-start switch.
        if train_counts.get(user_id, 0) < config.get("cold_start_threshold", 5):
            return content_scores
        return {
            rid: alpha * coll_scores.get(rid, 0.0) + (1.0 - alpha) * content_scores.get(rid, 0.0)
            for rid in candidate_ids
        }

    def evaluate_alpha(alpha):
        return evaluate.sampled_evaluate(
            lambda uid, ids: blend_scores(uid, ids, alpha), train, test, catalogue,
            n_negatives=args.negatives, k=args.k, seed=args.seed,
        )["precision@k"]

    print(f"sampled users={test.user_id.nunique()}  k={args.k}  negatives={args.negatives}")
    print("alpha sweep (sampled-pool precision):")
    curve = []
    for alpha in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        precision = evaluate_alpha(float(alpha))
        curve.append((float(alpha), precision))
        print(f"  alpha={alpha:.2f}  precision@{args.k}={precision:.6f}")
    best_alpha, best_precision = max(curve, key=lambda item: item[1])
    monotonic = all(next_precision >= precision - 1e-12
                    for (_, precision), (_, next_precision) in zip(curve, curve[1:]))
    print(f"alpha summary: best={best_alpha:.2f} ({best_precision:.6f}); "
          f"monotone through 1.00={monotonic}; "
          f"delta(0.95->1.00)={curve[-1][1] - curve[-2][1]:+.6f}")

    def ranked(user_id, candidate_ids, alpha, lambda_div):
        scores = blend_scores(user_id, candidate_ids, alpha)
        candidates = sorted(scores.items(), key=lambda item: -item[1])
        return [rid for rid, _ in reranker.rerank(
            candidates, tfidf_matrix, maps["recipe_id_to_row"], k=args.k, lambda_div=lambda_div
        )]

    rows = []
    # Existing implementation semantics: lambda=1 is no diversity pressure;
    # lambda=0 is maximum MMR diversity pressure.
    for label, alpha, lambda_div in [
        ("best_config", 1.0, 1.0),
        ("lambda_0_max_diversity", 1.0, 0.0),
        ("lambda_0.5_middle", 1.0, 0.5),
    ]:
        result = evaluate.sampled_evaluate_ranked(
            lambda uid, ids: ranked(uid, ids, alpha, lambda_div), train, test, catalogue,
            n_negatives=args.negatives, k=args.k, seed=args.seed,
            tfidf_matrix=tfidf_matrix, recipe_id_to_row=maps["recipe_id_to_row"], catalogue=catalogue,
        )
        rows.append({"config": label, "alpha": alpha, "lambda_div": lambda_div,
                     "precision": result["precision@k"], "coverage": result["coverage"],
                     "intra_list_diversity": result["intra_list_diversity"]})

    print("\nconfig          alpha  lambda_div  precision  coverage  diversity")
    for row in rows:
        print(f"{row['config']:<16} {row['alpha']:.2f}   {row['lambda_div']:.2f}       "
              f"{row['precision']:.6f}  {row['coverage']:.6f}  {row['intra_list_diversity']:.6f}")


if __name__ == "__main__":
    main()
