"""
Content-based recommender: TF-IDF over recipe text, cosine similarity between
a user's liked recipes and the catalogue.

Scores each candidate by its best match to any single one of the user's
liked recipes (max-similarity), rather than averaging all liked recipes into
one "centroid" profile first -- centroid averaging dilutes badly once a user
has liked more than a handful of different dishes. See recommend()'s
docstring for the concrete evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def fit(text: pd.Series, max_features: int = 20_000, min_valid_terms: int = 4) -> tuple[TfidfVectorizer, csr_matrix, np.ndarray]:
    """
    Fit a TF-IDF vectorizer over recipe text. Returns (vectorizer, matrix, valid_mask).

    valid_mask flags recipes whose vector has fewer than `min_valid_terms`
    nonzero entries. min_df strips rare, distinctive ingredient terms that
    appear in only one recipe -- which hollows out short/specific recipes
    down to just their common staple ingredients (sugar, butter, flour).
    A document left with 1-2 surviving terms gets nearly all its L2-normalised
    mass on that single axis, producing an artificially inflated cosine
    similarity against almost any profile that touches the same common term.
    Recipes below the threshold are excluded from content-based ranking
    entirely -- CF/popularity handle them instead, since content has no
    reliable signal for them.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.4,  # also drop near-universal terms (present in >40% of recipes)
    )
    matrix = vectorizer.fit_transform(text)
    nnz_per_doc = matrix.getnnz(axis=1)
    valid_mask = nnz_per_doc >= min_valid_terms

    print(f"content_based.fit: vocab={len(vectorizer.vocabulary_):,}, matrix={matrix.shape}")
    print(f"  mean nonzero terms/doc={nnz_per_doc.mean():.1f}, "
          f"excluded as degenerate (<{min_valid_terms} terms): {(~valid_mask).sum():,} "
          f"({(~valid_mask).mean()*100:.1f}%)")
    return vectorizer, matrix, valid_mask


def liked_items(
    user_id,
    train: pd.DataFrame,
    recipe_id_to_row: dict,
    max_liked: int = 50,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Rows and rating-weights of a user's liked recipes, capped to their
    `max_liked` highest-rated (ties broken by recency in `train`'s row order)
    to bound compute for very active users. Returns None if none found.
    """
    user_ratings = train[train["user_id"] == user_id].copy()
    user_ratings["row"] = user_ratings["recipe_id"].map(recipe_id_to_row)
    user_ratings = user_ratings.dropna(subset=["row"])
    if user_ratings.empty:
        return None

    user_ratings = user_ratings.sort_values("rating", ascending=False).head(max_liked)
    rows = user_ratings["row"].astype(int).to_numpy()
    weights = np.maximum(user_ratings["rating"].to_numpy(dtype=np.float32), 1.0)
    return rows, weights


def recommend(
    user_id,
    train: pd.DataFrame,
    tfidf_matrix: csr_matrix,
    recipe_id_to_row: dict,
    row_to_recipe_id: dict,
    seen: set,
    valid_mask: np.ndarray | None = None,
    n: int = 100,
    max_liked: int = 50,
) -> list[tuple]:
    """
    Top-n (recipe_id, score) candidates, scored by max-similarity: each
    candidate's score is its best cosine match to ANY single one of the
    user's liked recipes (weighted by that recipe's rating), not to one
    averaged "centroid" profile.

    Averaging all liked recipes into a single profile vector before scoring
    dilutes the signal badly once a user has more than a handful of likes
    across different dishes -- a near-duplicate of something they genuinely
    loved can end up scoring barely above unrelated recipes, because the
    average blurs it together with everything else they've ever rated.
    Scoring against each liked item individually and taking the best match
    avoids that: a strong match to any one dish stays a strong match.

    valid_mask (from fit()) excludes degenerate low-term-count recipes that
    would otherwise score artificially high regardless of user taste.
    """
    liked = liked_items(user_id, train, recipe_id_to_row, max_liked)
    if liked is None:
        return []
    liked_rows, weights = liked

    liked_vectors = tfidf_matrix[liked_rows]                     # (L, vocab)
    sims_matrix = cosine_similarity(liked_vectors, tfidf_matrix)  # (L, n_items)
    weighted_sims = sims_matrix * weights.reshape(-1, 1)
    scores = weighted_sims.max(axis=0)                            # (n_items,) best match per candidate

    if valid_mask is not None:
        scores = np.where(valid_mask, scores, -1.0)
    order = np.argsort(-scores)

    out = []
    for row in order:
        if scores[row] < 0:  # ran out of valid candidates
            break
        rid = row_to_recipe_id[row]
        if rid in seen:
            continue
        out.append((rid, float(scores[row])))
        if len(out) >= n:
            break
    return out


def score_candidates(
    user_id,
    train: pd.DataFrame,
    tfidf_matrix: csr_matrix,
    recipe_id_to_row: dict,
    candidate_recipe_ids: list,
    max_liked: int = 50,
) -> dict:
    """
    Score only the given candidate recipe_ids (for sampled-pool evaluation),
    using the same max-similarity logic as recommend() but restricted to a
    small pool instead of the full catalog.
    """
    liked = liked_items(user_id, train, recipe_id_to_row, max_liked)
    valid_candidates = [rid for rid in candidate_recipe_ids if rid in recipe_id_to_row]
    if liked is None or not valid_candidates:
        return {rid: 0.0 for rid in candidate_recipe_ids}

    liked_rows, weights = liked
    candidate_rows = [recipe_id_to_row[rid] for rid in valid_candidates]

    liked_vectors = tfidf_matrix[liked_rows]
    candidate_vectors = tfidf_matrix[candidate_rows]
    sims_matrix = cosine_similarity(liked_vectors, candidate_vectors)
    scores = (sims_matrix * weights.reshape(-1, 1)).max(axis=0)

    out = dict(zip(valid_candidates, scores))
    for rid in candidate_recipe_ids:
        out.setdefault(rid, 0.0)
    return out


def explain(
    recipe_id,
    user_liked_recipe_ids: list,
    vectorizer: TfidfVectorizer,
    tfidf_matrix: csr_matrix,
    recipe_id_to_row: dict,
    top_n: int = 3,
) -> list[str]:
    """
    Top overlapping TF-IDF terms between a recipe and the SINGLE liked recipe
    it matched best (consistent with recommend()'s max-similarity scoring —
    explains against the one dish that earned the recommendation, not an
    averaged blend of everything the user has ever liked).
    """
    row = recipe_id_to_row.get(recipe_id)
    if row is None:
        return []

    liked_rows = [recipe_id_to_row[r] for r in user_liked_recipe_ids if r in recipe_id_to_row]
    if not liked_rows:
        return []

    recipe_vec = tfidf_matrix[row]
    liked_vectors = tfidf_matrix[liked_rows]
    sims = cosine_similarity(recipe_vec, liked_vectors).ravel()
    best_liked_row = liked_rows[int(np.argmax(sims))]

    recipe_arr = recipe_vec.toarray().ravel()
    liked_arr = tfidf_matrix[best_liked_row].toarray().ravel()
    overlap = recipe_arr * liked_arr  # elementwise: high where both care about the term

    feature_names = vectorizer.get_feature_names_out()
    top_idx = [i for i in np.argsort(-overlap)[:top_n] if overlap[i] > 0]
    return [feature_names[i] for i in top_idx]
