"""
Data loading, cleaning, subsampling, temporal split, and matrix construction
for the Food.com Recipes & Interactions dataset.

Expected raw files (from the Kaggle dataset
"food-com-recipes-and-user-interactions"):
    RAW_recipes.csv
    RAW_interactions.csv
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_recipes(path: str | Path) -> pd.DataFrame:
    """Load RAW_recipes.csv and parse stringified list columns."""
    df = pd.read_csv(path)

    for col in ("ingredients", "tags", "steps"):
        if col in df.columns:
            df[col] = df[col].apply(_safe_literal_eval)

    df = df.rename(columns={"id": "recipe_id"})
    df["description"] = df.get("description", "").fillna("")
    df["name"] = df["name"].fillna("")
    return df


def load_interactions(path: str | Path) -> pd.DataFrame:
    """Load RAW_interactions.csv."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={"recipe_id": "recipe_id", "user_id": "user_id"})
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating", "user_id", "recipe_id", "date"])
    df["rating"] = df["rating"].astype(int)
    return df


def _safe_literal_eval(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    return val if isinstance(val, list) else []


# --------------------------------------------------------------------------
# Cleaning + subsampling
# --------------------------------------------------------------------------

def clean(recipes: pd.DataFrame, interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop null/duplicate/invalid records."""
    recipes = recipes.dropna(subset=["name"]).copy()
    recipes = recipes[recipes["ingredients"].map(len) > 0]
    recipes = recipes.drop_duplicates(subset="recipe_id")

    interactions = interactions[interactions["rating"].between(0, 5)].copy()
    interactions = interactions.drop_duplicates(subset=["user_id", "recipe_id", "date"])

    # keep only interactions pointing at recipes that survived cleaning
    interactions = interactions[interactions["recipe_id"].isin(recipes["recipe_id"])]
    return recipes, interactions


def subsample(
    recipes: pd.DataFrame,
    interactions: pd.DataFrame,
    min_ratings_per_recipe: int = 5,
    min_ratings_per_user: int = 5,
    n_recipes: int = 25_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter to recipes/users with enough signal, then cap catalogue size.
    Iterates the two filters a couple of times since each one changes the
    counts the other depends on.
    """
    inter = interactions.copy()

    for _ in range(3):
        recipe_counts = inter["recipe_id"].value_counts()
        keep_recipes = recipe_counts[recipe_counts >= min_ratings_per_recipe].index
        inter = inter[inter["recipe_id"].isin(keep_recipes)]

        user_counts = inter["user_id"].value_counts()
        keep_users = user_counts[user_counts >= min_ratings_per_user].index
        inter = inter[inter["user_id"].isin(keep_users)]

    # cap catalogue size, keeping the most-rated recipes
    if inter["recipe_id"].nunique() > n_recipes:
        top_recipes = inter["recipe_id"].value_counts().head(n_recipes).index
        inter = inter[inter["recipe_id"].isin(top_recipes)]

    recipes_out = recipes[recipes["recipe_id"].isin(inter["recipe_id"])].reset_index(drop=True)
    inter = inter.reset_index(drop=True)

    print(f"subsample: {recipes_out.shape[0]:,} recipes, "
          f"{inter['user_id'].nunique():,} users, "
          f"{inter.shape[0]:,} interactions")
    return recipes_out, inter


# --------------------------------------------------------------------------
# Feature text
# --------------------------------------------------------------------------

def build_text_column(recipes: pd.DataFrame) -> pd.Series:
    """Concatenate ingredients + description + tags into one text field per recipe."""
    def join_list(x):
        return " ".join(x) if isinstance(x, list) else ""

    text = (
        recipes["ingredients"].apply(join_list) + " "
        + recipes["description"].fillna("") + " "
        + recipes.get("tags", pd.Series([[]] * len(recipes))).apply(join_list)
    )
    return text.str.lower().str.replace(r"[^a-z0-9\s]", " ", regex=True)


# --------------------------------------------------------------------------
# Temporal split
# --------------------------------------------------------------------------

def temporal_split(interactions: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-user split: each user's most recent `test_frac` of interactions go to
    test, the rest to train. Users with too few interactions to split
    meaningfully (<5) go entirely to train.
    """
    train_parts, test_parts = [], []

    for _, g in interactions.groupby("user_id", sort=False):
        g = g.sort_values("date")
        if len(g) < 5:
            train_parts.append(g)
            continue
        n_test = max(1, int(len(g) * test_frac))
        train_parts.append(g.iloc[:-n_test])
        test_parts.append(g.iloc[-n_test:])

    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True) if test_parts else interactions.iloc[0:0]

    # sanity check: no test interaction should predate that user's train interactions
    assert _no_temporal_leakage(train, test), "temporal leakage detected in split"

    print(f"temporal_split: {len(train):,} train / {len(test):,} test interactions")
    return train, test


def _no_temporal_leakage(train: pd.DataFrame, test: pd.DataFrame) -> bool:
    train_max = train.groupby("user_id")["date"].max()
    test_min = test.groupby("user_id")["date"].min()
    common = train_max.index.intersection(test_min.index)
    if len(common) == 0:
        return True
    return bool((test_min.loc[common] >= train_max.loc[common]).all())


# --------------------------------------------------------------------------
# Matrix build
# --------------------------------------------------------------------------

def build_matrix(train: pd.DataFrame) -> tuple[csr_matrix, dict, dict]:
    """
    Build a sparse (n_users x n_items) rating matrix from training
    interactions. Returns the matrix plus id<->index maps in both directions.
    """
    user_ids = train["user_id"].unique()
    item_ids = train["recipe_id"].unique()

    user_index = {uid: i for i, uid in enumerate(user_ids)}
    item_index = {iid: i for i, iid in enumerate(item_ids)}

    rows = train["user_id"].map(user_index).values
    cols = train["recipe_id"].map(item_index).values
    vals = train["rating"].values.astype(np.float32)

    matrix = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(item_ids)))

    index_maps = {
        "user_index": user_index,
        "item_index": item_index,
        "index_user": {v: k for k, v in user_index.items()},
        "index_item": {v: k for k, v in item_index.items()},
    }
    print(f"build_matrix: shape={matrix.shape}, "
          f"density={matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.5f}")
    return matrix, index_maps["user_index"], index_maps["item_index"]


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_processed(recipes: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recipes.to_parquet(out_dir / "recipes.parquet", index=False)
    train.to_parquet(out_dir / "train.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    print(f"saved processed data to {out_dir}")


def load_processed(in_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    in_dir = Path(in_dir)
    recipes = pd.read_parquet(in_dir / "recipes.parquet")
    train = pd.read_parquet(in_dir / "train.parquet")
    test = pd.read_parquet(in_dir / "test.parquet")
    return recipes, train, test
