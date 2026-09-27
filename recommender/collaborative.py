"""
Collaborative filtering via matrix factorisation.

Two implementations are provided:
  - fit_svd / SVDModel: sklearn TruncatedSVD, fast, good default
  - fit_sgd / SGDMFModel: custom SGD with biases, better for the report if
    you want to write about tuning learning rate / regularisation

Pick ONE and use it consistently in engine.py — don't build both into the
final pipeline, just keep whichever you didn't pick as a documented option.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


# --------------------------------------------------------------------------
# Option A: TruncatedSVD (fast, pure-wheel, recommended default)
# --------------------------------------------------------------------------

class SVDModel:
    def __init__(self, svd: TruncatedSVD, user_factors: np.ndarray, item_factors: np.ndarray, global_mean: float):
        self.svd = svd
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.global_mean = global_mean

    def predict(self, user_row: int, item_row: int) -> float:
        return self.global_mean + float(self.user_factors[user_row] @ self.item_factors[item_row])

    def recommend_rows(self, user_row: int, seen_rows: set, n: int = 100) -> list[tuple[int, float]]:
        scores = self.item_factors @ self.user_factors[user_row]
        order = np.argsort(-scores)
        out = []
        for item_row in order:
            if item_row in seen_rows:
                continue
            out.append((int(item_row), float(self.global_mean + scores[item_row])))
            if len(out) >= n:
                break
        return out

    def score_rows(self, user_row: int, item_rows: list[int]) -> np.ndarray:
        """Score only the given item rows (for sampled-pool evaluation)."""
        item_rows = np.asarray(item_rows)
        return self.global_mean + self.item_factors[item_rows] @ self.user_factors[user_row]


def fit_svd(matrix: csr_matrix, n_factors: int = 50, random_state: int = 42) -> SVDModel:
    global_mean = matrix.data.mean() if matrix.nnz else 0.0
    centered = matrix.copy().astype(np.float32)
    centered.data -= global_mean  # mean-center only the observed entries

    svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
    user_factors = svd.fit_transform(centered)          # (n_users, k)
    item_factors = svd.components_.T                    # (n_items, k)

    print(f"collaborative.fit_svd: factors={n_factors}, "
          f"explained_variance={svd.explained_variance_ratio_.sum():.3f}")
    return SVDModel(svd, user_factors, item_factors, global_mean)


# --------------------------------------------------------------------------
# Option B: custom SGD matrix factorisation with biases
# --------------------------------------------------------------------------

class SGDMFModel:
    def __init__(self, P: np.ndarray, Q: np.ndarray, b_u: np.ndarray, b_i: np.ndarray, mu: float):
        self.P, self.Q, self.b_u, self.b_i, self.mu = P, Q, b_u, b_i, mu

    def predict(self, user_row: int, item_row: int) -> float:
        return float(self.mu + self.b_u[user_row] + self.b_i[item_row] + self.P[user_row] @ self.Q[item_row])

    def recommend_rows(self, user_row: int, seen_rows: set, n: int = 100) -> list[tuple[int, float]]:
        scores = self.mu + self.b_u[user_row] + self.b_i + self.Q @ self.P[user_row]
        order = np.argsort(-scores)
        out = []
        for item_row in order:
            if item_row in seen_rows:
                continue
            out.append((int(item_row), float(scores[item_row])))
            if len(out) >= n:
                break
        return out

    def score_rows(self, user_row: int, item_rows: list[int]) -> np.ndarray:
        """Score only the given item rows (for sampled-pool evaluation)."""
        item_rows = np.asarray(item_rows)
        return self.mu + self.b_u[user_row] + self.b_i[item_rows] + self.Q[item_rows] @ self.P[user_row]


def fit_sgd(
    matrix: csr_matrix,
    n_factors: int = 50,
    lr: float = 0.01,
    reg: float = 0.02,
    n_epochs: int = 15,
    random_state: int = 42,
) -> SGDMFModel:
    rng = np.random.default_rng(random_state)
    n_users, n_items = matrix.shape
    mu = float(matrix.data.mean()) if matrix.nnz else 0.0

    P = rng.normal(0, 0.1, (n_users, n_factors)).astype(np.float32)
    Q = rng.normal(0, 0.1, (n_items, n_factors)).astype(np.float32)
    b_u = np.zeros(n_users, dtype=np.float32)
    b_i = np.zeros(n_items, dtype=np.float32)

    coo = matrix.tocoo()
    rows, cols, vals = coo.row, coo.col, coo.data.astype(np.float32)
    order = np.arange(len(vals))

    for epoch in range(n_epochs):
        rng.shuffle(order)
        sq_err_sum = 0.0
        for idx in order:
            u, i, r = rows[idx], cols[idx], vals[idx]
            pred = mu + b_u[u] + b_i[i] + P[u] @ Q[i]
            err = r - pred
            sq_err_sum += err ** 2

            b_u[u] += lr * (err - reg * b_u[u])
            b_i[i] += lr * (err - reg * b_i[i])
            p_u = P[u].copy()
            P[u] += lr * (err * Q[i] - reg * P[u])
            Q[i] += lr * (err * p_u - reg * Q[i])

        rmse = np.sqrt(sq_err_sum / len(vals))
        print(f"  epoch {epoch + 1}/{n_epochs}  train RMSE={rmse:.4f}")

    return SGDMFModel(P, Q, b_u, b_i, mu)


# --------------------------------------------------------------------------
# Shared recommend() wrapper — works with either model type
# --------------------------------------------------------------------------

def recommend(
    model,
    user_id,
    user_index: dict,
    item_index: dict,
    index_item: dict,
    seen_recipe_ids: set,
    n: int = 100,
) -> list[tuple]:
    """Returns [(recipe_id, score), ...] for a user_id, or [] if unseen in training."""
    user_row = user_index.get(user_id)
    if user_row is None:
        return []  # not in the CF training matrix at all -> pure cold start

    seen_rows = {item_index[r] for r in seen_recipe_ids if r in item_index}
    rows_scores = model.recommend_rows(user_row, seen_rows, n)
    return [(index_item[row], score) for row, score in rows_scores]
