import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from recommender import content_based, evaluate, preprocessing, reranker


def _recipe_texts():
    # Every term occurs in at least two documents, while no term occurs in
    # enough documents to be removed by content_based.fit's min_df/max_df.
    return pd.Series([
        "chicken garlic ginger soy",
        "chicken garlic ginger soy sesame",
        "sesame tomato basil lemon",
        "tomato basil lemon olive",
        "apple pear cinnamon oat",
        "apple pear cinnamon oat",
        "beef pepper cumin onion",
        "beef pepper cumin onion",
        "salmon dill potato cream",
        "salmon dill potato cream",
        "rice coconut lime cilantro",
        "rice coconut lime cilantro",
        "pasta mushroom thyme parmesan",
        "pasta mushroom thyme parmesan",
        "bean paprika corn avocado",
        "bean paprika corn avocado",
    ])


@pytest.fixture
def fitted_recipes():
    vectorizer, matrix, valid_mask = content_based.fit(_recipe_texts())
    return vectorizer, matrix, valid_mask


def test_temporal_split_no_leakage():
    interactions = pd.DataFrame([
        {"user_id": 1, "recipe_id": i, "rating": 5, "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)}
        for i in range(1, 7)
    ] + [
        {"user_id": 2, "recipe_id": i + 20, "rating": 4, "date": pd.Timestamp("2025-02-01") + pd.Timedelta(days=i)}
        for i in range(1, 8)
    ] + [
        {"user_id": 3, "recipe_id": 99, "rating": 5, "date": pd.Timestamp("2025-03-01")},
    ])
    train, test = preprocessing.temporal_split(interactions, test_frac=0.2)

    for user_id in set(train.user_id) & set(test.user_id):
        assert test.loc[test.user_id == user_id, "date"].min() >= train.loc[train.user_id == user_id, "date"].max()


def test_precision_at_k_known_value():
    result = evaluate.precision_at_k([1, 2, 3, 4, 5], {1, 3, 5, 99}, k=5)
    assert result == pytest.approx(0.6)


def test_content_based_similar_recipes_score_high(fitted_recipes):
    _, matrix, _ = fitted_recipes
    similarity = cosine_similarity(matrix[0], matrix[1])[0, 0]
    assert similarity > 0.9


def test_content_based_unrelated_recipes_score_low(fitted_recipes):
    _, matrix, _ = fitted_recipes
    similarity = cosine_similarity(matrix[0], matrix[4])[0, 0]
    assert similarity < 0.3


def test_reranker_recency_filter():
    candidates = [(1, 0.9), (2, 0.8), (3, 0.7)]
    filtered = reranker.filter_recent(candidates, {2})
    assert [rid for rid, _ in filtered] == [1, 3]


def test_reranker_output_no_duplicates_respects_k(fitted_recipes):
    _, matrix, _ = fitted_recipes
    candidates = [(recipe_id, 1.0 - recipe_id / 100) for recipe_id in range(16)]
    recipe_id_to_row = {recipe_id: recipe_id for recipe_id in range(16)}
    result = reranker.rerank(candidates, matrix, recipe_id_to_row, k=10)

    result_ids = [recipe_id for recipe_id, _ in result]
    assert len(result_ids) == 10
    assert len(result_ids) == len(set(result_ids))
    assert set(result_ids) <= {recipe_id for recipe_id, _ in candidates}
