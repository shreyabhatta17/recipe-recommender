import numpy as np
import pandas as pd
import pytest
from django.contrib.auth import get_user_model
from django.urls import reverse
from scipy.sparse import csr_matrix

from api import recommender_state as state
from api.models import Interaction, Rating, Recipe
from recommender import collaborative, content_based


def _texts():
    return [
        "chicken garlic ginger soy", "chicken garlic ginger soy sesame",
        "sesame tomato basil lemon", "tomato basil lemon olive",
        "apple pear cinnamon oat", "apple pear cinnamon oat",
        "beef pepper cumin onion", "beef pepper cumin onion",
        "salmon dill potato cream", "salmon dill potato cream",
        "rice coconut lime cilantro", "rice coconut lime cilantro",
        "pasta mushroom thyme parmesan", "pasta mushroom thyme parmesan",
        "bean paprika corn avocado", "bean paprika corn avocado",
    ]


@pytest.fixture
def recommender_state(monkeypatch):
    _, matrix, valid_mask = content_based.fit(pd.Series(_texts()))
    recipe_ids = list(range(1, 17))
    recipe_id_to_row = {recipe_id: row for row, recipe_id in enumerate(recipe_ids)}
    row_to_recipe_id = dict(enumerate(recipe_ids))

    # The one-user model is enough to exercise the real hybrid engine without
    # loading the repository's large pickled artifacts.
    ratings = np.array([[5, 5, 4, 5, 4, 5]], dtype=np.float32)
    rating_matrix = csr_matrix(ratings)
    coll_model = collaborative.fit_svd(rating_matrix, n_factors=1)

    monkeypatch.setattr(state, "loaded", True)
    monkeypatch.setattr(state, "tfidf_matrix", matrix)
    monkeypatch.setattr(state, "valid_mask", valid_mask)
    monkeypatch.setattr(state, "coll_model", coll_model)
    monkeypatch.setattr(state, "user_index", {})
    monkeypatch.setattr(state, "item_index", {i: i - 1 for i in range(1, 7)})
    monkeypatch.setattr(state, "index_item", {i - 1: i for i in range(1, 7)})
    monkeypatch.setattr(state, "recipe_id_to_row", recipe_id_to_row)
    monkeypatch.setattr(state, "row_to_recipe_id", row_to_recipe_id)
    monkeypatch.setattr(state, "best_alpha", 0.5)
    monkeypatch.setattr(state, "best_lambda", 0.5)
    monkeypatch.setattr(state, "cold_start_threshold", 5)
    monkeypatch.setattr(state, "cb_kwargs", {
        "tfidf_matrix": matrix,
        "recipe_id_to_row": recipe_id_to_row,
        "row_to_recipe_id": row_to_recipe_id,
        "valid_mask": valid_mask,
    })
    monkeypatch.setattr(state, "coll_kwargs", {
        "user_index": state.user_index,
        "item_index": state.item_index,
        "index_item": state.index_item,
    })


@pytest.fixture
def recipes(db):
    return [Recipe.objects.create(recipe_id=i, title=f"Recipe {i}", minutes=10) for i in range(1, 17)]


def test_recommendations_endpoint_warm_user(client, db, recipes, recommender_state):
    user = get_user_model().objects.create_user(username="warm", password="password")
    Rating.objects.bulk_create([Rating(user=user, recipe=recipe, score=5) for recipe in recipes[:6]])
    state.user_index[user.id] = 0

    client.login(username="warm", password="password")
    response = client.get(reverse("recommendations-today"))

    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 10


def test_recommendations_endpoint_cold_start_user(client, db, recipes, recommender_state):
    user = get_user_model().objects.create_user(username="cold", password="password")
    popularity_user = get_user_model().objects.create_user(username="popular", password="password")
    Rating.objects.bulk_create([Rating(user=popularity_user, recipe=recipe, score=5) for recipe in recipes[:3]])

    client.login(username="cold", password="password")
    response = client.get(reverse("recommendations-today"))

    assert response.status_code == 200
    assert response.json()


def test_rating_endpoint_creates_rating(client, db, recipes):
    user = get_user_model().objects.create_user(username="rater", password="password")
    client.login(username="rater", password="password")

    response = client.post(
        reverse("ratings-create"),
        {"recipe_id": recipes[0].recipe_id, "score": 5},
        content_type="application/json",
    )

    assert response.status_code == 201
    assert response.json() == {"recipe_id": recipes[0].recipe_id, "score": 5}
    assert Rating.objects.filter(user=user, recipe=recipes[0], score=5).exists()


def test_rating_endpoint_rejects_out_of_range_score(client, db, recipes):
    get_user_model().objects.create_user(username="invalid-rater", password="password")
    client.login(username="invalid-rater", password="password")

    response = client.post(
        reverse("ratings-create"),
        {"recipe_id": recipes[0].recipe_id, "score": 7},
        content_type="application/json",
    )

    assert response.status_code == 400
    assert "score" in response.json()


def test_rating_endpoint_updates_existing_rating(client, db, recipes):
    user = get_user_model().objects.create_user(username="updating-rater", password="password")
    Rating.objects.create(user=user, recipe=recipes[0], score=2)
    client.login(username="updating-rater", password="password")

    response = client.post(
        reverse("ratings-create"),
        {"recipe_id": recipes[0].recipe_id, "score": 4},
        content_type="application/json",
    )

    assert response.status_code == 200
    assert response.json() == {"recipe_id": recipes[0].recipe_id, "score": 4}
    assert Rating.objects.filter(user=user, recipe=recipes[0]).count() == 1
    assert Rating.objects.get(user=user, recipe=recipes[0]).score == 4


def test_kitchen_view_shows_cooked_recipe_once(client, db, recipes):
    user = get_user_model().objects.create_user(username="kitchen-cook", password="password")
    Interaction.objects.create(user=user, recipe=recipes[0], action=Interaction.COOKED)
    Interaction.objects.create(user=user, recipe=recipes[0], action=Interaction.COOKED)

    client.login(username="kitchen-cook", password="password")
    response = client.get(reverse("kitchen"))

    assert response.status_code == 200
    assert "Recipe 1" in response.content.decode()
    assert response.content.decode().count("Recipe 1") == 1


def test_kitchen_view_shows_empty_state_for_new_user(client, db, recipes):
    user = get_user_model().objects.create_user(username="kitchen-new", password="password")

    client.login(username="kitchen-new", password="password")
    response = client.get(reverse("kitchen"))

    assert response.status_code == 200
    assert "You haven’t cooked anything yet" in response.content.decode()


def test_recipe_detail_view_shows_recipe_content(client, db, recipes):
    user = get_user_model().objects.create_user(username="detail-reader", password="password")
    recipes[0].ingredients = ["1 cup flour", "2 eggs"]
    recipes[0].steps = "['preheat the oven', 'mix the ingredients']"
    recipes[0].description = "A simple recipe description."
    recipes[0].save()

    client.login(username="detail-reader", password="password")
    response = client.get(reverse("recipe-detail", args=[recipes[0].recipe_id]))

    assert response.status_code == 200
    assert "Recipe 1" in response.content.decode()
    assert "1 cup flour" in response.content.decode()
    assert "2 eggs" in response.content.decode()
    assert "preheat the oven" in response.content.decode()
    assert "mix the ingredients" in response.content.decode()


def test_recipe_detail_view_shows_fallback_without_steps(client, db, recipes):
    user = get_user_model().objects.create_user(username="no-steps-reader", password="password")
    client.login(username="no-steps-reader", password="password")

    response = client.get(reverse("recipe-detail", args=[recipes[0].recipe_id]))

    assert response.status_code == 200
    assert "Not available" in response.content.decode()


def test_recipe_detail_view_returns_404_for_missing_recipe(client, db, recipes):
    user = get_user_model().objects.create_user(username="missing-reader", password="password")
    client.login(username="missing-reader", password="password")

    response = client.get(reverse("recipe-detail", args=[99999]))

    assert response.status_code == 404
