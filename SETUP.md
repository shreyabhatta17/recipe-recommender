# Setup

1. Create a virtualenv and install pinned dependencies:
   ```
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Copy your Colab artefacts into `artifacts/` (from `artifacts.zip` in Drive):
   ```
   artifacts/tfidf_vectorizer.pkl
   artifacts/tfidf_matrix.pkl
   artifacts/valid_mask.pkl
   artifacts/coll_model.pkl
   artifacts/index_maps.pkl
   artifacts/config.json
   ```

3. Copy `recipes.parquet` from Drive (`recipe_recommender/processed/recipes.parquet`)
   into `data/recipes.parquet` in this project.

4. Run migrations and import recipes:
   ```
   python manage.py migrate
   python manage.py import_recipes --path data/recipes.parquet
   ```

5. Create a superuser (for /admin/ and for testing the API):
   ```
   python manage.py createsuperuser
   ```

6. Run the dev server:
   ```
   python manage.py runserver
   ```

7. Test the endpoints (after logging in via a browser session or /admin/):
   - `GET /api/recommendations/today/`
   - `POST /api/interactions/` with `{"recipe": <id>, "action": "viewed"}`

## What's already verified working
This project was built and tested end-to-end with synthetic fake artefacts
before being handed over: migrations apply cleanly, `import_recipes` loads a
parquet file correctly, a warm user (existing ratings) gets real personalized
recommendations through the full engine pipeline (blend -> rerank), a
zero-history user correctly falls back to popularity instead of an empty
response, and interaction logging works. You're plugging in your real
artefacts, not untested code.
