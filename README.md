# Savorly

Savorly is a recipe discovery and recommendation app built with Django. It combines content-based recommendation with collaborative filtering, then re-ranks results for diversity before presenting a small daily recipe edit to each signed-in user.

The project is designed as a portfolio/capstone product layer around a recommendation pipeline trained in Colab.

## Features

- Personalized daily recommendations
- Cold-start fallback for users without ratings or interactions
- Recipe detail pages with ingredients, cooking steps, cook time, and description
- Five-star recipe ratings with create/update behavior
- “Cooked this” interaction tracking
- Personal Kitchen page containing previously cooked recipes
- Generated placeholder artwork when recipe photos are unavailable
- Responsive Django templates with Tailwind CSS via CDN
- Jazzmin-themed Django admin
- Session authentication and CSRF-protected POST requests

## Architecture

```text
Browser
  ├── Django templates + Tailwind CDN + vanilla JavaScript
  ├── GET /api/recommendations/today/
  ├── POST /api/interactions/
  └── POST /api/ratings/
          │
          ▼
      Django API
          │
          ▼
  Recommendation service
    ├── TF-IDF content similarity
    ├── Matrix-factorization collaborative filtering
    └── Blended scoring + diversity re-ranking
```

The ML artifacts are loaded once when Django starts. The runtime recommender reads the files in `artifacts/`; it does not retrain on server startup.

## Requirements

- Python 3.12 recommended
- A virtual environment
- The trained recommender artifacts
- The processed recipe dataset for importing or backfilling data

Install dependencies:

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows PowerShell
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Environment configuration

Copy the example environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Generate a real Django secret for `.env`:

```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

Then set:

```dotenv
SECRET_KEY=your-generated-secret
DEBUG=True
```

`.env` is ignored by Git. Never commit the real secret.

## Model artifacts and recipe data

The following files are required in `artifacts/` for recommendations to be available:

```text
tfidf_vectorizer.pkl
tfidf_matrix.pkl
valid_mask.pkl
coll_model.pkl
index_maps.pkl
config.json
```

The app can start without them, but `/api/recommendations/today/` returns `503` until they are present and load successfully.

Place the processed dataset at:

```text
data/recipes.parquet
```

The local `data/` directory is intentionally ignored because it contains a processed dataset export. The runtime artifacts are kept under version control because the app needs them and they are small enough for a normal Git repository.

## Database setup

Apply migrations:

```bash
python manage.py migrate
```

Import the recipe catalogue:

```bash
python manage.py import_recipes --path data/recipes.parquet
```

For an existing database, the targeted backfill commands can populate the display-only list fields without re-importing other recipe data:

```bash
python manage.py backfill_ingredients
python manage.py backfill_steps
```

Create an admin user:

```bash
python manage.py createsuperuser
```

## Run locally

```bash
python manage.py runserver
```

Then open [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

User-facing pages require authentication:

| Route | Purpose |
| --- | --- |
| `/` | Daily recommendations |
| `/kitchen/` | Recipes marked as cooked |
| `/recipe/<recipe_id>/` | Recipe detail page |
| `/accounts/login/` | Django session login |
| `/admin/` | Jazzmin admin panel |

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/recommendations/today/` | Returns the current user’s recommendations |
| `POST` | `/api/interactions/` | Records `viewed` or `cooked` interactions |
| `POST` | `/api/ratings/` | Creates or updates a `0`–`5` rating |

Example rating payload:

```json
{
  "recipe_id": 12345,
  "score": 5
}
```

All API routes require the authenticated Django session. POST requests require the Django CSRF token.

## Testing

Run the full test suite:

```bash
python -m pytest -q
```

The tests cover recommendation behavior, cold-start fallback, ratings, Kitchen deduplication, recipe detail rendering, authentication gates, and recommender utilities.

## Project layout

```text
api/             Django models, API views, serializers, admin, commands
artifacts/       Trained recommender files required at runtime
recommender/     Content, collaborative, evaluation, and re-ranking logic
static/          Custom CSS and vanilla JavaScript
templates/       Shared shell, recommendations, Kitchen, login, detail pages
tests/           Pytest coverage
webapp/          Django settings, URL configuration, and page views
```

## Notes

- Recipe photos are not required; missing images use generated placeholder art.
- The project does not include a public registration flow. Users can be created through Django admin or another local provisioning method.
- `db.sqlite3`, `.env`, virtual environments, caches, and local dataset exports are intentionally excluded from Git.
