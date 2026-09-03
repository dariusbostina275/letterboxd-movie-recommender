# Letterboxd Content-Based Movie Recommender

A portfolio-ready movie recommendation system that learns a taste profile from a
Letterboxd export and ranks unseen movies from TMDB. The project demonstrates a
complete machine-learning workflow: data ingestion, API enrichment, feature
engineering, vectorisation, similarity-based ranking, and human-readable
explanations.

## How it works

1. Load `ratings.csv` and `watched.csv` from a Letterboxd export.
2. Keep highly rated films (4 stars or above by default) as positive examples.
3. Enrich those films and a TMDB popular-movie candidate pool with genres,
   keywords, director, top-three cast members, and plot overview.
4. Build a **metadata soup**. Names are normalised into single tokens and the
   director is repeated to give that feature more influence.
5. Convert the combined text to TF-IDF vectors with English stop words removed.
6. Compute a rating-weighted average of favourite-film vectors to form the user
   profile.
7. Rank candidate vectors by cosine similarity to the profile and strictly remove
   titles already present in `watched.csv`.

Cosine similarity measures the angle between vectors rather than their magnitude:

```text
similarity(user, movie) = (user · movie) / (||user|| ||movie||)
```

This is a content-based system, so recommendations are driven entirely by movie
attributes and the user's own history. It does not require ratings from other
users, but it can be less effective at discovering films far outside the user's
existing taste profile.

## Repository structure

```text
letterboxd-movie-recommender/
├── data/
│   ├── raw/                 # ratings.csv and watched.csv (not committed)
│   ├── processed/           # optional generated datasets
│   └── cache/               # cached TMDB metadata
├── notebooks/               # exploratory analysis notebooks
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Letterboxd ingestion and TMDB integration
│   └── recommender.py       # feature engineering and recommendation model
├── tests/
│   ├── test_data_loader.py
│   └── test_recommender.py
├── .env.example
├── .gitignore
├── main.py                  # command-line entry point
├── requirements.txt
└── README.md
```

## Local setup

Prerequisites: Python 3.10 or newer, a Letterboxd export, and a TMDB v3 API key.

### 1. Clone the repository and create a virtual environment

```bash
git clone https://github.com/dariusbostina275/letterboxd-movie-recommender.git
cd letterboxd-movie-recommender
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Export Letterboxd data

In Letterboxd, open **Settings → Import & Export → Export Your Data**. Extract the
archive, then copy these files into `data/raw/`:

```text
data/raw/ratings.csv
data/raw/watched.csv
```

The CSV files are ignored by Git so personal viewing data is not published.

### 4. Configure TMDB

Create a TMDB API key, copy the example environment file, and replace the
placeholder value:

```bash
cp .env.example .env
```

Windows PowerShell equivalent:

```powershell
Copy-Item .env.example .env
```

```dotenv
TMDB_API_KEY=your_v3_api_key_here
```

Never commit `.env`; it is already included in `.gitignore`.

### 5. Run the recommender

```bash
python main.py
```

Useful options:

```bash
python main.py --top-n 20 --min-rating 4.5 --popular-pages 8
python main.py --help
```

The first run makes multiple TMDB requests and can take a minute. Movie metadata
is cached under `data/cache/`, so later runs are faster and use fewer API calls.

### 6. Run tests

```bash
pytest
```

Tests use synthetic data and do not require a TMDB key or network access.

## Design notes

- `LetterboxdDataLoader` owns local CSV validation and normalisation.
- `TMDBClient` owns authentication, request throttling, retries, response parsing,
  and caching.
- `ContentBasedRecommender` owns feature engineering, model fitting, watched-film
  exclusion, ranking, and explanations.
- `main.py` only orchestrates those components and handles command-line errors.

Scores are similarity values, not predicted star ratings or probabilities. A high
score means that a candidate's metadata resembles the learned taste profile.

## Ideas for extension

- Compare TF-IDF against sentence embeddings.
- Add diversity-aware re-ranking so the top results are less repetitive.
- Evaluate with a time-based holdout from the user's diary.
- Build a Streamlit interface and display posters from TMDB.
- Add CI with GitHub Actions for linting and tests.

## Data and attribution

This project expects data exported by the user and does not publish that data.
Movie metadata is supplied by TMDB. Follow the applicable Letterboxd and TMDB
terms when publishing or deploying the project.
