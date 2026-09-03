import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "private_data" / "letterboxd"
CACHE_PATH = DATA_DIR / "metadata" / "tmdb_movies.json"
OUTPUT_PATH = DATA_DIR / "results" / "recommendations.json"

load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("TMDB_API_KEY")

if not API_KEY:
    raise RuntimeError("TMDB_API_KEY is missing from .env")


def movie_key(row: dict) -> str:
    return f"{row['Name'].strip()}|{row['Year'].strip()}"


def load_ratings() -> dict[str, float]:
    ratings = {}

    with (DATA_DIR / "ratings.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["Rating"].strip():
                ratings[movie_key(row)] = float(row["Rating"])

    return ratings


def load_watched_keys() -> set[str]:
    with (DATA_DIR / "watched.csv").open(encoding="utf-8-sig", newline="") as file:
        return {movie_key(row) for row in csv.DictReader(file)}


ratings = load_ratings()
watched_keys = load_watched_keys()
metadata = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

genre_response = requests.get(
    "https://api.themoviedb.org/3/genre/movie/list",
    params={"api_key": API_KEY},
    timeout=10,
)
genre_response.raise_for_status()

genre_names = {
    genre["id"]: genre["name"]
    for genre in genre_response.json()["genres"]
}

genre_totals = defaultdict(float)
genre_counts = defaultdict(int)

for key, rating in ratings.items():
    movie = metadata.get(key)

    if not movie or movie["status"] != "matched":
        continue

    for genre_id in movie["genre_ids"]:
        genre_totals[genre_id] += rating - 2.5
        genre_counts[genre_id] += 1

genre_preferences = {
    genre_id: genre_totals[genre_id] / genre_counts[genre_id]
    for genre_id in genre_totals
}

recommendations = []

for key, movie in metadata.items():
    if movie["status"] != "matched":
        continue

    if "watchlist" not in movie["sources"]:
        continue

    if key in watched_keys or key in ratings:
        continue

    relevant_scores = [
        genre_preferences[genre_id]
        for genre_id in movie["genre_ids"]
        if genre_id in genre_preferences
    ]

    score = sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0

    strongest_genres = sorted(
        (
            genre_id
            for genre_id in movie["genre_ids"]
            if genre_id in genre_preferences
        ),
        key=lambda genre_id: genre_preferences[genre_id],
        reverse=True,
    )[:2]

    recommendations.append(
        {
            "title": movie["tmdb_title"],
            "release_date": movie["release_date"],
            "score": round(score, 3),
            "because_genres": [
                genre_names[genre_id] for genre_id in strongest_genres
            ],
        }
    )

recommendations.sort(key=lambda movie: movie["score"], reverse=True)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(
    json.dumps(
        {
            "model": "Average centered rating by genre",
            "recommendations": recommendations,
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print(f"Generated {len(recommendations)} recommendations.")
print("Saved results to private_data/letterboxd/results/recommendations.json")