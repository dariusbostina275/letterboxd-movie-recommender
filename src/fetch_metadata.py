import argparse
import csv
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "private_data" / "letterboxd"
CACHE_PATH = DATA_DIR / "metadata" / "tmdb_movies.json"

load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("TMDB_API_KEY")

if not API_KEY:
    raise RuntimeError("TMDB_API_KEY is missing from .env")


def load_movies(file_name: str, source: str, movies: dict) -> None:
    with (DATA_DIR / file_name).open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            title = row["Name"].strip()
            year = row["Year"].strip()
            key = f"{title}|{year}"

            movie = movies.setdefault(
                key,
                {"title": title, "year": year, "sources": []},
            )

            if source not in movie["sources"]:
                movie["sources"].append(source)


def choose_match(results: list, year: str) -> dict | None:
    for result in results:
        if result.get("release_date", "")[:4] == year:
            return result

    return results[0] if results else None


parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=3)
args = parser.parse_args()

movies = {}
load_movies("ratings.csv", "ratings", movies)
load_movies("watchlist.csv", "watchlist", movies)

CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}

added = 0

for key, movie in movies.items():
    if key in cache:
        continue

    response = requests.get(
        "https://api.themoviedb.org/3/search/movie",
        params={
            "api_key": API_KEY,
            "query": movie["title"],
            "year": movie["year"],
        },
        timeout=10,
    )
    response.raise_for_status()

    match = choose_match(response.json().get("results", []), movie["year"])

    if match:
        cache[key] = {
            **movie,
            "status": "matched",
            "tmdb_id": match["id"],
            "tmdb_title": match["title"],
            "release_date": match.get("release_date", ""),
            "genre_ids": match.get("genre_ids", []),
        }
    else:
        cache[key] = {**movie, "status": "not_found"}

    added += 1
    time.sleep(0.25)

    if added >= args.limit:
        break

CACHE_PATH.write_text(
    json.dumps(cache, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Added {added} cache entries.")
print(f"Cache now contains {len(cache)} entries.")