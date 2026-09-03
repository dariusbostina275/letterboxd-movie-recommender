import os
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("TMDB_API_KEY")

if not api_key:
    raise RuntimeError("TMDB_API_KEY is missing from .env")

response = requests.get(
    "https://api.themoviedb.org/3/configuration",
    params={"api_key": api_key},
    timeout=10,
)
response.raise_for_status()

print("TMDB connection successful.")