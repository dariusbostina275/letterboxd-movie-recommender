"""Load Letterboxd exports and enrich films with metadata from TMDB."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import requests
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)


class LetterboxdDataError(ValueError):
    """Raised when a Letterboxd export is missing or has an invalid schema."""


class TMDBAPIError(RuntimeError):
    """Raised when a TMDB request cannot be completed successfully."""


@dataclass(frozen=True)
class MovieMetadata:
    """Model-ready metadata for one movie."""

    tmdb_id: int
    title: str
    year: int | None
    genres: list[str]
    keywords: list[str]
    director: str
    cast: list[str]
    overview: str

    def to_record(self) -> dict[str, Any]:
        """Return a dictionary suitable for a pandas DataFrame or JSON cache."""

        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "MovieMetadata":
        """Construct metadata from a cache record."""

        return cls(
            tmdb_id=int(record["tmdb_id"]),
            title=str(record["title"]),
            year=int(record["year"]) if record.get("year") is not None else None,
            genres=[str(value) for value in record.get("genres", [])],
            keywords=[str(value) for value in record.get("keywords", [])],
            director=str(record.get("director", "")),
            cast=[str(value) for value in record.get("cast", [])],
            overview=str(record.get("overview", "")),
        )


class LetterboxdDataLoader:
    """Read and validate the relevant files from a Letterboxd export."""

    COLUMN_ALIASES = {
        "name": "title",
        "year": "year",
        "rating": "rating",
        "date": "date",
        "letterboxd uri": "letterboxd_uri",
    }

    def __init__(self, ratings_path: Path | str, watched_path: Path | str) -> None:
        self.ratings_path = Path(ratings_path)
        self.watched_path = Path(watched_path)

    @staticmethod
    def _read_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
        """Read a Letterboxd CSV and convert its headings to internal names."""

        if not path.is_file():
            raise LetterboxdDataError(f"Letterboxd file not found: {path}")

        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except (OSError, UnicodeError, pd.errors.ParserError) as exc:
            raise LetterboxdDataError(f"Could not read {path}: {exc}") from exc

        aliases = {
            column: LetterboxdDataLoader.COLUMN_ALIASES.get(
                str(column).strip().casefold(), str(column).strip().casefold()
            )
            for column in frame.columns
        }
        frame = frame.rename(columns=aliases)

        missing = required_columns.difference(frame.columns)
        if missing:
            names = ", ".join(sorted(missing))
            raise LetterboxdDataError(f"{path.name} is missing columns: {names}")

        frame["title"] = frame["title"].astype("string").str.strip()
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
        frame = frame[frame["title"].notna() & frame["title"].ne("")].copy()
        return frame

    def load_ratings(self) -> pd.DataFrame:
        """Load ratings with normalised title, year, and numeric rating columns."""

        ratings = self._read_csv(self.ratings_path, {"title", "year", "rating"})
        ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
        ratings = ratings.dropna(subset=["rating"])
        ratings = ratings[ratings["rating"].between(0.5, 5.0)].copy()
        if ratings.empty:
            raise LetterboxdDataError("ratings.csv contains no valid ratings.")
        return ratings

    def load_watched(self) -> pd.DataFrame:
        """Load the watched-film list used for strict recommendation exclusion."""

        watched = self._read_csv(self.watched_path, {"title", "year"})
        return watched.drop_duplicates(subset=["title", "year"]).reset_index(drop=True)

    @staticmethod
    def select_favourites(ratings: pd.DataFrame, min_rating: float = 4.0) -> pd.DataFrame:
        """Return films at or above ``min_rating`` for user-profile creation."""

        if not 0.5 <= min_rating <= 5.0:
            raise ValueError("min_rating must be between 0.5 and 5.0.")
        if not {"title", "year", "rating"}.issubset(ratings.columns):
            raise LetterboxdDataError(
                "Ratings data must contain title, year, and rating columns."
            )

        favourites = ratings.loc[ratings["rating"] >= min_rating].copy()
        favourites = favourites.sort_values("rating", ascending=False)
        return favourites.drop_duplicates(subset=["title", "year"]).reset_index(drop=True)


class JsonMovieCache:
    """Small persistent cache that avoids repeatedly downloading movie details."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._records: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            contents = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Ignoring unreadable TMDB cache %s: %s", self.path, exc)
            return {}
        return contents if isinstance(contents, dict) else {}

    def get(self, tmdb_id: int) -> MovieMetadata | None:
        """Return cached metadata, if present and valid."""

        record = self._records.get(str(tmdb_id))
        if record is None:
            return None
        try:
            return MovieMetadata.from_record(record)
        except (KeyError, TypeError, ValueError):
            LOGGER.warning("Ignoring invalid cache entry for TMDB movie %s", tmdb_id)
            return None

    def put(self, movie: MovieMetadata) -> None:
        """Store a movie and atomically update the JSON file."""

        self._records[str(movie.tmdb_id)] = movie.to_record()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(self._records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(self.path)


class TMDBClient:
    """Rate-limited client for movie search, details, credits, and keywords."""

    BASE_URL = "https://api.themoviedb.org/3"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache_path: Path | str = Path("data/cache/tmdb_movies.json"),
        min_request_interval: float = 0.25,
        timeout: float = 15.0,
        max_retries: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("TMDB_API_KEY", "")
        if not self.api_key:
            raise TMDBAPIError(
                "TMDB_API_KEY is missing. Copy .env.example to .env and add your key."
            )
        if min_request_interval < 0:
            raise ValueError("min_request_interval cannot be negative.")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1.")

        self.cache = JsonMovieCache(cache_path)
        self.min_request_interval = min_request_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.session.headers.update(
            {"Accept": "application/json", "User-Agent": "letterboxd-recommender/1.0"}
        )
        self._last_request_at = 0.0

    def close(self) -> None:
        """Close the underlying HTTP session."""

        self.session.close()

    def __enter__(self) -> "TMDBClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        delay = self.min_request_interval - elapsed
        if delay > 0:
            time.sleep(delay)

    def _request(self, endpoint: str, **params: Any) -> dict[str, Any]:
        """Send a request with throttling and retries for transient failures."""

        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        request_params = {"api_key": self.api_key, "language": "en-US", **params}
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            self._throttle()
            try:
                response = self.session.get(
                    url, params=request_params, timeout=self.timeout
                )
                self._last_request_at = time.monotonic()

                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", 1.0))
                    LOGGER.warning("TMDB rate limit reached; retrying in %.1fs", retry_after)
                    time.sleep(max(retry_after, self.min_request_interval))
                    continue
                if 500 <= response.status_code < 600:
                    raise requests.HTTPError(
                        f"TMDB server error {response.status_code}", response=response
                    )
                if 400 <= response.status_code < 500:
                    raise TMDBAPIError(
                        f"TMDB rejected {endpoint} with HTTP {response.status_code}. "
                        "Check the API key and request parameters."
                    )

                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise TMDBAPIError(f"Unexpected TMDB response from {endpoint}.")
                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    delay = 2**attempt
                    LOGGER.warning(
                        "TMDB request to %s failed (%s); retrying in %ss",
                        endpoint,
                        type(exc).__name__,
                        delay,
                    )
                    time.sleep(delay)

        raise TMDBAPIError(
            f"TMDB request failed after {self.max_retries} attempts: {endpoint}"
        ) from last_error

    @staticmethod
    def _release_year(result: Mapping[str, Any]) -> int | None:
        release_date = str(result.get("release_date") or "")
        try:
            return int(release_date[:4]) if len(release_date) >= 4 else None
        except ValueError:
            return None

    @staticmethod
    def _normalise_title(title: str) -> str:
        return "".join(character for character in title.casefold() if character.isalnum())

    def search_movie(self, title: str, year: int | None = None) -> int | None:
        """Find the most plausible TMDB movie ID for a Letterboxd title and year."""

        params: dict[str, Any] = {"query": title, "include_adult": "false"}
        if year is not None:
            params["primary_release_year"] = year
        payload = self._request("search/movie", **params)
        results = payload.get("results", [])

        # A year filter can occasionally hide a valid international release.
        if not results and year is not None:
            payload = self._request(
                "search/movie", query=title, include_adult="false"
            )
            results = payload.get("results", [])

        if not isinstance(results, list) or not results:
            return None

        wanted_title = self._normalise_title(title)

        def match_score(result: Mapping[str, Any]) -> tuple[int, float]:
            candidate_titles = (
                str(result.get("title") or ""),
                str(result.get("original_title") or ""),
            )
            title_match = any(
                self._normalise_title(candidate) == wanted_title
                for candidate in candidate_titles
            )
            candidate_year = self._release_year(result)
            year_match = year is not None and candidate_year == year
            quality = (2 if title_match else 0) + (1 if year_match else 0)
            return quality, float(result.get("popularity") or 0.0)

        best = max(results, key=match_score)
        quality, _ = match_score(best)
        if quality < 2:  # Require an exact normalised title or original-title match.
            return None
        try:
            return int(best["id"])
        except (KeyError, TypeError, ValueError):
            return None

    def get_movie_metadata(self, tmdb_id: int) -> MovieMetadata:
        """Fetch genres, keywords, director, cast, and overview for one movie."""

        cached = self.cache.get(tmdb_id)
        if cached is not None:
            return cached

        payload = self._request(
            f"movie/{tmdb_id}", append_to_response="credits,keywords"
        )
        credits = payload.get("credits") or {}
        keyword_data = payload.get("keywords") or {}
        keyword_items = keyword_data.get("keywords") or keyword_data.get("results") or []
        crew = credits.get("crew") or []
        cast = credits.get("cast") or []

        directors = [
            str(person.get("name"))
            for person in crew
            if person.get("job") == "Director" and person.get("name")
        ]
        movie = MovieMetadata(
            tmdb_id=int(payload.get("id", tmdb_id)),
            title=str(payload.get("title") or payload.get("original_title") or "Untitled"),
            year=self._release_year(payload),
            genres=[
                str(genre["name"])
                for genre in payload.get("genres", [])
                if genre.get("name")
            ],
            keywords=[
                str(keyword["name"])
                for keyword in keyword_items
                if keyword.get("name")
            ],
            director=directors[0] if directors else "",
            cast=[str(person["name"]) for person in cast[:3] if person.get("name")],
            overview=str(payload.get("overview") or ""),
        )
        self.cache.put(movie)
        return movie

    def enrich_letterboxd_movies(self, movies: pd.DataFrame) -> pd.DataFrame:
        """Match Letterboxd rows to TMDB and return their model-ready metadata."""

        required = {"title", "year"}
        if not required.issubset(movies.columns):
            raise LetterboxdDataError("Movies must contain title and year columns.")

        records: list[dict[str, Any]] = []
        for row in movies.to_dict(orient="records"):
            title = str(row["title"])
            raw_year = row.get("year")
            year = None if pd.isna(raw_year) else int(raw_year)
            try:
                tmdb_id = self.search_movie(title, year)
                if tmdb_id is None:
                    LOGGER.warning("No TMDB match found for %s (%s)", title, year or "?")
                    continue
                record = self.get_movie_metadata(tmdb_id).to_record()
            except TMDBAPIError as exc:
                LOGGER.warning("Skipping %s after a TMDB error: %s", title, exc)
                continue

            # Preserve profile information such as the user's rating.
            for key, value in row.items():
                if key not in {"title", "year"}:
                    record[key] = value
            record["letterboxd_title"] = title
            record["letterboxd_year"] = year
            records.append(record)

        return pd.DataFrame.from_records(records)

    def fetch_popular_movies(self, pages: int = 5) -> pd.DataFrame:
        """Build a metadata-rich candidate pool from TMDB popular movies."""

        if not 1 <= pages <= 20:
            raise ValueError("pages must be between 1 and 20.")

        movie_ids: list[int] = []
        for page in range(1, pages + 1):
            try:
                payload = self._request("movie/popular", page=page)
            except TMDBAPIError as exc:
                LOGGER.warning("Skipping TMDB popular page %s: %s", page, exc)
                continue
            for movie in payload.get("results", []):
                try:
                    movie_ids.append(int(movie["id"]))
                except (KeyError, TypeError, ValueError):
                    continue

        records: list[dict[str, Any]] = []
        for tmdb_id in dict.fromkeys(movie_ids):
            try:
                records.append(self.get_movie_metadata(tmdb_id).to_record())
            except TMDBAPIError as exc:
                LOGGER.warning("Skipping TMDB movie %s: %s", tmdb_id, exc)

        return pd.DataFrame.from_records(records)


def metadata_to_frame(movies: Iterable[MovieMetadata]) -> pd.DataFrame:
    """Convert an iterable of metadata objects into a DataFrame."""

    return pd.DataFrame.from_records(movie.to_record() for movie in movies)
