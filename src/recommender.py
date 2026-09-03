"""Content-based recommendation model built with TF-IDF and cosine similarity."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

METADATA_COLUMNS = {"tmdb_id", "title", "year", "genres", "keywords", "director", "cast", "overview"}


class RecommenderError(ValueError):
    """Raised when training or candidate data cannot be used by the model."""


@dataclass(frozen=True)
class RecommenderConfig:
    """Configuration for metadata feature weighting and vectorisation."""

    director_weight: int = 3
    genre_weight: int = 2
    max_features: int = 20_000

    def __post_init__(self) -> None:
        if self.director_weight < 1 or self.genre_weight < 1:
            raise ValueError("Feature weights must be positive integers.")
        if self.max_features < 1:
            raise ValueError("max_features must be positive.")


def _as_list(value: Any) -> list[str]:
    """Safely coerce a DataFrame cell containing metadata into a string list."""

    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _feature_token(value: str) -> str:
    """Normalise a multi-word named feature into one TF-IDF token."""

    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", ascii_value.casefold())


def _normalise_title(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


class ContentBasedRecommender:
    """Rank unseen movies by similarity to a rating-weighted taste profile."""

    def __init__(self, config: RecommenderConfig | None = None) -> None:
        self.config = config or RecommenderConfig()
        self.vectorizer: TfidfVectorizer | None = None
        self.user_profile: csr_matrix | None = None
        self._favourites: pd.DataFrame | None = None
        self._favourite_vectors: csr_matrix | None = None

    @staticmethod
    def _validate_metadata(frame: pd.DataFrame, label: str) -> None:
        missing = METADATA_COLUMNS.difference(frame.columns)
        if missing:
            raise RecommenderError(
                f"{label} metadata is missing columns: {', '.join(sorted(missing))}"
            )
        if frame.empty:
            raise RecommenderError(f"{label} metadata is empty.")

    def build_metadata_soup(self, row: pd.Series) -> str:
        """Combine natural language and weighted categorical movie metadata."""

        genres = [_feature_token(value) for value in _as_list(row.get("genres"))]
        keywords = [_feature_token(value) for value in _as_list(row.get("keywords"))]
        cast = [_feature_token(value) for value in _as_list(row.get("cast"))]
        director = _feature_token(str(row.get("director") or ""))

        tokens: list[str] = []
        tokens.extend(genres * self.config.genre_weight)
        tokens.extend(keywords)
        tokens.extend(cast)
        if director:
            tokens.extend([director] * self.config.director_weight)

        overview = str(row.get("overview") or "").strip()
        return " ".join([overview, *filter(None, tokens)]).strip()

    def fit(self, favourites: pd.DataFrame, candidates: pd.DataFrame) -> "ContentBasedRecommender":
        """Fit a shared TF-IDF space and create the user's preference vector."""

        self._validate_metadata(favourites, "Favourite")
        self._validate_metadata(candidates, "Candidate")

        favourite_data = favourites.copy().reset_index(drop=True)
        candidate_data = candidates.copy().reset_index(drop=True)
        favourite_soups = favourite_data.apply(self.build_metadata_soup, axis=1)
        candidate_soups = candidate_data.apply(self.build_metadata_soup, axis=1)
        corpus = pd.concat([favourite_soups, candidate_soups], ignore_index=True)

        if corpus.str.strip().eq("").all():
            raise RecommenderError("The supplied metadata contains no usable text.")

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=self.config.max_features,
            sublinear_tf=True,
        )
        matrix = self.vectorizer.fit_transform(corpus)
        favourite_count = len(favourite_data)
        favourite_vectors = matrix[:favourite_count].tocsr()

        ratings = pd.to_numeric(
            favourite_data.get("rating", pd.Series(1.0, index=favourite_data.index)),
            errors="coerce",
        ).fillna(1.0)
        # Positive weights retain rating preference without assuming a fixed scale.
        weights = ratings.to_numpy(dtype=float)
        weights = np.maximum(weights - weights.min() + 1.0, 1.0)
        weighted_sum = favourite_vectors.multiply(weights[:, np.newaxis]).sum(axis=0)

        self.user_profile = csr_matrix(np.asarray(weighted_sum) / weights.sum())
        self._favourites = favourite_data
        self._favourite_vectors = favourite_vectors
        return self

    @staticmethod
    def _watched_keys(watched: pd.DataFrame) -> tuple[set[tuple[str, int]], set[str]]:
        if not {"title", "year"}.issubset(watched.columns):
            raise RecommenderError("Watched data must contain title and year columns.")

        title_year: set[tuple[str, int]] = set()
        yearless_titles: set[str] = set()
        for row in watched[["title", "year"]].to_dict(orient="records"):
            title = _normalise_title(row["title"])
            if not title:
                continue
            if pd.isna(row["year"]):
                yearless_titles.add(title)
            else:
                title_year.add((title, int(row["year"])))
        return title_year, yearless_titles

    @classmethod
    def _exclude_watched(
        cls, candidates: pd.DataFrame, watched: pd.DataFrame
    ) -> pd.DataFrame:
        title_year, yearless_titles = cls._watched_keys(watched)

        def has_been_watched(row: pd.Series) -> bool:
            title = _normalise_title(row["title"])
            if title in yearless_titles:
                return True
            if pd.isna(row["year"]):
                # When the candidate year is unknown, prefer strict exclusion.
                return any(watched_title == title for watched_title, _ in title_year)
            return (title, int(row["year"])) in title_year

        mask = candidates.apply(has_been_watched, axis=1)
        return candidates.loc[~mask].copy()

    @staticmethod
    def _shared_values(candidate: pd.Series, favourite: pd.Series, column: str) -> list[str]:
        favourite_tokens = {_feature_token(value) for value in _as_list(favourite.get(column))}
        return [
            value
            for value in _as_list(candidate.get(column))
            if _feature_token(value) in favourite_tokens
        ]

    def _explain(self, candidate: pd.Series, favourite: pd.Series, score: float) -> str:
        reasons: list[str] = []
        candidate_director = str(candidate.get("director") or "")
        favourite_director = str(favourite.get("director") or "")
        if candidate_director and _feature_token(candidate_director) == _feature_token(favourite_director):
            reasons.append(f"same director ({candidate_director})")

        shared_genres = self._shared_values(candidate, favourite, "genres")
        if shared_genres:
            reasons.append(f"shared genres: {', '.join(shared_genres[:2])}")

        shared_cast = self._shared_values(candidate, favourite, "cast")
        if shared_cast:
            reasons.append(f"shared cast: {', '.join(shared_cast[:2])}")

        shared_keywords = self._shared_values(candidate, favourite, "keywords")
        if shared_keywords:
            reasons.append(f"shared themes: {', '.join(shared_keywords[:2])}")

        favourite_title = str(favourite["title"])
        if reasons:
            evidence = "; ".join(reasons[:2])
            return f"Similar to {favourite_title} — {evidence} (score {score:.3f})."
        return f"Its overall metadata is similar to {favourite_title} (score {score:.3f})."

    def recommend(
        self,
        candidates: pd.DataFrame,
        watched: pd.DataFrame,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Return ranked, unseen recommendations with nearest-film explanations."""

        if self.vectorizer is None or self.user_profile is None:
            raise RecommenderError("Call fit() before recommend().")
        if self._favourites is None or self._favourite_vectors is None:
            raise RecommenderError("The fitted profile is incomplete.")
        if top_n < 1:
            raise ValueError("top_n must be positive.")
        self._validate_metadata(candidates, "Candidate")

        unseen = self._exclude_watched(candidates, watched)
        unseen = unseen.drop_duplicates(subset=["tmdb_id"]).reset_index(drop=True)
        if unseen.empty:
            return pd.DataFrame(columns=[*candidates.columns, "similarity", "explanation"])

        soups = unseen.apply(self.build_metadata_soup, axis=1)
        candidate_vectors = self.vectorizer.transform(soups)
        profile_scores = cosine_similarity(candidate_vectors, self.user_profile).ravel()
        favourite_scores = cosine_similarity(candidate_vectors, self._favourite_vectors)
        nearest_indices = favourite_scores.argmax(axis=1)

        unseen["similarity"] = profile_scores
        unseen["explanation"] = [
            self._explain(
                unseen.iloc[index],
                self._favourites.iloc[int(nearest_indices[index])],
                float(profile_scores[index]),
            )
            for index in range(len(unseen))
        ]
        return (
            unseen.sort_values("similarity", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
