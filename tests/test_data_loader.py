"""Tests for local Letterboxd ingestion; no external API calls are made."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import LetterboxdDataError, LetterboxdDataLoader


def test_loader_normalises_letterboxd_columns(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    watched_path = tmp_path / "watched.csv"
    pd.DataFrame(
        [{"Date": "2026-01-01", "Name": "Arrival", "Year": 2016, "Rating": 4.5}]
    ).to_csv(ratings_path, index=False)
    pd.DataFrame([{"Date": "2026-01-01", "Name": "Arrival", "Year": 2016}]).to_csv(
        watched_path, index=False
    )

    loader = LetterboxdDataLoader(ratings_path, watched_path)

    assert loader.load_ratings().iloc[0]["title"] == "Arrival"
    assert loader.load_watched().iloc[0]["year"] == 2016


def test_loader_reports_missing_columns(tmp_path: Path) -> None:
    bad_path = tmp_path / "ratings.csv"
    pd.DataFrame([{"Name": "Arrival"}]).to_csv(bad_path, index=False)

    with pytest.raises(LetterboxdDataError, match="missing columns"):
        LetterboxdDataLoader._read_csv(bad_path, {"title", "year", "rating"})


def test_select_favourites_applies_threshold() -> None:
    ratings = pd.DataFrame(
        [
            {"title": "Arrival", "year": 2016, "rating": 4.5},
            {"title": "Tenet", "year": 2020, "rating": 3.0},
        ]
    )

    favourites = LetterboxdDataLoader.select_favourites(ratings, min_rating=4.0)

    assert favourites["title"].tolist() == ["Arrival"]
