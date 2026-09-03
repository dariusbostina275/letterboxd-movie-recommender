"""Unit tests for ranking and strict watched-film exclusion."""

import pandas as pd

from src.recommender import ContentBasedRecommender


def movie(
    tmdb_id: int,
    title: str,
    year: int,
    genres: list[str],
    keywords: list[str],
    director: str,
    cast: list[str],
    overview: str,
    rating: float | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "tmdb_id": tmdb_id,
        "title": title,
        "year": year,
        "genres": genres,
        "keywords": keywords,
        "director": director,
        "cast": cast,
        "overview": overview,
    }
    if rating is not None:
        record["rating"] = rating
    return record


def test_recommender_ranks_related_movie_and_excludes_watched() -> None:
    favourites = pd.DataFrame(
        [
            movie(
                1,
                "Favourite Space Film",
                2016,
                ["Science Fiction", "Drama"],
                ["first contact", "linguistics"],
                "Jane Director",
                ["Actor One"],
                "A linguist communicates with visitors from space.",
                5.0,
            )
        ]
    )
    candidates = pd.DataFrame(
        [
            movie(
                2,
                "Related Film",
                2024,
                ["Science Fiction", "Drama"],
                ["first contact"],
                "Jane Director",
                ["Actor Two"],
                "Visitors from space make contact with a scientist.",
            ),
            movie(
                3,
                "Unrelated Comedy",
                2023,
                ["Comedy"],
                ["wedding"],
                "Someone Else",
                ["Actor Three"],
                "A chaotic wedding weekend.",
            ),
        ]
    )
    watched = pd.DataFrame([{"title": "Unrelated Comedy", "year": 2023}])

    model = ContentBasedRecommender().fit(favourites, candidates)
    recommendations = model.recommend(candidates, watched, top_n=10)

    assert recommendations["title"].tolist() == ["Related Film"]
    assert "same director" in recommendations.iloc[0]["explanation"]


def test_watched_matching_is_case_and_punctuation_insensitive() -> None:
    favourites = pd.DataFrame(
        [movie(1, "Example", 2000, ["Drama"], [], "Director", [], "A story", 5.0)]
    )
    candidates = pd.DataFrame(
        [movie(2, "Spider-Man", 2002, ["Action"], [], "Director", [], "A hero")]
    )
    watched = pd.DataFrame([{"title": "spider man", "year": 2002}])

    recommendations = ContentBasedRecommender().fit(favourites, candidates).recommend(
        candidates, watched
    )

    assert recommendations.empty
