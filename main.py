"""Command-line entry point for the Letterboxd movie recommender."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data_loader import (
    LetterboxdDataError,
    LetterboxdDataLoader,
    TMDBAPIError,
    TMDBClient,
)
from src.recommender import ContentBasedRecommender, RecommenderError

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Recommend unseen movies from a Letterboxd export."
    )
    parser.add_argument("--ratings", type=Path, default=Path("data/raw/ratings.csv"))
    parser.add_argument("--watched", type=Path, default=Path("data/raw/watched.csv"))
    parser.add_argument(
        "--min-rating",
        type=float,
        default=4.0,
        help="Minimum Letterboxd rating used to build the profile (default: 4.0).",
    )
    parser.add_argument(
        "--popular-pages",
        type=int,
        default=5,
        help="Number of TMDB popular pages to use (20 movies per page).",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--cache", type=Path, default=Path("data/cache/tmdb_movies.json")
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def print_recommendations(recommendations: pd.DataFrame) -> None:
    """Render ranked recommendations in a readable console format."""

    # Kept separate from model code so another UI can replace the console later.
    if getattr(recommendations, "empty", True):
        print("\nNo unseen recommendations were found. Try more popular pages.")
        return

    print("\nYour movie recommendations")
    print("=" * 72)
    for rank, row in enumerate(recommendations.to_dict(orient="records"), start=1):
        raw_year = row.get("year")
        year = "Unknown year" if pd.isna(raw_year) else int(raw_year)
        score = float(row["similarity"])
        print(f"{rank:>2}. {row['title']} ({year}) — similarity {score:.1%}")
        print(f"    Why: {row['explanation']}")
        overview = str(row.get("overview") or "").strip()
        if overview:
            shortened = overview if len(overview) <= 160 else f"{overview[:157].rstrip()}..."
            print(f"    {shortened}")
        print()


def run(args: argparse.Namespace) -> int:
    """Execute the ingestion, enrichment, training, and ranking pipeline."""

    loader = LetterboxdDataLoader(args.ratings, args.watched)
    ratings = loader.load_ratings()
    watched = loader.load_watched()
    favourites = loader.select_favourites(ratings, args.min_rating)
    if favourites.empty:
        raise LetterboxdDataError(
            f"No movies have a rating of {args.min_rating:g} or higher."
        )

    LOGGER.info("Enriching %s favourite movies", len(favourites))
    with TMDBClient(cache_path=args.cache) as tmdb:
        favourite_metadata = tmdb.enrich_letterboxd_movies(favourites)
        LOGGER.info("Fetching %s TMDB popular pages", args.popular_pages)
        candidates = tmdb.fetch_popular_movies(args.popular_pages)

    if favourite_metadata.empty:
        raise RecommenderError("TMDB could not match any highly rated movies.")
    if candidates.empty:
        raise RecommenderError("TMDB returned no candidate movies.")

    recommender = ContentBasedRecommender()
    recommender.fit(favourite_metadata, candidates)
    recommendations = recommender.recommend(candidates, watched, args.top_n)
    print_recommendations(recommendations)
    return 0


def main() -> int:
    """Parse arguments and translate expected errors into friendly messages."""

    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        return run(args)
    except (LetterboxdDataError, TMDBAPIError, RecommenderError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Cancelled by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
