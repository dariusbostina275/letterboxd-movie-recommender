# Letterboxd Movie Recommender

An explainable Python movie recommender that uses a local Letterboxd data export and public TMDB metadata to rank movies from a personal watchlist.

The project is designed with privacy in mind: personal ratings, viewing history, API credentials, and generated recommendations remain local and are never committed to GitHub.

## How it works

1. Read local Letterboxd ratings, watched films, and watchlist files.
2. Fetch and cache public movie metadata from TMDB using title and year.
3. Calculate genre preferences from ratings.
4. Exclude films that have already been watched or rated.
5. Rank watchlist films and explain each result through its strongest matching genres.

The first model uses average centered ratings by genre:

```text
genre preference = average(rating - 2.5)