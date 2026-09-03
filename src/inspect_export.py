import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "private_data" / "letterboxd"

FILES = [
    "ratings.csv",
    "diary.csv",
    "watched.csv",
    "watchlist.csv",
]


def inspect_csv(file_path: Path) -> tuple[list[str], int]:
    with file_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        row_count = sum(1 for _ in reader)
        return reader.fieldnames or [], row_count


for file_name in FILES:
    file_path = DATA_DIR / file_name

    if not file_path.exists():
        print(f"{file_name}: missing")
        continue

    headers, row_count = inspect_csv(file_path)
    print(f"{file_name}: {row_count} rows")
    print(f"  Columns: {headers}")