"""Download the public UFCStats-derived source files."""

from pathlib import Path
from urllib.request import urlretrieve


FIGHTS_URL = (
    "https://raw.githubusercontent.com/komaksym/UFC-DataLab/main/"
    "data/stats/stats_raw.csv"
)
FIGHTERS_URL = (
    "https://raw.githubusercontent.com/komaksym/UFC-DataLab/main/"
    "data/external_data/raw_fighter_details.csv"
)


def main() -> None:
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    for url, filename in ((FIGHTS_URL, "fights.csv"), (FIGHTERS_URL, "fighters.csv")):
        destination = output_dir / filename
        print(f"Downloading {filename} ...")
        urlretrieve(url, destination)
        print(f"Saved {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

