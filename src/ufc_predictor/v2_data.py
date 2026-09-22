"""Immutable, versioned V2 data refresh and identity audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlretrieve
import uuid

import pandas as pd

from .fighter_matching import canonical_fighter_name


FIGHTS_URL = (
    "https://raw.githubusercontent.com/komaksym/UFC-DataLab/main/"
    "data/stats/stats_raw.csv"
)
FIGHTERS_URL = (
    "https://raw.githubusercontent.com/komaksym/UFC-DataLab/main/"
    "data/external_data/raw_fighter_details.csv"
)
IDENTITY_NAMESPACE = uuid.UUID("e1b5cc6d-968a-4ae4-aab7-848ecf22d7fa")
ALIAS_COLUMNS = ["alias_name", "canonical_fighter_name", "reason", "verified_at"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_aliases(path: Path | None = None) -> dict[str, str]:
    """Load reviewed name changes without guessing at fuzzy matches."""
    if path is None or not path.exists():
        return {}
    aliases = pd.read_csv(path).fillna("")
    missing = set(ALIAS_COLUMNS) - set(aliases.columns)
    if missing:
        raise RuntimeError(f"Alias table missing columns: {sorted(missing)}")
    return {
        canonical_fighter_name(row.alias_name): canonical_fighter_name(
            row.canonical_fighter_name
        )
        for row in aliases.itertuples(index=False)
        if str(row.alias_name).strip() and str(row.canonical_fighter_name).strip()
    }


def stable_fighter_id(name: str, aliases: dict[str, str] | None = None) -> str:
    """Project-stable ID; aliases can later map changed names to this ID."""
    canonical = canonical_fighter_name(name)
    canonical = (aliases or {}).get(canonical, canonical)
    return f"ufcp-{uuid.uuid5(IDENTITY_NAMESPACE, canonical)}"


def build_identity_table(
    fights: pd.DataFrame,
    fighters: pd.DataFrame,
    aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    names = set(fighters["fighter_name"].dropna().astype(str).str.strip())
    names.update(fights["red_fighter_name"].dropna().astype(str).str.strip())
    names.update(fights["blue_fighter_name"].dropna().astype(str).str.strip())
    rows = [
        {
            "fighter_id": stable_fighter_id(name, aliases),
            "canonical_name": (aliases or {}).get(
                canonical_fighter_name(name), canonical_fighter_name(name)
            ),
            "display_name": name.title() if name.isupper() else name,
            "alias_name": name,
            "identity_source": "UFC-DataLab/UFCStats-derived name",
        }
        for name in sorted(names, key=str.casefold)
    ]
    table = pd.DataFrame(rows)
    collisions = table.groupby("fighter_id")["canonical_name"].nunique()
    if (collisions > 1).any():
        raise RuntimeError("Stable fighter ID collision detected.")
    return table.drop_duplicates(["fighter_id", "alias_name"])


def audit_snapshot(
    snapshot_dir: Path,
    retrieved_at: str,
    alias_path: Path | None = None,
) -> dict:
    fights_path = snapshot_dir / "fights.csv"
    fighters_path = snapshot_dir / "fighters.csv"
    fights = pd.read_csv(fights_path, sep=";", low_memory=False)
    fighters = pd.read_csv(fighters_path)
    required_fight = {
        "event_date", "red_fighter_name", "blue_fighter_name",
        "red_fighter_result", "blue_fighter_result",
    }
    required_fighter = {"fighter_name", "DOB", "Height", "Reach"}
    if missing := required_fight - set(fights.columns):
        raise RuntimeError(f"Missing fight columns: {sorted(missing)}")
    if missing := required_fighter - set(fighters.columns):
        raise RuntimeError(f"Missing fighter columns: {sorted(missing)}")
    dates = pd.to_datetime(fights["event_date"], format="%d/%m/%Y", errors="coerce")
    if dates.isna().any():
        raise RuntimeError("Invalid event dates found in current snapshot.")
    if dates.max().date() > datetime.now(timezone.utc).date():
        raise RuntimeError("Current snapshot contains a future-dated completed fight.")
    aliases = load_aliases(alias_path)
    identities = build_identity_table(fights, fighters, aliases)
    identities.to_csv(snapshot_dir / "fighter_identities.csv", index=False)
    latest_rows = fights.loc[dates == dates.max(), [
        "event_name", "event_date", "red_fighter_name", "blue_fighter_name"
    ]]
    return {
        "retrieved_at": retrieved_at,
        "source": {
            "name": "UFC DataLab",
            "upstream": "UFCStats-derived public data",
            "fights_url": FIGHTS_URL,
            "fighters_url": FIGHTERS_URL,
        },
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in (fights_path, fighters_path, snapshot_dir / "fighter_identities.csv")
        },
        "fight_rows": int(len(fights)),
        "fighter_profile_rows": int(len(fighters)),
        "stable_fighter_ids": int(identities["fighter_id"].nunique()),
        "reviewed_aliases": int(len(aliases)),
        "date_range": [dates.min().date().isoformat(), dates.max().date().isoformat()],
        "newest_completed_fight_date": dates.max().date().isoformat(),
        "newest_event_names": sorted(latest_rows["event_name"].dropna().unique().tolist()),
        "missing_profile_fraction": {
            column: float(fighters[column].isna().mean())
            for column in ("DOB", "Height", "Reach", "Stance", "Weight")
        },
    }


def refresh(project_root: Path) -> Path:
    retrieved = datetime.now(timezone.utc)
    retrieved_at = retrieved.isoformat()
    stamp = retrieved.strftime("%Y%m%dT%H%M%SZ")
    current_root = project_root / "data" / "current"
    snapshot_dir = current_root / stamp
    if snapshot_dir.exists():
        raise FileExistsError(snapshot_dir)
    snapshot_dir.mkdir(parents=True)
    try:
        urlretrieve(FIGHTS_URL, snapshot_dir / "fights.csv")
        urlretrieve(FIGHTERS_URL, snapshot_dir / "fighters.csv")
        alias_path = current_root / "fighter_aliases.csv"
        if not alias_path.exists():
            alias_path.write_text(
                ",".join(ALIAS_COLUMNS) + "\n", encoding="utf-8"
            )
        metadata = audit_snapshot(snapshot_dir, retrieved_at, alias_path)
        (snapshot_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        current_root.mkdir(parents=True, exist_ok=True)
        (current_root / "LATEST.json").write_text(
            json.dumps({"snapshot": stamp, "retrieved_at": retrieved_at}, indent=2),
            encoding="utf-8",
        )
        return snapshot_dir
    except Exception:
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    snapshot = refresh(Path(args.project_root).resolve())
    print(snapshot)
    print((snapshot / "metadata.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
