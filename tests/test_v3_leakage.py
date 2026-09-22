"""Real-history temporal integrity tests for every V3-only feature."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ufc_predictor.v3_features import V3_EXTRA_FEATURE_NAMES, build_v3_dataset


ROOT = Path(__file__).resolve().parents[1]
FIGHTS = ROOT / "data" / "raw" / "fights.csv"
FIGHTERS = ROOT / "data" / "raw" / "fighters.csv"


def _target(frame: pd.DataFrame) -> pd.Series:
    mask = (
        (frame.fight_date == "2022-12-17")
        & frame[["fighter_a", "fighter_b"]].isin(["ALEX CACERES", "JULIAN EROSA"]).all(axis=1)
    )
    assert int(mask.sum()) == 1
    return frame.loc[mask].iloc[0]


def _bytes(row: pd.Series, names: list[str] = V3_EXTRA_FEATURE_NAMES) -> bytes:
    values = pd.to_numeric(row[names], errors="coerce").to_numpy(dtype="<f8")
    return np.nan_to_num(values, nan=-9.87654321012345e307).tobytes()


@pytest.fixture(scope="module")
def vectors(tmp_path_factory):
    temp = tmp_path_factory.mktemp("v3-leakage")
    raw = pd.read_csv(FIGHTS, sep=";", low_memory=False)
    dates = pd.to_datetime(raw.event_date, format="%d/%m/%Y", errors="coerce")
    reference = build_v3_dataset(FIGHTS, FIGHTERS)

    no_future_path = temp / "no_future.csv"
    raw.loc[dates.dt.year <= 2022].to_csv(no_future_path, sep=";", index=False)
    no_future = build_v3_dataset(no_future_path, FIGHTERS)

    current_removed = raw.copy()
    target = (
        (current_removed.event_date == "17/12/2022")
        & (current_removed.red_fighter_name == "ALEX CACERES")
        & (current_removed.blue_fighter_name == "JULIAN EROSA")
    )
    protected = {
        "red_fighter_name", "blue_fighter_name", "red_fighter_nickname",
        "blue_fighter_nickname", "event_date", "event_name", "event_location",
        "bout_type", "time_format", "referee",
    }
    current_removed.loc[target, [c for c in current_removed if c not in protected]] = pd.NA
    no_current_path = temp / "no_current.csv"
    current_removed.to_csv(no_current_path, sep=";", index=False)
    no_current = build_v3_dataset(no_current_path, FIGHTERS)
    return _target(reference), _target(no_future), _target(no_current), reference


def test_v3_future_mutation_is_exactly_invariant(vectors):
    reference, no_future, _, _ = vectors
    assert _bytes(reference) == _bytes(no_future)


def test_v3_current_bout_is_excluded(vectors):
    reference, _, no_current, _ = vectors
    assert _bytes(reference) == _bytes(no_current)


def test_v3_future_status_has_no_feature_path():
    assert not any("active" in name or "retired" in name for name in V3_EXTRA_FEATURE_NAMES)


def test_v3_ratings_stats_trends_and_divisions_are_future_invariant(vectors):
    reference, no_future, _, _ = vectors
    names = [
        name for name in V3_EXTRA_FEATURE_NAMES
        if any(token in name for token in (
            "glicko", "bradley", "rating", "adj_", "trend", "division",
            "recent", "debut", "title", "opponent",
        ))
    ]
    assert names
    assert _bytes(reference, names) == _bytes(no_future, names)


def test_v3_feature_timestamps_are_strictly_before_fights(vectors):
    *_, frame = vectors
    assert bool(
        (pd.to_datetime(frame.feature_timestamp, utc=True) < pd.to_datetime(frame.fight_timestamp, utc=True)).all()
    )
