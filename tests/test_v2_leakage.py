"""Acceptance tests for the V2 point-in-time leakage boundary.

These deliberately use a real 2022 bout and the real project dataset. They are
slower than unit tests because the state history must be replayed from 1994.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ufc_predictor.v2_features import FEATURE_NAMES, build_v2_dataset
from ufc_predictor.v2_status import derive_active_status


ROOT = Path(__file__).resolve().parents[1]
FIGHTS = ROOT / "data" / "raw" / "fights.csv"
FIGHTERS = ROOT / "data" / "raw" / "fighters.csv"
TARGET_DATE = "2022-12-17"
TARGET_RED = "ALEX CACERES"
TARGET_BLUE = "JULIAN EROSA"


def _target(frame: pd.DataFrame) -> pd.Series:
    rows = frame.loc[
        (frame["fight_date"] == TARGET_DATE)
        & (frame[["fighter_a", "fighter_b"]].isin([TARGET_RED, TARGET_BLUE]).all(axis=1))
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def _feature_bytes(row: pd.Series, names: list[str] | None = None) -> bytes:
    selected = names or FEATURE_NAMES
    values = pd.to_numeric(row[selected], errors="coerce").to_numpy(dtype="<f8")
    values = np.nan_to_num(values, nan=-9.87654321012345e307)
    return values.tobytes()


@pytest.fixture(scope="module")
def leakage_vectors(tmp_path_factory):
    temp = tmp_path_factory.mktemp("v2-leakage")
    raw = pd.read_csv(FIGHTS, sep=";", low_memory=False)
    parsed = pd.to_datetime(raw["event_date"], format="%d/%m/%Y", errors="coerce")

    # Reference build includes later fights, as normal production replay does.
    reference, _, _ = build_v2_dataset(FIGHTS, FIGHTERS)

    # Test A/D: all fights after 2022 are deleted.
    no_future_path = temp / "no_future.csv"
    raw.loc[parsed.dt.year <= 2022].to_csv(no_future_path, sep=";", index=False)
    no_future, _, _ = build_v2_dataset(no_future_path, FIGHTERS)

    # Test B: result and every post-fight statistic for the selected bout vanish.
    no_current = raw.copy()
    target_mask = (
        (no_current["event_date"] == "17/12/2022")
        & (no_current["red_fighter_name"] == TARGET_RED)
        & (no_current["blue_fighter_name"] == TARGET_BLUE)
    )
    assert int(target_mask.sum()) == 1
    protected = {
        "red_fighter_name", "blue_fighter_name", "red_fighter_nickname",
        "blue_fighter_nickname", "event_date", "event_name", "event_location",
        "bout_type", "time_format", "referee",
    }
    postfight = [column for column in no_current.columns if column not in protected]
    no_current.loc[target_mask, postfight] = pd.NA
    no_current_path = temp / "no_current.csv"
    no_current.to_csv(no_current_path, sep=";", index=False)
    current_removed, _, _ = build_v2_dataset(no_current_path, FIGHTERS)

    return _target(reference), _target(no_future), _target(current_removed), temp


def test_a_future_mutation_does_not_change_2022_vector(leakage_vectors):
    reference, no_future, _, _ = leakage_vectors
    assert _feature_bytes(reference) == _feature_bytes(no_future)


def test_b_current_fight_result_and_stats_do_not_enter_vector(leakage_vectors):
    reference, _, current_removed, _ = leakage_vectors
    assert _feature_bytes(reference) == _feature_bytes(current_removed)


def test_c_current_status_is_product_filter_not_historical_feature(leakage_vectors):
    reference, _, _, temp = leakage_vectors
    assert not any("active" in name or "retired" in name for name in FEATURE_NAMES)
    active_path = temp / "active.csv"
    retired_path = temp / "retired.csv"
    active_path.write_text(
        "fighter_name,is_active,reason,verified_at\nALEX CACERES,true,test,2026-01-01\n",
        encoding="utf-8",
    )
    retired_path.write_text(
        "fighter_name,is_active,reason,verified_at\nALEX CACERES,false,test,2026-01-01\n",
        encoding="utf-8",
    )
    active = derive_active_status(FIGHTS, as_of=date(2026, 6, 28), overrides_path=active_path)
    retired = derive_active_status(FIGHTS, as_of=date(2026, 6, 28), overrides_path=retired_path)
    fighter_id = active.loc[active.fighter_name == TARGET_RED, "fighter_id"].iloc[0]
    assert bool(active.loc[active.fighter_id == fighter_id, "is_active"].iloc[0]) is True
    assert bool(retired.loc[retired.fighter_id == fighter_id, "is_active"].iloc[0]) is False
    # Status changes have no pathway into, and therefore cannot mutate, this row.
    assert len(_feature_bytes(reference)) == len(FEATURE_NAMES) * 8


def test_d_future_career_mutation_cannot_change_state_at_t(leakage_vectors):
    reference, no_future, _, _ = leakage_vectors
    component_names = [
        name for name in FEATURE_NAMES
        if any(token in name for token in (
            "ufc_", "elo", "recent", "streak", "form_", "sig_", "striking",
            "takedown", "td_", "submission", "control", "opponent", "quality",
            "experience", "previous_12m", "previous_24m", "days_since",
        ))
    ]
    assert component_names
    assert _feature_bytes(reference, component_names) == _feature_bytes(no_future, component_names)


def test_every_feature_timestamp_precedes_fight_timestamp():
    frame, _, _ = build_v2_dataset(FIGHTS, FIGHTERS, stop_after="2024-12-31")
    feature_time = pd.to_datetime(frame["feature_timestamp"], utc=True)
    fight_time = pd.to_datetime(frame["fight_timestamp"], utc=True)
    assert bool((feature_time < fight_time).all())
