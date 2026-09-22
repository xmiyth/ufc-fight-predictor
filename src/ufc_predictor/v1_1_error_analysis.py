"""Categorize every V1.0 Rich Logistic error on the fixed chronological test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_dataset, division_distance, division_slug, normalize_name


TEST_START = "2022-11-05"
PROBABILITY_COLUMN = "Rich Logistic"


def _value(row: pd.Series, name: str) -> float | None:
    value = row.get(name)
    return None if pd.isna(value) else float(value)


def _difference(row: pd.Series, suffix: str) -> float | None:
    first, second = _value(row, f"fighter_a_{suffix}"), _value(row, f"fighter_b_{suffix}")
    return None if first is None or second is None else first - second


def flags(row: pd.Series) -> dict[str, bool | None]:
    fights_a = int(row["fighter_a_prior_ufc_fights"])
    fights_b = int(row["fighter_b_prior_ufc_fights"])
    age_gap, reach_gap = _difference(row, "age"), _difference(row, "reach")
    distance = division_distance(
        str(row.get("fighter_a_historical_division") or "other"),
        str(row.get("fighter_b_historical_division") or "other"),
    )
    booked = str(row.get("booked_division") or "other")
    historical = {
        str(row.get("fighter_a_historical_division") or "other"),
        str(row.get("fighter_b_historical_division") or "other"),
    }
    layoff_a = _value(row, "fighter_a_days_since_last_fight")
    layoff_b = _value(row, "fighter_b_days_since_last_fight")
    recent_a, recent_b = _value(row, "fighter_a_recent_win_pct"), _value(row, "fighter_b_recent_win_pct")
    career_a, career_b = _value(row, "fighter_a_career_win_pct"), _value(row, "fighter_b_career_win_pct")
    td_a, td_b = _value(row, "fighter_a_takedowns_per15"), _value(row, "fighter_b_takedowns_per15")
    sig_a, sig_b = _value(row, "fighter_a_sig_landed_pm"), _value(row, "fighter_b_sig_landed_pm")
    sub_a, sub_b = _value(row, "fighter_a_submission_rate"), _value(row, "fighter_b_submission_rate")
    sub_loss_a, sub_loss_b = _value(row, "fighter_a_submission_loss_rate"), _value(row, "fighter_b_submission_loss_rate")
    ko_a, ko_b = _value(row, "fighter_a_ko_rate"), _value(row, "fighter_b_ko_rate")
    defense_a, defense_b = _value(row, "fighter_a_striking_defense"), _value(row, "fighter_b_striking_defense")
    confidence = max(float(row[PROBABILITY_COLUMN]), 1.0 - float(row[PROBABILITY_COLUMN]))
    return {
        "ufc_debutant_or_very_low_experience": min(fights_a, fights_b) <= 2,
        "large_age_difference": age_gap is not None and abs(age_gap) >= 8.0,
        "large_reach_difference": reach_gap is not None and abs(reach_gap) >= 6.0,
        "large_size_difference": distance is not None and distance >= 2,
        "fighter_moving_weight_classes": booked != "other" and any(
            value != "other" and value != booked for value in historical
        ),
        "long_layoff": any(value is not None and value >= 540 for value in (layoff_a, layoff_b)),
        "short_notice": None,
        "recent_decline": any(
            recent is not None and career is not None and recent <= career - 0.20
            for recent, career in ((recent_a, career_a), (recent_b, career_b))
        ),
        "recent_improvement": any(
            recent is not None and career is not None and recent >= career + 0.20
            for recent, career in ((recent_a, career_a), (recent_b, career_b))
        ),
        "wrestler_vs_striker": (
            td_a is not None and td_b is not None and sig_a is not None and sig_b is not None
            and ((td_a >= 2 and td_b < 1 and sig_b >= 3.5) or (td_b >= 2 and td_a < 1 and sig_a >= 3.5))
        ),
        "grappler_vs_weak_submission_defense": (
            sub_a is not None and sub_b is not None and sub_loss_a is not None and sub_loss_b is not None
            and ((sub_a >= .35 and sub_loss_b >= .25) or (sub_b >= .35 and sub_loss_a >= .25))
        ),
        "power_puncher_vs_poor_defense": (
            ko_a is not None and ko_b is not None and defense_a is not None and defense_b is not None
            and ((ko_a >= .50 and defense_b <= .45) or (ko_b >= .50 and defense_a <= .45))
        ),
        "low_data_fighter": min(fights_a, fights_b) < 3,
        "close_50_50_fight": confidence < .55,
        "major_upset": confidence >= .70,
    }


def run(root: Path) -> dict:
    features, _, _ = build_dataset(
        str(root / "data" / "raw" / "fights.csv"),
        str(root / "data" / "raw" / "fighters.csv"),
    )
    features = features.loc[features.date >= TEST_START].copy()
    features["booked_division"] = features.apply(
        lambda row: next(
            (name.removeprefix("weight_class_") for name, value in row.items()
             if name.startswith("weight_class_") and value == 1.0),
            "other",
        ),
        axis=1,
    )
    predictions = pd.read_csv(root / "reports" / "model_test_predictions.csv")
    join = ["date", "fighter_a", "fighter_b"]
    for frame in (features, predictions):
        frame["fighter_a"] = frame.fighter_a.map(normalize_name)
        frame["fighter_b"] = frame.fighter_b.map(normalize_name)
    merged = features.merge(
        predictions[join + ["fighter_a_won", PROBABILITY_COLUMN]],
        on=join, suffixes=("", "_saved"), validate="one_to_one",
    )
    merged["v1_correct"] = (
        (merged[PROBABILITY_COLUMN] >= .5).astype(int) == merged.fighter_a_won_saved
    )
    errors = merged.loc[~merged.v1_correct].copy()
    flag_rows = [flags(row) for _, row in errors.iterrows()]
    categories = list(flag_rows[0])
    counts = {
        name: {
            "count": int(sum(row[name] is True for row in flag_rows)),
            "percent_of_errors": float(100 * sum(row[name] is True for row in flag_rows) / len(errors)),
            "available": not all(row[name] is None for row in flag_rows),
        }
        for name in categories
    }
    primary_order = [name for name in categories if name not in {"low_data_fighter", "short_notice"}]
    primary = []
    for row in flag_rows:
        primary.append(next((name for name in primary_order if row[name] is True), "other"))
    primary_counts = pd.Series(primary).value_counts().to_dict()
    output_rows = errors[join + ["fighter_a_won_saved", PROBABILITY_COLUMN]].copy()
    output_rows["primary_category"] = primary
    for name in categories:
        output_rows[name] = [row[name] for row in flag_rows]
    output_rows.to_csv(root / "reports" / "v1_1_error_rows.csv", index=False)
    report = {
        "baseline": "V1.0 Full Rich Logistic",
        "test_period": [str(errors.date.min()), str(errors.date.max())],
        "test_fights": int(len(merged)),
        "incorrect_predictions": int(len(errors)),
        "error_rate": float(len(errors) / len(merged)),
        "overlapping_category_counts": counts,
        "mutually_exclusive_primary_categories": {
            name: {"count": int(count), "percent_of_errors": float(100 * count / len(errors))}
            for name, count in primary_counts.items()
        },
        "short_notice_note": "Unavailable in the supplied historical files.",
        "thresholds": {
            "large_age_gap_years": 8,
            "large_reach_gap_inches": 6,
            "large_size_gap_divisions": 2,
            "long_layoff_days": 540,
            "close_confidence": "<55%",
            "major_upset_confidence": ">=70%",
        },
    }
    (root / "reports" / "v1_1_error_analysis.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run(Path(".").resolve())
