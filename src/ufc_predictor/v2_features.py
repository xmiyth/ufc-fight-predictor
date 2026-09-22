"""Strict point-in-time V2 fighter snapshots and matchup features.

All bouts on an event date are snapshotted before any bout from that date is
applied to fighter state. This avoids relying on unavailable event bout times.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import hashlib
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .features import (
    age_on,
    fight_duration_seconds,
    load_profiles,
    normalize_name,
    numeric_or_zero,
    parse_clock,
    parse_of,
)
from .v2_data import stable_fighter_id


LOCKBOX_START = "2025-01-01"
DECAY_HALF_LIVES = (180, 365, 548, 730, 1095)


@dataclass
class BoutRecord:
    bout_date: str
    won: int
    method: str
    duration_seconds: int
    scheduled_rounds: int
    sig_landed: float
    sig_attempted: float
    sig_absorbed: float
    sig_faced: float
    total_landed: float
    total_attempted: float
    td_landed: float
    td_attempted: float
    opponent_td_landed: float
    opponent_td_attempted: float
    submissions: float
    reversals: float
    control_seconds: float
    knockdowns: float
    head_landed: float
    body_landed: float
    leg_landed: float
    distance_landed: float
    clinch_landed: float
    ground_landed: float
    opponent_elo: float
    opponent_sig_landed_pm: float
    opponent_sig_absorbed_pm: float
    opponent_sig_accuracy: float
    opponent_striking_defense: float
    opponent_td_accuracy: float
    opponent_td_defense: float
    opponent_submissions_per15: float
    opponent_finish_rate: float
    weight_class: str


@dataclass
class V2FighterState:
    fighter_id: str
    display_name: str
    records: list[BoutRecord] = field(default_factory=list)
    elo: float = 1500.0
    elo_k16: float = 1500.0
    elo_k24: float = 1500.0
    elo_k48: float = 1500.0
    weight_elos: dict[str, float] = field(default_factory=dict)


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def scheduled_rounds(value: Any) -> int:
    text = str(value)
    for candidate in (5, 3):
        if text.strip().startswith(str(candidate)):
            return candidate
    return 3


def division_slug(value: Any) -> str:
    text = normalize_name(str(value)).replace("'", "")
    for phrase, slug in (
        ("womens strawweight", "womens_strawweight"),
        ("womens flyweight", "womens_flyweight"),
        ("womens bantamweight", "womens_bantamweight"),
        ("womens featherweight", "womens_featherweight"),
        ("light heavyweight", "light_heavyweight"),
        ("catch weight", "catch_weight"),
        ("open weight", "open_weight"),
        ("flyweight", "flyweight"),
        ("bantamweight", "bantamweight"),
        ("featherweight", "featherweight"),
        ("lightweight", "lightweight"),
        ("welterweight", "welterweight"),
        ("middleweight", "middleweight"),
        ("heavyweight", "heavyweight"),
    ):
        if phrase in text:
            return slug
    return "other"


def division_group(slug: str) -> str:
    if slug.startswith("womens_"):
        return "women"
    if slug in {"flyweight", "bantamweight", "featherweight", "lightweight"}:
        return "light"
    if slug in {"welterweight", "middleweight"}:
        return "middle"
    if slug in {"light_heavyweight", "heavyweight"}:
        return "heavy"
    return "other"


def _sum(records: list[BoutRecord], name: str, weights: list[float] | None = None) -> float:
    if weights is None:
        return sum(float(getattr(record, name)) for record in records)
    return sum(float(getattr(record, name)) * weight for record, weight in zip(records, weights))


def _weights(records: list[BoutRecord], as_of: date, half_life_days: int) -> list[float]:
    return [
        0.5 ** (max(0, (as_of - date.fromisoformat(record.bout_date)).days) / half_life_days)
        for record in records
    ]


def _performance(records: list[BoutRecord], weights: list[float] | None = None) -> dict[str, float]:
    if not records:
        return {
            "sig_landed_pm": 0.0, "sig_absorbed_pm": 0.0, "sig_accuracy": 0.0,
            "striking_defense": 0.0, "striking_differential_pm": 0.0,
            "strike_volume_pm": 0.0, "knockdowns_per15": 0.0,
            "takedowns_per15": 0.0, "takedown_accuracy": 0.0,
            "takedown_defense": 0.0, "submissions_per15": 0.0,
            "reversals_per15": 0.0, "control_pct": 0.0,
            "opponent_adjusted_accuracy": 0.0,
            "opponent_adjusted_striking_differential": 0.0,
            "opponent_adjusted_striking_defense": 0.0,
            "opponent_adjusted_takedown_offense": 0.0,
            "opponent_adjusted_takedown_defense": 0.0,
            "opponent_adjusted_submission_activity": 0.0,
            "head_strike_share": 0.0, "body_strike_share": 0.0,
            "leg_strike_share": 0.0, "distance_strike_share": 0.0,
            "clinch_strike_share": 0.0, "ground_strike_share": 0.0,
        }
    effective = weights if weights is not None else [1.0] * len(records)
    seconds = _sum(records, "duration_seconds", effective)
    minutes = seconds / 60.0
    sig_landed = _sum(records, "sig_landed", effective)
    sig_attempted = _sum(records, "sig_attempted", effective)
    sig_absorbed = _sum(records, "sig_absorbed", effective)
    sig_faced = _sum(records, "sig_faced", effective)
    td_landed = _sum(records, "td_landed", effective)
    td_attempted = _sum(records, "td_attempted", effective)
    opp_td_landed = _sum(records, "opponent_td_landed", effective)
    opp_td_attempted = _sum(records, "opponent_td_attempted", effective)
    average_opp_defense = safe_ratio(
        _sum(records, "opponent_striking_defense", effective), sum(effective)
    )
    average_opp_sig_landed = safe_ratio(
        _sum(records, "opponent_sig_landed_pm", effective), sum(effective)
    )
    average_opp_sig_absorbed = safe_ratio(
        _sum(records, "opponent_sig_absorbed_pm", effective), sum(effective)
    )
    average_opp_accuracy = safe_ratio(
        _sum(records, "opponent_sig_accuracy", effective), sum(effective)
    )
    average_opp_td_accuracy = safe_ratio(
        _sum(records, "opponent_td_accuracy", effective), sum(effective)
    )
    average_opp_td_defense = safe_ratio(
        _sum(records, "opponent_td_defense", effective), sum(effective)
    )
    average_opp_submissions = safe_ratio(
        _sum(records, "opponent_submissions_per15", effective), sum(effective)
    )
    accuracy = safe_ratio(sig_landed, sig_attempted)
    landed_pm = safe_ratio(sig_landed, minutes)
    absorbed_pm = safe_ratio(sig_absorbed, minutes)
    return {
        "sig_landed_pm": landed_pm,
        "sig_absorbed_pm": absorbed_pm,
        "sig_accuracy": accuracy,
        "striking_defense": 1.0 - safe_ratio(sig_absorbed, sig_faced),
        "striking_differential_pm": landed_pm - absorbed_pm,
        "strike_volume_pm": safe_ratio(sig_attempted, minutes),
        "knockdowns_per15": safe_ratio(_sum(records, "knockdowns", effective) * 15.0, minutes),
        "takedowns_per15": safe_ratio(td_landed * 15.0, minutes),
        "takedown_accuracy": safe_ratio(td_landed, td_attempted),
        "takedown_defense": 1.0 - safe_ratio(opp_td_landed, opp_td_attempted),
        "submissions_per15": safe_ratio(_sum(records, "submissions", effective) * 15.0, minutes),
        "reversals_per15": safe_ratio(_sum(records, "reversals", effective) * 15.0, minutes),
        "control_pct": safe_ratio(_sum(records, "control_seconds", effective), seconds),
        "opponent_adjusted_accuracy": accuracy - (1.0 - average_opp_defense),
        "opponent_adjusted_striking_differential": (
            landed_pm - average_opp_sig_absorbed
        ) - (absorbed_pm - average_opp_sig_landed),
        "opponent_adjusted_striking_defense": (
            1.0 - safe_ratio(sig_absorbed, sig_faced)
        ) - average_opp_accuracy,
        "opponent_adjusted_takedown_offense": safe_ratio(td_landed, td_attempted)
        - (1.0 - average_opp_td_defense),
        "opponent_adjusted_takedown_defense": (
            1.0 - safe_ratio(opp_td_landed, opp_td_attempted)
        ) - average_opp_td_accuracy,
        "opponent_adjusted_submission_activity": safe_ratio(
            _sum(records, "submissions", effective) * 15.0, minutes
        ) - average_opp_submissions,
        "head_strike_share": safe_ratio(_sum(records, "head_landed", effective), sig_landed),
        "body_strike_share": safe_ratio(_sum(records, "body_landed", effective), sig_landed),
        "leg_strike_share": safe_ratio(_sum(records, "leg_landed", effective), sig_landed),
        "distance_strike_share": safe_ratio(_sum(records, "distance_landed", effective), sig_landed),
        "clinch_strike_share": safe_ratio(_sum(records, "clinch_landed", effective), sig_landed),
        "ground_strike_share": safe_ratio(_sum(records, "ground_landed", effective), sig_landed),
    }


def fighter_snapshot(
    state: V2FighterState,
    profile: dict[str, Any],
    as_of: date,
    matchup_division: str,
) -> dict[str, float | None]:
    records = state.records
    fights = len(records)
    wins = sum(record.won for record in records)
    losses = fights - wins
    recent = [record.won for record in records[-5:]]
    methods = [record.method.casefold() for record in records if record.won]
    finish_wins = sum(not method.startswith("decision") for method in methods)
    ko_wins = sum("ko" in method for method in methods)
    sub_wins = sum("sub" in method for method in methods)
    decision_wins = sum(method.startswith("decision") for method in methods)
    last_date = date.fromisoformat(records[-1].bout_date) if records else None
    days_inactive = (as_of - last_date).days if last_date else None
    outcomes = recent
    win_streak = 0
    loss_streak = 0
    for outcome in reversed([record.won for record in records]):
        if outcome == 1 and loss_streak == 0:
            win_streak += 1
        elif outcome == 0 and win_streak == 0:
            loss_streak += 1
        else:
            break
    elo_recency = (
        1500.0
        if days_inactive is None
        else 1500.0 + (state.elo - 1500.0) * 0.5 ** (max(days_inactive, 0) / 730.0)
    )
    opponent_elos = [record.opponent_elo for record in records]
    won_elos = [record.opponent_elo for record in records if record.won]
    lost_elos = [record.opponent_elo for record in records if not record.won]
    result: dict[str, float | None] = {
        "age": age_on(profile.get("dob"), as_of),
        "height": profile.get("height"),
        "reach": profile.get("reach"),
        "ufc_wins": float(wins), "ufc_losses": float(losses),
        "ufc_fights": float(fights), "ufc_win_pct": safe_ratio(wins, fights),
        "last_1_win": float(records[-1].won) if records else 0.5,
        "recent_3_win_pct": safe_ratio(sum(outcomes[-3:]), len(outcomes[-3:]), 0.5),
        "recent_5_win_pct": safe_ratio(sum(outcomes), len(outcomes), 0.5),
        "current_win_streak": float(win_streak),
        "current_loss_streak": float(loss_streak),
        "days_since_last_fight": float(days_inactive) if days_inactive is not None else None,
        "fights_previous_12m": float(sum((as_of - date.fromisoformat(r.bout_date)).days <= 365 for r in records)),
        "fights_previous_24m": float(sum((as_of - date.fromisoformat(r.bout_date)).days <= 730 for r in records)),
        "finish_rate": safe_ratio(finish_wins, wins),
        "ko_tko_rate": safe_ratio(ko_wins, wins),
        "submission_rate": safe_ratio(sub_wins, wins),
        "decision_rate": safe_ratio(decision_wins, wins),
        "average_fight_duration": safe_ratio(
            sum(record.duration_seconds for record in records), fights * 60.0
        ),
        "five_round_experience": float(sum(record.scheduled_rounds == 5 for record in records)),
        "elo": state.elo,
        "elo_k16": state.elo_k16,
        "elo_k24": state.elo_k24,
        "elo_k48": state.elo_k48,
        "recency_adjusted_elo": elo_recency,
        "weight_class_elo": state.weight_elos.get(matchup_division, 1500.0),
        "average_opponent_elo": safe_ratio(sum(opponent_elos), len(opponent_elos), 1500.0),
        "average_beaten_opponent_elo": safe_ratio(sum(won_elos), len(won_elos), 1500.0),
        "average_lost_opponent_elo": safe_ratio(sum(lost_elos), len(lost_elos), 1500.0),
        "high_elo_wins": float(sum(record.won and record.opponent_elo >= 1600 for record in records)),
        "high_elo_losses": float(sum(not record.won and record.opponent_elo >= 1600 for record in records)),
        "quality_adjusted_win_rate": safe_ratio(
            sum(record.won * record.opponent_elo / 1500.0 for record in records), fights
        ),
        "quality_adjusted_finish_rate": safe_ratio(
            sum(
                record.won
                * (not record.method.casefold().startswith("decision"))
                * record.opponent_elo / 1500.0
                for record in records
            ),
            wins,
        ),
        **_performance(records),
    }
    age = result["age"]
    result["age_squared"] = None if age is None else float(age) ** 2
    result["age_x_experience"] = None if age is None else float(age) * fights
    layoff = result["days_since_last_fight"]
    result["age_x_layoff"] = None if age is None or layoff is None else float(age) * float(layoff) / 365.0
    for half_life in DECAY_HALF_LIVES:
        performance = _performance(records, _weights(records, as_of, half_life))
        suffix = f"decay_{half_life}d"
        for name, value in performance.items():
            result[f"{name}_{suffix}"] = value
        weights = _weights(records, as_of, half_life)
        result[f"form_{suffix}"] = safe_ratio(
            sum(record.won * weight for record, weight in zip(records, weights)),
            sum(weights),
            0.5,
        )
        result[f"opponent_elo_{suffix}"] = safe_ratio(
            sum(record.opponent_elo * weight for record, weight in zip(records, weights)),
            sum(weights),
            1500.0,
        )
    return result


SNAPSHOT_FEATURES = [
    "age", "age_squared", "height", "reach", "ufc_wins", "ufc_losses",
    "ufc_fights", "ufc_win_pct", "last_1_win", "recent_3_win_pct",
    "recent_5_win_pct", "current_win_streak", "current_loss_streak",
    "days_since_last_fight", "fights_previous_12m", "fights_previous_24m",
    "finish_rate", "ko_tko_rate", "submission_rate", "decision_rate",
    "average_fight_duration", "five_round_experience", "elo",
    "elo_k16", "elo_k24", "elo_k48",
    "recency_adjusted_elo", "weight_class_elo", "average_opponent_elo",
    "average_beaten_opponent_elo", "average_lost_opponent_elo",
    "high_elo_wins", "high_elo_losses", "quality_adjusted_win_rate",
    "quality_adjusted_finish_rate",
    "sig_landed_pm", "sig_absorbed_pm", "sig_accuracy", "striking_defense",
    "striking_differential_pm", "strike_volume_pm", "knockdowns_per15",
    "takedowns_per15", "takedown_accuracy", "takedown_defense",
    "submissions_per15", "reversals_per15", "control_pct",
    "opponent_adjusted_accuracy", "opponent_adjusted_striking_differential",
    "opponent_adjusted_striking_defense", "opponent_adjusted_takedown_offense",
    "opponent_adjusted_takedown_defense",
    "opponent_adjusted_submission_activity", "age_x_experience", "age_x_layoff",
    "head_strike_share", "body_strike_share", "leg_strike_share",
    "distance_strike_share", "clinch_strike_share", "ground_strike_share",
]
for _half_life in DECAY_HALF_LIVES:
    SNAPSHOT_FEATURES.extend(
        [
            f"{name}_decay_{_half_life}d"
            for name in (
                "sig_landed_pm", "sig_absorbed_pm", "sig_accuracy",
                "striking_defense", "striking_differential_pm", "strike_volume_pm",
                "knockdowns_per15", "takedowns_per15", "takedown_accuracy",
                "takedown_defense", "submissions_per15", "reversals_per15",
                "control_pct", "opponent_adjusted_accuracy",
                "opponent_adjusted_striking_differential",
                "opponent_adjusted_striking_defense",
                "opponent_adjusted_takedown_offense",
                "opponent_adjusted_takedown_defense",
                "opponent_adjusted_submission_activity",
                "head_strike_share", "body_strike_share", "leg_strike_share",
                "distance_strike_share", "clinch_strike_share", "ground_strike_share",
            )
        ]
        + [f"form_decay_{_half_life}d", f"opponent_elo_decay_{_half_life}d"]
    )

DIFFERENCE_FEATURES = [f"{name}_diff" for name in SNAPSHOT_FEATURES]
INTERACTION_FEATURES = [
    "a_td_offense_vs_b_defense", "b_td_offense_vs_a_defense",
    "a_striking_offense_vs_b_defense", "b_striking_offense_vs_a_defense",
    "a_submission_threat_vs_b_grappling", "b_submission_threat_vs_a_grappling",
]
CONTEXT_FEATURES = [
    "scheduled_rounds", "five_round_fight", "division_women", "division_light",
    "division_middle", "division_heavy", "division_other",
    "age_diff_x_division_light", "age_diff_x_division_heavy",
    "reach_diff_x_division_light", "wrestling_matchup_x_division_light",
    "wrestling_matchup_x_division_heavy",
]
FEATURE_NAMES = DIFFERENCE_FEATURES + INTERACTION_FEATURES + CONTEXT_FEATURES

METHOD_SNAPSHOT_FEATURES = [
    "finish_rate", "ko_tko_rate", "submission_rate", "decision_rate",
    "average_fight_duration", "ufc_fights", "recent_3_win_pct",
    "quality_adjusted_win_rate", "age", "sig_landed_pm", "sig_absorbed_pm",
    "sig_accuracy", "striking_defense", "knockdowns_per15",
    "takedowns_per15", "takedown_accuracy", "takedown_defense",
    "submissions_per15", "control_pct", "head_strike_share",
    "body_strike_share", "leg_strike_share", "distance_strike_share",
    "clinch_strike_share", "ground_strike_share",
]
METHOD_FEATURE_NAMES = (
    [f"{name}_mean" for name in METHOD_SNAPSHOT_FEATURES]
    + [f"{name}_abs_diff" for name in METHOD_SNAPSHOT_FEATURES]
    + ["scheduled_rounds", "five_round_fight", "division_women", "division_light",
       "division_middle", "division_heavy", "division_other"]
)

FEATURE_GROUPS = {
    "age": [name for name in FEATURE_NAMES if name.startswith("age")],
    "physical": [name for name in FEATURE_NAMES if name.startswith(("height", "reach"))],
    "recent_form": [name for name in FEATURE_NAMES if any(term in name for term in ("recent", "streak", "form_", "last_1"))],
    "activity": [name for name in FEATURE_NAMES if any(term in name for term in ("days_since", "previous_12m", "previous_24m"))],
    "career": [name for name in FEATURE_NAMES if any(term in name for term in ("ufc_", "finish_rate", "ko_tko_rate", "decision_rate", "five_round_experience"))],
    "striking": [name for name in FEATURE_NAMES if any(term in name for term in ("sig_", "striking_", "strike_volume", "knockdown", "accuracy"))],
    "grappling": [name for name in FEATURE_NAMES if any(term in name for term in ("td_", "takedown", "submission", "reversal", "control"))],
    "opponent_quality": [name for name in FEATURE_NAMES if any(term in name for term in ("elo", "quality", "opponent"))],
    "opponent_adjusted_performance": [
        name for name in FEATURE_NAMES if "opponent_adjusted" in name
    ],
    "elo_rating": [name for name in FEATURE_NAMES if "elo" in name],
    "matchup_interactions": INTERACTION_FEATURES,
    "context": CONTEXT_FEATURES,
}


def matchup_features(
    a: dict[str, float | None],
    b: dict[str, float | None],
    bout_type: str,
    rounds: int,
) -> dict[str, float | None]:
    result = {
        f"{name}_diff": (
            None if a.get(name) is None or b.get(name) is None
            else float(a[name]) - float(b[name])
        )
        for name in SNAPSHOT_FEATURES
    }
    result.update({
        "a_td_offense_vs_b_defense": float(a["takedown_accuracy"]) - float(b["takedown_defense"]),
        "b_td_offense_vs_a_defense": float(b["takedown_accuracy"]) - float(a["takedown_defense"]),
        "a_striking_offense_vs_b_defense": float(a["sig_landed_pm"]) * (1.0 - float(b["striking_defense"])),
        "b_striking_offense_vs_a_defense": float(b["sig_landed_pm"]) * (1.0 - float(a["striking_defense"])),
        "a_submission_threat_vs_b_grappling": float(a["submissions_per15"]) * (1.0 - float(b["takedown_defense"])),
        "b_submission_threat_vs_a_grappling": float(b["submissions_per15"]) * (1.0 - float(a["takedown_defense"])),
    })
    for name in METHOD_SNAPSHOT_FEATURES:
        a_value, b_value = a.get(name), b.get(name)
        if a_value is None or b_value is None:
            result[f"{name}_mean"] = None
            result[f"{name}_abs_diff"] = None
        else:
            result[f"{name}_mean"] = (float(a_value) + float(b_value)) / 2.0
            result[f"{name}_abs_diff"] = abs(float(a_value) - float(b_value))
    group = division_group(division_slug(bout_type))
    for candidate in ("women", "light", "middle", "heavy", "other"):
        result[f"division_{candidate}"] = float(group == candidate)
    result["scheduled_rounds"] = float(rounds)
    result["five_round_fight"] = float(rounds == 5)
    age_diff = result["age_diff"]
    reach_diff = result["reach_diff"]
    wrestling = result["a_td_offense_vs_b_defense"] - result["b_td_offense_vs_a_defense"]
    result["age_diff_x_division_light"] = None if age_diff is None else age_diff * result["division_light"]
    result["age_diff_x_division_heavy"] = None if age_diff is None else age_diff * result["division_heavy"]
    result["reach_diff_x_division_light"] = None if reach_diff is None else reach_diff * result["division_light"]
    result["wrestling_matchup_x_division_light"] = wrestling * result["division_light"]
    result["wrestling_matchup_x_division_heavy"] = wrestling * result["division_heavy"]
    return result


def _fight_id(row: pd.Series) -> str:
    raw = "|".join(str(row.get(name, "")) for name in (
        "event_date", "event_name", "red_fighter_name", "blue_fighter_name"
    ))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _snapshot_timestamp(bout_date: date) -> str:
    instant = datetime.combine(bout_date, datetime.min.time(), tzinfo=timezone.utc)
    return (instant - timedelta(microseconds=1)).isoformat()


def _profile_map(profiles_path: str | Path) -> dict[str, dict[str, Any]]:
    return load_profiles(str(profiles_path))


def build_v2_dataset(
    fights_path: str | Path,
    profiles_path: str | Path,
    *,
    include_labels: bool = True,
    stop_after: str | None = None,
) -> tuple[pd.DataFrame, dict[str, V2FighterState], dict[str, dict[str, Any]]]:
    fights = pd.read_csv(fights_path, sep=";", low_memory=False)
    fights["event_date_parsed"] = pd.to_datetime(
        fights["event_date"], format="%d/%m/%Y", errors="coerce"
    )
    fights = fights.dropna(subset=["event_date_parsed"])
    if stop_after:
        fights = fights[fights["event_date_parsed"] <= pd.Timestamp(stop_after)]
    fights = fights.sort_values(["event_date_parsed", "event_name"], kind="stable")
    profiles = _profile_map(profiles_path)
    states: dict[str, V2FighterState] = {}
    examples: list[dict[str, Any]] = []

    def state_for(name: str) -> V2FighterState:
        key = stable_fighter_id(name)
        if key not in states:
            display = str(name).strip().title() if str(name).isupper() else str(name).strip()
            states[key] = V2FighterState(key, display)
        return states[key]

    for timestamp, event_rows in fights.groupby("event_date_parsed", sort=True):
        bout_date = timestamp.date()
        pending_updates: list[
            tuple[pd.Series, V2FighterState, V2FighterState, dict, dict, dict, dict, float, float]
        ] = []
        for _, row in event_rows.iterrows():
            red_name = str(row["red_fighter_name"]).strip()
            blue_name = str(row["blue_fighter_name"]).strip()
            red_state, blue_state = state_for(red_name), state_for(blue_name)
            division = division_slug(row.get("bout_type"))
            red_profile = profiles.get(normalize_name(red_name), {})
            blue_profile = profiles.get(normalize_name(blue_name), {})
            red_before = fighter_snapshot(red_state, red_profile, bout_date, division)
            blue_before = fighter_snapshot(blue_state, blue_profile, bout_date, division)
            red_result = str(row.get("red_fighter_result", "")).strip().upper()
            blue_result = str(row.get("blue_fighter_result", "")).strip().upper()
            decisive = {red_result, blue_result} == {"W", "L"}
            swap = int(_fight_id(row), 16) % 2 == 0
            first, second = (blue_before, red_before) if swap else (red_before, blue_before)
            first_name, second_name = (blue_name, red_name) if swap else (red_name, blue_name)
            rounds = scheduled_rounds(row.get("time_format"))
            values = matchup_features(first, second, str(row.get("bout_type", "")), rounds)
            example = {
                "fight_id": _fight_id(row),
                "event_name": str(row.get("event_name", "")),
                "fight_date": bout_date.isoformat(),
                "fight_timestamp": datetime.combine(
                    bout_date, datetime.min.time(), tzinfo=timezone.utc
                ).isoformat(),
                "feature_timestamp": _snapshot_timestamp(bout_date),
                "fighter_a_id": stable_fighter_id(first_name),
                "fighter_b_id": stable_fighter_id(second_name),
                "fighter_a": first_name,
                "fighter_b": second_name,
                "weight_class": division,
                **values,
            }
            if include_labels:
                example["fighter_a_won"] = (
                    int((blue_result == "W") if swap else (red_result == "W"))
                    if decisive else None
                )
                method = str(row.get("method", "")).strip().casefold()
                if decisive:
                    if "ko" in method:
                        example["method_label"] = "KO/TKO"
                    elif "sub" in method:
                        example["method_label"] = "Submission"
                    elif method.startswith("decision"):
                        example["method_label"] = "Decision"
                    else:
                        example["method_label"] = None
                else:
                    example["method_label"] = None
            examples.append(example)
            pending_updates.append(
                (
                    row,
                    red_state,
                    blue_state,
                    red_before,
                    blue_before,
                    {name: float(getattr(red_state, name)) for name in ("elo", "elo_k16", "elo_k24", "elo_k48")},
                    {name: float(getattr(blue_state, name)) for name in ("elo", "elo_k16", "elo_k24", "elo_k48")},
                    red_state.weight_elos.get(division, 1500.0),
                    blue_state.weight_elos.get(division, 1500.0),
                )
            )

        # No same-date result or statistic can enter another same-date snapshot.
        for (
            row,
            red_state,
            blue_state,
            red_before,
            blue_before,
            red_pre_ratings,
            blue_pre_ratings,
            red_pre_weight_elo,
            blue_pre_weight_elo,
        ) in pending_updates:
            _apply_fight(
                row,
                red_state,
                blue_state,
                red_before,
                blue_before,
                red_pre_ratings,
                blue_pre_ratings,
                red_pre_weight_elo,
                blue_pre_weight_elo,
            )

    return pd.DataFrame(examples), states, profiles


def _apply_fight(
    row: pd.Series,
    red: V2FighterState,
    blue: V2FighterState,
    red_before: dict[str, Any],
    blue_before: dict[str, Any],
    red_pre_ratings: dict[str, float],
    blue_pre_ratings: dict[str, float],
    red_pre_weight_elo: float,
    blue_pre_weight_elo: float,
) -> None:
    red_result = str(row.get("red_fighter_result", "")).strip().upper()
    blue_result = str(row.get("blue_fighter_result", "")).strip().upper()
    if {red_result, blue_result} != {"W", "L"}:
        return
    duration = fight_duration_seconds(row)
    red_sig, blue_sig = parse_of(row.get("red_fighter_sig_str")), parse_of(row.get("blue_fighter_sig_str"))
    red_total, blue_total = parse_of(row.get("red_fighter_total_str")), parse_of(row.get("blue_fighter_total_str"))
    red_td, blue_td = parse_of(row.get("red_fighter_TD")), parse_of(row.get("blue_fighter_TD"))
    bout_date = pd.to_datetime(row["event_date_parsed"]).date().isoformat()
    division = division_slug(row.get("bout_type"))
    rounds = scheduled_rounds(row.get("time_format"))

    def record(prefix: str, own_sig, opp_sig, own_total, own_td, opp_td, won, opponent_elo, opponent_before):
        return BoutRecord(
            bout_date=bout_date, won=won, method=str(row.get("method", "")),
            duration_seconds=duration, scheduled_rounds=rounds,
            sig_landed=own_sig[0], sig_attempted=own_sig[1],
            sig_absorbed=opp_sig[0], sig_faced=opp_sig[1],
            total_landed=own_total[0], total_attempted=own_total[1],
            td_landed=own_td[0], td_attempted=own_td[1],
            opponent_td_landed=opp_td[0], opponent_td_attempted=opp_td[1],
            submissions=numeric_or_zero(row.get(f"{prefix}_fighter_sub_att")),
            reversals=numeric_or_zero(row.get(f"{prefix}_fighter_rev")),
            control_seconds=parse_clock(row.get(f"{prefix}_fighter_ctrl")),
            knockdowns=numeric_or_zero(row.get(f"{prefix}_fighter_KD")),
            head_landed=parse_of(row.get(f"{prefix}_fighter_sig_str_head"))[0],
            body_landed=parse_of(row.get(f"{prefix}_fighter_sig_str_body"))[0],
            leg_landed=parse_of(row.get(f"{prefix}_fighter_sig_str_leg"))[0],
            distance_landed=parse_of(row.get(f"{prefix}_fighter_sig_str_distance"))[0],
            clinch_landed=parse_of(row.get(f"{prefix}_fighter_sig_str_clinch"))[0],
            ground_landed=parse_of(row.get(f"{prefix}_fighter_sig_str_ground"))[0],
            opponent_elo=opponent_elo,
            opponent_sig_landed_pm=float(opponent_before["sig_landed_pm"]),
            opponent_sig_absorbed_pm=float(opponent_before["sig_absorbed_pm"]),
            opponent_sig_accuracy=float(opponent_before["sig_accuracy"]),
            opponent_striking_defense=float(opponent_before["striking_defense"]),
            opponent_td_accuracy=float(opponent_before["takedown_accuracy"]),
            opponent_td_defense=float(opponent_before["takedown_defense"]),
            opponent_submissions_per15=float(opponent_before["submissions_per15"]),
            opponent_finish_rate=float(opponent_before["finish_rate"]),
            weight_class=division,
        )

    red.records.append(record("red", red_sig, blue_sig, red_total, red_td, blue_td, int(red_result == "W"), blue_pre_ratings["elo"], blue_before))
    blue.records.append(record("blue", blue_sig, red_sig, blue_total, blue_td, red_td, int(blue_result == "W"), red_pre_ratings["elo"], red_before))
    # Accumulate changes from a shared pre-event rating. This matters for early
    # tournament cards where a fighter can appear more than once on one date.
    for attribute, k_factor in (("elo", 32.0), ("elo_k16", 16.0), ("elo_k24", 24.0), ("elo_k48", 48.0)):
        expected_red = 1.0 / (
            1.0 + 10.0 ** ((blue_pre_ratings[attribute] - red_pre_ratings[attribute]) / 400.0)
        )
        change = k_factor * (int(red_result == "W") - expected_red)
        setattr(red, attribute, getattr(red, attribute) + change)
        setattr(blue, attribute, getattr(blue, attribute) - change)
    expected_weight_red = 1.0 / (
        1.0 + 10.0 ** ((blue_pre_weight_elo - red_pre_weight_elo) / 400.0)
    )
    weight_change = 24.0 * (int(red_result == "W") - expected_weight_red)
    red.weight_elos[division] = red.weight_elos.get(division, 1500.0) + weight_change
    blue.weight_elos[division] = blue.weight_elos.get(division, 1500.0) - weight_change
