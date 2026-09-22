"""Advanced V3 point-in-time features layered over the frozen V2 feature path.

This module independently replays history for V3-only ratings, trajectories,
division changes, and opponent-adjusted performance. It merges those values by
fight_id with V2 point-in-time rows without modifying V1/V2 artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import fight_duration_seconds, numeric_or_zero, parse_clock, parse_of
from .v2_data import stable_fighter_id
from .v2_features import (
    FEATURE_NAMES,
    _fight_id,
    build_v2_dataset,
    division_slug,
    scheduled_rounds,
)


GLICKO_SCALE = 173.7178
GLICKO_TAU = 0.5
DIVISION_ORDER = {
    "womens_strawweight": 1, "flyweight": 2, "womens_flyweight": 2,
    "bantamweight": 3, "womens_bantamweight": 3, "featherweight": 4,
    "womens_featherweight": 4, "lightweight": 5, "welterweight": 6,
    "middleweight": 7, "light_heavyweight": 8, "heavyweight": 9,
}
DECAYS = (180, 365, 548, 730, 1095)


@dataclass
class V3Record:
    bout_date: str
    won: int
    finished: int
    division: str
    title_fight: int
    opponent_glicko: float
    opponent_bt: float
    sig_landed_pm: float
    sig_absorbed_pm: float
    sig_accuracy: float
    striking_defense: float
    sig_diff_pm: float
    strike_volume_pm: float
    td_per15: float
    td_absorbed_per15: float
    td_accuracy: float
    td_defense: float
    td_diff_per15: float
    submissions_per15: float
    control_pct: float
    knockdowns_per15: float
    adj_sig_landed: float
    adj_sig_absorbed: float
    adj_accuracy: float
    adj_defense: float
    adj_td_rate: float
    adj_td_accuracy: float
    adj_td_defense: float
    adj_submission: float
    adj_control: float
    adj_knockdown: float
    adj_finish: float
    post_glicko: float = 1500.0


@dataclass
class V3State:
    fighter_id: str
    records: list[V3Record] = field(default_factory=list)
    glicko_rating: float = 1500.0
    glicko_rd: float = 350.0
    glicko_volatility: float = 0.06
    bt_strength: float = 0.0
    division_bt: dict[str, float] = field(default_factory=dict)
    last_rating_date: date | None = None


def _ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, np.asarray(values, dtype=float), 1)[0])


def _inflated_rd(state: V3State, as_of: date) -> float:
    if state.last_rating_date is None:
        return state.glicko_rd
    periods = max(0.0, (as_of - state.last_rating_date).days / 30.0)
    phi = state.glicko_rd / GLICKO_SCALE
    inflated = math.sqrt(phi * phi + state.glicko_volatility**2 * periods)
    return min(350.0, inflated * GLICKO_SCALE)


def _glicko_update(
    rating: float,
    rd: float,
    volatility: float,
    opponent_rating: float,
    opponent_rd: float,
    score: float,
) -> tuple[float, float, float]:
    """One-opponent Glicko-2 update following Glickman's published algorithm."""
    mu = (rating - 1500.0) / GLICKO_SCALE
    phi = rd / GLICKO_SCALE
    mu_j = (opponent_rating - 1500.0) / GLICKO_SCALE
    phi_j = opponent_rd / GLICKO_SCALE
    g = 1.0 / math.sqrt(1.0 + 3.0 * phi_j * phi_j / math.pi**2)
    expected = 1.0 / (1.0 + math.exp(-g * (mu - mu_j)))
    variance = 1.0 / (g * g * expected * (1.0 - expected))
    delta = variance * g * (score - expected)
    a = math.log(volatility * volatility)

    def objective(x: float) -> float:
        exp_x = math.exp(x)
        numerator = exp_x * (delta * delta - phi * phi - variance - exp_x)
        denominator = 2.0 * (phi * phi + variance + exp_x) ** 2
        return numerator / denominator - (x - a) / (GLICKO_TAU * GLICKO_TAU)

    lower = a
    if delta * delta > phi * phi + variance:
        upper = math.log(delta * delta - phi * phi - variance)
    else:
        step = 1
        upper = a - step * GLICKO_TAU
        while objective(upper) < 0:
            step += 1
            upper = a - step * GLICKO_TAU
    f_lower, f_upper = objective(lower), objective(upper)
    while abs(upper - lower) > 1e-6:
        candidate = lower + (lower - upper) * f_lower / (f_upper - f_lower)
        f_candidate = objective(candidate)
        if f_candidate * f_upper <= 0:
            lower, f_lower = upper, f_upper
        else:
            f_lower /= 2.0
        upper, f_upper = candidate, f_candidate
    new_volatility = math.exp(lower / 2.0)
    phi_star = math.sqrt(phi * phi + new_volatility * new_volatility)
    new_phi = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / variance)
    new_mu = mu + new_phi * new_phi * g * (score - expected)
    return 1500.0 + GLICKO_SCALE * new_mu, GLICKO_SCALE * new_phi, new_volatility


def _averages(records: list[V3Record]) -> dict[str, float]:
    names = (
        "sig_landed_pm", "sig_absorbed_pm", "sig_accuracy", "striking_defense",
        "sig_diff_pm", "strike_volume_pm", "td_per15", "td_absorbed_per15",
        "td_accuracy", "td_defense", "td_diff_per15", "submissions_per15",
        "control_pct", "knockdowns_per15", "adj_sig_landed", "adj_sig_absorbed",
        "adj_accuracy", "adj_defense", "adj_td_rate", "adj_td_accuracy",
        "adj_td_defense", "adj_submission", "adj_control", "adj_knockdown",
        "adj_finish",
    )
    if not records:
        return {name: 0.0 for name in names}
    return {
        name: float(np.mean([getattr(record, name) for record in records]))
        for name in names
    }


def snapshot(state: V3State, as_of: date, division: str) -> dict[str, float]:
    records = state.records
    outcomes = [record.won for record in records]
    recent5 = records[-5:]
    averages = _averages(records)
    result: dict[str, float] = {
        "glicko_rating": state.glicko_rating,
        "glicko_rd": _inflated_rd(state, as_of),
        "glicko_volatility": state.glicko_volatility,
        "bradley_terry_strength": state.bt_strength,
        "rating_uncertainty": 1.0 / math.sqrt(len(records) + 1.0),
        "division_strength": state.division_bt.get(division, state.bt_strength * 0.5),
        "last_2_win_pct": _ratio(sum(outcomes[-2:]), len(outcomes[-2:]), 0.5),
        "time_since_ufc_debut_days": float((as_of - date.fromisoformat(records[0].bout_date)).days) if records else 0.0,
        "fights_current_division": float(sum(record.division == division for record in records)),
        "first_fight_new_division": float(bool(records) and records[-1].division != division),
        "recent_division_change": float(bool(records) and any(record.division != division for record in records[-2:])),
        "division_move_direction": float(
            DIVISION_ORDER.get(division, 0) - DIVISION_ORDER.get(records[-1].division, 0)
        ) if records else 0.0,
        "title_fight_experience": float(sum(record.title_fight for record in records)),
        "highest_rated_opponent_beaten": max(
            [record.opponent_glicko for record in records if record.won], default=1500.0
        ),
        "recent_opponent_rating": float(np.mean([record.opponent_glicko for record in recent5])) if recent5 else 1500.0,
        "recent_finish_rate": _ratio(sum(record.finished for record in recent5), len(recent5)),
        **averages,
    }
    adjusted_outcomes = [
        record.won * record.opponent_glicko / 1500.0 for record in records
    ]
    result["opponent_adjusted_recent_3"] = _ratio(
        sum(adjusted_outcomes[-3:]), len(adjusted_outcomes[-3:]), 0.5
    )
    result["opponent_adjusted_recent_5"] = _ratio(
        sum(adjusted_outcomes[-5:]), len(adjusted_outcomes[-5:]), 0.5
    )
    trend_sources = {
        "striking_diff_trend": [record.sig_diff_pm for record in recent5],
        "defense_trend": [record.striking_defense for record in recent5],
        "takedown_trend": [record.td_diff_per15 for record in recent5],
        "rating_trend": [record.post_glicko for record in recent5],
        "opponent_quality_trend": [record.opponent_glicko for record in recent5],
    }
    result.update({name: _slope(values) for name, values in trend_sources.items()})
    for half_life in DECAYS:
        weights = [
            0.5 ** ((as_of - date.fromisoformat(record.bout_date)).days / half_life)
            for record in records
        ]
        result[f"opponent_adjusted_form_decay_{half_life}d"] = _ratio(
            sum(value * weight for value, weight in zip(adjusted_outcomes, weights)),
            sum(weights),
            0.5,
        )
    return result


V3_SNAPSHOT_FEATURES = [
    "glicko_rating", "glicko_rd", "glicko_volatility",
    "bradley_terry_strength", "rating_uncertainty", "division_strength",
    "last_2_win_pct", "time_since_ufc_debut_days", "fights_current_division",
    "first_fight_new_division", "recent_division_change", "division_move_direction",
    "title_fight_experience", "highest_rated_opponent_beaten",
    "recent_opponent_rating", "recent_finish_rate", "opponent_adjusted_recent_3",
    "opponent_adjusted_recent_5", "striking_diff_trend", "defense_trend",
    "takedown_trend", "rating_trend", "opponent_quality_trend",
    "sig_landed_pm", "sig_absorbed_pm", "sig_accuracy", "striking_defense",
    "sig_diff_pm", "strike_volume_pm", "td_per15", "td_absorbed_per15",
    "td_accuracy", "td_defense", "td_diff_per15", "submissions_per15",
    "control_pct", "knockdowns_per15", "adj_sig_landed", "adj_sig_absorbed",
    "adj_accuracy", "adj_defense", "adj_td_rate", "adj_td_accuracy",
    "adj_td_defense", "adj_submission", "adj_control", "adj_knockdown",
    "adj_finish",
]
for _half_life in DECAYS:
    V3_SNAPSHOT_FEATURES.append(f"opponent_adjusted_form_decay_{_half_life}d")

V3_INTERACTIONS = [
    "striking_offense_defense_matchup", "reverse_striking_offense_defense_matchup",
    "wrestling_offense_defense_matchup", "reverse_wrestling_offense_defense_matchup",
    "submission_grappling_matchup", "reverse_submission_grappling_matchup",
    "pressure_efficiency_matchup", "knockdown_threat_matchup",
    "any_division_change", "both_low_history", "rating_advantage_x_uncertainty",
]
V3_EXTRA_FEATURE_NAMES = [f"v3_{name}_diff" for name in V3_SNAPSHOT_FEATURES] + [
    f"v3_{name}" for name in V3_INTERACTIONS
]
V3_FEATURE_NAMES = FEATURE_NAMES + V3_EXTRA_FEATURE_NAMES

V3_FEATURE_GROUPS = {
    "fighter_ratings": [name for name in V3_FEATURE_NAMES if any(token in name for token in ("elo", "glicko", "bradley_terry", "division_strength", "rating_"))],
    "opponent_quality": [name for name in V3_FEATURE_NAMES if any(token in name for token in ("opponent", "quality", "highest_rated"))],
    "striking": [name for name in V3_FEATURE_NAMES if any(token in name for token in ("sig_", "striking", "knockdown", "pressure"))],
    "wrestling": [name for name in V3_FEATURE_NAMES if any(token in name for token in ("td_", "takedown", "wrestling"))],
    "grappling": [name for name in V3_FEATURE_NAMES if any(token in name for token in ("submission", "control", "reversal", "grappling"))],
    "age": [name for name in V3_FEATURE_NAMES if "age" in name],
    "physical": [name for name in V3_FEATURE_NAMES if any(token in name for token in ("height", "reach"))],
    "recent_form": [name for name in V3_FEATURE_NAMES if any(token in name for token in ("recent", "last_", "form_decay", "streak"))],
    "trend": [name for name in V3_FEATURE_NAMES if "trend" in name],
    "activity": [name for name in V3_FEATURE_NAMES if any(token in name for token in ("days_since", "previous_12m", "previous_24m", "debut"))],
    "division_change": [name for name in V3_FEATURE_NAMES if "division" in name],
    "matchup_interactions": [f"v3_{name}" for name in V3_INTERACTIONS],
}


def _raw_performance(row: pd.Series, prefix: str, opponent_prefix: str) -> dict[str, float]:
    duration = max(1, fight_duration_seconds(row))
    minutes = duration / 60.0
    sig = parse_of(row.get(f"{prefix}_fighter_sig_str"))
    opp_sig = parse_of(row.get(f"{opponent_prefix}_fighter_sig_str"))
    td = parse_of(row.get(f"{prefix}_fighter_TD"))
    opp_td = parse_of(row.get(f"{opponent_prefix}_fighter_TD"))
    return {
        "sig_landed_pm": _ratio(sig[0], minutes),
        "sig_absorbed_pm": _ratio(opp_sig[0], minutes),
        "sig_accuracy": _ratio(sig[0], sig[1]),
        "striking_defense": 1.0 - _ratio(opp_sig[0], opp_sig[1]),
        "sig_diff_pm": _ratio(sig[0] - opp_sig[0], minutes),
        "strike_volume_pm": _ratio(sig[1], minutes),
        "td_per15": _ratio(td[0] * 15.0, minutes),
        "td_absorbed_per15": _ratio(opp_td[0] * 15.0, minutes),
        "td_accuracy": _ratio(td[0], td[1]),
        "td_defense": 1.0 - _ratio(opp_td[0], opp_td[1]),
        "td_diff_per15": _ratio((td[0] - opp_td[0]) * 15.0, minutes),
        "submissions_per15": _ratio(numeric_or_zero(row.get(f"{prefix}_fighter_sub_att")) * 15.0, minutes),
        "control_pct": _ratio(parse_clock(row.get(f"{prefix}_fighter_ctrl")), duration),
        "knockdowns_per15": _ratio(numeric_or_zero(row.get(f"{prefix}_fighter_KD")) * 15.0, minutes),
    }


def _extra_matchup(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    result = {
        f"v3_{name}_diff": float(a[name]) - float(b[name])
        for name in V3_SNAPSHOT_FEATURES
    }
    result.update({
        "v3_striking_offense_defense_matchup": a["sig_landed_pm"] - b["striking_defense"],
        "v3_reverse_striking_offense_defense_matchup": b["sig_landed_pm"] - a["striking_defense"],
        "v3_wrestling_offense_defense_matchup": a["td_accuracy"] - b["td_defense"],
        "v3_reverse_wrestling_offense_defense_matchup": b["td_accuracy"] - a["td_defense"],
        "v3_submission_grappling_matchup": a["submissions_per15"] - b["td_defense"],
        "v3_reverse_submission_grappling_matchup": b["submissions_per15"] - a["td_defense"],
        "v3_pressure_efficiency_matchup": (a["strike_volume_pm"] * a["sig_accuracy"]) - (b["strike_volume_pm"] * b["sig_accuracy"]),
        "v3_knockdown_threat_matchup": a["knockdowns_per15"] - b["knockdowns_per15"],
        "v3_any_division_change": float(bool(a["first_fight_new_division"] or b["first_fight_new_division"])),
        "v3_both_low_history": float(a["rating_uncertainty"] > 0.5 and b["rating_uncertainty"] > 0.5),
        "v3_rating_advantage_x_uncertainty": (a["glicko_rating"] - b["glicko_rating"]) * ((a["glicko_rd"] + b["glicko_rd"]) / 700.0),
    })
    return result


def build_v3_dataset(
    fights_path: str | Path,
    profiles_path: str | Path,
    *,
    stop_after: str | None = None,
) -> pd.DataFrame:
    base, _, _ = build_v2_dataset(fights_path, profiles_path, stop_after=stop_after)
    fights = pd.read_csv(fights_path, sep=";", low_memory=False)
    fights["event_date_parsed"] = pd.to_datetime(fights.event_date, format="%d/%m/%Y", errors="coerce")
    fights = fights.dropna(subset=["event_date_parsed"])
    if stop_after:
        fights = fights.loc[fights.event_date_parsed <= pd.Timestamp(stop_after)]
    fights = fights.sort_values(["event_date_parsed", "event_name"], kind="stable")
    states: dict[str, V3State] = {}
    extras: list[dict[str, Any]] = []

    def state_for(name: str) -> V3State:
        fighter_id = stable_fighter_id(name)
        if fighter_id not in states:
            states[fighter_id] = V3State(fighter_id)
        return states[fighter_id]

    for timestamp, rows in fights.groupby("event_date_parsed", sort=True):
        as_of = timestamp.date()
        pending = []
        for _, row in rows.iterrows():
            red, blue = state_for(row.red_fighter_name), state_for(row.blue_fighter_name)
            division = division_slug(row.get("bout_type"))
            red_snapshot, blue_snapshot = snapshot(red, as_of, division), snapshot(blue, as_of, division)
            swap = int(_fight_id(row), 16) % 2 == 0
            first, second = (blue_snapshot, red_snapshot) if swap else (red_snapshot, blue_snapshot)
            extras.append({"fight_id": _fight_id(row), **_extra_matchup(first, second)})
            pending.append((row, red, blue, red_snapshot, blue_snapshot, division, as_of))
        for row, red, blue, red_before, blue_before, division, as_of in pending:
            _apply(row, red, blue, red_before, blue_before, division, as_of)
    extra_frame = pd.DataFrame(extras)
    # A few historical rows share the legacy fight hash. Pair repeated hashes
    # by their stable chronological occurrence rather than multiplying rows.
    base["_v3_occurrence"] = base.groupby("fight_id", sort=False).cumcount()
    extra_frame["_v3_occurrence"] = extra_frame.groupby("fight_id", sort=False).cumcount()
    merged = base.merge(
        extra_frame,
        on=["fight_id", "_v3_occurrence"],
        how="left",
        validate="one_to_one",
    )
    return merged.drop(columns="_v3_occurrence")


def _apply(
    row: pd.Series,
    red: V3State,
    blue: V3State,
    red_before: dict[str, float],
    blue_before: dict[str, float],
    division: str,
    as_of: date,
) -> None:
    red_result = str(row.get("red_fighter_result", "")).strip().upper()
    blue_result = str(row.get("blue_fighter_result", "")).strip().upper()
    if {red_result, blue_result} != {"W", "L"}:
        return
    red_score = float(red_result == "W")
    red_perf = _raw_performance(row, "red", "blue")
    blue_perf = _raw_performance(row, "blue", "red")
    method = str(row.get("method", "")).casefold()
    finished = int(not method.startswith("decision"))
    title = int("title" in str(row.get("bout_type", "")).casefold())
    red_rd, blue_rd = red_before["glicko_rd"], blue_before["glicko_rd"]
    red_new = _glicko_update(red_before["glicko_rating"], red_rd, red.glicko_volatility, blue_before["glicko_rating"], blue_rd, red_score)
    blue_new = _glicko_update(blue_before["glicko_rating"], blue_rd, blue.glicko_volatility, red_before["glicko_rating"], red_rd, 1.0 - red_score)

    def make_record(
        own: dict[str, float], opponent: dict[str, float], before: dict[str, float],
        opponent_before: dict[str, float], won: int, post_glicko: float,
    ) -> V3Record:
        return V3Record(
            bout_date=as_of.isoformat(), won=won, finished=finished,
            division=division, title_fight=title,
            opponent_glicko=opponent_before["glicko_rating"],
            opponent_bt=opponent_before["bradley_terry_strength"],
            **own,
            adj_sig_landed=own["sig_landed_pm"] - opponent_before["sig_absorbed_pm"],
            adj_sig_absorbed=opponent_before["sig_landed_pm"] - own["sig_absorbed_pm"],
            adj_accuracy=own["sig_accuracy"] - (1.0 - opponent_before["striking_defense"]),
            adj_defense=own["striking_defense"] - opponent_before["sig_accuracy"],
            adj_td_rate=own["td_per15"] - opponent_before["td_absorbed_per15"],
            adj_td_accuracy=own["td_accuracy"] - (1.0 - opponent_before["td_defense"]),
            adj_td_defense=own["td_defense"] - opponent_before["td_accuracy"],
            adj_submission=own["submissions_per15"] - opponent_before["submissions_per15"],
            adj_control=own["control_pct"] - opponent_before["control_pct"],
            adj_knockdown=own["knockdowns_per15"] * (1.0 + opponent_before["striking_defense"]),
            adj_finish=finished * opponent_before["glicko_rating"] / 1500.0,
            post_glicko=post_glicko,
        )

    red.records.append(make_record(red_perf, blue_perf, red_before, blue_before, int(red_score), red_new[0]))
    blue.records.append(make_record(blue_perf, red_perf, blue_before, red_before, int(1-red_score), blue_new[0]))
    # Accumulate from shared pre-event values for tournament cards.
    red.glicko_rating += red_new[0] - red_before["glicko_rating"]
    blue.glicko_rating += blue_new[0] - blue_before["glicko_rating"]
    red.glicko_rd, red.glicko_volatility = red_new[1], red_new[2]
    blue.glicko_rd, blue.glicko_volatility = blue_new[1], blue_new[2]
    expected = 1.0 / (1.0 + math.exp(-(red_before["bradley_terry_strength"] - blue_before["bradley_terry_strength"])))
    change = 0.12 * (red_score - expected)
    red.bt_strength += change
    blue.bt_strength -= change
    red_div = red_before["division_strength"]
    blue_div = blue_before["division_strength"]
    div_expected = 1.0 / (1.0 + math.exp(-(red_div - blue_div)))
    div_change = 0.10 * (red_score - div_expected)
    red.division_bt[division] = red_div + div_change
    blue.division_bt[division] = blue_div - div_change
    red.last_rating_date = as_of
    blue.last_rating_date = as_of
