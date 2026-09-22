"""Local FastAPI interface for the frozen UFC Predictor V2.1 ensemble."""

from __future__ import annotations

from datetime import date
import hashlib
import math
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
import joblib
import pandas as pd
from pydantic import BaseModel, Field
import uvicorn

from .event_sources import Event, EventFight, create_event_source
from .features import (
    FighterState,
    difference_features,
    division_rank,
    fighter_snapshot,
    method_features,
    normalize_name,
)
from .fighter_matching import FighterNameMatcher
from .prediction_history import PredictionHistory
from .winner_v2 import feature_frame, analyze
from .fighter_portraits import portrait


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "v2.1-accuracy" / "ufc_predictor_v2_1_accuracy.joblib"
PROFILE_MODEL_PATH = PROJECT_ROOT / "models" / "v1.1" / "ufc_predictor_v1_1.joblib"
INDEX_PATH = PROJECT_ROOT / "web" / "index.html"
METHOD_MODEL_PATH = (
    PROJECT_ROOT / "models" / "method_v1.0" / "method_predictor_v1_0.joblib"
)
EXPECTED_MODEL_SHA256 = "5C0725ED3BEC1FF9B82C9DB548963BC27279CB9DCB6852A462A7F443CB059416"
EXPECTED_METHOD_MODEL_SHA256 = "1877349434ED89329B3DDD8B8644085E1E5DEDAA2F9E492CF6B7072647CF8D3C"
MIN_COMPLETED_FIGHTS = 0
MODEL_VERSION = "V2.1-accuracy"
METHOD_MODEL_VERSION = "Method V2.0"

CATEGORY_EXPLANATIONS = {
    "Age": "The fighters' age profiles shift the historical model toward {fighter}.",
    "Physical attributes": "The recorded height and reach matchup contributes in {fighter}'s direction.",
    "Recent form": "Recent UFC results contribute in {fighter}'s direction.",
    "Experience": "The accumulated UFC fight and result history favors {fighter} in the model.",
    "Strength of competition": "Pre-fight rating and opponent-strength history favor {fighter}.",
    "Striking": "Historical striking output, accuracy, and power metrics favor {fighter}.",
    "Striking defense": "Historical significant-strike defense and absorption metrics favor {fighter}.",
    "Wrestling": "The historical takedown-performance matchup favors {fighter}.",
    "Takedown defense": "The historical takedown-resistance profile favors {fighter}.",
    "Submission threat": "Historical submission activity and outcomes favor {fighter}.",
    "Finishing profile": "Historical finish and method tendencies shift the model toward {fighter}.",
    "Fight activity": "The time-since-last-fight profile favors {fighter}.",
}

FEATURE_LABELS = {
    "age_diff": "Age matchup",
    "height_diff": "Height advantage",
    "reach_diff": "Reach advantage",
    "ufc_wins_diff": "UFC wins",
    "ufc_losses_diff": "UFC loss history",
    "ufc_win_pct_diff": "UFC win percentage",
    "recent_3_win_pct_diff": "Recent 3-fight form",
    "recent_5_win_pct_diff": "Recent 5-fight form",
    "current_win_streak_diff": "Current win streak",
    "current_loss_streak_diff": "Current loss streak",
    "sig_str_accuracy_diff": "Significant-strike accuracy",
    "sig_str_landed_pm_diff": "Significant-strike output",
    "sig_str_absorbed_pm_diff": "Significant strikes absorbed",
    "striking_differential_pm_diff": "Striking differential",
    "striking_defense_diff": "Striking defense",
    "knockdowns_per_fight_diff": "Knockdown rate",
    "head_strike_share_diff": "Head-strike tendency",
    "body_strike_share_diff": "Body-strike tendency",
    "leg_strike_share_diff": "Leg-strike tendency",
    "distance_strike_share_diff": "Distance striking style",
    "clinch_strike_share_diff": "Clinch striking style",
    "ground_strike_share_diff": "Ground striking style",
    "takedown_accuracy_diff": "Takedown accuracy",
    "takedown_defense_diff": "Takedown defense",
    "submission_attempts_per15_diff": "Submission activity",
    "finish_rate_diff": "Finish rate",
    "ko_tko_win_pct_diff": "KO/TKO history",
    "submission_win_pct_diff": "Submission history",
    "decision_win_pct_diff": "Decision history",
    "days_since_last_fight_diff": "Time since last fight",
    "ufc_fights_diff": "UFC experience",
    "elo_diff": "Pre-fight Elo strength",
    "average_opponent_elo_diff": "Opponent quality faced",
    "average_beaten_opponent_elo_diff": "Quality of defeated opponents",
    "quality_adjusted_win_score_diff": "Quality-adjusted wins",
    "fighter_a_td_accuracy_vs_b_defense": "Fighter A takedown matchup",
    "fighter_b_td_accuracy_vs_a_defense": "Fighter B takedown matchup",
    "fighter_a_striking_offense_vs_b_defense": "Fighter A striking matchup",
    "fighter_b_striking_offense_vs_a_defense": "Fighter B striking matchup",
}


def _load_frozen_model() -> dict[str, Any]:
    lock = json.loads(MODEL_PATH.with_name("ARCHITECTURE_LOCK.json").read_text())
    for filename, key in (("fights.csv", "raw_fights_sha256"), ("fighters.csv", "raw_fighters_sha256")):
        actual = hashlib.sha256((PROJECT_ROOT / "data" / "raw" / filename).read_bytes()).hexdigest().upper()
        if actual != lock["checksums"][key]:
            raise RuntimeError("Fighter data differs from the frozen V2.1 snapshot; refusing to serve predictions.")
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest().upper()
    if digest != EXPECTED_MODEL_SHA256:
        raise RuntimeError("UFC Predictor Model V2.1 failed its checksum verification.")
    artifact = joblib.load(MODEL_PATH)
    if artifact.get("model_version") != MODEL_VERSION:
        raise RuntimeError("Unexpected V2.1 ensemble artifact.")
    # Small inference batches do not need training-time parallelism.
    artifact["components"]["xgboost"].named_steps["model"].set_params(n_jobs=1)
    return artifact


ARTIFACT = _load_frozen_model()
PROFILE_ARTIFACT = joblib.load(PROFILE_MODEL_PATH)
EVALUATION = json.loads(MODEL_PATH.with_name("HOLDOUT_EVALUATION.json").read_text())
FEATURE_NAMES = ARTIFACT["feature_names"]
PROFILES = PROFILE_ARTIFACT["profiles"]
STATES = {
    name: FighterState.from_dict(values)
    for name, values in PROFILE_ARTIFACT["fighter_states"].items()
}
for _fighter_key in PROFILES:
    STATES.setdefault(_fighter_key, FighterState())


def _load_method_model() -> dict[str, Any]:
    digest = hashlib.sha256(METHOD_MODEL_PATH.read_bytes()).hexdigest().upper()
    if digest != EXPECTED_METHOD_MODEL_SHA256:
        raise RuntimeError("Method V2.0 failed its checksum verification.")
    artifact = joblib.load(METHOD_MODEL_PATH)
    if artifact.get("model_version") != METHOD_MODEL_VERSION:
        raise RuntimeError("Unexpected method model artifact.")
    return artifact


METHOD_ARTIFACT = _load_method_model()
METHOD_PIPELINE = METHOD_ARTIFACT["pipeline"]
METHOD_FEATURE_NAMES = METHOD_ARTIFACT["feature_names"]
METHOD_CLASS_NAMES = METHOD_ARTIFACT["class_names"]
METHOD_PROFILES = METHOD_ARTIFACT["profiles"]
METHOD_STATES = {
    name: FighterState.from_dict(values)
    for name, values in METHOD_ARTIFACT["fighter_states"].items()
}
for _fighter_key in METHOD_PROFILES:
    METHOD_STATES.setdefault(_fighter_key, FighterState())


def _display_name(key: str) -> str:
    name = str(PROFILES.get(key, {}).get("name") or key.title()).strip()
    return name.title() if name.isupper() else name


def _completed_fights(key: str) -> int:
    state = STATES[key]
    return state.wins + state.losses


def _prediction_mode(key: str) -> str:
    fights = _completed_fights(key)
    return "DEBUTANT" if fights == 0 else "LOW_SAMPLE" if fights <= 2 else "NORMAL"


def _low_data_warning(key_a: str, key_b: str) -> str | None:
    messages = []
    for key in (key_a, key_b):
        fights = _completed_fights(key)
        if fights == 0:
            messages.append(f"{_display_name(key)} is making their UFC debut")
        elif fights <= 2:
            messages.append(
                f"{_display_name(key)} has only {fights} previous UFC fight{'s' if fights != 1 else ''}"
            )
    return "Limited UFC data: " + "; ".join(messages) + "." if messages else None


# Expose every unique fighter found in the frozen historical state. History
# sufficiency is checked only when a prediction is requested.
FIGHTERS = sorted({_display_name(key) for key in STATES}, key=str.casefold)
FIGHTER_MATCHER = FighterNameMatcher(
    (key, _display_name(key)) for key in STATES
)
EVENT_SOURCE = create_event_source(PROJECT_ROOT)
PREDICTION_HISTORY = PredictionHistory(PROJECT_ROOT / "data" / "prediction_history.json")


class MatchupRequest(BaseModel):
    fighter_a: str = Field(min_length=1, max_length=100)
    fighter_b: str = Field(min_length=1, max_length=100)


def _resolve_fighter(value: str) -> str:
    key = normalize_name(value)
    if key not in STATES:
        raise HTTPException(status_code=400, detail=f"Fighter not found: {value}")
    return key


def _model_analysis(
    first_key: str, second_key: str, as_of: date,
    bout_type: str | None = None, scheduled_rounds: int = 3,
) -> tuple[float, dict[str, float], dict[str, Any], dict[str, Any]]:
    first = fighter_snapshot(first_key, as_of, STATES, PROFILES)
    second = fighter_snapshot(second_key, as_of, STATES, PROFILES)
    inferred = bout_type or _inferred_fighter_division(first_key) or _inferred_fighter_division(second_key) or "Other"
    selected = feature_frame(PROJECT_ROOT, ARTIFACT, first_key, second_key, as_of, inferred, scheduled_rounds)
    probability, contributions = analyze(ARTIFACT, selected)
    return probability, contributions, first, second


def _symmetric_model_analysis(
    first_key: str, second_key: str, as_of: date,
    bout_type: str | None = None, scheduled_rounds: int = 3,
) -> tuple[float, dict[str, float], dict[str, Any], dict[str, Any]]:
    """Score both orientations so fighter ordering cannot change the answer."""
    bout_type = bout_type or _inferred_bout_type(first_key, second_key)
    forward, forward_parts, first, second = _model_analysis(
        first_key, second_key, as_of, bout_type, scheduled_rounds
    )
    reverse, reverse_parts, _, _ = _model_analysis(
        second_key, first_key, as_of, bout_type, scheduled_rounds
    )
    probability = 0.5 * (forward + (1.0 - reverse))
    contributions = {
        name: 0.5 * (forward_parts.get(name, 0.0) - reverse_parts.get(name, 0.0))
        for name in set(forward_parts) | set(reverse_parts)
    }
    return probability, contributions, first, second


def _extreme_size_adjustment(
    probability: float, first_key: str, second_key: str
) -> tuple[float, float]:
    """Apply a symmetric size prior only for 3+ division fantasy matchups.

    UFC history contains too few extreme cross-division bouts to estimate this
    relationship. The log-odds increment is isolated from ordinary same- and
    adjacent-division predictions and returned for API auditing.
    """
    first_division = _inferred_fighter_division(first_key)
    second_division = _inferred_fighter_division(second_key)
    if first_division is None or second_division is None:
        return probability, 0.0
    first_rank = division_rank(first_division)
    second_rank = division_rank(second_division)
    if first_rank is None or second_rank is None:
        return probability, 0.0
    rank_gap = first_rank - second_rank
    if abs(rank_gap) < 3:
        return probability, 0.0
    increment = math.copysign(0.45 * (abs(rank_gap) - 2.0), rank_gap)
    clipped = min(max(probability, 1e-6), 1.0 - 1e-6)
    adjusted = 1.0 / (
        1.0 + math.exp(-(math.log(clipped / (1.0 - clipped)) + increment))
    )
    return adjusted, increment


def _explanation_factors(
    contributions: dict[str, float], display_a: str, display_b: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def category(feature: str) -> str:
        if feature.startswith("age_"):
            return "Age"
        if feature.startswith(("height_", "reach_")):
            return "Physical attributes"
        if "recent_" in feature or "streak" in feature:
            return "Recent form"
        if feature.startswith(("ufc_fights", "ufc_wins", "ufc_losses", "ufc_win_pct")):
            return "Experience"
        if any(term in feature for term in ("elo", "opponent", "quality_adjusted")):
            return "Strength of competition"
        if "submission" in feature:
            return "Submission threat"
        if "takedown_defense" in feature:
            return "Takedown defense"
        if "td_accuracy" in feature or "takedown_accuracy" in feature:
            return "Wrestling"
        if any(term in feature for term in ("striking_defense", "absorbed")):
            return "Striking defense"
        if any(term in feature for term in ("sig_", "strike_", "striking", "knockdown")):
            return "Striking"
        if any(term in feature for term in ("finish_rate", "ko_tko", "decision_win")):
            return "Finishing profile"
        if "days_since" in feature:
            return "Fight activity"
        return "Experience"

    grouped: dict[str, float] = {}
    grouped_features: dict[str, list[str]] = {}
    for feature, contribution in contributions.items():
        if not math.isfinite(contribution):
            continue
        name = category(feature)
        grouped[name] = grouped.get(name, 0.0) + contribution
        grouped_features.setdefault(name, []).append(feature)

    def strongest(sign: int, limit: int, fighter: str) -> list[dict[str, Any]]:
        eligible = [(name, value) for name, value in grouped.items() if value * sign > 0]
        eligible.sort(key=lambda item: abs(item[1]), reverse=True)
        return [
            {
                "feature": grouped_features[name][0],
                "features": grouped_features[name],
                "category": name,
                "label": name,
                "advantage": (
                    "Strong" if abs(value) >= 0.50
                    else "Moderate" if abs(value) >= 0.20
                    else "Small"
                ),
                "explanation": CATEGORY_EXPLANATIONS[name].format(fighter=fighter),
                "log_odds_contribution": round(abs(value), 4),
                "supports": fighter,
            }
            for name, value in eligible[:limit]
        ]

    return strongest(1, 5, display_a), strongest(-1, 3, display_b)


def _inferred_bout_type(first_key: str, second_key: str) -> str:
    weights = [
        PROFILES.get(key, {}).get("weight") for key in (first_key, second_key)
        if PROFILES.get(key, {}).get("weight") is not None
    ]
    weight = max(weights) if weights else None
    if weight is None:
        return ""
    divisions = [
        (125, "Flyweight"), (135, "Bantamweight"), (145, "Featherweight"),
        (155, "Lightweight"), (170, "Welterweight"), (185, "Middleweight"),
        (205, "Light Heavyweight"), (999, "Heavyweight"),
    ]
    return next(name for ceiling, name in divisions if weight <= ceiling)


def _inferred_fighter_division(key: str) -> str | None:
    weight = PROFILES.get(key, {}).get("weight")
    if weight is None:
        return None
    divisions = [
        (125, "flyweight"), (135, "bantamweight"), (145, "featherweight"),
        (155, "lightweight"), (170, "welterweight"), (185, "middleweight"),
        (205, "light_heavyweight"), (999, "heavyweight"),
    ]
    return next(name for ceiling, name in divisions if float(weight) <= ceiling)


def _method_prediction(
    winner_key: str,
    opponent_key: str,
    winner_snapshot: dict[str, Any],
    opponent_snapshot: dict[str, Any],
    bout_type: str | None = None,
    scheduled_rounds: int = 3,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    bout_type = bout_type or _inferred_bout_type(winner_key, opponent_key)
    winner_snapshot = dict(winner_snapshot)
    opponent_snapshot = dict(opponent_snapshot)
    winner_snapshot["historical_division"] = (
        winner_snapshot.get("historical_division")
        or _inferred_fighter_division(winner_key)
    )
    opponent_snapshot["historical_division"] = (
        opponent_snapshot.get("historical_division")
        or _inferred_fighter_division(opponent_key)
    )
    values = method_features(
        winner_snapshot,
        opponent_snapshot,
        bout_type,
        f"{scheduled_rounds} Rnd " + "(" + "-".join(["5"] * scheduled_rounds) + ")",
    )
    frame = pd.DataFrame([values])[METHOD_FEATURE_NAMES]
    raw = METHOD_PIPELINE.predict_proba(frame)[0]
    class_indices = list(METHOD_PIPELINE.classes_)
    probabilities = [float(raw[class_indices.index(index)]) for index in range(3)]
    ranked = sorted(
        (
            {"method": name, "probability": round(probability * 100.0, 1)}
            for name, probability in zip(METHOD_CLASS_NAMES, probabilities)
        ),
        key=lambda row: row["probability"],
        reverse=True,
    )
    top = ranked[0]["probability"]
    confidence = "Low" if top < 50.0 else "Moderate" if top < 65.0 else "High"
    distance = values.get("division_distance")
    numeric_distance = None if distance is None or pd.isna(distance) else int(distance)
    mismatch = (
        "UNKNOWN" if numeric_distance is None else
        "NORMAL" if numeric_distance == 0 else
        "MODERATE" if numeric_distance == 1 else
        "LARGE" if numeric_distance == 2 else "EXTREME"
    )
    reliability = (
        "HIGH" if numeric_distance == 0 else
        "MEDIUM" if numeric_distance == 1 else "LOW"
    )
    context = {
        "winner_division": next(
            name.removeprefix("winner_division_")
            for name, value in values.items()
            if name.startswith("winner_division_") and value == 1.0
        ),
        "opponent_division": next(
            name.removeprefix("opponent_division_")
            for name, value in values.items()
            if name.startswith("opponent_division_") and value == 1.0
        ),
        "division_distance": numeric_distance,
        "mismatch_category": mismatch,
        "matchup_confidence": reliability,
        "estimated_weight_gap_lbs": (
            None if values.get("estimated_weight_gap") is None
            else round(float(values["estimated_weight_gap"]), 1)
        ),
        "relative_weight_gap": (
            None if values.get("relative_weight_gap") is None
            else round(float(values["relative_weight_gap"]), 3)
        ),
        "warning": (
            "Low-confidence fantasy matchup — fighters normally compete several weight classes apart."
            if reliability == "LOW" else None
        ),
    }
    return ranked, confidence, context


def predict_matchup(
    fighter_a: str,
    fighter_b: str,
    *,
    bout_type: str | None = None,
    scheduled_rounds: int = 3,
    as_of: date | None = None,
) -> dict[str, Any]:
    key_a = _resolve_fighter(fighter_a)
    key_b = _resolve_fighter(fighter_b)
    if key_a == key_b:
        raise HTTPException(status_code=400, detail="Choose two different fighters.")

    prediction_date = as_of or date.today()
    probability_a, contributions, snapshot_a, snapshot_b = _symmetric_model_analysis(
        key_a, key_b, prediction_date, bout_type, scheduled_rounds
    )
    probability_a, size_log_odds_adjustment = _extreme_size_adjustment(
        probability_a, key_a, key_b
    )
    probability_b = 1.0 - probability_a
    display_a, display_b = _display_name(key_a), _display_name(key_b)
    factors_a, factors_b = _explanation_factors(contributions, display_a, display_b)
    method_snapshot_a = fighter_snapshot(
        key_a, prediction_date, METHOD_STATES, METHOD_PROFILES
    )
    method_snapshot_b = fighter_snapshot(
        key_b, prediction_date, METHOD_STATES, METHOD_PROFILES
    )
    methods_a, confidence_a, matchup_a = _method_prediction(
        key_a, key_b, method_snapshot_a, method_snapshot_b,
        bout_type, scheduled_rounds,
    )
    methods_b, confidence_b, matchup_b = _method_prediction(
        key_b, key_a, method_snapshot_b, method_snapshot_a,
        bout_type, scheduled_rounds,
    )
    if probability_a >= probability_b:
        winner_key, opponent_key = key_a, key_b
        winner_factors = factors_a
        predicted_winner = display_a
        method_probabilities, method_confidence = methods_a, confidence_a
    else:
        winner_key, opponent_key = key_b, key_a
        winner_factors = factors_b
        predicted_winner = display_b
        method_probabilities, method_confidence = methods_b, confidence_b
    matchup = matchup_a
    method_maps = {
        display_a: {row["method"]: row["probability"] / 100.0 for row in methods_a},
        display_b: {row["method"]: row["probability"] / 100.0 for row in methods_b},
    }
    outcome_probabilities = sorted(
        [
            {
                "fighter": fighter,
                "method": method,
                "probability": round(win_probability * method_probability * 100.0, 2),
            }
            for fighter, win_probability in (
                (display_a, probability_a), (display_b, probability_b)
            )
            for method, method_probability in method_maps[fighter].items()
        ],
        key=lambda row: row["probability"],
        reverse=True,
    )
    winner_probability = max(probability_a, probability_b) * 100.0
    mode_a, mode_b = _prediction_mode(key_a), _prediction_mode(key_b)
    low_data_warning = _low_data_warning(key_a, key_b)
    return {
        "fighter_a": display_a,
        "fighter_b": display_b,
        "fighter_a_probability": round(probability_a * 100.0, 1),
        "fighter_b_probability": round(probability_b * 100.0, 1),
        "predicted_winner": predicted_winner,
        "winner_confidence": (
            "High" if winner_probability >= 70.0
            else "Moderate" if winner_probability >= 57.0
            else "Close fight"
        ),
        "method_probabilities": method_probabilities,
        "conditional_method_probabilities": {
            "fighter_a": methods_a,
            "fighter_b": methods_b,
        },
        "outcome_probabilities": outcome_probabilities,
        "outcome_probability_total": round(
            sum(row["probability"] for row in outcome_probabilities), 2
        ),
        "most_likely_result": {
            "fighter": outcome_probabilities[0]["fighter"],
            "method": outcome_probabilities[0]["method"],
            "probability": outcome_probabilities[0]["probability"],
        },
        "predicted_method": method_probabilities[0]["method"],
        "method_confidence": method_confidence,
        "method_model_version": METHOD_MODEL_VERSION,
        "method_context": (
            f"{scheduled_rounds}-round bout"
            + (f", {bout_type}" if bout_type else " (weight class inferred)")
        ),
        "weight_class_matchup": matchup,
        "mismatch_category": matchup["mismatch_category"],
        "matchup_confidence": matchup["matchup_confidence"],
        "matchup_warning": matchup["warning"],
        "extreme_size_log_odds_adjustment": round(size_log_odds_adjustment, 4),
        "fighter_a_prediction_mode": mode_a,
        "fighter_b_prediction_mode": mode_b,
        "prediction_mode": (
            "DEBUTANT" if "DEBUTANT" in {mode_a, mode_b}
            else "LOW_SAMPLE" if "LOW_SAMPLE" in {mode_a, mode_b}
            else "NORMAL"
        ),
        "fighter_a_prior_ufc_fights": _completed_fights(key_a),
        "fighter_b_prior_ufc_fights": _completed_fights(key_b),
        "limited_ufc_data": low_data_warning is not None,
        "low_data_warning": low_data_warning,
        "factors_for_fighter_a": factors_a,
        "factors_for_fighter_b": factors_b,
        "winner_explanation": winner_factors,
        "explanation_method": "Ensemble sensitivity to training-median feature replacement; effects are not additive.",
        "model_version": MODEL_VERSION,
        "historical_data_through": ARTIFACT["trained_through"],
    }


def _event_summary(event: Event) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "event_name": event.event_name,
        "event_date": event.event_date,
        "location": event.location,
        "fight_count": len(event.fights),
        "headliners": ([
            {"name": event.fights[0].fighter_a, "espn_id": event.fights[0].fighter_a_espn_id},
            {"name": event.fights[0].fighter_b, "espn_id": event.fights[0].fighter_b_espn_id},
        ] if event.fights else []),
        "starts_at": event.starts_at,
        "source_url": event.source_url,
        "historical_data_through": ARTIFACT["trained_through"],
    }


def _unavailable_fight(fight: EventFight, reason: str) -> dict[str, Any]:
    return {
        **fight.to_dict(),
        "prediction_status": "unavailable",
        "prediction_unavailable_reason": reason,
        "prediction": None,
    }


def _event_fight(event: Event, fight: EventFight) -> dict[str, Any]:
    match_a = FIGHTER_MATCHER.match(fight.fighter_a)
    match_b = FIGHTER_MATCHER.match(fight.fighter_b)
    if match_a.status != "matched" or match_b.status != "matched":
        return _unavailable_fight(
            fight, "Insufficient UFC historical data for one or both fighters."
        )
    if match_a.matched_key == match_b.matched_key:
        return _unavailable_fight(fight, "A fighter cannot be matched against themself.")
    if (
        _completed_fights(match_a.matched_key) < MIN_COMPLETED_FIGHTS
        or _completed_fights(match_b.matched_key) < MIN_COMPLETED_FIGHTS
    ):
        return _unavailable_fight(
            fight, "Insufficient UFC historical data for one or both fighters."
        )

    existing = PREDICTION_HISTORY.get_prediction(
        event.event_id, fight.fight_id, MODEL_VERSION
    )
    if existing:
        saved = existing["prediction"]
        if saved["fighter_a"] != match_a.matched_name or saved["fighter_b"] != match_b.matched_name:
            return _unavailable_fight(fight, "The card changed; the old prediction is retained only in history.")
        return {
            **fight.to_dict(),
            "prediction_status": "locked",
            "prediction_locked_at": existing["prediction_timestamp"],
            "prediction": existing["prediction"],
        }
    if event.event_date <= date.today().isoformat():
        return _unavailable_fight(
            fight, "No prediction was published before the event date."
        )

    cutoffs = [ARTIFACT["trained_through"], PROFILE_ARTIFACT["last_data_date"],
               METHOD_ARTIFACT.get("last_data_date", METHOD_ARTIFACT.get("trained_through", ""))]
    if event.event_date <= max(cutoffs):
        return _unavailable_fight(fight, "The event must be after all model and fighter-data cutoffs.")
    if hasattr(EVENT_SOURCE, "can_publish") and not EVENT_SOURCE.can_publish():
        return _unavailable_fight(fight, "Waiting for a fresh schedule before publishing a prediction.")

    prediction = predict_matchup(
        match_a.matched_name,
        match_b.matched_name,
        bout_type=fight.weight_class,
        scheduled_rounds=fight.number_of_rounds,
        as_of=date.fromisoformat(event.event_date),
    )
    prediction["schedule_source"] = EVENT_SOURCE.source_name
    prediction["schedule_fetched_at"] = getattr(EVENT_SOURCE, "fetched_at", None)
    prediction["model_sha256"] = EXPECTED_MODEL_SHA256
    prediction["method_model_sha256"] = EXPECTED_METHOD_MODEL_SHA256
    record, _ = PREDICTION_HISTORY.lock_prediction(
        _event_summary(event), fight.to_dict(), prediction
    )
    return {
        **fight.to_dict(),
        "prediction_status": "locked",
        "prediction_locked_at": record["prediction_timestamp"],
        "prediction": record["prediction"],
    }


def _event_detail(event: Event) -> dict[str, Any]:
    return {**_event_summary(event), "fights": [_event_fight(event, fight) for fight in event.fights]}


app = FastAPI(title="UFC Fight Predictor", version=MODEL_VERSION)


@app.get("/api/fighter-portrait", include_in_schema=False)
def fighter_portrait(name: str = Query(min_length=1, max_length=100),
                     espn_id: str | None = Query(default=None, pattern=r"^\d{1,12}$")):
    path = portrait(name, espn_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Portrait unavailable")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/", include_in_schema=False)
@app.get("/events", include_in_schema=False)
@app.get("/predict", include_in_schema=False)
@app.get("/results", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(INDEX_PATH)


@app.get("/event/{event_id}", include_in_schema=False)
def event_page(event_id: str) -> FileResponse:
    return FileResponse(INDEX_PATH)


@app.get("/api/fighters")
def list_fighters() -> dict[str, Any]:
    return {
        "fighters": FIGHTERS,
        "count": len(FIGHTERS),
        "minimum_historical_fights": MIN_COMPLETED_FIGHTS,
        "supports_ufc_debutants": True,
        "model_version": MODEL_VERSION,
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "ok", "model_version": MODEL_VERSION,
        "method_model_version": METHOD_MODEL_VERSION,
    }


@app.get("/api/events")
def upcoming_events() -> dict[str, Any]:
    events = EVENT_SOURCE.upcoming_events()
    summaries = [_event_summary(event) for event in events]
    return {
        "source": EVENT_SOURCE.source_name,
        "schedule_fetched_at": getattr(EVENT_SOURCE, "fetched_at", None),
        "schedule_warning": getattr(EVENT_SOURCE, "last_error", None),
        "events": summaries,
        "count": len(summaries),
        "next_event": summaries[0] if summaries else None,
    }


@app.get("/api/events/{event_id}")
def event_detail(event_id: str) -> dict[str, Any]:
    event = next(
        (item for item in EVENT_SOURCE.upcoming_events() if item.event_id == event_id),
        None,
    )
    if event is None:
        raise HTTPException(status_code=404, detail="Upcoming event not found.")
    return {"source": EVENT_SOURCE.source_name, "event": _event_detail(event)}


@app.get("/api/results")
def prediction_results() -> dict[str, Any]:
    predictions = PREDICTION_HISTORY.list_predictions()
    completed = [item for item in predictions if item.get("actual_winner") is not None]
    winner_scored = [item for item in completed if item.get("correct_winner") is not None]
    method_scored = [item for item in completed if item.get("correct_method") is not None]
    return {
        "model_version": MODEL_VERSION,
        "historical_accuracy": EVALUATION["metrics"]["accuracy"],
        "historical_test_fights": EVALUATION["holdout_fights"],
        "historical_data_through": ARTIFACT["trained_through"],
        "tracked_predictions": len(predictions),
        "completed_predictions": len(completed),
        "winner_accuracy": (
            sum(bool(item["correct_winner"]) for item in winner_scored) / len(winner_scored)
            if winner_scored else None
        ),
        "method_accuracy": (
            sum(bool(item["correct_method"]) for item in method_scored) / len(method_scored)
            if method_scored else None
        ),
        "predictions": sorted(
            predictions,
            key=lambda item: (item["event_date"], item["prediction_timestamp"]),
            reverse=True,
        ),
    }


@app.post("/api/predict")
def predict(request: MatchupRequest) -> dict[str, Any]:
    return predict_matchup(request.fighter_a, request.fighter_b)


def main() -> None:
    uvicorn.run("ufc_predictor.web:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
