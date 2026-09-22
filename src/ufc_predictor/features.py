"""Turn chronological raw bouts into pre-fight model features.

The ordering in ``build_dataset`` is the leakage safeguard: snapshot first,
then update each fighter with the current bout's outcome and statistics.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import date
import hashlib
import re
from typing import Any

import pandas as pd


BASE_FEATURES = [
    "age_diff",
    "height_diff",
    "reach_diff",
    "ufc_wins_diff",
    "ufc_losses_diff",
    "ufc_win_pct_diff",
    "recent_3_win_pct_diff",
    "recent_5_win_pct_diff",
    "current_win_streak_diff",
    "current_loss_streak_diff",
    "sig_str_accuracy_diff",
    "sig_str_landed_pm_diff",
    "sig_str_absorbed_pm_diff",
    "striking_differential_pm_diff",
    "striking_defense_diff",
    "knockdowns_per_fight_diff",
    "head_strike_share_diff",
    "body_strike_share_diff",
    "leg_strike_share_diff",
    "distance_strike_share_diff",
    "clinch_strike_share_diff",
    "ground_strike_share_diff",
    "takedowns_per15_diff",
    "takedown_accuracy_diff",
    "takedown_defense_diff",
    "submission_attempts_per15_diff",
    "control_time_per15_diff",
    "finish_rate_diff",
    "ko_tko_win_pct_diff",
    "submission_win_pct_diff",
    "decision_win_pct_diff",
    "days_since_last_fight_diff",
    "ufc_fights_diff",
    "elo_diff",
    "average_opponent_elo_diff",
    "average_beaten_opponent_elo_diff",
    "quality_adjusted_win_score_diff",
    "ufc_debutant_diff",
    "ko_tko_wins_diff",
    "submission_wins_diff",
    "decision_wins_diff",
]

STANCE_FEATURES = [
    "fighter_a_orthodox", "fighter_a_southpaw", "fighter_a_switch",
    "fighter_b_orthodox", "fighter_b_southpaw", "fighter_b_switch",
    "same_stance",
]

WEIGHT_CLASSES = [
    "womens_strawweight", "womens_flyweight", "womens_bantamweight",
    "womens_featherweight", "flyweight", "bantamweight", "featherweight",
    "lightweight", "welterweight", "middleweight", "light_heavyweight",
    "heavyweight", "catch_weight", "open_weight", "other",
]
WEIGHT_CLASS_FEATURES = [f"weight_class_{name}" for name in WEIGHT_CLASSES]

MATCHUP_FEATURES = [
    "fighter_a_td_accuracy_vs_b_defense",
    "fighter_b_td_accuracy_vs_a_defense",
    "fighter_a_striking_offense_vs_b_defense",
    "fighter_b_striking_offense_vs_a_defense",
]

LOW_DATA_FEATURES = [
    "fighter_a_ufc_debutant", "fighter_b_ufc_debutant",
    "fighter_a_ufc_fights", "fighter_b_ufc_fights",
]

FEATURE_NAMES = (
    BASE_FEATURES + STANCE_FEATURES + WEIGHT_CLASS_FEATURES
    + MATCHUP_FEATURES + LOW_DATA_FEATURES
)

METHOD_ABSOLUTE_FEATURES = [
    "age", "ufc_win_pct", "recent_3_win_pct", "recent_5_win_pct",
    "current_win_streak", "current_loss_streak", "sig_str_accuracy",
    "sig_str_landed_pm", "sig_str_absorbed_pm", "striking_differential_pm",
    "striking_defense", "knockdowns_per_fight", "takedowns_per15",
    "takedown_accuracy", "takedown_defense", "submission_attempts_per15",
    "control_time_per15", "finish_rate", "ko_tko_win_pct",
    "submission_win_pct", "decision_win_pct", "ko_tko_loss_pct",
    "submission_loss_pct", "decision_loss_pct", "recent_finish_rate",
    "recent_ko_tko_rate", "recent_submission_rate", "recent_decision_rate",
    "average_fight_duration_minutes",
    "ufc_fights", "elo", "average_opponent_elo", "quality_adjusted_win_score",
]
METHOD_MATCHUP_FEATURES = [
    "winner_td_accuracy_vs_opponent_defense",
    "opponent_td_accuracy_vs_winner_defense",
    "winner_striking_offense_vs_opponent_defense",
    "opponent_striking_offense_vs_winner_defense",
]
METHOD_STYLE_FEATURES = (
    [f"winner_{name}" for name in METHOD_ABSOLUTE_FEATURES]
    + [f"opponent_{name}" for name in METHOD_ABSOLUTE_FEATURES]
    + METHOD_MATCHUP_FEATURES
    + ["scheduled_rounds", "five_round_fight"]
)
METHOD_WEIGHT_FEATURES = (
    [f"winner_division_{name}" for name in WEIGHT_CLASSES]
    + [f"opponent_division_{name}" for name in WEIGHT_CLASSES]
    + [
        "winner_division_rank", "opponent_division_rank", "division_distance",
        "same_division", "one_division_apart", "two_plus_divisions_apart",
        "mismatch_moderate", "mismatch_large", "mismatch_extreme",
        "winner_reference_weight", "opponent_reference_weight",
        "estimated_weight_gap", "relative_weight_gap", "winner_size_advantage",
    ]
)
METHOD_PHYSICAL_INTERACTION_FEATURES = [
    "height_diff", "reach_diff", "age_diff", "bmi_proxy_diff", "same_stance",
    "winner_size_x_ko_rate", "winner_size_x_submission_rate",
    "winner_size_x_takedowns_per15", "winner_size_x_opponent_ko_loss_rate",
    "winner_size_x_opponent_submission_loss_rate",
    "reach_advantage_x_striking_accuracy", "division_distance_x_finish_rate",
    "division_distance_x_decision_rate",
]
METHOD_FEATURE_NAMES = (
    METHOD_STYLE_FEATURES + METHOD_WEIGHT_FEATURES + METHOD_PHYSICAL_INTERACTION_FEATURES
)

# Exact richer-evaluation predecessor, retained for an apples-to-apples benchmark.
CURRENT_MODEL_FEATURES = [
    "age_diff", "height_diff", "reach_diff", "ufc_wins_diff",
    "ufc_losses_diff", "ufc_win_pct_diff", "recent_3_win_pct_diff",
    "recent_5_win_pct_diff", "sig_str_accuracy_diff", "sig_str_landed_pm_diff",
    "sig_str_absorbed_pm_diff", "striking_differential_pm_diff",
    "takedown_accuracy_diff", "takedown_defense_diff",
    "submission_attempts_per15_diff", "finish_rate_diff",
    "days_since_last_fight_diff", "ufc_fights_diff", "elo_diff",
]


def normalize_name(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def safe_ratio(
    numerator: float, denominator: float, default: float | None = 0.0
) -> float | None:
    return numerator / denominator if denominator else default


def nullable_difference(first: Any, second: Any) -> float | None:
    return None if first is None or second is None else float(first) - float(second)


def striking_matchup(offense: Any, defense: Any) -> float | None:
    return None if offense is None or defense is None else float(offense) * (1.0 - float(defense))


def parse_of(value: Any) -> tuple[float, float]:
    """Parse UFCStats values such as '31 of 65'."""
    match = re.match(r"\s*(\d+(?:\.\d+)?)\s+of\s+(\d+(?:\.\d+)?)", str(value))
    return (float(match.group(1)), float(match.group(2))) if match else (0.0, 0.0)


def parse_clock(value: Any) -> int:
    try:
        minutes, seconds = str(value).split(":")
        return int(minutes) * 60 + int(seconds)
    except (ValueError, TypeError):
        return 0


def numeric_or_zero(value: Any) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    return 0.0 if pd.isna(parsed) else float(parsed)


def fight_duration_seconds(row: pd.Series) -> int:
    """Elapsed bout time, assuming standard five-minute UFC rounds."""
    try:
        final_round = int(row["round"])
    except (ValueError, TypeError):
        return 0
    return max(0, (final_round - 1) * 300 + parse_clock(row["time"]))


def parse_height_inches(value: Any) -> float | None:
    match = re.match(r"\s*(\d+)\s*'\s*(\d+)", str(value))
    return float(int(match.group(1)) * 12 + int(match.group(2))) if match else None


def parse_reach_inches(value: Any) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else None


def parse_weight_lbs(value: Any) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else None


@dataclass
class FighterState:
    wins: int = 0
    losses: int = 0
    sig_landed: float = 0.0
    sig_attempted: float = 0.0
    sig_absorbed: float = 0.0
    sig_faced_attempted: float = 0.0
    seconds: float = 0.0
    td_landed: float = 0.0
    td_attempted: float = 0.0
    opponent_td_landed: float = 0.0
    opponent_td_attempted: float = 0.0
    sub_attempts: float = 0.0
    knockdowns: float = 0.0
    head_landed: float = 0.0
    body_landed: float = 0.0
    leg_landed: float = 0.0
    distance_landed: float = 0.0
    clinch_landed: float = 0.0
    ground_landed: float = 0.0
    control_seconds: float = 0.0
    finish_wins: int = 0
    ko_tko_wins: int = 0
    submission_wins: int = 0
    decision_wins: int = 0
    ko_tko_losses: int = 0
    submission_losses: int = 0
    decision_losses: int = 0
    current_win_streak: int = 0
    current_loss_streak: int = 0
    elo: float = 1500.0
    opponent_elo_sum: float = 0.0
    opponents_faced: int = 0
    beaten_opponent_elo_sum: float = 0.0
    quality_win_points: float = 0.0
    last_fight_date: str | None = None
    recent_results: deque[int] = field(default_factory=lambda: deque(maxlen=5))
    recent_methods: deque[str] = field(default_factory=lambda: deque(maxlen=5))
    last_division: str | None = None

    def performance_snapshot(self) -> dict[str, float | None]:
        fights = self.wins + self.losses
        minutes = self.seconds / 60.0
        has_fights = fights > 0
        return {
            "ufc_wins": float(self.wins),
            "ufc_losses": float(self.losses),
            "ufc_win_pct": safe_ratio(self.wins, fights) if has_fights else None,
            "recent_3_win_pct": safe_ratio(
                sum(list(self.recent_results)[-3:]),
                len(list(self.recent_results)[-3:]),
                None,
            ),
            "recent_5_win_pct": safe_ratio(
                sum(self.recent_results), len(self.recent_results), None
            ),
            "current_win_streak": float(self.current_win_streak),
            "current_loss_streak": float(self.current_loss_streak),
            "sig_str_accuracy": safe_ratio(self.sig_landed, self.sig_attempted, None),
            "sig_str_landed_pm": safe_ratio(self.sig_landed, minutes, None),
            "sig_str_absorbed_pm": safe_ratio(self.sig_absorbed, minutes, None),
            "striking_differential_pm": safe_ratio(
                self.sig_landed - self.sig_absorbed, minutes, None
            ),
            "striking_defense": safe_ratio(
                self.sig_faced_attempted - self.sig_absorbed,
                self.sig_faced_attempted,
                None,
            ),
            "knockdowns_per_fight": safe_ratio(self.knockdowns, fights, None),
            "head_strike_share": safe_ratio(self.head_landed, self.sig_landed, None),
            "body_strike_share": safe_ratio(self.body_landed, self.sig_landed, None),
            "leg_strike_share": safe_ratio(self.leg_landed, self.sig_landed, None),
            "distance_strike_share": safe_ratio(self.distance_landed, self.sig_landed, None),
            "clinch_strike_share": safe_ratio(self.clinch_landed, self.sig_landed, None),
            "ground_strike_share": safe_ratio(self.ground_landed, self.sig_landed, None),
            "takedowns_per15": safe_ratio(self.td_landed * 15.0, minutes, None),
            "takedown_accuracy": safe_ratio(self.td_landed, self.td_attempted, None),
            "takedown_defense": safe_ratio(
                self.opponent_td_attempted - self.opponent_td_landed,
                self.opponent_td_attempted,
                None,
            ),
            "submission_attempts_per15": safe_ratio(self.sub_attempts * 15.0, minutes, None),
            "control_time_per15": safe_ratio(self.control_seconds * 15.0, self.seconds, None),
            "finish_rate": safe_ratio(self.finish_wins, self.wins, None),
            "ko_tko_win_pct": safe_ratio(self.ko_tko_wins, self.wins, None),
            "submission_win_pct": safe_ratio(self.submission_wins, self.wins, None),
            "decision_win_pct": safe_ratio(self.decision_wins, self.wins, None),
            "ko_tko_loss_pct": safe_ratio(self.ko_tko_losses, self.losses, None),
            "submission_loss_pct": safe_ratio(self.submission_losses, self.losses, None),
            "decision_loss_pct": safe_ratio(self.decision_losses, self.losses, None),
            "recent_finish_rate": safe_ratio(
                sum(value != "Decision" for value in self.recent_methods),
                len(self.recent_methods),
                None,
            ),
            "recent_ko_tko_rate": safe_ratio(
                sum(value == "KO/TKO" for value in self.recent_methods),
                len(self.recent_methods),
                None,
            ),
            "recent_submission_rate": safe_ratio(
                sum(value == "Submission" for value in self.recent_methods),
                len(self.recent_methods),
                None,
            ),
            "recent_decision_rate": safe_ratio(
                sum(value == "Decision" for value in self.recent_methods),
                len(self.recent_methods),
                None,
            ),
            "average_fight_duration_minutes": safe_ratio(self.seconds, fights * 60.0, None),
            "ufc_fights": float(fights),
            "ufc_debutant": float(fights == 0),
            "ko_tko_wins": float(self.ko_tko_wins),
            "submission_wins": float(self.submission_wins),
            "decision_wins": float(self.decision_wins),
            "elo": self.elo,
            "average_opponent_elo": safe_ratio(
                self.opponent_elo_sum, self.opponents_faced, None
            ),
            "average_beaten_opponent_elo": safe_ratio(
                self.beaten_opponent_elo_sum, self.wins, None
            ),
            "quality_adjusted_win_score": safe_ratio(
                self.quality_win_points, fights, None
            ),
        }

    def serializable(self) -> dict[str, Any]:
        result = asdict(self)
        result["recent_results"] = list(self.recent_results)
        result["recent_methods"] = list(self.recent_methods)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FighterState":
        values = dict(values)
        values["recent_results"] = deque(values.get("recent_results", []), maxlen=5)
        values["recent_methods"] = deque(values.get("recent_methods", []), maxlen=5)
        return cls(**values)


def load_profiles(path: str) -> dict[str, dict[str, Any]]:
    raw = pd.read_csv(path)
    profiles: dict[str, dict[str, Any]] = {}
    for row in raw.to_dict(orient="records"):
        key = normalize_name(row["fighter_name"])
        dob = pd.to_datetime(row.get("DOB"), errors="coerce")
        candidate = {
            "name": str(row["fighter_name"]).strip(),
            "height": parse_height_inches(row.get("Height")),
            "reach": parse_reach_inches(row.get("Reach")),
            "weight": parse_weight_lbs(row.get("Weight")),
            "stance": (
                None if pd.isna(row.get("Stance")) else str(row.get("Stance")).strip().casefold()
            ),
            "dob": None if pd.isna(dob) else dob.date().isoformat(),
        }
        # Prefer the duplicate with more complete physical data.
        if key not in profiles or sum(v is not None for v in candidate.values()) > sum(
            v is not None for v in profiles[key].values()
        ):
            profiles[key] = candidate
    return profiles


def age_on(dob_iso: str | None, fight_date: date) -> float | None:
    if not dob_iso:
        return None
    born = date.fromisoformat(dob_iso)
    return (fight_date - born).days / 365.2425


def fighter_snapshot(
    fighter_name: str,
    fight_date: date,
    states: dict[str, FighterState],
    profiles: dict[str, dict[str, Any]],
) -> dict[str, float | None]:
    key = normalize_name(fighter_name)
    profile = profiles.get(key, {})
    return {
        "age": age_on(profile.get("dob"), fight_date),
        "height": profile.get("height"),
        "reach": profile.get("reach"),
        "stance": profile.get("stance"),
        "days_since_last_fight": (
            None
            if states[key].last_fight_date is None
            else float((fight_date - date.fromisoformat(states[key].last_fight_date)).days)
        ),
        "historical_division": states[key].last_division,
        **states[key].performance_snapshot(),
    }


def weight_class_features(bout_type: str | None) -> dict[str, float]:
    slug = division_slug(bout_type)
    return {f"weight_class_{name}": float(name == slug) for name in WEIGHT_CLASSES}


DIVISION_REFERENCE_WEIGHTS = {
    "womens_strawweight": 115.0, "womens_flyweight": 125.0,
    "womens_bantamweight": 135.0, "womens_featherweight": 145.0,
    "flyweight": 125.0, "bantamweight": 135.0, "featherweight": 145.0,
    "lightweight": 155.0, "welterweight": 170.0, "middleweight": 185.0,
    "light_heavyweight": 205.0, "heavyweight": 265.0,
}
MEN_DIVISIONS = [
    "flyweight", "bantamweight", "featherweight", "lightweight",
    "welterweight", "middleweight", "light_heavyweight", "heavyweight",
]
WOMEN_DIVISIONS = [
    "womens_strawweight", "womens_flyweight", "womens_bantamweight",
    "womens_featherweight",
]


def division_slug(bout_type: str | None) -> str:
    """Normalize UFC bout labels without using any post-fight information."""
    text = normalize_name(bout_type or "").replace("'", "")
    slug = "other"
    checks = [
        ("womens strawweight", "womens_strawweight"),
        ("womens flyweight", "womens_flyweight"),
        ("womens bantamweight", "womens_bantamweight"),
        ("womens featherweight", "womens_featherweight"),
        ("light heavyweight", "light_heavyweight"),
        ("catch weight", "catch_weight"),
        ("open weight", "open_weight"),
        ("flyweight", "flyweight"), ("bantamweight", "bantamweight"),
        ("featherweight", "featherweight"), ("lightweight", "lightweight"),
        ("welterweight", "welterweight"), ("middleweight", "middleweight"),
        ("heavyweight", "heavyweight"),
    ]
    for phrase, candidate in checks:
        if phrase in text:
            slug = candidate
            break
    return slug


def division_rank(slug: str) -> float | None:
    divisions = WOMEN_DIVISIONS if slug.startswith("womens_") else MEN_DIVISIONS
    return float(divisions.index(slug)) if slug in divisions else None


def division_distance(first: str, second: str) -> float | None:
    first_rank, second_rank = division_rank(first), division_rank(second)
    if first_rank is None or second_rank is None:
        return None
    if first.startswith("womens_") != second.startswith("womens_"):
        return None
    return abs(first_rank - second_rank)


def parse_scheduled_rounds(value: Any) -> float:
    match = re.match(r"\s*(\d+)\s+Rnd", str(value), flags=re.IGNORECASE)
    return float(match.group(1)) if match else 3.0


def method_category(value: Any) -> str | None:
    text = str(value).strip().casefold()
    if "ko" in text:
        return "KO/TKO"
    if "submission" in text or text == "sub":
        return "Submission"
    if text.startswith("decision"):
        return "Decision"
    return None


def method_features(
    winner: dict[str, Any],
    opponent: dict[str, Any],
    bout_type: str | None,
    time_format: Any,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for feature in METHOD_ABSOLUTE_FEATURES:
        result[f"winner_{feature}"] = winner.get(feature)
        result[f"opponent_{feature}"] = opponent.get(feature)
    result.update(
        {
            "winner_td_accuracy_vs_opponent_defense": (
                nullable_difference(winner["takedown_accuracy"], opponent["takedown_defense"])
            ),
            "opponent_td_accuracy_vs_winner_defense": (
                nullable_difference(opponent["takedown_accuracy"], winner["takedown_defense"])
            ),
            "winner_striking_offense_vs_opponent_defense": (
                striking_matchup(winner["sig_str_landed_pm"], opponent["striking_defense"])
            ),
            "opponent_striking_offense_vs_winner_defense": (
                striking_matchup(opponent["sig_str_landed_pm"], winner["striking_defense"])
            ),
        }
    )
    booked_division = division_slug(bout_type)
    winner_division = str(winner.get("historical_division") or booked_division)
    opponent_division = str(opponent.get("historical_division") or booked_division)
    if winner_division not in WEIGHT_CLASSES:
        winner_division = booked_division
    if opponent_division not in WEIGHT_CLASSES:
        opponent_division = booked_division
    for name in WEIGHT_CLASSES:
        result[f"winner_division_{name}"] = float(winner_division == name)
        result[f"opponent_division_{name}"] = float(opponent_division == name)
    winner_rank = division_rank(winner_division)
    opponent_rank = division_rank(opponent_division)
    distance = division_distance(winner_division, opponent_division)
    winner_weight = DIVISION_REFERENCE_WEIGHTS.get(winner_division)
    opponent_weight = DIVISION_REFERENCE_WEIGHTS.get(opponent_division)
    weight_gap = (
        None if winner_weight is None or opponent_weight is None
        else winner_weight - opponent_weight
    )
    relative_gap = (
        None if weight_gap is None
        else weight_gap / min(winner_weight, opponent_weight)
    )
    result.update(
        {
            "winner_division_rank": winner_rank,
            "opponent_division_rank": opponent_rank,
            "division_distance": distance,
            "same_division": None if distance is None else float(distance == 0),
            "one_division_apart": None if distance is None else float(distance == 1),
            "two_plus_divisions_apart": None if distance is None else float(distance >= 2),
            "mismatch_moderate": None if distance is None else float(distance == 1),
            "mismatch_large": None if distance is None else float(distance == 2),
            "mismatch_extreme": None if distance is None else float(distance >= 3),
            "winner_reference_weight": winner_weight,
            "opponent_reference_weight": opponent_weight,
            "estimated_weight_gap": weight_gap,
            "relative_weight_gap": relative_gap,
            "winner_size_advantage": relative_gap,
        }
    )
    height_diff = (
        None if winner.get("height") is None or opponent.get("height") is None
        else float(winner["height"]) - float(opponent["height"])
    )
    reach_diff = (
        None if winner.get("reach") is None or opponent.get("reach") is None
        else float(winner["reach"]) - float(opponent["reach"])
    )
    age_diff = (
        None if winner.get("age") is None or opponent.get("age") is None
        else float(winner["age"]) - float(opponent["age"])
    )
    winner_bmi = (
        None if winner_weight is None or not winner.get("height")
        else winner_weight * 703.0 / float(winner["height"]) ** 2
    )
    opponent_bmi = (
        None if opponent_weight is None or not opponent.get("height")
        else opponent_weight * 703.0 / float(opponent["height"]) ** 2
    )
    size = relative_gap
    result.update(
        {
            "height_diff": height_diff,
            "reach_diff": reach_diff,
            "age_diff": age_diff,
            "bmi_proxy_diff": (
                None if winner_bmi is None or opponent_bmi is None
                else winner_bmi - opponent_bmi
            ),
            "same_stance": float(
                bool(winner.get("stance"))
                and winner.get("stance") == opponent.get("stance")
            ),
            "winner_size_x_ko_rate": None if size is None or winner["ko_tko_win_pct"] is None else size * float(winner["ko_tko_win_pct"]),
            "winner_size_x_submission_rate": None if size is None or winner["submission_win_pct"] is None else size * float(winner["submission_win_pct"]),
            "winner_size_x_takedowns_per15": None if size is None or winner["takedowns_per15"] is None else size * float(winner["takedowns_per15"]),
            "winner_size_x_opponent_ko_loss_rate": None if size is None or opponent["ko_tko_loss_pct"] is None else size * float(opponent["ko_tko_loss_pct"]),
            "winner_size_x_opponent_submission_loss_rate": None if size is None or opponent["submission_loss_pct"] is None else size * float(opponent["submission_loss_pct"]),
            "reach_advantage_x_striking_accuracy": None if reach_diff is None or winner["sig_str_accuracy"] is None else reach_diff * float(winner["sig_str_accuracy"]),
            "division_distance_x_finish_rate": None if distance is None or winner["finish_rate"] is None else distance * float(winner["finish_rate"]),
            "division_distance_x_decision_rate": None if distance is None or winner["decision_win_pct"] is None else distance * float(winner["decision_win_pct"]),
        }
    )
    rounds = parse_scheduled_rounds(time_format)
    result["scheduled_rounds"] = rounds
    result["five_round_fight"] = float(rounds >= 5)
    return result


def difference_features(
    a: dict[str, Any], b: dict[str, Any], bout_type: str | None = None
) -> dict[str, float | None]:
    bases = [name.removesuffix("_diff") for name in BASE_FEATURES]
    result: dict[str, float | None] = {
        f"{base}_diff": (
            None if a.get(base) is None or b.get(base) is None else a[base] - b[base]
        )
        for base in bases
    }
    stance_a, stance_b = str(a.get("stance") or ""), str(b.get("stance") or "")
    for stance in ("orthodox", "southpaw", "switch"):
        result[f"fighter_a_{stance}"] = float(stance_a == stance)
        result[f"fighter_b_{stance}"] = float(stance_b == stance)
    result["same_stance"] = float(bool(stance_a) and stance_a == stance_b)
    result.update(weight_class_features(bout_type))
    result.update(
        {
            "fighter_a_td_accuracy_vs_b_defense": (
                nullable_difference(a["takedown_accuracy"], b["takedown_defense"])
            ),
            "fighter_b_td_accuracy_vs_a_defense": (
                nullable_difference(b["takedown_accuracy"], a["takedown_defense"])
            ),
            "fighter_a_striking_offense_vs_b_defense": (
                striking_matchup(a["sig_str_landed_pm"], b["striking_defense"])
            ),
            "fighter_b_striking_offense_vs_a_defense": (
                striking_matchup(b["sig_str_landed_pm"], a["striking_defense"])
            ),
        }
    )
    result.update(
        {
            "fighter_a_ufc_debutant": float(a["ufc_fights"] == 0),
            "fighter_b_ufc_debutant": float(b["ufc_fights"] == 0),
            "fighter_a_ufc_fights": float(a["ufc_fights"]),
            "fighter_b_ufc_fights": float(b["ufc_fights"]),
        }
    )
    return result


def should_swap(row: pd.Series) -> bool:
    identity = "|".join(
        [str(row["event_date"]), str(row["red_fighter_name"]), str(row["blue_fighter_name"])]
    )
    return hashlib.sha256(identity.encode("utf-8")).digest()[0] % 2 == 0


def _update_state(
    state: FighterState,
    own_sig: tuple[float, float],
    opp_sig: tuple[float, float],
    own_td: tuple[float, float],
    opp_td: tuple[float, float],
    submissions: float,
    knockdowns: float,
    strike_locations: dict[str, float],
    control_seconds: int,
    duration: int,
    result: str,
    method: str,
    bout_date: date,
    opponent_pre_fight_elo: float,
    bout_division: str,
) -> None:
    state.sig_landed += own_sig[0]
    state.sig_attempted += own_sig[1]
    state.sig_absorbed += opp_sig[0]
    state.sig_faced_attempted += opp_sig[1]
    state.td_landed += own_td[0]
    state.td_attempted += own_td[1]
    state.opponent_td_landed += opp_td[0]
    state.opponent_td_attempted += opp_td[1]
    state.sub_attempts += submissions
    state.knockdowns += knockdowns
    state.head_landed += strike_locations["head"]
    state.body_landed += strike_locations["body"]
    state.leg_landed += strike_locations["leg"]
    state.distance_landed += strike_locations["distance"]
    state.clinch_landed += strike_locations["clinch"]
    state.ground_landed += strike_locations["ground"]
    state.control_seconds += control_seconds
    state.seconds += duration
    state.opponent_elo_sum += opponent_pre_fight_elo
    state.opponents_faced += 1
    if result == "W":
        state.wins += 1
        state.recent_results.append(1)
        state.current_win_streak += 1
        state.current_loss_streak = 0
        state.beaten_opponent_elo_sum += opponent_pre_fight_elo
        state.quality_win_points += opponent_pre_fight_elo / 1500.0
        if not method.casefold().startswith("decision"):
            state.finish_wins += 1
        method_lower = method.casefold()
        if "ko" in method_lower:
            state.ko_tko_wins += 1
        elif "sub" in method_lower:
            state.submission_wins += 1
        elif method_lower.startswith("decision"):
            state.decision_wins += 1
    elif result == "L":
        state.losses += 1
        state.recent_results.append(0)
        state.current_loss_streak += 1
        state.current_win_streak = 0
        method_lower = method.casefold()
        if "ko" in method_lower:
            state.ko_tko_losses += 1
        elif "sub" in method_lower:
            state.submission_losses += 1
        elif method_lower.startswith("decision"):
            state.decision_losses += 1
    category = method_category(method)
    if result in {"W", "L"} and category is not None:
        state.recent_methods.append(category)
    state.last_fight_date = bout_date.isoformat()
    if bout_division in DIVISION_REFERENCE_WEIGHTS:
        state.last_division = bout_division


def _update_elo(red: FighterState, blue: FighterState, red_result: str, k: float = 32.0) -> None:
    """Update both ratings after a decisive fight; ratings are snapshotted first."""
    if red_result not in {"W", "L"}:
        return
    expected_red = 1.0 / (1.0 + 10.0 ** ((blue.elo - red.elo) / 400.0))
    actual_red = 1.0 if red_result == "W" else 0.0
    change = k * (actual_red - expected_red)
    red.elo += change
    blue.elo -= change


def build_dataset(
    fights_path: str, profiles_path: str
) -> tuple[pd.DataFrame, dict[str, FighterState], dict[str, dict[str, Any]]]:
    fights = pd.read_csv(fights_path, sep=";", low_memory=False)
    fights["event_date"] = pd.to_datetime(
        fights["event_date"], format="%d/%m/%Y", errors="coerce"
    )
    fights = fights.dropna(subset=["event_date"]).sort_values("event_date", kind="stable")
    profiles = load_profiles(profiles_path)
    states: dict[str, FighterState] = defaultdict(FighterState)
    examples: list[dict[str, Any]] = []

    for _, row in fights.iterrows():
        red_name, blue_name = str(row["red_fighter_name"]), str(row["blue_fighter_name"])
        red_key, blue_key = normalize_name(red_name), normalize_name(blue_name)
        bout_date = row["event_date"].date()

        # IMPORTANT: these snapshots happen before this row updates either fighter.
        red_before = fighter_snapshot(red_name, bout_date, states, profiles)
        blue_before = fighter_snapshot(blue_name, bout_date, states, profiles)
        red_result = str(row.get("red_fighter_result", "")).strip().upper()
        blue_result = str(row.get("blue_fighter_result", "")).strip().upper()
        method = str(row.get("method", "")).strip()
        bout_division = division_slug(str(row.get("bout_type", "")))

        if {red_result, blue_result} == {"W", "L"}:
            swap = should_swap(row)
            first, second = (blue_before, red_before) if swap else (red_before, blue_before)
            first_name, second_name = (blue_name, red_name) if swap else (red_name, blue_name)
            first_won = (blue_result == "W") if swap else (red_result == "W")
            winner_before = red_before if red_result == "W" else blue_before
            opponent_before = blue_before if red_result == "W" else red_before
            winner_name = red_name if red_result == "W" else blue_name
            examples.append(
                {
                    "date": bout_date.isoformat(),
                    "fighter_a": first_name,
                    "fighter_b": second_name,
                    "fighter_a_prior_ufc_fights": int(first["ufc_fights"]),
                    "fighter_b_prior_ufc_fights": int(second["ufc_fights"]),
                    "fighter_a_age": first.get("age"),
                    "fighter_b_age": second.get("age"),
                    "fighter_a_reach": first.get("reach"),
                    "fighter_b_reach": second.get("reach"),
                    "fighter_a_height": first.get("height"),
                    "fighter_b_height": second.get("height"),
                    "fighter_a_days_since_last_fight": first.get("days_since_last_fight"),
                    "fighter_b_days_since_last_fight": second.get("days_since_last_fight"),
                    "fighter_a_historical_division": first.get("historical_division"),
                    "fighter_b_historical_division": second.get("historical_division"),
                    "fighter_a_recent_win_pct": first.get("recent_3_win_pct"),
                    "fighter_b_recent_win_pct": second.get("recent_3_win_pct"),
                    "fighter_a_career_win_pct": first.get("ufc_win_pct"),
                    "fighter_b_career_win_pct": second.get("ufc_win_pct"),
                    "fighter_a_takedowns_per15": first.get("takedowns_per15"),
                    "fighter_b_takedowns_per15": second.get("takedowns_per15"),
                    "fighter_a_sig_landed_pm": first.get("sig_str_landed_pm"),
                    "fighter_b_sig_landed_pm": second.get("sig_str_landed_pm"),
                    "fighter_a_submission_rate": first.get("submission_win_pct"),
                    "fighter_b_submission_rate": second.get("submission_win_pct"),
                    "fighter_a_submission_loss_rate": first.get("submission_loss_pct"),
                    "fighter_b_submission_loss_rate": second.get("submission_loss_pct"),
                    "fighter_a_ko_rate": first.get("ko_tko_win_pct"),
                    "fighter_b_ko_rate": second.get("ko_tko_win_pct"),
                    "fighter_a_striking_defense": first.get("striking_defense"),
                    "fighter_b_striking_defense": second.get("striking_defense"),
                    **difference_features(first, second, str(row.get("bout_type", ""))),
                    "fighter_a_won": int(first_won),
                    **method_features(
                        winner_before,
                        opponent_before,
                        str(row.get("bout_type", "")),
                        row.get("time_format"),
                    ),
                    "actual_winner": winner_name,
                    "method_label": method_category(method),
                }
            )

        red_sig = parse_of(row.get("red_fighter_sig_str"))
        blue_sig = parse_of(row.get("blue_fighter_sig_str"))
        red_td = parse_of(row.get("red_fighter_TD"))
        blue_td = parse_of(row.get("blue_fighter_TD"))
        red_locations = {
            location: parse_of(row.get(f"red_fighter_sig_str_{location}"))[0]
            for location in ("head", "body", "leg", "distance", "clinch", "ground")
        }
        blue_locations = {
            location: parse_of(row.get(f"blue_fighter_sig_str_{location}"))[0]
            for location in ("head", "body", "leg", "distance", "clinch", "ground")
        }
        duration = fight_duration_seconds(row)
        _update_state(
            states[red_key], red_sig, blue_sig, red_td, blue_td,
            numeric_or_zero(row.get("red_fighter_sub_att")),
            numeric_or_zero(row.get("red_fighter_KD")), red_locations,
            parse_clock(row.get("red_fighter_ctrl")),
            duration, red_result, method, bout_date, float(blue_before["elo"]),
            bout_division,
        )
        _update_state(
            states[blue_key], blue_sig, red_sig, blue_td, red_td,
            numeric_or_zero(row.get("blue_fighter_sub_att")),
            numeric_or_zero(row.get("blue_fighter_KD")), blue_locations,
            parse_clock(row.get("blue_fighter_ctrl")),
            duration, blue_result, method, bout_date, float(red_before["elo"]),
            bout_division,
        )
        _update_elo(states[red_key], states[blue_key], red_result)

    return pd.DataFrame(examples), dict(states), profiles
