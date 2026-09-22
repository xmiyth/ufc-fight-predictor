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

FEATURE_NAMES = BASE_FEATURES + STANCE_FEATURES + WEIGHT_CLASS_FEATURES + MATCHUP_FEATURES

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


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


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
    current_win_streak: int = 0
    current_loss_streak: int = 0
    elo: float = 1500.0
    opponent_elo_sum: float = 0.0
    opponents_faced: int = 0
    beaten_opponent_elo_sum: float = 0.0
    quality_win_points: float = 0.0
    last_fight_date: str | None = None
    recent_results: deque[int] = field(default_factory=lambda: deque(maxlen=5))

    def performance_snapshot(self) -> dict[str, float]:
        fights = self.wins + self.losses
        minutes = self.seconds / 60.0
        return {
            "ufc_wins": float(self.wins),
            "ufc_losses": float(self.losses),
            "ufc_win_pct": safe_ratio(self.wins, fights, 0.5),
            "recent_3_win_pct": safe_ratio(
                sum(list(self.recent_results)[-3:]),
                len(list(self.recent_results)[-3:]),
                0.5,
            ),
            "recent_5_win_pct": safe_ratio(
                sum(self.recent_results), len(self.recent_results), 0.5
            ),
            "current_win_streak": float(self.current_win_streak),
            "current_loss_streak": float(self.current_loss_streak),
            "sig_str_accuracy": safe_ratio(self.sig_landed, self.sig_attempted),
            "sig_str_landed_pm": safe_ratio(self.sig_landed, minutes),
            "sig_str_absorbed_pm": safe_ratio(self.sig_absorbed, minutes),
            "striking_differential_pm": safe_ratio(
                self.sig_landed - self.sig_absorbed, minutes
            ),
            "striking_defense": safe_ratio(
                self.sig_faced_attempted - self.sig_absorbed,
                self.sig_faced_attempted,
                0.5,
            ),
            "knockdowns_per_fight": safe_ratio(self.knockdowns, fights),
            "head_strike_share": safe_ratio(self.head_landed, self.sig_landed),
            "body_strike_share": safe_ratio(self.body_landed, self.sig_landed),
            "leg_strike_share": safe_ratio(self.leg_landed, self.sig_landed),
            "distance_strike_share": safe_ratio(self.distance_landed, self.sig_landed),
            "clinch_strike_share": safe_ratio(self.clinch_landed, self.sig_landed),
            "ground_strike_share": safe_ratio(self.ground_landed, self.sig_landed),
            "takedowns_per15": safe_ratio(self.td_landed * 15.0, minutes),
            "takedown_accuracy": safe_ratio(self.td_landed, self.td_attempted),
            "takedown_defense": safe_ratio(
                self.opponent_td_attempted - self.opponent_td_landed,
                self.opponent_td_attempted,
                0.5,
            ),
            "submission_attempts_per15": safe_ratio(self.sub_attempts * 15.0, minutes),
            "control_time_per15": safe_ratio(self.control_seconds * 15.0, self.seconds),
            "finish_rate": safe_ratio(self.finish_wins, self.wins),
            "ko_tko_win_pct": safe_ratio(self.ko_tko_wins, self.wins),
            "submission_win_pct": safe_ratio(self.submission_wins, self.wins),
            "decision_win_pct": safe_ratio(self.decision_wins, self.wins),
            "ufc_fights": float(fights),
            "elo": self.elo,
            "average_opponent_elo": safe_ratio(
                self.opponent_elo_sum, self.opponents_faced, 1500.0
            ),
            "average_beaten_opponent_elo": safe_ratio(
                self.beaten_opponent_elo_sum, self.wins, 1500.0
            ),
            "quality_adjusted_win_score": safe_ratio(
                self.quality_win_points, fights
            ),
        }

    def serializable(self) -> dict[str, Any]:
        result = asdict(self)
        result["recent_results"] = list(self.recent_results)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FighterState":
        values = dict(values)
        values["recent_results"] = deque(values.get("recent_results", []), maxlen=5)
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
        **states[key].performance_snapshot(),
    }


def weight_class_features(bout_type: str | None) -> dict[str, float]:
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
    return {f"weight_class_{name}": float(name == slug) for name in WEIGHT_CLASSES}


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
                a["takedown_accuracy"] - b["takedown_defense"]
            ),
            "fighter_b_td_accuracy_vs_a_defense": (
                b["takedown_accuracy"] - a["takedown_defense"]
            ),
            "fighter_a_striking_offense_vs_b_defense": (
                a["sig_str_landed_pm"] * (1.0 - b["striking_defense"])
            ),
            "fighter_b_striking_offense_vs_a_defense": (
                b["sig_str_landed_pm"] * (1.0 - a["striking_defense"])
            ),
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
    state.last_fight_date = bout_date.isoformat()


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

        if {red_result, blue_result} == {"W", "L"}:
            swap = should_swap(row)
            first, second = (blue_before, red_before) if swap else (red_before, blue_before)
            first_name, second_name = (blue_name, red_name) if swap else (red_name, blue_name)
            first_won = (blue_result == "W") if swap else (red_result == "W")
            examples.append(
                {
                    "date": bout_date.isoformat(),
                    "fighter_a": first_name,
                    "fighter_b": second_name,
                    **difference_features(first, second, str(row.get("bout_type", ""))),
                    "fighter_a_won": int(first_won),
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
        )
        _update_state(
            states[blue_key], blue_sig, red_sig, blue_td, red_td,
            numeric_or_zero(row.get("blue_fighter_sub_att")),
            numeric_or_zero(row.get("blue_fighter_KD")), blue_locations,
            parse_clock(row.get("blue_fighter_ctrl")),
            duration, blue_result, method, bout_date, float(red_before["elo"]),
        )
        _update_elo(states[red_key], states[blue_key], red_result)

    return pd.DataFrame(examples), dict(states), profiles
