"""Predict a matchup using the latest historical state in the trained artifact."""

from __future__ import annotations

import argparse
from datetime import date
import difflib

import joblib
import pandas as pd

from .features import FighterState, difference_features, fighter_snapshot, normalize_name


def resolve_fighter(name: str, profiles: dict, states: dict) -> str:
    key = normalize_name(name)
    if key in states:
        return key
    choices = sorted(set(profiles) | set(states))
    suggestions = difflib.get_close_matches(key, choices, n=3, cutoff=0.6)
    readable = [profiles.get(item, {}).get("name", item.title()) for item in suggestions]
    hint = f" Did you mean: {', '.join(readable)}?" if readable else ""
    raise ValueError(f"Fighter not found in historical bouts: {name}.{hint}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fighter_a")
    parser.add_argument("fighter_b")
    parser.add_argument("--model", default="models/best_model.joblib")
    parser.add_argument("--as-of", default=date.today().isoformat(), help="YYYY-MM-DD")
    parser.add_argument(
        "--weight-class", default="", help="Optional bout type, e.g. Lightweight Bout"
    )
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    states = {
        name: FighterState.from_dict(values)
        for name, values in artifact["fighter_states"].items()
    }
    profiles = artifact["profiles"]
    key_a = resolve_fighter(args.fighter_a, profiles, states)
    key_b = resolve_fighter(args.fighter_b, profiles, states)
    if key_a == key_b:
        raise ValueError("Choose two different fighters.")

    prediction_date = date.fromisoformat(args.as_of)
    snapshot_a = fighter_snapshot(key_a, prediction_date, states, profiles)
    snapshot_b = fighter_snapshot(key_b, prediction_date, states, profiles)
    features = pd.DataFrame([difference_features(snapshot_a, snapshot_b, args.weight_class)])
    probability_a = float(
        artifact["pipeline"].predict_proba(features[artifact["feature_names"]])[0, 1]
    )

    display_a = profiles.get(key_a, {}).get("name", args.fighter_a)
    display_b = profiles.get(key_b, {}).get("name", args.fighter_b)
    print(f"Fighter A ({display_a}): {probability_a:.1%}")
    print(f"Fighter B ({display_b}): {1.0 - probability_a:.1%}")
    print(f"Historical statistics through: {artifact['last_data_date']}")


if __name__ == "__main__":
    main()
