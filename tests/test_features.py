from collections import defaultdict
from datetime import date

import pandas as pd

from ufc_predictor.features import (
    FighterState,
    difference_features,
    fighter_snapshot,
    parse_height_inches,
    parse_of,
    build_dataset,
)


def test_parsers():
    assert parse_of("31 of 65") == (31.0, 65.0)
    assert parse_of("---") == (0.0, 0.0)
    assert parse_height_inches("5' 11\"") == 71.0


def test_snapshot_only_contains_existing_history():
    states = defaultdict(FighterState)
    profiles = {
        "alpha": {"dob": "1990-01-01", "height": 72.0, "reach": 74.0},
        "beta": {"dob": "1992-01-01", "height": 70.0, "reach": 71.0},
    }
    before = difference_features(
        fighter_snapshot("alpha", date(2020, 1, 1), states, profiles),
        fighter_snapshot("beta", date(2020, 1, 1), states, profiles),
    )
    assert before["ufc_wins_diff"] == 0.0
    states["alpha"].wins = 1
    states["alpha"].recent_results.append(1)
    after = difference_features(
        fighter_snapshot("alpha", date(2020, 2, 1), states, profiles),
        fighter_snapshot("beta", date(2020, 2, 1), states, profiles),
    )
    assert after["ufc_wins_diff"] == 1.0
    # Beta is a debutant: unknown form is missing, not a fabricated 50% prior.
    assert after["recent_5_win_pct_diff"] is None
    assert after["recent_3_win_pct_diff"] is None


def test_debutant_performance_is_missing_not_zero():
    snapshot = fighter_snapshot(
        "alpha",
        date(2020, 1, 1),
        defaultdict(FighterState),
        {"alpha": {"dob": "1990-01-01", "height": 72.0, "reach": 74.0}},
    )
    assert snapshot["ufc_fights"] == 0.0
    assert snapshot["ufc_debutant"] == 1.0
    for feature in (
        "ufc_win_pct", "sig_str_landed_pm", "sig_str_absorbed_pm",
        "sig_str_accuracy", "striking_defense", "takedown_accuracy",
        "takedown_defense", "submission_attempts_per15", "finish_rate",
    ):
        assert snapshot[feature] is None


def test_current_fight_is_not_in_its_own_features(tmp_path):
    fights = pd.DataFrame(
        [
            {
                "red_fighter_name": "Alpha", "blue_fighter_name": "Beta",
                "event_date": "01/01/2020", "red_fighter_result": "W",
                "blue_fighter_result": "L", "method": "KO/TKO", "round": 1,
                "time": "1:00", "red_fighter_sig_str": "10 of 20",
                "blue_fighter_sig_str": "5 of 15", "red_fighter_TD": "0 of 0",
                "blue_fighter_TD": "0 of 1", "red_fighter_sub_att": 0,
                "blue_fighter_sub_att": 0,
            },
            {
                "red_fighter_name": "Alpha", "blue_fighter_name": "Beta",
                "event_date": "01/02/2020", "red_fighter_result": "L",
                "blue_fighter_result": "W", "method": "Decision - Unanimous", "round": 3,
                "time": "5:00", "red_fighter_sig_str": "20 of 40",
                "blue_fighter_sig_str": "30 of 50", "red_fighter_TD": "1 of 2",
                "blue_fighter_TD": "2 of 3", "red_fighter_sub_att": 0,
                "blue_fighter_sub_att": 1,
            },
        ]
    )
    profiles = pd.DataFrame(
        [
            {"fighter_name": "Alpha", "Height": "6' 0\"", "Reach": "74\"", "DOB": "Jan 01, 1990"},
            {"fighter_name": "Beta", "Height": "5' 10\"", "Reach": "71\"", "DOB": "Jan 01, 1992"},
        ]
    )
    fights_path, profiles_path = tmp_path / "fights.csv", tmp_path / "fighters.csv"
    fights.to_csv(fights_path, sep=";", index=False)
    profiles.to_csv(profiles_path, index=False)

    data, _, _ = build_dataset(str(fights_path), str(profiles_path))
    assert len(data) == 2
    assert data.iloc[0]["ufc_wins_diff"] == 0.0
    assert abs(data.iloc[1]["ufc_wins_diff"]) == 1.0
    assert abs(data.iloc[1]["elo_diff"]) == 32.0
