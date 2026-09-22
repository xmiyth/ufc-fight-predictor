import pytest

from ufc_predictor import web
from ufc_predictor.features import (
    DIVISION_REFERENCE_WEIGHTS,
    division_distance,
    division_slug,
)


SANITY_MATCHUPS = [
    ("Jon Jones", "Demetrious Johnson"),
    ("Jon Jones", "Brandon Moreno"),
    ("Tom Aspinall", "Alexandre Pantoja"),
    ("Islam Makhachev", "Charles Oliveira"),
    ("Alex Pereira", "Israel Adesanya"),
    ("Merab Dvalishvili", "Sean O'Malley"),
    ("Max Holloway", "Dustin Poirier"),
]


def test_division_order_and_reference_weight_proxy():
    assert division_slug("UFC Light Heavyweight Title Bout") == "light_heavyweight"
    assert division_distance("flyweight", "heavyweight") == 7
    assert division_distance("womens_strawweight", "womens_bantamweight") == 2
    assert division_distance("womens_flyweight", "flyweight") is None
    assert DIVISION_REFERENCE_WEIGHTS["lightweight"] == 155.0
    assert DIVISION_REFERENCE_WEIGHTS["heavyweight"] == 265.0


@pytest.mark.parametrize(("fighter_a", "fighter_b"), SANITY_MATCHUPS)
def test_sanity_matchups_return_conditional_and_joint_distributions(fighter_a, fighter_b):
    result = web.predict_matchup(fighter_a, fighter_b)
    assert result["method_model_version"] == "Method V2.0"
    assert result["mismatch_category"] in {"NORMAL", "MODERATE", "LARGE", "EXTREME"}
    assert result["matchup_confidence"] in {"HIGH", "MEDIUM", "LOW"}
    assert len(result["outcome_probabilities"]) == 6
    assert result["outcome_probability_total"] == pytest.approx(100.0, abs=0.15)
    for side in ("fighter_a", "fighter_b"):
        probabilities = result["conditional_method_probabilities"][side]
        assert sum(row["probability"] for row in probabilities) == pytest.approx(100.0, abs=0.15)


@pytest.mark.parametrize(
    ("fighter_a", "fighter_b"),
    [
        ("Jon Jones", "Demetrious Johnson"),
        ("Jon Jones", "Brandon Moreno"),
        ("Tom Aspinall", "Alexandre Pantoja"),
    ],
)
def test_extreme_mismatches_are_flagged_and_not_decision_dominated(fighter_a, fighter_b):
    result = web.predict_matchup(fighter_a, fighter_b)
    decision = next(
        row["probability"] for row in result["method_probabilities"]
        if row["method"] == "Decision"
    )
    assert result["mismatch_category"] == "EXTREME"
    assert result["matchup_confidence"] == "LOW"
    assert result["matchup_warning"]
    assert decision < 40.0


def test_same_fighter_method_distribution_responds_to_progressive_size_gap():
    opponents = [
        "Brandon Moreno",       # flyweight
        "Sean O'Malley",       # bantamweight
        "Dustin Poirier",      # lightweight
        "Kamaru Usman",        # welterweight
        "Alex Pereira",        # light heavyweight
        "Tom Aspinall",        # heavyweight
    ]
    distances, distributions = [], []
    for opponent in opponents:
        result = web.predict_matchup("Demetrious Johnson", opponent)
        distances.append(result["weight_class_matchup"]["division_distance"])
        distributions.append(
            tuple(
                row["probability"]
                for row in result["conditional_method_probabilities"]["fighter_a"]
            )
        )
    assert distances == sorted(distances)
    assert distances[0] == 0 and distances[-1] >= 3
    assert len(set(distributions)) == len(distributions)
    assert max(values[0] for values in distributions) - min(values[0] for values in distributions) > 5.0
