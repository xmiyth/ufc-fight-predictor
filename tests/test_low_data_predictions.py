import pytest

from ufc_predictor import web


def _assert_prediction(fighter_a, fighter_b):
    result = web.predict_matchup(fighter_a, fighter_b)
    assert result["fighter_a_probability"] + result["fighter_b_probability"] == 100.0
    assert result["limited_ufc_data"] is True
    assert result["low_data_warning"].startswith("Limited UFC data:")
    assert len(result["outcome_probabilities"]) == 6
    return result


def test_established_fighter_vs_debutant():
    result = _assert_prediction("Islam Makhachev", "Bob Sapp")
    assert result["fighter_a_prediction_mode"] == "NORMAL"
    assert result["fighter_b_prediction_mode"] == "DEBUTANT"
    assert result["fighter_b_prior_ufc_fights"] == 0


def test_debutant_vs_debutant():
    result = _assert_prediction("Bob Sapp", "Aaron Jeffery")
    assert result["prediction_mode"] == "DEBUTANT"
    assert result["fighter_a_prediction_mode"] == "DEBUTANT"
    assert result["fighter_b_prediction_mode"] == "DEBUTANT"


def test_exactly_one_ufc_fight_vs_veteran():
    result = _assert_prediction("Frank Hamaker", "Islam Makhachev")
    assert result["fighter_a_prior_ufc_fights"] == 1
    assert result["fighter_a_prediction_mode"] == "LOW_SAMPLE"
    assert result["fighter_b_prediction_mode"] == "NORMAL"


def test_exactly_two_ufc_fights_vs_veteran():
    result = _assert_prediction("Orlando Wiet", "Islam Makhachev")
    assert result["fighter_a_prior_ufc_fights"] == 2
    assert result["fighter_a_prediction_mode"] == "LOW_SAMPLE"
    assert result["fighter_b_prediction_mode"] == "NORMAL"


def test_extreme_weight_mismatch_involving_debutant():
    result = _assert_prediction("Bob Sapp", "Brandon Moreno")
    assert result["mismatch_category"] == "EXTREME"
    assert result["matchup_confidence"] == "LOW"
    assert result["weight_class_matchup"]["division_distance"] >= 3


def test_low_data_probability_is_not_confidence_shrunk():
    today = __import__("datetime").date.today()
    probability, _, _, _ = web._symmetric_model_analysis(
        "bob sapp", "aaron jeffery", today
    )
    adjusted, increment = web._extreme_size_adjustment(
        probability, "bob sapp", "aaron jeffery"
    )
    result = web.predict_matchup("Bob Sapp", "Aaron Jeffery")
    assert increment == 0.0
    assert adjusted == probability
    assert result["fighter_a_probability"] == pytest.approx(
        round(probability * 100.0, 1)
    )


@pytest.mark.parametrize(
    "fighter_a,fighter_b",
    [
        ("Islam Makhachev", "Charles Oliveira"),
        ("Bob Sapp", "Brandon Moreno"),
        ("Frank Hamaker", "Islam Makhachev"),
    ],
)
def test_swapping_fighters_inverts_winner_probability(fighter_a, fighter_b):
    forward = web.predict_matchup(fighter_a, fighter_b)
    reverse = web.predict_matchup(fighter_b, fighter_a)
    assert forward["fighter_a_probability"] == pytest.approx(
        reverse["fighter_b_probability"], abs=0.1
    )
    assert forward["fighter_b_probability"] == pytest.approx(
        reverse["fighter_a_probability"], abs=0.1
    )


def test_extreme_size_adjustment_is_isolated_and_directional():
    result = web.predict_matchup("Bob Sapp", "Brandon Moreno")
    reverse = web.predict_matchup("Brandon Moreno", "Bob Sapp")
    assert result["extreme_size_log_odds_adjustment"] > 0
    assert reverse["extreme_size_log_odds_adjustment"] < 0
    assert result["fighter_a_probability"] > 50.0
