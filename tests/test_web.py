import hashlib
from datetime import date
import math

import pytest
from fastapi import HTTPException

from ufc_predictor import web


def test_deployed_winner_checksum_is_valid():
    digest = hashlib.sha256(web.MODEL_PATH.read_bytes()).hexdigest().upper()
    assert digest == web.EXPECTED_MODEL_SHA256


def test_separate_method_model_checksum_is_valid():
    digest = hashlib.sha256(web.METHOD_MODEL_PATH.read_bytes()).hexdigest().upper()
    assert digest == web.EXPECTED_METHOD_MODEL_SHA256


def test_prediction_uses_real_frozen_fighter_histories():
    result = web.predict_matchup("Islam Makhachev", "Ilia Topuria")
    assert result["model_version"] == "V2.1-accuracy"
    assert result["historical_data_through"] == "2026-06-27"
    assert result["fighter_a_probability"] + result["fighter_b_probability"] == 100.0
    assert result["predicted_winner"] in {result["fighter_a"], result["fighter_b"]}
    assert {row["method"] for row in result["method_probabilities"]} == {
        "KO/TKO", "Submission", "Decision"
    }
    assert sum(row["probability"] for row in result["method_probabilities"]) == pytest.approx(
        100.0, abs=0.2
    )
    assert 3 <= len(result["factors_for_fighter_a"]) <= 5
    assert 2 <= len(result["factors_for_fighter_b"]) <= 3
    assert all("log_odds_contribution" in row for row in result["winner_explanation"])


def test_winner_uses_both_frozen_ensemble_components():
    from ufc_predictor.winner_v2 import feature_frame
    frame = feature_frame(web.PROJECT_ROOT, web.ARTIFACT,
                          "islam makhachev", "ilia topuria", date.today(), "Lightweight", 3)
    expected = sum(web.ARTIFACT["weights"][name] * model.predict_proba(frame)[0, 1]
                   for name, model in web.ARTIFACT["components"].items())
    probability, contributions, _, _ = web._model_analysis(
        "islam makhachev", "ilia topuria", date.today(), "Lightweight", 3
    )
    assert probability == pytest.approx(expected, abs=1e-7)
    assert all(math.isfinite(value) for value in contributions.values())


def test_displayed_evaluation_matches_deployed_model():
    result = web.prediction_results()
    assert result["model_version"] == web.ARTIFACT["model_version"]
    assert result["historical_accuracy"] == pytest.approx(484 / 772)
    assert result["historical_test_fights"] == 772


def test_same_fighter_is_rejected():
    with pytest.raises(HTTPException, match="different fighters"):
        web.predict_matchup("Islam Makhachev", "Islam Makhachev")


def test_all_fighters_are_unique_alphabetical_and_cover_a_to_z():
    assert len(web.FIGHTERS) == len(web.STATES)
    assert web.FIGHTERS == sorted(set(web.FIGHTERS), key=str.casefold)
    for initial in "MSTVZ":
        assert any(name.casefold().startswith(initial.casefold()) for name in web.FIGHTERS)
