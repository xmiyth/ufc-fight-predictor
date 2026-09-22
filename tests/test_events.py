import json
from datetime import date, timedelta

import pytest

from ufc_predictor.event_sources.base import Event
from ufc_predictor.event_sources.local_json import LocalJsonEventSource
from ufc_predictor.fighter_matching import FighterNameMatcher, canonical_fighter_name
from ufc_predictor.prediction_history import PredictionHistory
from ufc_predictor import web


def _write_events(path, events):
    path.write_text(json.dumps({"events": events}), encoding="utf-8")


def test_event_parser_validates_and_orders_real_card_fields(tmp_path):
    path = tmp_path / "events.json"
    _write_events(
        path,
        [
            {
                "event_id": "later",
                "event_name": "Later Event",
                "event_date": "2030-02-01",
                "location": "London",
                "fights": [{
                    "fight_id": "main", "fighter_a": "A", "fighter_b": "B",
                    "weight_class": "Lightweight", "number_of_rounds": 5,
                    "bout_order": 1, "card_section": "Main Event",
                }],
            },
            {
                "event_id": "sooner", "event_name": "Sooner Event",
                "event_date": "2030-01-01", "fights": [],
            },
        ],
    )
    events = LocalJsonEventSource(path).upcoming_events(date(2029, 1, 1))
    assert [event.event_id for event in events] == ["sooner", "later"]
    assert events[1].fights[0].number_of_rounds == 5
    assert events[1].fights[0].weight_class == "Lightweight"


def test_event_parser_rejects_invalid_round_count():
    with pytest.raises(ValueError, match="3 or 5"):
        Event.from_dict({
            "event_id": "bad", "event_name": "Bad", "event_date": "2030-01-01",
            "fights": [{"fighter_a": "A", "fighter_b": "B", "number_of_rounds": 4}],
        })


def test_fighter_matching_is_accent_and_punctuation_tolerant_but_not_fuzzy():
    matcher = FighterNameMatcher([
        ("josé aldo", "José Aldo"),
        ("sean o'malley", "Sean O'Malley"),
    ])
    assert canonical_fighter_name("  José  Aldo ") == "jose aldo"
    assert matcher.match("Jose Aldo").matched_key == "josé aldo"
    assert matcher.match("Sean O Malley").matched_key == "sean o'malley"
    assert matcher.match("Sean Malley").status == "not_found"


def test_unknown_event_fighter_returns_unavailable_without_fake_statistics(tmp_path, monkeypatch):
    future = (date.today() + timedelta(days=30)).isoformat()
    path = tmp_path / "events.json"
    _write_events(path, [{
        "event_id": "debut", "event_name": "Verified Card", "event_date": future,
        "fights": [{"fight_id": "one", "fighter_a": "Unknown Debutant", "fighter_b": "Ilia Topuria"}],
    }])
    monkeypatch.setattr(web, "EVENT_SOURCE", LocalJsonEventSource(path))
    monkeypatch.setattr(web, "PREDICTION_HISTORY", PredictionHistory(tmp_path / "history.json"))
    response = web.event_detail("debut")
    fight = response["event"]["fights"][0]
    assert fight["prediction_status"] == "unavailable"
    assert fight["prediction"] is None
    assert "Insufficient UFC historical data" in fight["prediction_unavailable_reason"]


def test_first_event_prediction_is_locked_and_not_silently_replaced(tmp_path):
    history = PredictionHistory(tmp_path / "history.json")
    event = {"event_id": "event", "event_name": "Event", "event_date": "2030-01-01"}
    fight = {"fight_id": "fight"}
    first = {
        "model_version": "V1.0", "fighter_a": "A", "fighter_b": "B",
        "fighter_a_probability": 60.0, "fighter_b_probability": 40.0,
        "predicted_winner": "A", "predicted_method": "Decision",
        "method_model_version": "Method V1.0",
    }
    replacement = {**first, "fighter_a_probability": 10.0, "predicted_winner": "B"}
    original, created = history.lock_prediction(event, fight, first)
    locked, replaced = history.lock_prediction(event, fight, replacement)
    assert created is True
    assert replaced is False
    assert locked == original
    assert locked["fighter_a_probability"] == 60.0
    assert locked["actual_winner"] is None
    assert locked["correct_winner"] is None


def test_known_event_fighters_receive_and_lock_automatic_prediction(tmp_path, monkeypatch):
    future = (date.today() + timedelta(days=30)).isoformat()
    path = tmp_path / "events.json"
    _write_events(path, [{
        "event_id": "known", "event_name": "Verified Card", "event_date": future,
        "fights": [{
            "fight_id": "main", "fighter_a": "Islam Makhachev",
            "fighter_b": "Ilia Topuria", "weight_class": "Lightweight",
            "number_of_rounds": 5, "card_section": "Main Event",
        }],
    }])
    history = PredictionHistory(tmp_path / "history.json")
    monkeypatch.setattr(web, "EVENT_SOURCE", LocalJsonEventSource(path))
    monkeypatch.setattr(web, "PREDICTION_HISTORY", history)
    fight = web.event_detail("known")["event"]["fights"][0]
    assert fight["prediction_status"] == "locked"
    assert fight["prediction"]["model_version"] == "V2.1-accuracy"
    assert fight["prediction"]["method_context"].startswith("5-round")
    assert len(history.list_predictions()) == 1
    assert web.event_detail("known")["event"]["fights"][0]["prediction"] == fight["prediction"]


def test_frontend_has_route_views_and_mobile_layout():
    html = web.INDEX_PATH.read_text(encoding="utf-8")
    for route in ('href="/events"', 'href="/predict"', 'href="/results"'):
        assert route in html
    assert "@media(max-width:760px)" in html
    assert "item.explanation" in html
    assert "log_odds_contribution" not in html
