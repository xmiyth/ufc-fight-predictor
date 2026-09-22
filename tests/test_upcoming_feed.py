import copy
from datetime import date, timedelta
import json

from ufc_predictor.event_sources.espn import EspnEventSource, parse_schedule
from ufc_predictor.event_sources.base import Event
from ufc_predictor.prediction_history import PredictionHistory
from ufc_predictor import web


def feed():
    status = {'type': {'state': 'pre', 'name': 'STATUS_SCHEDULED', 'completed': False}}
    return {'events': [{'id': '1', 'name': 'UFC Fight Night: Test', 'date': '2030-01-10T22:00Z',
                        'status': status, 'competitions': [{'id': '2', 'status': status,
                        'type': {'abbreviation': 'W Flyweight'}, 'format': {'regulation': {'periods': 5}},
                        'competitors': [{'order': 1, 'athlete': {'displayName': 'A'}},
                                        {'order': 2, 'athlete': {'displayName': 'B'}}]}]}]}


def test_schedule_allowlist_ignores_outcomes_odds_and_statistics():
    payload = feed()
    expected = parse_schedule(payload, date(2030, 1, 1))
    bout = payload['events'][0]['competitions'][0]
    bout.update(odds=[{'value': 999}], statistics={'postfight': 123}, score=42)
    bout['competitors'][0].update(winner=True, records=[{'summary': '999-0'}], statistics=[999])
    assert parse_schedule(payload, date(2030, 1, 1)) == expected
    assert expected[0].fights[0].weight_class == "Women's Flyweight"
    serialized = json.dumps(expected[0].to_dict())
    assert 'statistics' not in serialized and 'winner' not in serialized and 'records' not in serialized


def test_replacements_and_rescheduling_receive_different_prediction_identity():
    payload = feed()
    original = parse_schedule(payload, date(2030, 1, 1))[0].fights[0].fight_id
    replaced = copy.deepcopy(payload)
    replaced['events'][0]['competitions'][0]['competitors'][1]['athlete']['displayName'] = 'C'
    assert parse_schedule(replaced, date(2030, 1, 1))[0].fights[0].fight_id != original
    payload['events'][0]['date'] = '2030-01-11T22:00Z'
    assert parse_schedule(payload, date(2030, 1, 1))[0].fights[0].fight_id != original


def test_live_completed_cancelled_and_non_ufc_events_are_excluded():
    for state, name in [('in', 'STATUS_IN_PROGRESS'), ('post', 'STATUS_FINAL'), ('pre', 'STATUS_CANCELED')]:
        payload = feed()
        payload['events'][0]['status']['type'].update(state=state, name=name)
        assert parse_schedule(payload, date(2030, 1, 1)) == []
    payload = feed()
    payload['events'][0]['name'] = "Dana White's Contender Series"
    assert parse_schedule(payload, date(2030, 1, 1)) == []


def test_stale_cache_is_visible_but_cannot_publish_new_picks(tmp_path, monkeypatch):
    cache = tmp_path / 'cache.json'
    cache.write_text(json.dumps({'fetched_epoch': 0, 'fetched_at': '2020-01-01T00:00:00+00:00',
                                'events': [e.to_dict() for e in parse_schedule(feed(), date(2030, 1, 1))]}))
    def unavailable(*args, **kwargs):
        raise OSError('offline')
    monkeypatch.setattr('ufc_predictor.event_sources.espn.urlopen', unavailable)
    source = EspnEventSource(cache)
    assert len(source.upcoming_events(date(2030, 1, 1))) == 1
    assert source.last_error and not source.can_publish()


def test_same_day_event_cannot_get_a_new_pre_fight_prediction(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'PREDICTION_HISTORY', PredictionHistory(tmp_path / 'history.json'))
    event = Event.from_dict({'event_id': 'today', 'event_name': 'UFC Test', 'event_date': date.today().isoformat(),
                            'fights': [{'fighter_a': 'Islam Makhachev', 'fighter_b': 'Ilia Topuria'}]})
    result = web._event_fight(event, event.fights[0])
    assert result['prediction_status'] == 'unavailable'
    assert web.PREDICTION_HISTORY.list_predictions() == []


def test_future_data_cutoff_blocks_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'PREDICTION_HISTORY', PredictionHistory(tmp_path / 'history.json'))
    future = (date.today() + timedelta(days=30)).isoformat()
    monkeypatch.setitem(web.ARTIFACT, 'trained_through', future)
    event = Event.from_dict({'event_id': 'future', 'event_name': 'UFC Test', 'event_date': future,
                            'fights': [{'fighter_a': 'Islam Makhachev', 'fighter_b': 'Ilia Topuria'}]})
    assert 'cutoffs' in web._event_fight(event, event.fights[0])['prediction_unavailable_reason']
    assert web.PREDICTION_HISTORY.list_predictions() == []
