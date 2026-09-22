"""Upcoming schedule only: never import records, odds, scores or fight stats."""

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from threading import Lock
import time
from urllib.request import Request, urlopen

from .base import Event, EventSource

URL = 'https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard'


def scheduled(item):
    status = item.get('status', {}).get('type', {})
    return status.get('state') == 'pre' and status.get('name') == 'STATUS_SCHEDULED' and not status.get('completed')


def parse_schedule(payload, today):
    events = []
    for raw in payload['events']:
        if not raw.get('name', '').startswith('UFC ') or not scheduled(raw):
            continue
        start = datetime.fromisoformat(raw['date'].replace('Z', '+00:00'))
        if start.date() < today:
            continue
        fights = []
        competitions = raw.get('competitions', [])
        for bout in reversed(competitions):
            if not scheduled(bout):
                continue
            competitors = sorted(bout.get('competitors', []), key=lambda x: x.get('order', 0))
            names = [x.get('athlete', {}).get('displayName', '').strip() for x in competitors]
            rounds = bout.get('format', {}).get('regulation', {}).get('periods')
            if len(names) != 2 or any(not n or 'tba' in n.lower() or 'tbd' in n.lower() for n in names) or rounds not in (3, 5):
                continue
            division = bout.get('type', {}).get('abbreviation')
            if not division:
                continue
            if division.startswith('W '):
                division = "Women's " + division[2:]
            # ESPN bout IDs can survive replacement opponents and rescheduling.
            revision = hashlib.sha256(json.dumps([names, raw['date'], division, rounds]).encode()).hexdigest()[:16]
            fights.append(dict(fight_id=f"espn-{bout['id']}-{revision}",
                               fighter_a_espn_id=competitors[0].get('id'),
                               fighter_b_espn_id=competitors[1].get('id'),
                               fighter_a=names[0], fighter_b=names[1], weight_class=division,
                               number_of_rounds=rounds, bout_order=len(fights) + 1,
                               card_section='Scheduled bout'))
        venue = (competitions[0].get('venue', {}) if competitions else raw.get('venue', {}))
        address = venue.get('address', {})
        location = ', '.join(str(x) for x in [venue.get('fullName'), address.get('city'), address.get('country')] if x)
        events.append(Event.from_dict(dict(event_id=f"espn-{raw['id']}", event_name=raw['name'],
                                          event_date=start.date().isoformat(), location=location,
                                          starts_at=start.isoformat(),
                                          source_url=f"https://www.espn.com/mma/fightcenter/_/id/{raw['id']}", fights=fights)))
    return sorted(events, key=lambda e: (e.event_date, e.event_id))


class EspnEventSource(EventSource):
    source_name = 'ESPN UFC schedule'

    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()
        self._retry_after = 0.0
        self.last_error = None
        self.fetched_at = None

    def _read(self):
        try:
            return json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {}

    def upcoming_events(self, as_of=None):
        today = as_of or date.today()
        with self._lock:
            cache = self._read()
            age = time.time() - cache.get('fetched_epoch', 0)
            if age > 3600 and time.monotonic() >= self._retry_after:
                try:
                    end = today + timedelta(days=120)
                    url = f'{URL}?dates={today:%Y%m%d}-{end:%Y%m%d}&limit=1000'
                    request = Request(url, headers={'User-Agent': 'UFC-Predictor/1.0', 'Accept': 'application/json'})
                    with urlopen(request, timeout=15) as response:
                        payload = json.load(response)
                    if not isinstance(payload.get('events'), list):
                        raise ValueError('Schedule response has no events list')
                    events = parse_schedule(payload, today)
                    cache = dict(fetched_epoch=time.time(), fetched_at=datetime.now(timezone.utc).isoformat(),
                                 events=[e.to_dict() for e in events])
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = self.path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(cache, indent=2), encoding='utf-8')
                    temporary.replace(self.path)
                    self.last_error = None
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self.last_error = 'Schedule refresh unavailable; showing the last saved schedule.'
                    self._retry_after = time.monotonic() + 300
            self.fetched_at = cache.get('fetched_at')
            return [Event.from_dict(e) for e in cache.get('events', []) if e['event_date'] >= today.isoformat()]

    def can_publish(self):
        # Do not lock new picks against a card that could no longer be current.
        return self.fetched_at is not None and self.last_error is None and (
            datetime.now(timezone.utc) - datetime.fromisoformat(self.fetched_at)
        ).total_seconds() <= 3600
