"""Validated local JSON event source used when no trustworthy feed is configured."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path

from .base import Event, EventSource


class LocalJsonEventSource(EventSource):
    source_name = "Local verified JSON"

    def __init__(self, path: Path):
        self.path = path

    def all_events(self) -> list[Event]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        raw_events = payload.get("events", []) if isinstance(payload, dict) else payload
        events = [Event.from_dict(item) for item in raw_events]
        ids = [event.event_id for event in events]
        if len(ids) != len(set(ids)):
            raise ValueError("Upcoming event IDs must be unique.")
        return sorted(events, key=lambda event: (event.event_date, event.event_id))

    def upcoming_events(self, as_of: date | None = None) -> list[Event]:
        cutoff = (as_of or date.today()).isoformat()
        return [event for event in self.all_events() if event.event_date >= cutoff]

