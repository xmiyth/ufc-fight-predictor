"""Swappable upcoming-event data sources."""

from .base import Event, EventFight, EventSource
from .current_source import create_event_source

__all__ = ["Event", "EventFight", "EventSource", "create_event_source"]
