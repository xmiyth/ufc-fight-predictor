"""Single configuration point for the active upcoming-event source."""

from pathlib import Path
import os

from .base import EventSource
from .local_json import LocalJsonEventSource
from .espn import EspnEventSource


def create_event_source(project_root: Path) -> EventSource:
    if os.environ.get("UFC_EVENT_SOURCE") == "local":
        return LocalJsonEventSource(project_root / "data" / "upcoming_events.json")
    return EspnEventSource(project_root / "data" / "upcoming_espn.json")
