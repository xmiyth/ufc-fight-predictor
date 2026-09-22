"""Source-independent upcoming UFC event contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class EventFight:
    fight_id: str
    fighter_a: str
    fighter_b: str
    weight_class: str | None = None
    number_of_rounds: int = 3
    bout_order: int | None = None
    card_section: str = "Main Card"
    fighter_a_espn_id: str | None = None
    fighter_b_espn_id: str | None = None

    @classmethod
    def from_dict(cls, values: dict[str, Any], index: int) -> "EventFight":
        fighter_a = str(values.get("fighter_a", "")).strip()
        fighter_b = str(values.get("fighter_b", "")).strip()
        if not fighter_a or not fighter_b:
            raise ValueError("Every event fight requires fighter_a and fighter_b.")
        rounds = int(values.get("number_of_rounds") or 3)
        if rounds not in {3, 5}:
            raise ValueError("number_of_rounds must be 3 or 5.")
        return cls(
            fight_id=str(values.get("fight_id") or f"fight-{index + 1}").strip(),
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            weight_class=(str(values["weight_class"]).strip() if values.get("weight_class") else None),
            number_of_rounds=rounds,
            bout_order=(int(values["bout_order"]) if values.get("bout_order") is not None else None),
            card_section=str(values.get("card_section") or "Main Card").strip(),
            fighter_a_espn_id=values.get("fighter_a_espn_id"),
            fighter_b_espn_id=values.get("fighter_b_espn_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Event:
    event_id: str
    event_name: str
    event_date: str
    location: str | None
    fights: tuple[EventFight, ...]
    starts_at: str | None = None
    source_url: str | None = None

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Event":
        event_id = str(values.get("event_id", "")).strip()
        event_name = str(values.get("event_name", "")).strip()
        event_date = str(values.get("event_date", "")).strip()
        if not event_id or not event_name or not event_date:
            raise ValueError("Each event requires event_id, event_name, and event_date.")
        date.fromisoformat(event_date)
        parsed_fights = [
            EventFight.from_dict(fight, index)
            for index, fight in enumerate(values.get("fights", []))
        ]
        fights = tuple(
            sorted(
                parsed_fights,
                key=lambda fight: (
                    fight.bout_order is None,
                    fight.bout_order if fight.bout_order is not None else 10_000,
                ),
            )
        )
        fight_ids = [fight.fight_id for fight in fights]
        if len(fight_ids) != len(set(fight_ids)):
            raise ValueError(f"Duplicate fight_id in event {event_id}.")
        return cls(
            event_id=event_id,
            event_name=event_name,
            event_date=event_date,
            location=(str(values["location"]).strip() if values.get("location") else None),
            fights=fights,
            starts_at=values.get("starts_at"),
            source_url=values.get("source_url"),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["fights"] = [fight.to_dict() for fight in self.fights]
        return result


class EventSource(ABC):
    """Adapter interface; website code never depends on a particular feed."""

    source_name: str

    @abstractmethod
    def upcoming_events(self, as_of: date | None = None) -> list[Event]:
        """Return known events on or after ``as_of`` in chronological order."""
