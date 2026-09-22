"""Conservative event-name matching against frozen-model fighter identities."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


def canonical_fighter_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).casefold())
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", ascii_text).split())


@dataclass(frozen=True)
class FighterNameMatch:
    requested_name: str
    matched_name: str | None
    matched_key: str | None
    status: str


class FighterNameMatcher:
    def __init__(self, identities: Iterable[tuple[str, str]]):
        self._lookup: dict[str, list[tuple[str, str]]] = {}
        for key, display_name in identities:
            self._lookup.setdefault(canonical_fighter_name(display_name), []).append(
                (key, display_name)
            )

    def match(self, value: str) -> FighterNameMatch:
        matches = self._lookup.get(canonical_fighter_name(value), [])
        if len(matches) == 1:
            key, display = matches[0]
            return FighterNameMatch(value, display, key, "matched")
        if len(matches) > 1:
            return FighterNameMatch(value, None, None, "ambiguous")
        return FighterNameMatch(value, None, None, "not_found")

