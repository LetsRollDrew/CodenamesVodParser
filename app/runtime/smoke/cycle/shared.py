"""Shared clue-matching helpers for smoke cycle parsing"""

from __future__ import annotations

from typing import Sequence

from app.parsers.gamelog import ClueEvent, GuessEvent


def _normalize_clue_text(value: str | None) -> str:
    return "".join(character for character in str(value or "").upper() if character.isalpha())


def _clue_signature_strict_match(
    left_text: str | None,
    left_count: str | None,
    right_text: str | None,
    right_count: str | None,
) -> bool:
    normalized_left = _normalize_clue_text(left_text)
    normalized_right = _normalize_clue_text(right_text)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left != normalized_right:
        return False
    normalized_left_count = str(left_count or "").casefold()
    normalized_right_count = str(right_count or "").casefold()
    if not normalized_left_count or not normalized_right_count:
        return True
    return normalized_left_count == normalized_right_count


def _latest_clue_event(events: Sequence[ClueEvent | GuessEvent]) -> ClueEvent | None:
    for event in reversed(events):
        if isinstance(event, ClueEvent):
            return event
    return None
