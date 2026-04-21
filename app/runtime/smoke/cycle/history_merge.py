"""Visible-history merge helpers for smoke cycle parsing"""

from __future__ import annotations

from difflib import SequenceMatcher

from app.core.models import CardColor
from app.parsers.gamelog import ClueEvent, GuessEvent

from .shared import _clue_signature_strict_match, _normalize_clue_text


def _append_new_visible_events(
    history: list[ClueEvent | GuessEvent],
    previous_visible: list[ClueEvent | GuessEvent],
    current_visible: list[ClueEvent | GuessEvent],
) -> tuple[list[ClueEvent | GuessEvent], list[ClueEvent | GuessEvent], list[ClueEvent | GuessEvent]]:
    if not current_visible:
        return history, list(previous_visible), []
    if _is_fully_historical_visible_slice(history, current_visible):
        return history, list(current_visible), []
    current_visible = _trim_historical_visible_prefix(history, current_visible)
    if not current_visible:
        return history, list(previous_visible), []
    current_visible = _drop_stale_leading_clue_label(history, current_visible)
    if not current_visible:
        return history, list(previous_visible), []
    stale_merge = _merge_stale_visible_turn_residue(history, current_visible)
    if stale_merge is not None:
        return stale_merge

    overlap = 0
    max_overlap = min(len(previous_visible), len(current_visible))
    for candidate in range(max_overlap, -1, -1):
        if _event_slices_equivalent(previous_visible[-candidate:], current_visible[:candidate]):
            overlap = candidate
            break

    history_updated = list(history)
    new_events: list[ClueEvent | GuessEvent] = []
    overlap_slice = current_visible[:overlap]
    for index, event in enumerate(overlap_slice):
        if isinstance(event, ClueEvent):
            replacement_index = _find_matching_visible_clue_index(
                history_updated,
                event,
                has_following_visible_guesses=any(
                    isinstance(remaining_event, GuessEvent)
                    for remaining_event in overlap_slice[index + 1 :]
                ),
            )
            if replacement_index is None:
                continue
            existing = history_updated[replacement_index]
            if isinstance(existing, ClueEvent) and _clue_event_preferred(event, existing):
                history_updated[replacement_index] = _merge_clue_event(existing, event)
            continue
        if isinstance(event, GuessEvent):
            replacement_index = _find_matching_guess_index(history_updated, event)
            if replacement_index is None:
                continue
            existing = history_updated[replacement_index]
            if isinstance(existing, GuessEvent) and _guess_event_preferred(event, existing):
                history_updated[replacement_index] = _merge_guess_event(existing, event)
    new_slice = current_visible[overlap:]
    for index, event in enumerate(new_slice):
        if isinstance(event, ClueEvent):
            replacement_index = _find_matching_visible_clue_index(
                history_updated,
                event,
                has_following_visible_guesses=any(
                    isinstance(remaining_event, GuessEvent)
                    for remaining_event in new_slice[index + 1 :]
                ),
            )
            if replacement_index is not None:
                existing = history_updated[replacement_index]
                if isinstance(existing, ClueEvent) and _clue_event_preferred(event, existing):
                    history_updated[replacement_index] = _merge_clue_event(existing, event)
            continue
        if isinstance(event, GuessEvent):
            replacement_index = _find_matching_guess_index(history_updated, event)
            if replacement_index is not None:
                existing = history_updated[replacement_index]
                if isinstance(existing, GuessEvent) and _guess_event_preferred(event, existing):
                    history_updated[replacement_index] = _merge_guess_event(existing, event)
                continue
        history_updated.append(event)
        new_events.append(event)
    return history_updated, list(current_visible), new_events


def _event_signature(event: ClueEvent | GuessEvent) -> tuple[str, str, str, str]:
    if isinstance(event, ClueEvent):
        return (
            "clue",
            event.team_color.value,
            _normalize_clue_text(event.clue_text),
            str(event.clue_count or "").casefold(),
        )
    return ("guess", event.player_name, event.word, event.card_color.value)


def _should_replace_locked_board_state(
    *,
    current_board_state: object | None,
    candidate_board_state: object | None,
    board_locked: bool,
) -> bool:
    if candidate_board_state is None:
        return False
    if current_board_state is None:
        return True

    candidate_quality = getattr(candidate_board_state, "quality", None)
    current_quality = getattr(current_board_state, "quality", None)
    if candidate_quality is None or current_quality is None:
        from app.runtime.smoke.observations import _board_quality

        candidate_quality = _board_quality(candidate_board_state)
        current_quality = _board_quality(current_board_state)
    if not board_locked:
        return candidate_quality >= current_quality
    if candidate_quality < current_quality + 20:
        return False
    return _board_alpha_cell_count(candidate_board_state) >= _board_alpha_cell_count(current_board_state)


def _board_alpha_cell_count(board_state: object | None) -> int:
    return sum(
        1
        for cell in getattr(board_state, "cells", [])
        if str(getattr(cell, "word", "")).strip().isalpha() and len(str(getattr(cell, "word", "")).strip()) >= 3
    )


def _event_slices_equivalent(
    previous_slice: list[ClueEvent | GuessEvent],
    current_slice: list[ClueEvent | GuessEvent],
) -> bool:
    if len(previous_slice) != len(current_slice):
        return False
    return all(_events_equivalent(previous, current) for previous, current in zip(previous_slice, current_slice))


def _events_equivalent(previous: ClueEvent | GuessEvent, current: ClueEvent | GuessEvent) -> bool:
    if isinstance(previous, ClueEvent) and isinstance(current, ClueEvent):
        return _clue_events_equivalent(previous, current)
    if isinstance(previous, GuessEvent) and isinstance(current, GuessEvent):
        return _guess_events_equivalent(previous, current)
    return False


def _clue_events_equivalent(previous: ClueEvent, current: ClueEvent) -> bool:
    if previous.team_color != current.team_color:
        return False
    previous_text = _normalize_clue_text(previous.clue_text)
    current_text = _normalize_clue_text(current.clue_text)
    if _clue_signature_strict_match(previous_text, previous.clue_count, current_text, current.clue_count):
        return True
    if previous.clue_count == current.clue_count and (
        previous_text in current_text or current_text in previous_text
    ):
        return True
    if min(len(previous_text), len(current_text)) >= 5 and (
        previous_text in current_text or current_text in previous_text
    ):
        return True
    similarity = SequenceMatcher(a=previous_text, b=current_text).ratio()
    if previous.clue_count == current.clue_count and similarity >= 0.55:
        return True
    if similarity >= 0.9:
        return True
    return False


def _find_matching_clue_index(
    history: list[ClueEvent | GuessEvent],
    candidate: ClueEvent,
) -> int | None:
    for index in range(len(history) - 1, -1, -1):
        event = history[index]
        if not isinstance(event, ClueEvent):
            continue
        if event.team_color != candidate.team_color:
            break
        if candidate.timestamp_sec - event.timestamp_sec > 240.0:
            break
        if _clue_events_equivalent(event, candidate):
            return index
    return None


def _find_matching_clue_index_anywhere(
    history: list[ClueEvent | GuessEvent],
    candidate: ClueEvent,
    *,
    max_age_sec: float = 600.0,
) -> int | None:
    for index in range(len(history) - 1, -1, -1):
        event = history[index]
        if not isinstance(event, ClueEvent):
            continue
        if candidate.timestamp_sec - event.timestamp_sec > max_age_sec:
            break
        if _clue_events_equivalent(event, candidate):
            return index
    return None


def _find_matching_visible_clue_index(
    history: list[ClueEvent | GuessEvent],
    candidate: ClueEvent,
    *,
    has_following_visible_guesses: bool,
) -> int | None:
    direct_match = _find_matching_clue_index(history, candidate)
    if direct_match is not None:
        return direct_match
    prior_match = _find_matching_clue_index_anywhere(history, candidate, max_age_sec=1200.0)
    if prior_match is None:
        return None
    if _find_next_clue_index(history, prior_match) is None:
        return None
    return prior_match


def _find_next_clue_index(history: list[ClueEvent | GuessEvent], start_index: int) -> int | None:
    for index in range(start_index + 1, len(history)):
        if isinstance(history[index], ClueEvent):
            return index
    return None


def _is_stale_banner_residue(
    history: list[ClueEvent | GuessEvent],
    candidate: ClueEvent,
) -> bool:
    matching_index = _find_matching_clue_index_anywhere(history, candidate, max_age_sec=1200.0)
    if matching_index is None:
        return False
    next_clue_index = _find_next_clue_index(history, matching_index)
    if next_clue_index is None:
        return False
    return _is_recent_stale_residue_candidate(
        history,
        matching_clue_index=matching_index,
        next_clue_index=next_clue_index,
        timestamp_sec=candidate.timestamp_sec,
    )


def _merge_stale_visible_turn_residue(
    history: list[ClueEvent | GuessEvent],
    current_visible: list[ClueEvent | GuessEvent],
) -> tuple[list[ClueEvent | GuessEvent], list[ClueEvent | GuessEvent], list[ClueEvent | GuessEvent]] | None:
    leading_event = current_visible[0]
    if not isinstance(leading_event, ClueEvent):
        return None

    target_clue_index = _find_matching_clue_index_anywhere(history, leading_event)
    if target_clue_index is None:
        return None
    next_clue_index = _find_next_clue_index(history, target_clue_index)
    if next_clue_index is None:
        return None
    if not _is_recent_stale_residue_candidate(
        history,
        matching_clue_index=target_clue_index,
        next_clue_index=next_clue_index,
        timestamp_sec=leading_event.timestamp_sec,
    ):
        return None

    history_updated = list(history)
    insertion_index = next_clue_index
    new_events: list[ClueEvent | GuessEvent] = []
    for event in current_visible[1:]:
        if not isinstance(event, GuessEvent):
            continue
        replacement_index = _find_matching_guess_index(history_updated, event)
        if replacement_index is not None:
            existing = history_updated[replacement_index]
            if isinstance(existing, GuessEvent) and _guess_event_preferred(event, existing):
                history_updated[replacement_index] = _merge_guess_event(existing, event)
            continue
        history_updated.insert(insertion_index, event)
        new_events.append(event)
        insertion_index += 1
    return history_updated, list(current_visible), new_events


def _is_fully_historical_visible_slice(
    history: list[ClueEvent | GuessEvent],
    current_visible: list[ClueEvent | GuessEvent],
) -> bool:
    if len(current_visible) < 2:
        return False
    leading_event = current_visible[0]
    if not isinstance(leading_event, ClueEvent):
        return False

    matching_clue_index = _find_matching_clue_index_anywhere(history, leading_event, max_age_sec=1200.0)
    if matching_clue_index is None:
        return False
    if _find_next_clue_index(history, matching_clue_index) is None:
        return False

    saw_guess = False
    for event in current_visible[1:]:
        if not isinstance(event, GuessEvent):
            continue
        saw_guess = True
        if _find_matching_guess_index_anywhere(history, event) is None:
            return False
    return saw_guess


def _trim_historical_visible_prefix(
    history: list[ClueEvent | GuessEvent],
    current_visible: list[ClueEvent | GuessEvent],
) -> list[ClueEvent | GuessEvent]:
    if not history or len(current_visible) < 2:
        return list(current_visible)

    matched_history_index = -1
    matched_prefix_length = 0
    for event in current_visible:
        next_index = _find_next_equivalent_history_index(history, event, after_index=matched_history_index)
        if next_index is None:
            break
        matched_history_index = next_index
        matched_prefix_length += 1

    if matched_prefix_length == 0:
        return list(current_visible)
    if matched_prefix_length >= len(current_visible):
        return []

    suffix = current_visible[matched_prefix_length:]
    if not suffix or not isinstance(suffix[0], ClueEvent):
        return list(current_visible)

    return list(suffix)


def _find_next_equivalent_history_index(
    history: list[ClueEvent | GuessEvent],
    candidate: ClueEvent | GuessEvent,
    *,
    after_index: int,
) -> int | None:
    for index in range(after_index + 1, len(history)):
        if _events_equivalent(history[index], candidate):
            return index
    return None


def _find_matching_guess_index(
    history: list[ClueEvent | GuessEvent],
    candidate: GuessEvent,
) -> int | None:
    seen_current_turn = False
    for index in range(len(history) - 1, -1, -1):
        event = history[index]
        if isinstance(event, ClueEvent):
            if seen_current_turn:
                break
            seen_current_turn = True
            continue
        if candidate.timestamp_sec - event.timestamp_sec > 20.0 and seen_current_turn:
            break
        if not isinstance(event, GuessEvent):
            continue
        if event.word != candidate.word:
            continue
        same_word_color_upgrade = (
            event.card_color != candidate.card_color
            and CardColor.NEUTRAL in {event.card_color, candidate.card_color}
        )
        if not same_word_color_upgrade and abs(candidate.timestamp_sec - event.timestamp_sec) > 20.0:
            continue
        if (
            event.player_name != "unknown"
            and candidate.player_name != "unknown"
            and event.player_name != candidate.player_name
        ):
            continue
        return index
    return None


def _find_matching_guess_index_anywhere(
    history: list[ClueEvent | GuessEvent],
    candidate: GuessEvent,
    *,
    max_age_sec: float = 1200.0,
) -> int | None:
    for index in range(len(history) - 1, -1, -1):
        event = history[index]
        if not isinstance(event, GuessEvent):
            continue
        if candidate.timestamp_sec - event.timestamp_sec > max_age_sec:
            break
        if event.word != candidate.word:
            continue
        same_word_color_upgrade = (
            event.card_color != candidate.card_color
            and CardColor.NEUTRAL in {event.card_color, candidate.card_color}
        )
        if event.card_color != candidate.card_color and not same_word_color_upgrade:
            continue
        if (
            event.player_name != "unknown"
            and candidate.player_name != "unknown"
            and event.player_name != candidate.player_name
        ):
            continue
        return index
    return None


def _guess_events_equivalent(previous: GuessEvent, current: GuessEvent) -> bool:
    if previous.word != current.word or previous.card_color != current.card_color:
        return False
    if abs(current.timestamp_sec - previous.timestamp_sec) > 20.0:
        return False
    if previous.player_name == "unknown" or current.player_name == "unknown":
        return True
    return previous.player_name == current.player_name


def _is_recent_stale_residue_candidate(
    history: list[ClueEvent | GuessEvent],
    *,
    matching_clue_index: int,
    next_clue_index: int,
    timestamp_sec: float,
) -> bool:
    matching_clue = history[matching_clue_index]
    next_clue = history[next_clue_index]
    if not isinstance(matching_clue, ClueEvent) or not isinstance(next_clue, ClueEvent):
        return False
    if timestamp_sec - matching_clue.timestamp_sec > 150.0:
        return False
    if timestamp_sec - next_clue.timestamp_sec > 45.0:
        return False
    return True


def _drop_stale_leading_clue_label(
    history: list[ClueEvent | GuessEvent],
    current_visible: list[ClueEvent | GuessEvent],
) -> list[ClueEvent | GuessEvent]:
    if len(current_visible) < 2:
        return list(current_visible)
    leading_event = current_visible[0]
    if not isinstance(leading_event, ClueEvent):
        return list(current_visible)
    if not any(isinstance(event, GuessEvent) for event in current_visible[1:]):
        return list(current_visible)
    matching_index = _find_matching_clue_index_anywhere(history, leading_event, max_age_sec=1200.0)
    if matching_index is None:
        return list(current_visible)
    next_clue_index = _find_next_clue_index(history, matching_index)
    if next_clue_index is None:
        return list(current_visible)
    matching_clue = history[matching_index]
    if not isinstance(matching_clue, ClueEvent):
        return list(current_visible)
    if leading_event.timestamp_sec - matching_clue.timestamp_sec <= 150.0:
        return list(current_visible)
    return list(current_visible[1:])


def _guess_event_preferred(candidate: GuessEvent, existing: GuessEvent) -> bool:
    if existing.card_color != candidate.card_color:
        if existing.card_color.value == "neutral" and candidate.card_color.value != "neutral":
            return True
        if candidate.card_color.value == "neutral" and existing.card_color.value != "neutral":
            return False
        return False
    if existing.player_name == "unknown" and candidate.player_name != "unknown":
        return True
    if candidate.confidence > existing.confidence + 0.02:
        return True
    return candidate.timestamp_sec > existing.timestamp_sec and candidate.confidence >= existing.confidence


def _merge_guess_event(existing: GuessEvent, candidate: GuessEvent) -> GuessEvent:
    return candidate.model_copy(
        update={
            "timestamp_sec": existing.timestamp_sec,
            "sequence_index": existing.sequence_index,
        }
    )


def _clue_event_preferred(candidate: ClueEvent, existing: ClueEvent) -> bool:
    if existing.spymaster_name == "unknown" and candidate.spymaster_name != "unknown":
        return True
    if _clue_count_quality(candidate.clue_count) > _clue_count_quality(existing.clue_count):
        return True
    if len(candidate.clue_text) > len(existing.clue_text) and candidate.confidence >= existing.confidence - 0.05:
        return True
    if candidate.confidence > existing.confidence + 0.03:
        return True
    return candidate.timestamp_sec > existing.timestamp_sec and candidate.confidence >= existing.confidence


def _merge_clue_event(existing: ClueEvent, candidate: ClueEvent) -> ClueEvent:
    chosen_text = candidate.clue_text
    existing_text = existing.clue_text.upper()
    candidate_text = candidate.clue_text.upper()
    if len(existing_text) > len(candidate_text) and (
        candidate_text in existing_text or SequenceMatcher(a=existing_text, b=candidate_text).ratio() >= 0.55
    ):
        chosen_text = existing.clue_text

    chosen_count = existing.clue_count
    existing_count_quality = _clue_count_quality(existing.clue_count)
    candidate_count_quality = _clue_count_quality(candidate.clue_count)
    if candidate_count_quality > existing_count_quality:
        existing_count_normalized = str(existing.clue_count or "").casefold()
        if not (
            str(candidate.clue_count or "").casefold() == "infinity"
            and existing_count_normalized.isdigit()
            and int(existing_count_normalized) <= 4
        ):
            chosen_count = candidate.clue_count
    elif candidate_count_quality == existing_count_quality:
        if existing_count_quality <= 1:
            chosen_count = candidate.clue_count
        elif (
            candidate.clue_count != existing.clue_count
            and candidate.confidence > existing.confidence + 0.12
            and _should_prefer_candidate_clue_count(existing, candidate)
        ):
            chosen_count = candidate.clue_count

    chosen_spymaster = candidate.spymaster_name
    if existing.spymaster_name != "unknown" and candidate.spymaster_name == "unknown":
        chosen_spymaster = existing.spymaster_name

    chosen_team = candidate.team_color
    if existing.spymaster_name != "unknown" and candidate.spymaster_name == "unknown":
        chosen_team = existing.team_color

    return candidate.model_copy(
        update={
            "team_color": chosen_team,
            "spymaster_name": chosen_spymaster,
            "clue_text": chosen_text,
            "clue_count": chosen_count,
            "confidence": max(existing.confidence, candidate.confidence),
            "timestamp_sec": existing.timestamp_sec,
            "sequence_index": existing.sequence_index,
        }
    )


def _clue_count_quality(value: str) -> int:
    normalized = str(value or "").casefold()
    if normalized == "infinity":
        return 3
    if normalized.isdigit() and normalized != "0":
        return 2
    if normalized == "0":
        return 0
    return 1


def _should_prefer_candidate_clue_count(existing: ClueEvent, candidate: ClueEvent) -> bool:
    existing_text = existing.clue_text.upper()
    candidate_text = candidate.clue_text.upper()
    if not existing_text or not candidate_text:
        return False
    if existing_text == candidate_text:
        return True
    if existing_text in candidate_text or candidate_text in existing_text:
        return True
    return SequenceMatcher(a=existing_text, b=candidate_text).ratio() >= 0.9
