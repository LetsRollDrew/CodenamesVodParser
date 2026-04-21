"""Visible-log filtering and commit helpers for smoke cycle parsing"""

from __future__ import annotations

from typing import Sequence

from app.parsers.banner import _banner_signatures_equivalent
from app.parsers.gamelog import ClueEvent, GuessEvent
from app.runtime.smoke.runtime_models import BannerObservation, SmokeRuntimeState

from .clue_window import (
    _classify_frame_phase,
    _should_suppress_clue_candidate_for_phase,
    _should_suppress_same_team_clue_candidate,
    _visible_clue_matches_active_window,
    _visible_clue_matches_banner,
)
from .history_merge import _append_new_visible_events, _event_signature
from .shared import _latest_clue_event


def _should_probe_visible_log_from_state(
    runtime_state: SmokeRuntimeState,
    *,
    log_changed: bool,
    banner_observation: BannerObservation,
) -> bool:
    if log_changed:
        return True
    if not banner_observation.clue_text:
        return False
    if runtime_state.visible_log.quiet_log_frame_count < 2:
        return False

    latest_visible_clue = _latest_clue_event(runtime_state.visible_log.previous_visible_events)
    if latest_visible_clue is None:
        return True
    if not _visible_clue_matches_banner(latest_visible_clue, banner_observation):
        return True

    active_window = runtime_state.active_clue_window
    if active_window is None:
        return True
    if not _banner_signatures_equivalent(active_window.signature, banner_observation.signature):
        return True

    return (
        len(runtime_state.visible_log.previous_visible_events) <= 1
        and runtime_state.visible_log.quiet_log_frame_count >= 4
    )


def _filter_visible_log_events_for_runtime(
    runtime_state: SmokeRuntimeState,
    events: Sequence[ClueEvent | GuessEvent],
    *,
    banner_observation: BannerObservation,
) -> list[ClueEvent | GuessEvent]:
    normalized_events = list(events)
    if not normalized_events:
        return []
    frame_phase = _classify_frame_phase(banner_observation.top_banner_text)

    latest_clue_index = max(
        (index for index, event in enumerate(normalized_events) if isinstance(event, ClueEvent)),
        default=-1,
    )
    if latest_clue_index < 0:
        return normalized_events

    clue_events = [
        event
        for event in normalized_events
        if isinstance(event, ClueEvent)
    ]
    if banner_observation.clue_text and not any(
        _visible_clue_matches_banner(event, banner_observation) for event in clue_events
    ):
        active_window = runtime_state.active_clue_window
        if active_window is None or not any(
            _visible_clue_matches_active_window(event, active_window) for event in clue_events
        ):
            return []

    latest_clue = normalized_events[latest_clue_index]
    assert isinstance(latest_clue, ClueEvent)
    has_following_guesses = any(
        isinstance(event, GuessEvent) for event in normalized_events[latest_clue_index + 1 :]
    )
    if _should_suppress_clue_candidate_for_phase(
        runtime_state,
        latest_clue,
        banner_observation=banner_observation,
        frame_phase=frame_phase,
    ):
        return list(normalized_events[latest_clue_index + 1 :])
    if _should_suppress_same_team_clue_candidate(
        runtime_state,
        latest_clue,
        banner_observation=banner_observation,
        has_following_visible_guesses=has_following_guesses,
    ):
        return list(normalized_events[latest_clue_index + 1 :])

    if has_following_guesses or latest_clue.confidence >= 0.45:
        return normalized_events

    if _visible_clue_matches_banner(latest_clue, banner_observation):
        return normalized_events
    active_window = runtime_state.active_clue_window
    if active_window is not None and _visible_clue_matches_active_window(latest_clue, active_window):
        return normalized_events
    return []


def _commit_visible_log_events(
    runtime_state: SmokeRuntimeState,
    current_visible_events: Sequence[ClueEvent | GuessEvent],
    *,
    timestamp_sec: float,
) -> list[ClueEvent | GuessEvent]:
    runtime_state.visible_log.current_row_signatures = [
        _event_signature(event) for event in current_visible_events
    ]
    _prune_recent_visible_row_registry(runtime_state, timestamp_sec=timestamp_sec)
    if not current_visible_events:
        return []

    row_keys = _visible_row_registry_keys(runtime_state, current_visible_events)
    if row_keys and all(key in runtime_state.visible_log.recent_row_registry for key in row_keys):
        runtime_state.visible_log.previous_visible_events = list(current_visible_events)
        _remember_visible_row_keys(runtime_state, row_keys=row_keys, timestamp_sec=timestamp_sec)
        return []

    runtime_state.history, runtime_state.visible_log.previous_visible_events, new_events = _append_new_visible_events(
        runtime_state.history,
        runtime_state.visible_log.previous_visible_events,
        list(current_visible_events),
    )
    _remember_visible_row_keys(runtime_state, row_keys=row_keys, timestamp_sec=timestamp_sec)
    return new_events


def _visible_row_registry_keys(
    runtime_state: SmokeRuntimeState,
    events: Sequence[ClueEvent | GuessEvent],
) -> list[tuple[tuple[str, str | None] | None, tuple[str, str, str, str]]]:
    active_context = None if runtime_state.active_clue_window is None else runtime_state.active_clue_window.signature
    current_context = active_context
    keys: list[tuple[tuple[str, str | None] | None, tuple[str, str, str, str]]] = []
    for event in events:
        if isinstance(event, ClueEvent):
            current_context = (event.clue_text, event.clue_count)
        keys.append((current_context, _event_signature(event)))
    return keys


def _remember_visible_row_keys(
    runtime_state: SmokeRuntimeState,
    *,
    row_keys: Sequence[tuple[tuple[str, str | None] | None, tuple[str, str, str, str]]],
    timestamp_sec: float,
) -> None:
    for row_key in row_keys:
        runtime_state.visible_log.recent_row_registry[row_key] = timestamp_sec


def _prune_recent_visible_row_registry(
    runtime_state: SmokeRuntimeState,
    *,
    timestamp_sec: float,
    max_age_sec: float = 180.0,
) -> None:
    stale_keys = [
        key
        for key, last_seen in runtime_state.visible_log.recent_row_registry.items()
        if timestamp_sec - last_seen > max_age_sec
    ]
    for key in stale_keys:
        runtime_state.visible_log.recent_row_registry.pop(key, None)
