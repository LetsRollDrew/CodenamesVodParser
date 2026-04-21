"""Clue-window state helpers for smoke cycle parsing"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Literal, Sequence

from app.core.models import CardColor
from app.parsers.banner import _banner_signatures_equivalent
from app.parsers.gamelog import ClueEvent, GuessEvent
from app.runtime.smoke.runtime_models import (
    ActiveClueWindow,
    BannerObservation,
    PendingClueCandidate,
    PendingGuessEvidence,
    SmokeRuntimeState,
)
from app.vision.detection.frame_detectors import is_winner_banner_text
from app.vision.ocr.preprocessing import collapse_whitespace

from .history_merge import _clue_events_equivalent, _find_matching_clue_index_anywhere
from .shared import _clue_signature_strict_match, _latest_clue_event, _normalize_clue_text

FramePhase = Literal[
    "spymaster_entering_clue",
    "operatives_guessing",
    "guess_confirm",
    "between_turns",
    "endgame",
]


def _visible_clue_matches_banner(
    clue_event: ClueEvent,
    banner_observation: BannerObservation,
) -> bool:
    if banner_observation.signature is None:
        return False
    return _banner_signatures_equivalent(
        (clue_event.clue_text, clue_event.clue_count),
        banner_observation.signature,
    )


def _visible_clue_matches_active_window(
    clue_event: ClueEvent,
    active_window: ActiveClueWindow,
) -> bool:
    if clue_event.team_color != active_window.team_color:
        return False
    return _banner_signatures_equivalent(
        (clue_event.clue_text, clue_event.clue_count),
        active_window.signature,
    )


def _split_visible_clue_and_trailing_events(
    events: Sequence[ClueEvent | GuessEvent],
) -> tuple[ClueEvent | None, list[ClueEvent | GuessEvent]]:
    normalized_events = list(events)
    latest_clue_index = max(
        (index for index, event in enumerate(normalized_events) if isinstance(event, ClueEvent)),
        default=-1,
    )
    if latest_clue_index < 0:
        return None, normalized_events
    latest_clue = normalized_events[latest_clue_index]
    assert isinstance(latest_clue, ClueEvent)
    return latest_clue, [latest_clue, *normalized_events[latest_clue_index + 1 :]]


def _visible_clue_requires_pending_promotion(
    runtime_state: SmokeRuntimeState,
    clue_event: ClueEvent,
) -> bool:
    active_window = runtime_state.active_clue_window
    if active_window is not None and _clue_signature_exactly_matches_window(clue_event, active_window):
        return False
    return _find_matching_clue_index_anywhere(runtime_state.history, clue_event) is None


def _observe_clue_candidate(
    runtime_state: SmokeRuntimeState,
    candidate: ClueEvent,
    *,
    source: str,
    frame_phase: FramePhase,
    banner_observation: BannerObservation,
    timestamp_sec: float,
    saw_same_frame_guesses: bool,
) -> ClueEvent | None:
    if _find_matching_clue_index_anywhere(runtime_state.history, candidate) is not None:
        if runtime_state.pending_clue is not None and _pending_clue_matches_candidate(runtime_state.pending_clue, candidate):
            runtime_state.pending_clue = None
        return None

    contamination_flags = _clue_candidate_contamination_flags(
        runtime_state,
        candidate,
        banner_observation=banner_observation,
    )
    banner_confirmed = source == "banner" or _visible_clue_matches_banner(candidate, banner_observation)
    visible_log_confirmed = source == "visible_log"
    pending = runtime_state.pending_clue
    if pending is not None and not _pending_clue_matches_candidate(pending, candidate):
        runtime_state.pending_clue = None
        pending = None

    if pending is None:
        pending = PendingClueCandidate(
            event=candidate,
            first_seen_ts=timestamp_sec,
            last_seen_ts=timestamp_sec,
            phase_first_seen=frame_phase,
            last_phase=frame_phase,
            best_confidence=candidate.confidence,
            sources={source},
            banner_confirmed=banner_confirmed,
            visible_log_confirmed=visible_log_confirmed,
            contamination_flags=set(contamination_flags),
            saw_same_frame_guesses=saw_same_frame_guesses,
        )
        runtime_state.pending_clue = pending
    else:
        if timestamp_sec > pending.last_seen_ts + 1e-6:
            pending.consecutive_frames += 1
        pending.last_seen_ts = timestamp_sec
        pending.last_phase = frame_phase
        pending.best_confidence = max(pending.best_confidence, candidate.confidence)
        pending.sources.add(source)
        pending.banner_confirmed = pending.banner_confirmed or banner_confirmed
        pending.visible_log_confirmed = pending.visible_log_confirmed or visible_log_confirmed
        pending.contamination_flags.update(contamination_flags)
        pending.saw_same_frame_guesses = pending.saw_same_frame_guesses or saw_same_frame_guesses
        if candidate.confidence >= pending.event.confidence:
            pending.event = candidate

    if not _should_open_new_turn_from_candidate(
        runtime_state,
        pending.event,
        banner_observation=banner_observation,
        frame_phase=frame_phase,
    ):
        return None
    if not _should_promote_pending_clue(runtime_state, pending, frame_phase=frame_phase):
        return None

    promoted = pending.event.model_copy(
        update={"confidence": max(pending.event.confidence, pending.best_confidence)}
    )
    runtime_state.pending_clue = None
    return promoted


def _pending_clue_matches_candidate(
    pending: PendingClueCandidate,
    candidate: ClueEvent,
) -> bool:
    return pending.event.team_color == candidate.team_color and _clue_events_equivalent(pending.event, candidate)


def _clue_candidate_contamination_flags(
    runtime_state: SmokeRuntimeState,
    candidate: ClueEvent,
    *,
    banner_observation: BannerObservation,
) -> set[str]:
    flags: set[str] = set()
    normalized_text = _normalize_clue_text(candidate.clue_text)
    if _is_contaminated_clue_candidate(runtime_state, candidate):
        flags.add("contaminated")
    if len(normalized_text) < 4 and not _visible_clue_matches_banner(candidate, banner_observation):
        flags.add("short_unconfirmed")
    if str(candidate.clue_count or "").casefold() == "infinity" and not _visible_clue_matches_banner(candidate, banner_observation):
        flags.add("infinity_unconfirmed")
    active_window = runtime_state.active_clue_window
    if active_window is not None and candidate.team_color != active_window.team_color:
        active_text = _normalize_clue_text(active_window.clue_text)
        if active_text and SequenceMatcher(a=normalized_text, b=active_text).ratio() >= 0.68:
            flags.add("cross_team_overlap")
    return flags


def _should_promote_pending_clue(
    runtime_state: SmokeRuntimeState,
    pending: PendingClueCandidate,
    *,
    frame_phase: FramePhase,
) -> bool:
    if pending.contamination_flags:
        return False

    first_clue = not any(isinstance(event, ClueEvent) for event in runtime_state.history)
    if pending.visible_log_confirmed and not pending.banner_confirmed and pending.saw_same_frame_guesses:
        return pending.consecutive_frames >= 3 and pending.best_confidence >= 0.9
    if pending.banner_confirmed and pending.visible_log_confirmed:
        return True
    if pending.banner_confirmed:
        return first_clue or pending.consecutive_frames >= 2
    if pending.visible_log_confirmed:
        if first_clue and pending.best_confidence >= 0.85 and frame_phase in {"spymaster_entering_clue", "between_turns"}:
            return True
        return pending.consecutive_frames >= 2 and (first_clue or frame_phase == "spymaster_entering_clue")
    return False


def _should_open_new_turn_from_candidate(
    runtime_state: SmokeRuntimeState,
    candidate: ClueEvent,
    *,
    banner_observation: BannerObservation,
    frame_phase: FramePhase,
) -> bool:
    active_window = runtime_state.active_clue_window
    latest_committed_clue = _latest_clue_event(runtime_state.history)
    if (
        latest_committed_clue is not None
        and candidate.timestamp_sec - latest_committed_clue.timestamp_sec < 8.0
        and not _clue_events_equivalent(candidate, latest_committed_clue)
        and not _visible_clue_matches_banner(candidate, banner_observation)
    ):
        return False
    if active_window is None:
        return True
    if _clue_signature_exactly_matches_window(candidate, active_window):
        return True
    if frame_phase in {"operatives_guessing", "guess_confirm", "endgame"}:
        return False
    if _banner_supports_active_window_strict(active_window, banner_observation):
        return False
    return _active_window_is_closed(
        runtime_state,
        active_window,
        frame_phase=frame_phase,
        timestamp_sec=candidate.timestamp_sec,
    )


def _active_window_is_closed(
    runtime_state: SmokeRuntimeState,
    active_window: ActiveClueWindow,
    *,
    frame_phase: FramePhase,
    timestamp_sec: float,
) -> bool:
    if frame_phase in {"operatives_guessing", "guess_confirm", "endgame"}:
        return False
    if _active_window_has_attached_guesses(runtime_state.history, active_window):
        return True
    return timestamp_sec - active_window.last_seen_ts >= 30.0


def _clear_stale_pending_clue(
    runtime_state: SmokeRuntimeState,
    *,
    frame_phase: FramePhase,
    timestamp_sec: float,
) -> None:
    pending = runtime_state.pending_clue
    if pending is None:
        return
    if (
        not any(isinstance(event, ClueEvent) for event in runtime_state.history)
        and pending.visible_log_confirmed
        and not pending.contamination_flags
        and pending.best_confidence >= 0.85
        and frame_phase in {"operatives_guessing", "guess_confirm"}
    ):
        return
    if frame_phase in {"operatives_guessing", "guess_confirm", "endgame"}:
        runtime_state.pending_clue = None
        return
    if timestamp_sec - pending.last_seen_ts > 4.0:
        runtime_state.pending_clue = None


def _update_active_clue_window(
    runtime_state: SmokeRuntimeState,
    *,
    source: str,
    events: Sequence[ClueEvent | GuessEvent],
    timestamp_sec: float,
) -> None:
    clue_event = next((event for event in reversed(events) if isinstance(event, ClueEvent)), None)
    if clue_event is None:
        return

    current_window = runtime_state.active_clue_window
    if (
        current_window is not None
        and _clue_signature_exactly_matches_window(clue_event, current_window)
    ):
        current_window.last_seen_ts = timestamp_sec
        current_window.confidence = max(current_window.confidence, clue_event.confidence)
        current_window.sources.add(source)
        return

    if (
        current_window is not None
        and source == "visible_log"
        and _is_low_confidence_fragment_of_active_clue(current_window, clue_event)
    ):
        current_window.last_seen_ts = timestamp_sec
        current_window.sources.add(source)
        return

    if _find_matching_clue_index_anywhere(runtime_state.history, clue_event) is None:
        return

    runtime_state.active_clue_window = ActiveClueWindow(
        clue_text=clue_event.clue_text,
        clue_count=clue_event.clue_count,
        team_color=clue_event.team_color,
        first_seen_ts=timestamp_sec,
        last_seen_ts=timestamp_sec,
        confidence=clue_event.confidence,
        sources={source},
    )


def _should_suppress_same_team_clue_candidate(
    runtime_state: SmokeRuntimeState,
    candidate: ClueEvent,
    *,
    banner_observation: BannerObservation,
    has_following_visible_guesses: bool,
) -> bool:
    active_window = runtime_state.active_clue_window
    if active_window is None or active_window.team_color is None:
        return False
    if candidate.team_color != active_window.team_color:
        return False
    if _clue_signature_exactly_matches_window(candidate, active_window):
        return False

    previous_visible_clue = _latest_clue_event(runtime_state.visible_log.previous_visible_events)
    candidate_repeated = (
        previous_visible_clue is not None
        and previous_visible_clue.team_color == candidate.team_color
        and previous_visible_clue.clue_text == candidate.clue_text
        and previous_visible_clue.clue_count == candidate.clue_count
    )
    if candidate_repeated and candidate.confidence >= max(0.7, active_window.confidence + 0.08):
        return False

    banner_supports_active = _banner_supports_active_window_strict(active_window, banner_observation)
    candidate_is_fragment = _is_low_confidence_fragment_of_active_clue(active_window, candidate)
    no_turn_boundary = candidate.timestamp_sec - active_window.last_seen_ts <= 45.0
    active_has_guesses = _active_window_has_attached_guesses(runtime_state.history, active_window)
    candidate_is_review_worthy = candidate.confidence < 0.45

    if banner_supports_active and no_turn_boundary and candidate_is_fragment:
        return True
    if banner_supports_active and candidate_is_review_worthy and not candidate_repeated:
        return True
    if active_has_guesses and candidate_is_fragment:
        return True
    if no_turn_boundary and candidate_is_fragment and not has_following_visible_guesses:
        return True
    return False


def _should_suppress_clue_candidate_for_phase(
    runtime_state: SmokeRuntimeState,
    candidate: ClueEvent,
    *,
    banner_observation: BannerObservation,
    frame_phase: FramePhase,
) -> bool:
    active_window = runtime_state.active_clue_window
    matches_active = active_window is not None and _clue_signature_exactly_matches_window(candidate, active_window)
    matches_banner = _visible_clue_matches_banner(candidate, banner_observation)

    if frame_phase in {"operatives_guessing", "guess_confirm"}:
        return not matches_active and not matches_banner

    if _is_contaminated_clue_candidate(runtime_state, candidate):
        return not matches_active and not matches_banner

    if frame_phase == "between_turns":
        if active_window is None and not any(isinstance(event, ClueEvent) for event in runtime_state.history):
            return False
        return not matches_active and not matches_banner

    return False


def _banner_supports_active_window_strict(
    active_window: ActiveClueWindow,
    banner_observation: BannerObservation,
) -> bool:
    if banner_observation.signature is None:
        return False
    return _clue_signature_strict_match(
        active_window.clue_text,
        active_window.clue_count,
        banner_observation.clue_text,
        banner_observation.clue_count,
    )


def _clue_signature_exactly_matches_window(
    clue_event: ClueEvent,
    active_window: ActiveClueWindow,
) -> bool:
    if clue_event.team_color != active_window.team_color:
        return False
    return _clue_signature_strict_match(
        clue_event.clue_text,
        clue_event.clue_count,
        active_window.clue_text,
        active_window.clue_count,
    )


def _is_low_confidence_fragment_of_active_clue(
    active_window: ActiveClueWindow,
    candidate: ClueEvent,
) -> bool:
    if candidate.team_color != active_window.team_color:
        return False

    active_text = _normalize_clue_text(active_window.clue_text)
    candidate_text = _normalize_clue_text(candidate.clue_text)
    if not active_text or not candidate_text or active_text == candidate_text:
        return False

    shorter_fragment = candidate_text in active_text and len(candidate_text) < len(active_text)
    longer_overlap = active_text in candidate_text and len(active_text) < len(candidate_text)
    similarity = SequenceMatcher(a=active_text, b=candidate_text).ratio()
    shorter_and_weaker = (
        len(candidate_text) <= len(active_text) - 2
        and candidate.confidence <= max(0.55, active_window.confidence - 0.2)
    )
    count_conflict = (
        active_window.clue_count is not None
        and candidate.clue_count is not None
        and str(active_window.clue_count).casefold() != str(candidate.clue_count).casefold()
    )
    if shorter_fragment and shorter_and_weaker:
        return True
    if shorter_fragment and count_conflict and candidate.confidence < active_window.confidence:
        return True
    if shorter_fragment and similarity >= 0.55 and candidate.confidence < 0.5:
        return True
    if longer_overlap and candidate.confidence < active_window.confidence - 0.15:
        return True
    return False


def _is_contaminated_clue_candidate(
    runtime_state: SmokeRuntimeState,
    candidate: ClueEvent,
) -> bool:
    normalized_text = _normalize_clue_text(candidate.clue_text)
    if not normalized_text:
        return True
    if len(normalized_text) > 16:
        return True
    if "GAMELOG" in normalized_text:
        return True

    normalized_speaker = _normalize_clue_text(candidate.spymaster_name)
    if normalized_speaker:
        if normalized_text == normalized_speaker:
            return True
        if SequenceMatcher(a=normalized_text, b=normalized_speaker).ratio() >= 0.78:
            return True

    active_window = runtime_state.active_clue_window
    if active_window is not None:
        active_text = _normalize_clue_text(active_window.clue_text)
        if active_text and active_text != normalized_text:
            if normalized_text in active_text or active_text in normalized_text:
                return True
            if candidate.confidence < max(0.75, active_window.confidence) and (
                SequenceMatcher(a=normalized_text, b=active_text).ratio() >= 0.78
            ):
                return True
    return False


def _classify_frame_phase(top_banner_text: str) -> FramePhase:
    normalized = collapse_whitespace(top_banner_text).upper()
    if not normalized:
        return "between_turns"
    if is_winner_banner_text(normalized):
        return "endgame"
    if "TAP TO CONFIRM" in normalized or "CONFIRM YOUR CHOICE" in normalized:
        return "guess_confirm"
    if (
        "TAP ON CARD" in normalized
        or "MATCH THE CLUE" in normalized
        or "ARE GUESSING" in normalized
        or "GUESSING" in normalized
    ):
        return "operatives_guessing"
    if (
        "GIVING A CLUE" in normalized
        or "GIVE YOU A CLUE" in normalized
        or "GIVE A CLUE" in normalized
        or "THINKING OF A CLUE" in normalized
    ):
        return "spymaster_entering_clue"
    return "between_turns"


def _phase_allows_new_clue_commit(frame_phase: FramePhase) -> bool:
    return frame_phase == "spymaster_entering_clue"


def _phase_allows_board_reveal_scan(frame_phase: FramePhase) -> bool:
    return frame_phase in {"operatives_guessing", "guess_confirm"}


def _observe_pending_assassin_guess(
    runtime_state: SmokeRuntimeState,
    *,
    evidence_events: Sequence[GuessEvent],
    frame_phase: FramePhase,
    winner_visible: bool,
    timestamp_sec: float,
) -> GuessEvent | None:
    black_candidates = [
        event
        for event in evidence_events
        if event.card_color is CardColor.BLACK
    ]
    if not black_candidates:
        pending = runtime_state.pending_assassin_guess
        if pending is not None and timestamp_sec - pending.last_seen_ts > 4.0:
            runtime_state.pending_assassin_guess = None
        return None

    candidate = max(black_candidates, key=lambda event: event.confidence)
    pending = runtime_state.pending_assassin_guess
    if pending is None or pending.event.word != candidate.word:
        pending = PendingGuessEvidence(
            event=candidate,
            source="assassin_probe",
            first_seen_ts=timestamp_sec,
            last_seen_ts=timestamp_sec,
            confirmations=1,
        )
        runtime_state.pending_assassin_guess = pending
    else:
        if timestamp_sec > pending.last_seen_ts + 1e-6:
            pending.confirmations += 1
        pending.last_seen_ts = timestamp_sec
        if candidate.confidence >= pending.event.confidence:
            pending.event = candidate

    if winner_visible:
        runtime_state.pending_assassin_guess = None
        return pending.event
    if frame_phase in {"operatives_guessing", "guess_confirm", "endgame"} and pending.confirmations >= 2 and pending.event.confidence >= 0.45:
        runtime_state.pending_assassin_guess = None
        return pending.event
    return None


def _active_window_has_attached_guesses(
    history: Sequence[ClueEvent | GuessEvent],
    active_window: ActiveClueWindow,
) -> bool:
    matching_index = next(
        (
            index
            for index in range(len(history) - 1, -1, -1)
            if isinstance(history[index], ClueEvent)
            and _clue_signature_strict_match(
                history[index].clue_text,
                history[index].clue_count,
                active_window.clue_text,
                active_window.clue_count,
            )
            and history[index].team_color == active_window.team_color
        ),
        None,
    )
    if matching_index is None:
        return False
    return any(isinstance(event, GuessEvent) for event in history[matching_index + 1 :])
