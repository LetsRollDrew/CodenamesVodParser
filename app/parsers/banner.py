"""Banner and clue helpers for smoke parsing"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np

from app.core.models import PlayerRosterEntry, TeamColor
from app.infra.roi_config import ROIConfig, crop_roi
from app.parsers.gamelog import ClueEvent, GuessEvent, _normalize_clue_count_token
from app.vision.ocr.preprocessing import OCRBackend, collapse_whitespace, detect_text_across_variants


def _focus_visible_events_on_latest_turn(
    events: list[ClueEvent | GuessEvent],
) -> list[ClueEvent | GuessEvent]:
    clue_indexes = [
        index
        for index, event in enumerate(events)
        if isinstance(event, ClueEvent)
    ]
    if not clue_indexes:
        return list(events)
    latest_clue_index = clue_indexes[-1]

    return list(events[latest_clue_index:])


def _should_probe_visible_game_log(
    *,
    log_changed: bool,
    quiet_log_frame_count: int,
    clue_banner_text: str | None = None,
    clue_banner_count: str | None = None,
    previous_visible_events: list[ClueEvent | GuessEvent] | None = None,
) -> bool:
    if log_changed:
        return True
    if not clue_banner_text:
        return False
    if quiet_log_frame_count < 2 or quiet_log_frame_count % 2 != 0:
        return False
    if quiet_log_frame_count <= 10:
        return True

    latest_visible_clue = _last_clue_event(previous_visible_events or [])
    if latest_visible_clue is None:
        return True
    if not _should_override_clue_text(latest_visible_clue.clue_text, clue_banner_text):
        return True
    if clue_banner_count and not _should_override_clue_count(latest_visible_clue.clue_count, clue_banner_count):
        return True
    return len(previous_visible_events or []) <= 1 and quiet_log_frame_count >= 4


def _suppress_unconfirmed_visible_clue(
    events: list[ClueEvent | GuessEvent],
    *,
    clue_banner_text: str | None = None,
    clue_banner_count: str | None = None,
) -> list[ClueEvent | GuessEvent]:
    latest_clue_index = max(
        (index for index, event in enumerate(events) if isinstance(event, ClueEvent)),
        default=-1,
    )
    if latest_clue_index < 0:
        return list(events)
    clue_event = events[latest_clue_index]
    assert isinstance(clue_event, ClueEvent)
    has_guess_rows = any(isinstance(event, GuessEvent) for event in events[latest_clue_index + 1:])
    if has_guess_rows or clue_event.confidence >= 0.45:
        return list(events)

    banner_confirms_text = bool(
        clue_banner_text and _should_override_clue_text(clue_event.clue_text, clue_banner_text)
    )
    banner_confirms_count = bool(
        clue_banner_count and _should_override_clue_count(clue_event.clue_count, clue_banner_count)
    )
    if clue_banner_text and not banner_confirms_text:
        banner_confirms_count = False
    if banner_confirms_text or banner_confirms_count:
        return list(events)
    return []


def _drop_visible_events_when_banner_conflicts(
    events: list[ClueEvent | GuessEvent],
    *,
    clue_banner_text: str | None = None,
    clue_banner_count: str | None = None,
) -> list[ClueEvent | GuessEvent]:
    if not clue_banner_text:
        return list(events)

    clue_events = [
        event
        for event in events
        if isinstance(event, ClueEvent)
    ]
    if not clue_events:
        return list(events)

    banner_text = str(clue_banner_text or "").strip()
    if len(banner_text) < 4:
        return list(events)

    for clue_event in clue_events:
        if not _should_override_clue_text(clue_event.clue_text, banner_text):
            continue
        if (
            clue_banner_count
            and not _should_override_clue_count(clue_event.clue_count, clue_banner_count)
        ):
            continue
        return list(events)

    return []


def _banner_signatures_equivalent(
    previous_signature: tuple[str, str | None] | None,
    current_signature: tuple[str, str | None] | None,
) -> bool:
    if previous_signature is None or current_signature is None:
        return False
    previous_text, previous_count = previous_signature
    current_text, current_count = current_signature
    if _should_override_clue_text(previous_text, current_text):
        return True
    if previous_text == current_text and (
        previous_count == current_count or previous_count is None or current_count is None
    ):
        return True
    return False


def _refine_visible_events_from_banner(
    events: list[ClueEvent | GuessEvent],
    *,
    top_banner_text: str,
    roster: list[PlayerRosterEntry],
    clue_banner_text: str | None = None,
    clue_banner_count: str | None = None,
) -> list[ClueEvent | GuessEvent]:
    active_team = _infer_active_guessing_team(top_banner_text, roster)
    latest_clue_index = max(
        (index for index, event in enumerate(events) if isinstance(event, ClueEvent)),
        default=-1,
    )

    refined: list[ClueEvent | GuessEvent] = []
    for index, event in enumerate(events):
        if isinstance(event, ClueEvent) and index == latest_clue_index:
            update: dict[str, object] = {}
            if active_team is not None and event.team_color != active_team:
                update["team_color"] = active_team
            effective_team = update.get("team_color", event.team_color)
            team_spymaster = next(
                (
                    player.display_name
                    for player in roster
                    if player.team_color == effective_team and player.role.value == "spymaster"
                ),
                None,
            )
            if team_spymaster is not None and event.spymaster_name == "unknown":
                update["spymaster_name"] = team_spymaster
            if clue_banner_text and _should_override_clue_text(event.clue_text, clue_banner_text):
                update["clue_text"] = clue_banner_text
            if clue_banner_count and _should_override_clue_count(event.clue_count, clue_banner_count):
                update["clue_count"] = clue_banner_count
            refined.append(event.model_copy(update=update) if update else event)
        else:
            refined.append(event)
    return refined


def _build_banner_clue_event(
    *,
    clue_text: str | None,
    clue_count: str | None,
    top_banner_text: str,
    roster: list[PlayerRosterEntry],
    history: list[ClueEvent | GuessEvent],
    left_counter: int | None,
    right_counter: int | None,
    timestamp_sec: float,
) -> ClueEvent | None:
    if not clue_text or not clue_count:
        return None
    normalized_clue_text = "".join(character for character in str(clue_text).upper() if character.isalpha())
    if len(normalized_clue_text) < 4 or len(normalized_clue_text) > 16:
        return None
    normalized_count = str(clue_count or "").casefold()
    if normalized_count in {"0", "unknown"}:
        return None
    if normalized_count == "infinity" and len(normalized_clue_text) < 5:
        return None
    roster_names = [
        "".join(character for character in player.display_name.upper() if character.isalpha())
        for player in roster
        if player.display_name
    ]
    if any(
        normalized_clue_text == roster_name
        or SequenceMatcher(a=normalized_clue_text, b=roster_name).ratio() >= 0.78
        for roster_name in roster_names
        if roster_name
    ):
        return None

    team_color = _infer_active_guessing_team(top_banner_text, roster)
    previous_clue = _last_clue_event(history)
    if team_color is None:
        if (
            previous_clue is not None
            and _should_override_clue_text(previous_clue.clue_text, normalized_clue_text)
        ):
            return None
        if previous_clue is not None:
            team_color = _opposite_team(previous_clue.team_color)
    if team_color is None:
        return None

    spymaster_name = next(
        (
            player.display_name
            for player in roster
            if player.team_color == team_color and player.role.value == "spymaster"
        ),
        "unknown",
    )
    return ClueEvent(
        team_color=team_color,
        spymaster_name=spymaster_name,
        clue_text=normalized_clue_text,
        clue_count=clue_count,
        timestamp_sec=timestamp_sec,
        sequence_index=0,
        confidence=0.97,
    )


def _parse_center_clue_banner(
    frame: np.ndarray,
    *,
    roi_config: ROIConfig,
    ocr_backend: OCRBackend,
) -> tuple[str | None, str | None]:
    region = roi_config.optional("clue_banner_region")
    if region is not None:
        clue_frame = crop_roi(frame, region)
    else:
        height, width = frame.shape[:2]
        top = int(height * 0.86)
        bottom = int(height * 0.95)
        left = int(width * 0.31)
        right = int(width * 0.81)
        clue_frame = frame[top:bottom, left:right]
    if clue_frame.size == 0:
        return None, None

    height, width = clue_frame.shape[:2]
    text_left = max(0, int(width * 0.04))
    text_right = max(text_left + 1, int(width * 0.84))
    text_top = max(0, int(height * 0.08))
    text_bottom = max(text_top + 1, int(height * 0.90))
    text_strip = clue_frame[text_top:text_bottom, text_left:text_right].copy()

    text_attempt = detect_text_across_variants(
        ocr_backend,
        text_strip,
        profile="banner",
        hint="center_clue_banner:text",
    )
    detections = text_attempt.detections
    if not detections:
        detections = detect_text_across_variants(
            ocr_backend,
            clue_frame,
            profile="banner",
            hint="center_clue_banner",
        ).detections
    if not detections:
        return None, _parse_center_clue_count_bubble(clue_frame, ocr_backend)

    word_candidates: list[tuple[float, int, str]] = []
    count_candidate: tuple[float, str] | None = None
    for detection in sorted(detections, key=lambda item: item.box.left):
        text = collapse_whitespace(detection.text).strip()
        if not text:
            continue
        normalized_count = _normalize_clue_count_token(text)
        if normalized_count is not None:
            if count_candidate is None or detection.confidence > count_candidate[0]:
                count_candidate = (detection.confidence, normalized_count)
            continue
        alpha_text = "".join(character for character in text.upper() if character.isalpha())
        if len(alpha_text) >= 3:
            word_candidates.append((detection.confidence, len(alpha_text), alpha_text))

    clue_text = None if not word_candidates else max(word_candidates, key=lambda item: (item[0], item[1]))[2]
    clue_count = None if count_candidate is None else count_candidate[1]
    if clue_count is None:
        clue_count = _parse_center_clue_count_bubble(clue_frame, ocr_backend)
    return clue_text, clue_count


def _parse_center_clue_count_bubble(
    clue_frame: np.ndarray,
    ocr_backend: OCRBackend,
) -> str | None:
    height, width = clue_frame.shape[:2]
    if height <= 0 or width <= 0:
        return None
    bubble_left = int(width * 0.84)
    bubble_right = max(bubble_left + 1, int(width * 0.995))
    bubble_top = max(0, int(height * 0.02))
    bubble_bottom = max(bubble_top + 1, int(height * 0.98))
    bubble = clue_frame[bubble_top:bubble_bottom, bubble_left:bubble_right].copy()
    detection = max(
        detect_text_across_variants(
            ocr_backend,
            bubble,
            profile="counter",
            hint="center_clue_count",
        ).detections,
        key=lambda item: item.confidence,
        default=None,
    )
    if detection is None:
        return None
    return _normalize_clue_count_token(detection.text)


def _should_override_clue_count(existing_count: str, candidate_count: str) -> bool:
    existing = str(existing_count or "").casefold()
    candidate = str(candidate_count or "").casefold()
    if not candidate:
        return False
    if not existing:
        return True
    if existing == candidate:
        return True
    return existing in {"0", "unknown"}


def _should_override_clue_text(existing_text: str, candidate_text: str) -> bool:
    existing = str(existing_text or "").upper()
    candidate = str(candidate_text or "").upper()
    if not candidate:
        return False
    if not existing:
        return True
    if existing == candidate:
        return True
    if existing in candidate or candidate in existing:
        return True
    return SequenceMatcher(a=existing, b=candidate).ratio() >= 0.65


def _infer_active_guessing_team(
    top_banner_text: str,
    roster: list[PlayerRosterEntry],
) -> TeamColor | None:
    normalized = top_banner_text.upper()
    if "GUESSING" not in normalized:
        return None

    tokens = {
        token
        for token in re.split(r"[^A-Z0-9]+", normalized)
        if token
    }

    team_matches: dict[TeamColor, tuple[int, int]] = {}
    for player in roster:
        if player.role.value != "operative":
            continue
        name = player.display_name.strip()
        if not name:
            continue
        normalized_name = name.upper()
        if len(normalized_name) <= 3:
            matched = normalized_name in tokens
        else:
            matched = normalized_name in normalized
        if not matched:
            continue
        current_count, current_max_len = team_matches.get(player.team_color, (0, 0))
        team_matches[player.team_color] = (
            current_count + 1,
            max(current_max_len, len(normalized_name)),
        )

    if not team_matches:
        return None
    ordered_matches = sorted(
        team_matches.items(),
        key=lambda item: (item[1][0], item[1][1]),
        reverse=True,
    )
    best_team, (best_count, best_name_len) = ordered_matches[0]
    if len(ordered_matches) >= 2 and ordered_matches[1][1][0] == best_count:
        return None
    if best_count <= 0:
        return None
    if best_count == 1 and best_name_len <= 3:
        return None
    return best_team


def _last_clue_event(events: list[ClueEvent | GuessEvent]) -> ClueEvent | None:
    for event in reversed(events):
        if isinstance(event, ClueEvent):
            return event
    return None


def _opposite_team(team_color: TeamColor) -> TeamColor:
    return TeamColor.RED if team_color is TeamColor.BLUE else TeamColor.BLUE
