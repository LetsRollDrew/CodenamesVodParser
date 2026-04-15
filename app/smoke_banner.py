"""Banner and clue helpers for smoke parsing."""

from __future__ import annotations

import numpy as np

from app.gamelog_parser import ClueEvent, GuessEvent
from app.models import PlayerRosterEntry, TeamColor
from app.ocr import OCRBackend, collapse_whitespace, detect_text_with_fallback, prepare_ocr_image
from app.roi_config import ROIConfig, crop_roi


def _refine_visible_events_from_banner(
    events: list[ClueEvent | GuessEvent],
    *,
    top_banner_text: str,
    roster: list[PlayerRosterEntry],
    clue_banner_text: str | None = None,
    clue_banner_count: str | None = None,
) -> list[ClueEvent | GuessEvent]:
    active_team = _infer_active_guessing_team(top_banner_text, roster)

    refined: list[ClueEvent | GuessEvent] = []
    for event in events:
        if isinstance(event, ClueEvent):
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
            if clue_banner_text:
                update["clue_text"] = clue_banner_text
            if clue_banner_count and clue_banner_count.isdigit():
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
    if not clue_text or not clue_count or not clue_count.isdigit():
        return None

    team_color = _infer_active_guessing_team(top_banner_text, roster)
    if team_color is None:
        previous_clue = _last_clue_event(history)
        if previous_clue is not None:
            team_color = _opposite_team(previous_clue.team_color)
    if (
        team_color is None
        and _last_clue_event(history) is None
        and left_counter is not None
        and right_counter is not None
        and left_counter != right_counter
        and max(left_counter, right_counter) >= 9
    ):
        team_color = TeamColor.BLUE if left_counter > right_counter else TeamColor.RED
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
        clue_text=clue_text,
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

    detections = detect_text_with_fallback(
        ocr_backend,
        clue_frame,
        prepared_image=clue_frame,
        hint="center_clue_banner",
    )
    if not detections:
        return None, None

    word_candidates: list[tuple[float, int, str]] = []
    count_candidate: tuple[float, str] | None = None
    for detection in sorted(detections, key=lambda item: item.box.left):
        text = collapse_whitespace(detection.text).strip()
        if not text:
            continue
        alpha_text = "".join(character for character in text.upper() if character.isalpha())
        if text.isdigit():
            if count_candidate is None or detection.confidence > count_candidate[0]:
                count_candidate = (detection.confidence, text)
            continue
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
    bubble = clue_frame[:, int(width * 0.88) : max(int(width * 0.995), int(width * 0.88) + 1)]
    detections = detect_text_with_fallback(
        ocr_backend,
        bubble,
        prepared_image=prepare_ocr_image(bubble, profile="counter"),
        hint="center_clue_count",
    )
    detection = max(detections, key=lambda item: item.confidence, default=None)
    if detection is None:
        return None
    digits = "".join(character for character in detection.text if character.isdigit())
    if not digits:
        return None
    return digits[-1]


def _infer_active_guessing_team(
    top_banner_text: str,
    roster: list[PlayerRosterEntry],
) -> TeamColor | None:
    normalized = top_banner_text.upper()
    if "GUESSING" not in normalized:
        return None

    team_matches: dict[TeamColor, int] = {}
    for player in roster:
        if player.role.value != "operative":
            continue
        name = player.display_name.strip()
        if not name:
            continue
        if name.upper() in normalized:
            team_matches[player.team_color] = team_matches.get(player.team_color, 0) + 1

    if not team_matches:
        return None
    return max(team_matches.items(), key=lambda item: item[1])[0]


def _last_clue_event(events: list[ClueEvent | GuessEvent]) -> ClueEvent | None:
    for event in reversed(events):
        if isinstance(event, ClueEvent):
            return event
    return None


def _opposite_team(team_color: TeamColor) -> TeamColor:
    return TeamColor.RED if team_color is TeamColor.BLUE else TeamColor.BLUE
