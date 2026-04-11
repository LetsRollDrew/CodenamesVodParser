"""Parse the Codenames game log into clue and guess events."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from app.board_parser import snap_word_to_board
from app.models import BoardState, CardColor, OCRDetection, PlayerRosterEntry, TeamColor
from app.ocr import OCRBackend, collapse_whitespace
from app.roi_config import ROIConfig, crop_roi


class GameLogEvent(BaseModel):
    event_type: Literal["clue", "guess"]
    timestamp_sec: float = Field(ge=0)
    sequence_index: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)


class ClueEvent(GameLogEvent):
    event_type: Literal["clue"] = "clue"
    team_color: TeamColor
    spymaster_name: str
    clue_text: str
    clue_count: str


class GuessEvent(GameLogEvent):
    event_type: Literal["guess"] = "guess"
    player_name: str
    word: str
    card_color: CardColor


def log_has_changed(
    previous_log_frame: np.ndarray | None,
    current_log_frame: np.ndarray,
    *,
    minimum_change_ratio: float = 0.01,
    pixel_delta_threshold: float = 12.0,
) -> bool:
    """Return whether the visible game log changed enough to justify OCR."""

    if previous_log_frame is None:
        return True
    if previous_log_frame.shape != current_log_frame.shape:
        return True

    absolute_delta = np.abs(
        current_log_frame.astype(np.int16) - previous_log_frame.astype(np.int16)
    )
    changed_pixels = np.any(absolute_delta >= pixel_delta_threshold, axis=-1)
    change_ratio = float(changed_pixels.mean())
    return change_ratio >= minimum_change_ratio


def parse_game_log(
    frame: np.ndarray,
    roi_config: ROIConfig,
    ocr_backend: OCRBackend,
    roster: Sequence[PlayerRosterEntry],
    board_state: BoardState,
    *,
    timestamp_sec: float,
    avatar_matches_by_row: dict[int, str] | None = None,
) -> list[ClueEvent | GuessEvent]:
    """Parse clue and guess events from the configured game-log ROI."""

    log_frame = crop_roi(frame, roi_config.require("game_log_region"))
    detections = sorted(
        (
            detection.model_copy(update={"text": collapse_whitespace(detection.text)})
            for detection in ocr_backend.detect_text(log_frame, hint="game_log")
        ),
        key=lambda item: (item.box.center_y, item.box.left),
    )
    rows = _group_detections_by_row(detections)

    events: list[ClueEvent | GuessEvent] = []
    for row_index, row in enumerate(rows):
        avatar_match_name = None if avatar_matches_by_row is None else avatar_matches_by_row.get(row_index)
        parsed = _parse_row(
            log_frame,
            row,
            roster,
            board_state,
            timestamp_sec=timestamp_sec,
            sequence_index=row_index,
            avatar_match_name=avatar_match_name,
        )
        if parsed is not None:
            events.append(parsed)

    return events


def _group_detections_by_row(detections: Sequence[OCRDetection]) -> list[list[OCRDetection]]:
    if not detections:
        return []

    heights = sorted(detection.box.height for detection in detections)
    median_height = heights[len(heights) // 2]
    row_gap_threshold = max(float(median_height) * 0.85, 12.0)

    rows: list[list[OCRDetection]] = [[detections[0]]]
    current_center = detections[0].box.center_y
    for detection in detections[1:]:
        if detection.box.center_y - current_center > row_gap_threshold:
            rows.append([detection])
            current_center = detection.box.center_y
            continue
        rows[-1].append(detection)
        current_center = sum(item.box.center_y for item in rows[-1]) / len(rows[-1])

    return [sorted(row, key=lambda item: item.box.left) for row in rows]


def _parse_row(
    log_frame: np.ndarray,
    row: Sequence[OCRDetection],
    roster: Sequence[PlayerRosterEntry],
    board_state: BoardState,
    *,
    timestamp_sec: float,
    sequence_index: int,
    avatar_match_name: str | None,
) -> ClueEvent | GuessEvent | None:
    if not row:
        return None

    roster_lookup = {
        collapse_whitespace(player.display_name).casefold(): player
        for player in roster
    }
    player_detections = [
        (detection, roster_lookup[detection.text.casefold()])
        for detection in row
        if detection.text.casefold() in roster_lookup
    ]
    count_detection = next(
        (detection for detection in row if _is_clue_count_token(detection.text)),
        None,
    )

    if count_detection is not None:
        spymaster_entry = (
            next((entry for _, entry in player_detections if entry.role.value == "spymaster"), None)
            or next((entry for _, entry in player_detections), None)
        )
        clue_text_detection = max(
            (
                detection
                for detection in row
                if detection is not count_detection
                and detection.text.casefold() not in roster_lookup
            ),
            key=lambda item: (item.confidence, len(item.text)),
            default=None,
        )
        if clue_text_detection is None:
            return None
        team_color = (
            spymaster_entry.team_color
            if spymaster_entry is not None
            else _sample_team_color(log_frame, _row_box(row))
        )
        spymaster_name = spymaster_entry.display_name if spymaster_entry is not None else "unknown"
        confidence_parts = [count_detection.confidence, clue_text_detection.confidence]
        if spymaster_entry is not None and player_detections:
            confidence_parts.append(player_detections[0][0].confidence)
        return ClueEvent(
            team_color=team_color,
            spymaster_name=spymaster_name,
            clue_text=clue_text_detection.text.upper(),
            clue_count=count_detection.text,
            timestamp_sec=timestamp_sec,
            sequence_index=sequence_index,
            confidence=sum(confidence_parts) / len(confidence_parts),
        )

    guessed_word_detection, snapped_word = _resolve_guess_word(row, board_state)
    if guessed_word_detection is None or snapped_word is None:
        return None

    resolved_player = None
    if avatar_match_name is not None:
        resolved_player = next(
            (entry for entry in roster if entry.display_name == avatar_match_name),
            None,
        )
    if resolved_player is None and player_detections:
        resolved_player = player_detections[0][1]
    if resolved_player is None:
        return None

    chip_box = _expand_box_for_guess_chip(guessed_word_detection.box, log_frame.shape)
    card_color = _sample_card_color(log_frame, chip_box)
    confidence_parts = [guessed_word_detection.confidence]
    if player_detections:
        confidence_parts.append(player_detections[0][0].confidence)
    if avatar_match_name is not None and not player_detections:
        confidence_parts.append(0.95)
    confidence_parts.append(_card_color_confidence(log_frame, chip_box, card_color))

    return GuessEvent(
        player_name=resolved_player.display_name,
        word=snapped_word,
        card_color=card_color,
        timestamp_sec=timestamp_sec,
        sequence_index=sequence_index,
        confidence=sum(confidence_parts) / len(confidence_parts),
    )


def _resolve_guess_word(
    row: Sequence[OCRDetection],
    board_state: BoardState,
) -> tuple[OCRDetection | None, str | None]:
    best_detection: OCRDetection | None = None
    best_word: str | None = None
    best_confidence = -1.0
    for detection in row:
        snapped = snap_word_to_board(detection.text, board_state)
        if snapped is None:
            continue
        if detection.confidence > best_confidence:
            best_detection = detection
            best_word = snapped
            best_confidence = detection.confidence
    return best_detection, best_word


def _expand_box_for_guess_chip(box, frame_shape: tuple[int, ...]):
    height, width = frame_shape[:2]
    pad_x = max(int(box.width * 0.10), 3)
    pad_y = max(int(box.height * 0.20), 3)
    left = max(box.left - pad_x, 0)
    right = min(box.right + pad_x, width)
    top = max(box.top - pad_y, 0)
    bottom = min(box.bottom + pad_y, height)
    return box.model_copy(update={"left": left, "right": right, "top": top, "bottom": bottom})


def _row_box(row: Sequence[OCRDetection]):
    left = min(item.box.left for item in row)
    right = max(item.box.right for item in row)
    top = min(item.box.top for item in row)
    bottom = max(item.box.bottom for item in row)
    return row[0].box.model_copy(update={"left": left, "right": right, "top": top, "bottom": bottom})


def _sample_team_color(log_frame: np.ndarray, row_box) -> TeamColor:
    row_region = log_frame[row_box.top : row_box.bottom, row_box.left : row_box.right]
    if row_region.size == 0:
        return TeamColor.BLUE
    mean_bgr = row_region.reshape(-1, row_region.shape[-1]).mean(axis=0)
    return TeamColor.RED if float(mean_bgr[2]) > float(mean_bgr[0]) else TeamColor.BLUE


def _sample_card_color(log_frame: np.ndarray, chip_box) -> CardColor:
    region = log_frame[chip_box.top : chip_box.bottom, chip_box.left : chip_box.right]
    if region.size == 0:
        return CardColor.NEUTRAL
    mean_bgr = region.reshape(-1, region.shape[-1]).mean(axis=0)
    blue, green, red = map(float, mean_bgr[:3])
    brightness = (blue + green + red) / 3.0

    if brightness < 55 and max(blue, green, red) < 80:
        return CardColor.BLACK
    if red > (blue + 35) and red > (green + 20):
        return CardColor.RED
    if blue > (red + 35) and blue > (green + 15):
        return CardColor.BLUE
    return CardColor.NEUTRAL


def _card_color_confidence(log_frame: np.ndarray, chip_box, card_color: CardColor) -> float:
    region = log_frame[chip_box.top : chip_box.bottom, chip_box.left : chip_box.right]
    if region.size == 0:
        return 0.0
    mean_bgr = region.reshape(-1, region.shape[-1]).mean(axis=0)
    blue, green, red = map(float, mean_bgr[:3])
    brightness = (blue + green + red) / 3.0
    if card_color is CardColor.BLACK:
        return max(0.0, min(1.0, 1.0 - (brightness / 80.0)))
    if card_color is CardColor.RED:
        return max(0.0, min(1.0, (red - max(blue, green)) / 180.0 + 0.5))
    if card_color is CardColor.BLUE:
        return max(0.0, min(1.0, (blue - max(red, green)) / 180.0 + 0.5))
    channel_delta = max(abs(red - blue), abs(red - green), abs(blue - green))
    return max(0.0, min(1.0, 1.0 - (channel_delta / 120.0)))


def _is_clue_count_token(text: str) -> bool:
    stripped = text.strip()
    return stripped.isdigit() or stripped in {"∞", "INF", "INFINITY"}
