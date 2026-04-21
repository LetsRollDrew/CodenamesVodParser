"""Parse the Codenames game log into clue and guess events"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from app.core.models import BoardState, CardColor, ImageBoundingBox, OCRDetection, PlayerRosterEntry, TeamColor
from app.parsers.board import snap_word_to_board
from app.infra.roi_config import ROIConfig, crop_roi
from app.vision.ocr.preprocessing import (
    OCRBackend,
    collapse_whitespace,
    detect_text_across_variants,
    normalize_count_like_token,
    prepare_ocr_image,
)


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


@dataclass(frozen=True, slots=True)
class ParsedRowField:
    name: str
    box: ImageBoundingBox | None
    detections: list[OCRDetection] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(collapse_whitespace(detection.text) for detection in self.detections).strip()

    @property
    def confidence(self) -> float:
        if not self.detections:
            return 0.0
        return sum(float(detection.confidence) for detection in self.detections) / len(self.detections)


@dataclass(frozen=True, slots=True)
class RowSignature:
    row_type: str
    primary_text: str
    secondary_text: str | None = None
    team_color: TeamColor | None = None


@dataclass(frozen=True, slots=True)
class ParsedLogRow:
    row_index: int
    row_box: ImageBoundingBox
    fields: dict[str, ParsedRowField]
    signature: RowSignature
    confidence: float
    row_detections: list[OCRDetection] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ParsedClueRow(ParsedLogRow):
    team_color: TeamColor = TeamColor.BLUE
    spymaster_name: str = "unknown"
    clue_text: str = ""
    clue_count: str = ""


@dataclass(frozen=True, slots=True)
class ParsedGuessRow(ParsedLogRow):
    player_name: str = "unknown"
    word: str = ""
    card_color: CardColor = CardColor.NEUTRAL


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

    absolute_delta = np.abs(current_log_frame.astype(np.int16) - previous_log_frame.astype(np.int16))
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

    log_roi = roi_config.require("game_log_region")
    log_frame = crop_roi(frame, log_roi)
    global_detections = sorted(
        (
            detection.model_copy(update={"text": collapse_whitespace(detection.text)})
            for detection in ocr_backend.detect_text(
                prepare_ocr_image(log_frame, profile="game_log"),
                hint="game_log",
            )
        ),
        key=lambda item: (item.box.center_y, item.box.left),
    )

    structured_rows = _parse_structured_rows(
        log_frame,
        ocr_backend,
        roster,
        board_state,
        timestamp_sec=timestamp_sec,
        global_detections=global_detections,
        avatar_matches_by_row=avatar_matches_by_row,
    )

    events: list[ClueEvent | GuessEvent] = []
    pending_support_players: list[PlayerRosterEntry] = []
    for row in structured_rows:
        if isinstance(row, ParsedClueRow):
            events.append(
                ClueEvent(
                    team_color=row.team_color,
                    spymaster_name=row.spymaster_name,
                    clue_text=row.clue_text,
                    clue_count=row.clue_count,
                    timestamp_sec=timestamp_sec,
                    sequence_index=row.row_index,
                    confidence=row.confidence,
                )
            )
            pending_support_players = []
            continue
        if isinstance(row, ParsedGuessRow):
            player_name = row.player_name
            if player_name == "unknown" and len(pending_support_players) == 1:
                player_name = pending_support_players[0].display_name
            events.append(
                GuessEvent(
                    player_name=player_name,
                    word=row.word,
                    card_color=row.card_color,
                    timestamp_sec=timestamp_sec,
                    sequence_index=row.row_index,
                    confidence=row.confidence,
                )
            )
            pending_support_players = []
            continue
        pending_support_players = _extract_support_players(
            row.row_detections,
            {
                collapse_whitespace(player.display_name).casefold(): player
                for player in roster
            },
        )

    return events


def _parse_structured_rows(
    log_frame: np.ndarray,
    ocr_backend: OCRBackend,
    roster: Sequence[PlayerRosterEntry],
    board_state: BoardState,
    *,
    timestamp_sec: float,
    global_detections: Sequence[OCRDetection],
    avatar_matches_by_row: dict[int, str] | None,
) -> list[ParsedLogRow]:
    row_boxes = _detect_row_containers(log_frame, global_detections)
    if not row_boxes:
        row_boxes = [
            clamped_box
            for row in _group_detections_by_row(global_detections)
            if (clamped_box := _clamp_box_to_frame(_row_box(row), log_frame.shape)) is not None
        ]

    roster_lookup = {
        collapse_whitespace(player.display_name).casefold(): player
        for player in roster
    }
    parsed_rows: list[ParsedLogRow] = []
    for row_index, row_box in enumerate(row_boxes):
        row_detections = [
            detection for detection in global_detections if _box_intersects(detection.box, row_box)
        ]
        avatar_match_name = None if avatar_matches_by_row is None else avatar_matches_by_row.get(row_index)
        parsed_row = _parse_structured_row(
            log_frame,
            row_box,
            row_detections,
            ocr_backend,
            roster,
            roster_lookup,
            board_state,
            timestamp_sec=timestamp_sec,
            row_index=row_index,
            avatar_match_name=avatar_match_name,
        )
        if parsed_row is not None:
            parsed_rows.append(parsed_row)
    return parsed_rows


def _parse_structured_row(
    log_frame: np.ndarray,
    row_box: ImageBoundingBox,
    row_detections: Sequence[OCRDetection],
    ocr_backend: OCRBackend,
    roster: Sequence[PlayerRosterEntry],
    roster_lookup: dict[str, PlayerRosterEntry],
    board_state: BoardState,
    *,
    timestamp_sec: float,
    row_index: int,
    avatar_match_name: str | None,
) -> ParsedLogRow | None:
    del timestamp_sec
    fields = _ocr_row_fields(log_frame, row_box, row_index=row_index, ocr_backend=ocr_backend, row_detections=row_detections)
    name_field = fields["name"]
    main_field = fields["main"]
    count_field = fields["count"]
    result_field = fields["result"]

    count_detection, normalized_count = _find_clue_count_detection(count_field.detections or row_detections)
    team_strip_box = fields["team_strip"].box or _fallback_team_strip_box(row_box, log_frame.shape)
    sampled_team_color = _sample_team_color(log_frame, team_strip_box)
    player_detections = _extract_player_detections(name_field.detections or row_detections, roster_lookup)

    if normalized_count is not None:
        spymaster_entry = (
            next((entry for _, entry in player_detections if entry.role.value == "spymaster"), None)
            or next((entry for _, entry in player_detections), None)
        )
        if spymaster_entry is None and avatar_match_name is not None:
            spymaster_entry = next(
                (entry for entry in roster if entry.display_name == avatar_match_name and entry.role.value == "spymaster"),
                None,
            )
        if spymaster_entry is None:
            spymaster_entry = _match_best_roster_entry(
                (detection.text for detection in name_field.detections or row_detections),
                [
                    entry
                    for entry in roster
                    if entry.role.value == "spymaster" and entry.team_color == sampled_team_color
                ],
                minimum_similarity=0.55,
            )

        clue_text, clue_confidence = _extract_clue_text_from_detections(main_field.detections or row_detections)
        if not clue_text:
            return ParsedLogRow(
                row_index=row_index,
                row_box=row_box,
                fields=fields,
                signature=RowSignature("unknown", main_field.text or name_field.text),
                confidence=max(main_field.confidence, name_field.confidence, count_field.confidence),
                row_detections=list(row_detections),
            )
        if SequenceMatcher(a=clue_text, b="GAMELOG").ratio() >= 0.8:
            return None
        _, board_like_similarity = _best_board_word_match(clue_text, board_state, minimum_similarity=0.0)
        if board_like_similarity < 0.92:
            team_color = spymaster_entry.team_color if spymaster_entry is not None else sampled_team_color
            spymaster_name = spymaster_entry.display_name if spymaster_entry is not None else "unknown"
            confidence_parts = [count_field.confidence or (count_detection.confidence if count_detection else 0.0), clue_confidence]
            if spymaster_entry is not None and player_detections:
                confidence_parts.append(player_detections[0][0].confidence)
            confidence = max(0.0, min(1.0, sum(confidence_parts) / max(len(confidence_parts), 1)))
            return ParsedClueRow(
                row_index=row_index,
                row_box=row_box,
                fields=fields,
                row_detections=list(row_detections),
                signature=RowSignature("clue", clue_text, normalized_count, team_color),
                confidence=confidence,
                team_color=team_color,
                spymaster_name=spymaster_name,
                clue_text=clue_text,
                clue_count=normalized_count,
            )

    guessed_word_detection, snapped_word = _resolve_guess_word(main_field.detections or row_detections, board_state)
    if guessed_word_detection is None or snapped_word is None:
        return ParsedLogRow(
            row_index=row_index,
            row_box=row_box,
            fields=fields,
            signature=RowSignature("unknown", main_field.text or name_field.text),
            confidence=max(main_field.confidence, name_field.confidence),
            row_detections=list(row_detections),
        )

    resolved_player = None
    if avatar_match_name is not None:
        resolved_player = next(
            (entry for entry in roster if entry.display_name == avatar_match_name),
            None,
        )
    if resolved_player is None and player_detections:
        resolved_player = player_detections[0][1]

    result_box = (
        result_field.box
        if result_field.detections and result_field.box is not None
        else _box_for_guess_result(row_box, guessed_word_detection.box, log_frame.shape)
    )
    card_color = _sample_card_color(log_frame, result_box)
    confidence_parts = [guessed_word_detection.confidence, _card_color_confidence(log_frame, result_box, card_color)]
    if player_detections:
        confidence_parts.append(player_detections[0][0].confidence)
    if avatar_match_name is not None and not player_detections:
        confidence_parts.append(0.9)
    confidence = max(0.0, min(1.0, sum(confidence_parts) / len(confidence_parts)))
    player_name = resolved_player.display_name if resolved_player is not None else "unknown"
    return ParsedGuessRow(
        row_index=row_index,
        row_box=row_box,
        fields=fields,
        row_detections=list(row_detections),
        signature=RowSignature("guess", snapped_word, player_name),
        confidence=confidence,
        player_name=player_name,
        word=snapped_word,
        card_color=card_color,
    )


def _ocr_row_fields(
    log_frame: np.ndarray,
    row_box: ImageBoundingBox,
    *,
    row_index: int,
    ocr_backend: OCRBackend,
    row_detections: Sequence[OCRDetection],
) -> dict[str, ParsedRowField]:
    row_width = row_box.width
    row_height = row_box.height
    field_boxes = {
        "team_strip": _make_row_field_box(
            log_frame.shape,
            left=row_box.left,
            top=row_box.top,
            right=row_box.left + max(1, int(row_width * 0.10)),
            bottom=row_box.bottom,
        ),
        "name": _make_row_field_box(
            log_frame.shape,
            left=row_box.left,
            top=row_box.top,
            right=row_box.left + max(1, int(row_width * 0.34)),
            bottom=row_box.bottom,
        ),
        "main": _make_row_field_box(
            log_frame.shape,
            left=row_box.left + max(1, int(row_width * 0.24)),
            top=row_box.top,
            right=row_box.left + max(1, int(row_width * 0.80)),
            bottom=row_box.bottom,
        ),
        "count": _make_row_field_box(
            log_frame.shape,
            left=row_box.left + max(1, int(row_width * 0.78)),
            top=row_box.top + max(0, int(row_height * 0.02)),
            right=row_box.left + max(1, int(row_width * 0.98)),
            bottom=row_box.bottom,
        ),
        "result": _make_row_field_box(
            log_frame.shape,
            left=row_box.left + max(1, int(row_width * 0.74)),
            top=row_box.top,
            right=row_box.left + max(1, int(row_width * 0.98)),
            bottom=row_box.bottom,
        ),
    }

    profile_by_field = {
        "name": "name_strip",
        "main": "game_log",
        "count": "counter",
        "result": "game_log",
    }

    fields: dict[str, ParsedRowField] = {}
    for field_name, field_box in field_boxes.items():
        if field_name == "team_strip":
            detections = _intersecting_detections(row_detections, field_box)
        else:
            detections = _detect_row_field(
                log_frame,
                field_box,
                field_name=field_name,
                row_index=row_index,
                ocr_backend=ocr_backend,
                fallback_detections=row_detections,
                profile=profile_by_field[field_name],  # type: ignore[arg-type]
            )
        fields[field_name] = ParsedRowField(name=field_name, box=field_box, detections=detections)

    return fields


def _detect_row_field(
    log_frame: np.ndarray,
    field_box: ImageBoundingBox | None,
    *,
    field_name: str,
    row_index: int,
    ocr_backend: OCRBackend,
    fallback_detections: Sequence[OCRDetection],
    profile: Literal["name_strip", "counter", "game_log"],
) -> list[OCRDetection]:
    if field_box is None:
        return []
    intersecting = _intersecting_detections(fallback_detections, field_box)
    if intersecting:
        return intersecting
    crop = log_frame[field_box.top : field_box.bottom, field_box.left : field_box.right].copy()
    if crop.size == 0:
        return []
    detections = detect_text_across_variants(
        ocr_backend,
        crop,
        profile=profile,
        hint=f"game_log:{row_index}:{field_name}",
    ).detections
    if detections:
        return [_offset_detection_box(detection, left_offset=field_box.left, top_offset=field_box.top) for detection in detections]
    return []


def _detect_row_containers(
    log_frame: np.ndarray,
    detections: Sequence[OCRDetection],
) -> list[ImageBoundingBox]:
    if log_frame.size == 0:
        return []
    if not detections and float(np.std(log_frame)) < 1.5:
        return []

    grayscale = log_frame.mean(axis=2) if log_frame.ndim == 3 else log_frame.astype(np.float32)
    row_activity = grayscale.mean(axis=1)
    row_contrast = grayscale.std(axis=1)
    combined_activity = row_activity + (row_contrast * 1.5)
    threshold = max(10.0, float(np.percentile(combined_activity, 60)) * 0.65)
    active_rows = combined_activity >= threshold

    segments: list[tuple[int, int]] = []
    start: int | None = None
    for row_index, is_active in enumerate(active_rows):
        if is_active and start is None:
            start = row_index
            continue
        if not is_active and start is not None:
            if row_index - start >= 10:
                segments.append((start, row_index))
            start = None
    if start is not None and len(active_rows) - start >= 10:
        segments.append((start, len(active_rows)))

    boxes: list[ImageBoundingBox] = []
    for top, bottom in segments:
        segment = grayscale[top:bottom]
        if segment.size == 0:
            continue
        column_activity = segment.mean(axis=0)
        column_threshold = max(6.0, float(np.percentile(column_activity, 55)) * 0.7)
        active_cols = np.where(column_activity >= column_threshold)[0]
        if active_cols.size == 0:
            left = 0
            right = log_frame.shape[1]
        else:
            left = max(int(active_cols[0]) - 2, 0)
            right = min(int(active_cols[-1]) + 3, log_frame.shape[1])
        boxes.append(ImageBoundingBox(left=left, top=top, right=max(left + 1, right), bottom=bottom))

    if boxes:
        grouped_rows = _group_detections_by_row(detections)
        if len(boxes) == 1 and len(grouped_rows) > 1:
            return [
                clamped_box
                for row in grouped_rows
                if (clamped_box := _clamp_box_to_frame(_row_box(row), log_frame.shape)) is not None
            ]
        return boxes
    if detections:
        return [
            clamped_box
            for row in _group_detections_by_row(detections)
            if (clamped_box := _clamp_box_to_frame(_row_box(row), log_frame.shape)) is not None
        ]
    return []


def _box_intersects(left: ImageBoundingBox, right: ImageBoundingBox) -> bool:
    return not (
        left.right <= right.left
        or right.right <= left.left
        or left.bottom <= right.top
        or right.bottom <= left.top
    )


def _intersecting_detections(
    detections: Sequence[OCRDetection],
    box: ImageBoundingBox,
) -> list[OCRDetection]:
    return [
        detection
        for detection in detections
        if box.left <= detection.box.center_x <= box.right and box.top <= detection.box.center_y <= box.bottom
    ]


def _offset_detection_box(
    detection: OCRDetection,
    *,
    left_offset: int,
    top_offset: int,
) -> OCRDetection:
    box = detection.box
    return detection.model_copy(
        update={
            "box": ImageBoundingBox(
                left=box.left + left_offset,
                top=box.top + top_offset,
                right=box.right + left_offset,
                bottom=box.bottom + top_offset,
            )
        }
    )


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


def _extract_support_players(
    row: Sequence[OCRDetection],
    roster_lookup: dict[str, PlayerRosterEntry],
) -> list[PlayerRosterEntry]:
    support_players = [
        entry
        for _, entry in _extract_player_detections(row, roster_lookup)
        if entry.role.value == "operative"
    ]
    deduped: list[PlayerRosterEntry] = []
    seen_names: set[str] = set()
    for entry in support_players:
        if entry.display_name in seen_names:
            continue
        deduped.append(entry)
        seen_names.add(entry.display_name)
    return deduped


def _extract_player_detections(
    row: Sequence[OCRDetection],
    roster_lookup: dict[str, PlayerRosterEntry],
) -> list[tuple[OCRDetection, PlayerRosterEntry]]:
    return [
        (detection, matched_player)
        for detection in row
        if (matched_player := _match_roster_entry(detection.text, roster_lookup)) is not None
    ]


def _resolve_guess_word(
    row: Sequence[OCRDetection],
    board_state: BoardState,
) -> tuple[OCRDetection | None, str | None]:
    best_detection: OCRDetection | None = None
    best_word: str | None = None
    best_confidence = -1.0
    for detection in row:
        snapped, similarity = _best_board_word_match(detection.text, board_state)
        if snapped is None:
            continue
        score = float(detection.confidence) + (similarity * 0.35)
        if score > best_confidence:
            best_detection = detection
            best_word = snapped
            best_confidence = score
    return best_detection, best_word


def _best_board_word_match(
    candidate: str,
    board_state: BoardState,
    *,
    minimum_similarity: float = 0.68,
) -> tuple[str | None, float]:
    # Guess resolution is intentionally constrained to the current visible board.
    # Do not widen this to the global default dictionary.
    normalized = collapse_whitespace(candidate).upper()
    if not normalized:
        return None, 0.0

    best_word: str | None = None
    best_similarity = 0.0
    for word in board_state.words:
        if not word:
            continue
        similarity = SequenceMatcher(a=normalized, b=word).ratio()
        if similarity > best_similarity:
            best_similarity = similarity
            best_word = word

    if best_word is None or best_similarity < minimum_similarity:
        return None, best_similarity
    snapped = snap_word_to_board(normalized, board_state, minimum_similarity=minimum_similarity)
    return snapped or best_word, best_similarity


def _make_row_field_box(
    frame_shape: tuple[int, ...],
    *,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> ImageBoundingBox | None:
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return None

    clamped_left = max(0, min(left, width))
    clamped_top = max(0, min(top, height))
    if clamped_left >= width or clamped_top >= height:
        return None
    clamped_right = max(clamped_left + 1, min(max(right, clamped_left + 1), width))
    clamped_bottom = max(clamped_top + 1, min(max(bottom, clamped_top + 1), height))
    if clamped_right <= clamped_left or clamped_bottom <= clamped_top:
        return None
    return ImageBoundingBox(left=clamped_left, right=clamped_right, top=clamped_top, bottom=clamped_bottom)


def _clamp_box_to_frame(box: ImageBoundingBox, frame_shape: tuple[int, ...]) -> ImageBoundingBox | None:
    height, width = frame_shape[:2]
    left = max(0, min(box.left, width))
    right = max(0, min(box.right, width))
    top = max(0, min(box.top, height))
    bottom = max(0, min(box.bottom, height))
    if right <= left or bottom <= top:
        return None
    return ImageBoundingBox(left=left, right=right, top=top, bottom=bottom)


def _row_box(row: Sequence[OCRDetection]) -> ImageBoundingBox:
    left = min(item.box.left for item in row)
    right = max(item.box.right for item in row)
    top = min(item.box.top for item in row)
    bottom = max(item.box.bottom for item in row)
    return ImageBoundingBox(left=left, right=right, top=top, bottom=bottom)


def _fallback_team_strip_box(
    row_box: ImageBoundingBox,
    frame_shape: tuple[int, ...],
) -> ImageBoundingBox:
    fallback = _make_row_field_box(
        frame_shape,
        left=row_box.left,
        top=row_box.top,
        right=row_box.left + max(1, int(row_box.width * 0.08)),
        bottom=row_box.bottom,
    )
    return fallback or row_box


def _sample_team_color(log_frame: np.ndarray, sample_box: ImageBoundingBox) -> TeamColor:
    region = log_frame[sample_box.top : sample_box.bottom, sample_box.left : sample_box.right]
    if region.size == 0:
        return TeamColor.BLUE
    mean_bgr = region.reshape(-1, region.shape[-1]).mean(axis=0)
    return TeamColor.RED if float(mean_bgr[2]) > float(mean_bgr[0]) else TeamColor.BLUE


def _box_for_guess_result(
    row_box: ImageBoundingBox,
    word_box: ImageBoundingBox,
    frame_shape: tuple[int, ...],
) -> ImageBoundingBox:
    height, width = frame_shape[:2]
    left = min(max(word_box.right + 2, row_box.left + int(row_box.width * 0.72)), width - 1)
    right = min(width, max(left + 1, row_box.right))
    return ImageBoundingBox(left=left, right=right, top=row_box.top, bottom=min(row_box.bottom, height))


def _sample_card_color(log_frame: np.ndarray, chip_box: ImageBoundingBox) -> CardColor:
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


def _card_color_confidence(log_frame: np.ndarray, chip_box: ImageBoundingBox, card_color: CardColor) -> float:
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


def _find_clue_count_detection(
    row: Sequence[OCRDetection],
) -> tuple[OCRDetection | None, str | None]:
    best_detection: OCRDetection | None = None
    best_normalized: str | None = None
    best_score: tuple[int, int, float] | None = None
    for detection in row:
        normalized = normalize_count_like_token(detection.text)
        if normalized is None:
            continue
        stripped = collapse_whitespace(detection.text).upper().strip(".,:;()[]{}")
        score = (detection.box.right, int(stripped.isdigit()), float(detection.confidence))
        if best_score is None or score > best_score:
            best_detection = detection
            best_normalized = normalized
            best_score = score
    return best_detection, best_normalized


def _normalize_clue_count_token(text: str) -> str | None:
    return normalize_count_like_token(text)


def _extract_clue_text_from_detections(
    detections: Sequence[OCRDetection],
) -> tuple[str | None, float]:
    fragments: list[tuple[int, float, str]] = []
    for detection in detections:
        normalized = collapse_whitespace(detection.text).upper()
        alpha_text = "".join(character for character in normalized if character.isalpha())
        if not alpha_text:
            continue
        fragments.append((detection.box.left, float(detection.confidence), alpha_text))

    if not fragments:
        return None, 0.0

    fragments.sort(key=lambda item: item[0])
    joined = "".join(fragment for _, _, fragment in fragments)
    if len(joined) < 2:
        return None, 0.0
    confidence = sum(confidence for _, confidence, _ in fragments) / len(fragments)
    return joined, confidence


def _match_roster_entry(
    text: str,
    roster_lookup: dict[str, PlayerRosterEntry],
    *,
    minimum_similarity: float = 0.72,
) -> PlayerRosterEntry | None:
    normalized = collapse_whitespace(text).casefold()
    if not normalized:
        return None
    if normalized in roster_lookup:
        return roster_lookup[normalized]

    adaptive_minimum = minimum_similarity
    if len(normalized) <= 2:
        adaptive_minimum = min(adaptive_minimum, 0.5)
    elif len(normalized) <= 4:
        adaptive_minimum = min(adaptive_minimum, 0.62)

    best_key = None
    best_similarity = 0.0
    for key in roster_lookup:
        similarity = SequenceMatcher(a=normalized, b=key).ratio()
        if similarity > best_similarity:
            best_similarity = similarity
            best_key = key

    if best_key is None or best_similarity < adaptive_minimum:
        return None
    return roster_lookup[best_key]


def _match_best_roster_entry(
    texts: Sequence[str] | list[str] | tuple[str, ...] | object,
    candidates: Sequence[PlayerRosterEntry],
    *,
    minimum_similarity: float,
) -> PlayerRosterEntry | None:
    candidate_list = list(candidates)
    if not candidate_list:
        return None

    best_entry: PlayerRosterEntry | None = None
    best_similarity = 0.0
    for text in texts:
        normalized = collapse_whitespace(str(text)).casefold()
        if not normalized:
            continue
        for entry in candidate_list:
            similarity = SequenceMatcher(
                a=normalized,
                b=collapse_whitespace(entry.display_name).casefold(),
            ).ratio()
            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

    if best_entry is None or best_similarity < minimum_similarity:
        return None
    return best_entry
