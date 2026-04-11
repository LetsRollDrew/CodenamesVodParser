"""Board parsing for the initial unrevealed 5x5 Codenames grid."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

import numpy as np

from app.models import BoardCell, BoardState, ImageBoundingBox
from app.ocr import OCRBackend, pick_best_detection
from app.roi_config import ROIConfig, crop_roi

BOARD_GRID_SIZE = 5


@dataclass(frozen=True, slots=True)
class BoardGridLayout:
    outer_margin_x: float = 0.028
    outer_margin_y: float = 0.028
    gap_x: float = 0.017
    gap_y: float = 0.023
    word_strip_x: float = 0.09
    word_strip_y: float = 0.44
    word_strip_width: float = 0.82
    word_strip_height: float = 0.24


DEFAULT_BOARD_GRID_LAYOUT = BoardGridLayout()


def normalize_board_word(text: str) -> str:
    """Normalize OCR output into a canonical board-word form."""

    collapsed = " ".join(text.replace("\n", " ").split())
    return collapsed.strip().upper()


def estimate_board_cell_boxes(
    board_frame: np.ndarray,
    *,
    layout: BoardGridLayout = DEFAULT_BOARD_GRID_LAYOUT,
) -> list[ImageBoundingBox]:
    """Estimate the 25 board cell boxes from the overall board crop."""

    height, width = board_frame.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("board_frame must be non-empty")

    usable_width = width * (1.0 - (2 * layout.outer_margin_x) - (4 * layout.gap_x))
    usable_height = height * (1.0 - (2 * layout.outer_margin_y) - (4 * layout.gap_y))
    cell_width = usable_width / BOARD_GRID_SIZE
    cell_height = usable_height / BOARD_GRID_SIZE

    boxes: list[ImageBoundingBox] = []
    for row in range(BOARD_GRID_SIZE):
        for col in range(BOARD_GRID_SIZE):
            left = int(round(width * layout.outer_margin_x + col * (cell_width + (width * layout.gap_x))))
            top = int(round(height * layout.outer_margin_y + row * (cell_height + (height * layout.gap_y))))
            right = int(round(left + cell_width))
            bottom = int(round(top + cell_height))
            boxes.append(ImageBoundingBox(left=left, top=top, right=right, bottom=bottom))

    return boxes


def crop_word_strip(
    board_frame: np.ndarray,
    box: ImageBoundingBox,
    *,
    layout: BoardGridLayout = DEFAULT_BOARD_GRID_LAYOUT,
) -> np.ndarray:
    """Crop the darker text strip inside a single board cell."""

    cell = board_frame[box.top : box.bottom, box.left : box.right]
    height, width = cell.shape[:2]
    left = int(round(width * layout.word_strip_x))
    top = int(round(height * layout.word_strip_y))
    right = int(round(left + (width * layout.word_strip_width)))
    bottom = int(round(top + (height * layout.word_strip_height)))
    return cell[top:bottom, left:right].copy()


def parse_board_words(
    frame: np.ndarray,
    roi_config: ROIConfig,
    ocr_backend: OCRBackend,
    *,
    layout: BoardGridLayout = DEFAULT_BOARD_GRID_LAYOUT,
) -> BoardState:
    """Parse the 25 board words from the configured board ROI."""

    board_frame = crop_roi(frame, roi_config.require("board_region"))
    boxes = estimate_board_cell_boxes(board_frame, layout=layout)
    cells: list[BoardCell] = []

    for index, box in enumerate(boxes):
        row = index // BOARD_GRID_SIZE
        col = index % BOARD_GRID_SIZE
        hint = f"board:{row}:{col}"
        word_strip = crop_word_strip(board_frame, box, layout=layout)
        detection = pick_best_detection(ocr_backend.detect_text(word_strip, hint=hint))
        word = normalize_board_word(detection.text) if detection else ""
        confidence = detection.confidence if detection else 0.0
        cells.append(
            BoardCell(
                row=row,
                col=col,
                word=word,
                confidence=confidence,
                box=box,
            )
        )

    return BoardState(cells=cells)


def snap_word_to_board(
    candidate: str,
    board_state: BoardState,
    *,
    minimum_similarity: float = 0.75,
) -> str | None:
    """Snap a noisy OCR guess back to the closest known board word."""

    normalized = normalize_board_word(candidate)
    if not normalized:
        return None

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
        return None
    return best_word


def board_dictionary(board_state: BoardState) -> set[str]:
    """Return the current game's normalized board dictionary."""

    return {word for word in board_state.words if word}
