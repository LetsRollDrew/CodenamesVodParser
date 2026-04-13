"""Board parsing for the initial unrevealed 5x5 Codenames grid."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from app.models import BoardCell, BoardState, ImageBoundingBox
from app.ocr import OCRBackend, detect_text_with_fallback, pick_best_detection, prepare_ocr_image
from app.roi_config import ROIConfig, crop_roi

BOARD_GRID_SIZE = 5


@dataclass(frozen=True, slots=True)
class BoardGridLayout:
    outer_margin_x: float = 0.028
    outer_margin_y: float = 0.028
    gap_x: float = 0.017
    gap_y: float = 0.023
    word_strip_x: float = 0.05
    word_strip_y: float = 0.48
    word_strip_width: float = 0.90
    word_strip_height: float = 0.34


DEFAULT_BOARD_GRID_LAYOUT = BoardGridLayout()
DEFAULT_WORD_LIST_PATH = Path(__file__).with_name("data") / "codenames_default_words.txt"


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
    if height <= 0 or width <= 0:
        return cell.copy()

    left = int(round(width * layout.word_strip_x))
    right = int(round(width * (layout.word_strip_x + layout.word_strip_width)))

    search_top = int(round(height * layout.word_strip_y))
    search_bottom = int(round(height * 0.92))
    band_height = max(int(round(height * layout.word_strip_height)), 1)

    grayscale = cell.mean(axis=2) if cell.ndim == 3 else cell.astype(np.float32)
    if search_bottom - search_top <= band_height:
        top = max(min(search_top, height - band_height), 0)
    else:
        row_darkness = 255.0 - grayscale[search_top:search_bottom].mean(axis=1)
        kernel = np.ones(band_height, dtype=np.float32) / float(band_height)
        scores = np.convolve(row_darkness, kernel, mode="valid")
        best_offset = int(np.argmax(scores)) if scores.size else 0
        top = search_top + best_offset

    bottom = min(top + band_height, height)

    inset_x = max(int(round((right - left) * 0.03)), 1)
    inset_y = max(int(round((bottom - top) * 0.06)), 1)
    return cell[
        max(top + inset_y, 0) : max(bottom - inset_y, top + inset_y + 1),
        max(left + inset_x, 0) : max(right - inset_x, left + inset_x + 1),
    ].copy()


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
        detection = _detect_board_cell_word(
            board_frame,
            box,
            ocr_backend=ocr_backend,
            hint=hint,
            layout=layout,
        )
        word = normalize_board_word(detection.text) if detection else ""
        if word:
            word = _refine_board_word_from_default_dictionary(word)
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


def _detect_board_cell_word(
    board_frame: np.ndarray,
    box: ImageBoundingBox,
    *,
    ocr_backend: OCRBackend,
    hint: str,
    layout: BoardGridLayout,
) -> OCRDetection | None:
    best_detection: OCRDetection | None = None
    best_score = float("-inf")
    for variant_index, strip in enumerate(_iter_word_strip_candidates(board_frame, box, layout=layout)):
        if strip.size == 0:
            continue
        prepared_strip = prepare_ocr_image(strip, profile="board_word")
        detection = pick_best_detection(
            detect_text_with_fallback(
                ocr_backend,
                strip,
                prepared_image=prepared_strip,
                hint=f"{hint}:v{variant_index}",
            )
        )
        if detection is None:
            continue
        score = _board_word_detection_score(detection)
        if score > best_score:
            best_detection = detection
            best_score = score
    return best_detection


def _iter_word_strip_candidates(
    board_frame: np.ndarray,
    box: ImageBoundingBox,
    *,
    layout: BoardGridLayout,
) -> Iterable[np.ndarray]:
    yield crop_word_strip(board_frame, box, layout=layout)

    cell = board_frame[box.top : box.bottom, box.left : box.right]
    height, width = cell.shape[:2]
    if height <= 0 or width <= 0:
        return

    left = int(round(width * 0.05))
    right = int(round(width * 0.95))
    band_height = max(int(round(height * 0.28)), 1)
    for top_ratio in (0.34, 0.40, 0.46, 0.52):
        top = int(round(height * top_ratio))
        bottom = min(top + band_height, height)
        inset_x = max(int(round((right - left) * 0.03)), 1)
        inset_y = max(int(round((bottom - top) * 0.06)), 1)
        yield cell[
            max(top + inset_y, 0) : max(bottom - inset_y, top + inset_y + 1),
            max(left + inset_x, 0) : max(right - inset_x, left + inset_x + 1),
        ].copy()


def _board_word_detection_score(detection: OCRDetection) -> float:
    word = normalize_board_word(detection.text)
    score = float(detection.confidence) * 100.0
    if word.isalpha():
        score += 30.0
    if 3 <= len(word) <= 12:
        score += 15.0
    vowels = sum(1 for character in word if character in "AEIOUY")
    if vowels == 0:
        score -= 20.0
    else:
        ratio = vowels / max(len(word), 1)
        if 0.20 <= ratio <= 0.65:
            score += 8.0
        else:
            score -= 4.0
    if any(character.isdigit() for character in word):
        score -= 15.0
    return score


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


@lru_cache(maxsize=1)
def default_word_dictionary() -> tuple[str, ...]:
    """Return the canonical default Codenames word list, if available."""

    if not DEFAULT_WORD_LIST_PATH.exists():
        return tuple()
    words: list[str] = []
    for raw_line in DEFAULT_WORD_LIST_PATH.read_text(encoding="utf-8").splitlines():
        word = normalize_board_word(raw_line)
        if word:
            words.append(word)
    return tuple(words)


def board_dictionary(board_state: BoardState) -> set[str]:
    """Return the current game's normalized board dictionary."""

    return {word for word in board_state.words if word}


def _refine_board_word_from_default_dictionary(word: str) -> str:
    normalized = normalize_board_word(word)
    if not normalized:
        return normalized

    dictionary = default_word_dictionary()
    if not dictionary or normalized in dictionary:
        return normalized

    best_word: str | None = None
    best_similarity = 0.0
    second_best = 0.0
    for candidate in dictionary:
        similarity = SequenceMatcher(a=normalized, b=candidate).ratio()
        if similarity > best_similarity:
            second_best = best_similarity
            best_similarity = similarity
            best_word = candidate
        elif similarity > second_best:
            second_best = similarity

    if best_word is None:
        return normalized

    margin = best_similarity - second_best
    if best_similarity >= 0.94 and margin >= 0.03:
        return best_word

    if (
        len(normalized) >= 6
        and len(best_word) - len(normalized) <= 2
        and best_word.startswith(normalized)
        and margin >= 0.02
    ):
        return best_word

    return normalized
