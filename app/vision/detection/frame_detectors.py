"""Frame-level detectors used by the game reconstruction pipeline"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.core.models import CardColor
from app.infra.roi_config import ROIConfig, crop_roi
from app.vision.ocr.preprocessing import (
    OCRAttemptResult,
    OCRBackend,
    collapse_whitespace,
    detect_text_across_variants,
    normalize_count_like_token,
    pick_best_detection,
    prepare_ocr_image,
)

_SETUP_PANEL_REGION = (0.26, 0.20, 0.50, 0.44)
_START_BUTTON_REGION = (0.33, 0.90, 0.36, 0.07)


@dataclass(frozen=True, slots=True)
class CounterObservation:
    value: int | None
    confidence: float
    raw_text: str
    variant_name: str | None
    score: float


def _crop_relative(frame: np.ndarray, *, x: float, y: float, width: float, height: float) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    left = max(0, min(frame_width - 1, int(round(frame_width * x))))
    top = max(0, min(frame_height - 1, int(round(frame_height * y))))
    right = max(left + 1, min(frame_width, int(round(frame_width * (x + width)))))
    bottom = max(top + 1, min(frame_height, int(round(frame_height * (y + height)))))
    return frame[top:bottom, left:right].copy()


def parse_setup_panel_text(frame: np.ndarray, ocr_backend: OCRBackend) -> str:
    """Parse lobby/setup text from the central settings panel."""

    setup_frame = _crop_relative(
        frame,
        x=_SETUP_PANEL_REGION[0],
        y=_SETUP_PANEL_REGION[1],
        width=_SETUP_PANEL_REGION[2],
        height=_SETUP_PANEL_REGION[3],
    )
    detections = ocr_backend.detect_text(
        prepare_ocr_image(setup_frame, profile="banner"),
        hint="setup_panel",
    )
    return " ".join(collapse_whitespace(item.text).upper() for item in detections)


def start_game_button_visible(frame: np.ndarray) -> bool:
    """Return whether the green pregame start button is visible near the bottom center."""

    button_frame = _crop_relative(
        frame,
        x=_START_BUTTON_REGION[0],
        y=_START_BUTTON_REGION[1],
        width=_START_BUTTON_REGION[2],
        height=_START_BUTTON_REGION[3],
    )
    if button_frame.size == 0:
        return False
    blue = button_frame[:, :, 0].astype(np.int16)
    green = button_frame[:, :, 1].astype(np.int16)
    red = button_frame[:, :, 2].astype(np.int16)
    green_mask = (green >= 100) & (green >= red + 25) & (green >= blue + 25)
    return float(green_mask.mean()) >= 0.10


def parse_counter_value(counter_frame: np.ndarray, ocr_backend: OCRBackend, *, hint: str) -> int | None:
    """Parse a numeric counter value from a counter crop."""

    return parse_counter_observation(counter_frame, ocr_backend, hint=hint).value


def parse_counter_observation(
    counter_frame: np.ndarray,
    ocr_backend: OCRBackend,
    *,
    hint: str,
) -> CounterObservation:
    """Parse one counter crop and return the normalized reading plus OCR provenance."""

    attempt = detect_text_across_variants(
        ocr_backend,
        counter_frame,
        profile="counter",
        hint=hint,
    )
    return _counter_observation_from_attempt(counter_frame, attempt)


def parse_game_counter_observations(
    frame: np.ndarray,
    roi_config: ROIConfig,
    ocr_backend: OCRBackend,
) -> tuple[CounterObservation, CounterObservation]:
    """Parse both remaining-card counters with OCR provenance."""

    left_observation = parse_counter_observation(
        crop_roi(frame, roi_config.require("left_counter_region")),
        ocr_backend,
        hint="left_counter",
    )
    right_observation = parse_counter_observation(
        crop_roi(frame, roi_config.require("right_counter_region")),
        ocr_backend,
        hint="right_counter",
    )
    return left_observation, right_observation


def parse_game_counters(
    frame: np.ndarray,
    roi_config: ROIConfig,
    ocr_backend: OCRBackend,
) -> tuple[int | None, int | None]:
    """Parse the left and right remaining-card counters from a frame."""

    left_observation, right_observation = parse_game_counter_observations(frame, roi_config, ocr_backend)
    return left_observation.value, right_observation.value


def has_play_next_game(frame: np.ndarray, roi_config: ROIConfig, ocr_backend: OCRBackend) -> bool:
    """Return whether the end banner region contains 'Play next game'."""

    banner_frame = crop_roi(frame, roi_config.require("end_banner_region"))
    detections = ocr_backend.detect_text(
        prepare_ocr_image(banner_frame, profile="banner"),
        hint="end_banner",
    )
    banner_text = " ".join(collapse_whitespace(item.text).upper() for item in detections)
    return "PLAY NEXT GAME" in banner_text


def parse_top_banner_text(frame: np.ndarray, roi_config: ROIConfig, ocr_backend: OCRBackend) -> str:
    """Parse the top banner text once for reuse within a frame pass."""

    top_frame = crop_roi(frame, roi_config.require("top_banner_region"))
    detections = ocr_backend.detect_text(
        prepare_ocr_image(top_frame, profile="banner"),
        hint="top_banner",
    )
    return " ".join(collapse_whitespace(item.text).upper() for item in detections)


def is_winner_banner_text(banner_text: str) -> bool:
    """Return whether a parsed top banner contains a winner message."""

    return "YOUR TEAM WINS" in banner_text or "OPPOSING TEAM WINS" in banner_text


def winner_banner_visible(frame: np.ndarray, roi_config: ROIConfig, ocr_backend: OCRBackend) -> bool:
    """Return whether the top banner contains a winner message."""

    return is_winner_banner_text(parse_top_banner_text(frame, roi_config, ocr_backend))


def is_setup_screen_text(top_text: str) -> bool:
    """Return whether parsed top-banner text matches the setup/lobby screen."""

    return any(
        token in top_text
        for token in (
            "START GAME",
            "GAME SETTINGS",
            "JOIN TEAM",
            "WORD PACKS",
            "CLASSIC",
        )
    )


def is_setup_panel_text(panel_text: str) -> bool:
    """Return whether parsed central-panel text matches the setup/lobby screen."""

    normalized = collapse_whitespace(panel_text).upper()
    if "GAME SETTINGS" in normalized:
        return True
    if "WORD PACKS" in normalized:
        return True
    return "TIMER" in normalized


def setup_screen_visible(frame: np.ndarray, roi_config: ROIConfig, ocr_backend: OCRBackend) -> bool:
    """Return whether the setup/lobby screen is visible."""

    top_banner_text = parse_top_banner_text(frame, roi_config, ocr_backend)
    if is_winner_banner_text(top_banner_text):
        return False
    if is_setup_screen_text(top_banner_text):
        return True
    if start_game_button_visible(frame):
        return True
    if has_play_next_game(frame, roi_config, ocr_backend):
        return False
    return is_setup_panel_text(parse_setup_panel_text(frame, ocr_backend))


def make_board_fingerprint(frame: np.ndarray, roi_config: ROIConfig, *, hash_size: int = 8) -> str:
    """Compute a simple perceptual fingerprint for the board region."""

    board_frame = crop_roi(frame, roi_config.require("board_region"))
    grayscale = Image.fromarray(board_frame).convert("L").resize((hash_size, hash_size))
    pixels = np.asarray(grayscale, dtype=np.float32)
    threshold = float(pixels.mean())
    bits = "".join("1" if value >= threshold else "0" for value in pixels.flatten())
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def detect_assassin_in_events(card_colors: list[CardColor]) -> bool:
    """Return whether the parsed events include an assassin reveal."""

    return CardColor.BLACK in card_colors


def _counter_observation_from_attempt(
    counter_frame: np.ndarray,
    attempt: OCRAttemptResult,
) -> CounterObservation:
    detection = pick_best_detection(attempt.detections)
    raw_text = "" if detection is None else collapse_whitespace(detection.text)
    normalized_value = None if detection is None else _normalize_counter_value(detection.text)
    confidence = 0.0 if detection is None else float(detection.confidence)

    heuristic_digit, heuristic_confidence = _estimate_counter_digit(counter_frame)
    if heuristic_digit is not None and (
        normalized_value is None
        or confidence < 0.96
        or (normalized_value in {0, 1} and heuristic_digit not in {0, 1})
    ):
        normalized_value = heuristic_digit
        confidence = max(confidence, heuristic_confidence)
        if not raw_text:
            raw_text = str(heuristic_digit)

    return CounterObservation(
        value=normalized_value,
        confidence=max(0.0, min(1.0, confidence)),
        raw_text=raw_text,
        variant_name=attempt.variant_label,
        score=attempt.score,
    )


def _normalize_counter_value(text: str) -> int | None:
    normalized = collapse_whitespace(text).upper().strip(".,:;()[]{}")
    if not normalized:
        return None
    digits = "".join(character for character in normalized if character.isdigit())
    if not digits:
        if normalized in {"I", "L", "|"}:
            return 1
        return None
    if len(digits) > 1:
        digits = digits[-1]
    return int(digits)


def _estimate_counter_digit(counter_frame: np.ndarray) -> tuple[int | None, float]:
    if counter_frame.size == 0:
        return None, 0.0

    grayscale = counter_frame.mean(axis=2) if counter_frame.ndim == 3 else counter_frame.astype(np.float32)
    percentile_90 = float(np.percentile(grayscale, 90))
    best_digit: int | None = None
    best_confidence = 0.0
    for multiplier in (0.72, 0.78, 0.84, 0.9):
        candidate_digit, candidate_confidence = _estimate_counter_digit_for_threshold(
            grayscale,
            bright_threshold=max(150.0, percentile_90 * multiplier),
        )
        if candidate_confidence > best_confidence:
            best_digit = candidate_digit
            best_confidence = candidate_confidence
    return best_digit, best_confidence


def _estimate_counter_digit_for_threshold(
    grayscale: np.ndarray,
    *,
    bright_threshold: float,
) -> tuple[int | None, float]:
    bright_mask = grayscale >= bright_threshold
    coordinates = np.argwhere(bright_mask)
    if coordinates.size == 0:
        return None, 0.0

    top = int(coordinates[:, 0].min())
    bottom = int(coordinates[:, 0].max()) + 1
    left = int(coordinates[:, 1].min())
    right = int(coordinates[:, 1].max()) + 1
    digit_mask = bright_mask[top:bottom, left:right]
    if digit_mask.size == 0:
        return None, 0.0

    digit_height, digit_width = digit_mask.shape
    if digit_height < 6 or digit_width < 3:
        return None, 0.0

    hole_count, hole_centers = _count_mask_holes(digit_mask)
    aspect_ratio = digit_width / max(digit_height, 1)
    lower_left_density = float(digit_mask[digit_height // 2 :, : max(1, digit_width // 3)].mean())
    lower_right_density = float(
        digit_mask[digit_height // 2 :, max(0, digit_width - max(1, digit_width // 3)) :].mean()
    )

    if hole_count >= 2:
        return 8, 0.98
    if hole_count == 1:
        hole_center_y = hole_centers[0][0] / max(digit_height, 1)
        if lower_left_density < lower_right_density * 0.7 and hole_center_y <= 0.58:
            return 9, 0.9
        if lower_left_density > lower_right_density * 1.3 and hole_center_y >= 0.42:
            return 6, 0.86
        return 0, 0.82
    if aspect_ratio <= 0.48:
        return 1, 0.82
    return None, 0.0


def _count_mask_holes(mask: np.ndarray) -> tuple[int, list[tuple[float, float]]]:
    height, width = mask.shape
    inverse_mask = ~mask
    visited = np.zeros_like(inverse_mask, dtype=bool)
    hole_centers: list[tuple[float, float]] = []
    hole_count = 0
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for row in range(height):
        for col in range(width):
            if not inverse_mask[row, col] or visited[row, col]:
                continue

            stack = [(row, col)]
            visited[row, col] = True
            component: list[tuple[int, int]] = []
            touches_border = False
            while stack:
                current_row, current_col = stack.pop()
                component.append((current_row, current_col))
                if current_row in {0, height - 1} or current_col in {0, width - 1}:
                    touches_border = True
                for row_delta, col_delta in neighbors:
                    next_row = current_row + row_delta
                    next_col = current_col + col_delta
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if visited[next_row, next_col] or not inverse_mask[next_row, next_col]:
                        continue
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))

            if touches_border or len(component) < 10:
                continue
            hole_count += 1
            average_row = sum(item_row for item_row, _ in component) / len(component)
            average_col = sum(item_col for _, item_col in component) / len(component)
            hole_centers.append((average_row, average_col))

    return hole_count, hole_centers
