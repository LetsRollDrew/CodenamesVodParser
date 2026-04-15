"""Frame-level detectors used by the game reconstruction pipeline."""

from __future__ import annotations

import numpy as np
from PIL import Image

from app.models import CardColor
from app.ocr import OCRBackend, collapse_whitespace, detect_text_with_fallback, pick_best_detection, prepare_ocr_image
from app.roi_config import ROIConfig, crop_roi

_SETUP_PANEL_REGION = (0.26, 0.20, 0.50, 0.44)
_START_BUTTON_REGION = (0.33, 0.90, 0.36, 0.07)


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
    detections = detect_text_with_fallback(
        ocr_backend,
        setup_frame,
        prepared_image=prepare_ocr_image(setup_frame, profile="banner"),
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

    detections = detect_text_with_fallback(
        ocr_backend,
        counter_frame,
        prepared_image=prepare_ocr_image(counter_frame, profile="counter"),
        hint=hint,
    )
    detection = pick_best_detection(detections)
    if detection is None:
        return None

    digits = "".join(character for character in detection.text if character.isdigit())
    if not digits:
        return None
    if len(digits) > 1:
        digits = digits[-1]
    return int(digits)


def parse_game_counters(
    frame: np.ndarray,
    roi_config: ROIConfig,
    ocr_backend: OCRBackend,
) -> tuple[int | None, int | None]:
    """Parse the left and right remaining-card counters from a frame."""

    left_counter = parse_counter_value(
        crop_roi(frame, roi_config.require("left_counter_region")),
        ocr_backend,
        hint="left_counter",
    )
    right_counter = parse_counter_value(
        crop_roi(frame, roi_config.require("right_counter_region")),
        ocr_backend,
        hint="right_counter",
    )
    return left_counter, right_counter


def has_play_next_game(frame: np.ndarray, roi_config: ROIConfig, ocr_backend: OCRBackend) -> bool:
    """Return whether the end banner region contains 'Play next game'."""

    banner_frame = crop_roi(frame, roi_config.require("end_banner_region"))
    detections = detect_text_with_fallback(
        ocr_backend,
        banner_frame,
        prepared_image=prepare_ocr_image(banner_frame, profile="banner"),
        hint="end_banner",
    )
    banner_text = " ".join(collapse_whitespace(item.text).upper() for item in detections)
    return "PLAY NEXT GAME" in banner_text


def parse_top_banner_text(frame: np.ndarray, roi_config: ROIConfig, ocr_backend: OCRBackend) -> str:
    """Parse the top banner text once for reuse within a frame pass."""

    top_frame = crop_roi(frame, roi_config.require("top_banner_region"))
    detections = detect_text_with_fallback(
        ocr_backend,
        top_frame,
        prepared_image=prepare_ocr_image(top_frame, profile="banner"),
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
    if is_winner_banner_text(top_banner_text) or has_play_next_game(frame, roi_config, ocr_backend):
        return False
    if is_setup_screen_text(top_banner_text):
        return True
    if start_game_button_visible(frame):
        return True
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
