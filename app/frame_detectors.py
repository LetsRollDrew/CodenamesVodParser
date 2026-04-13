"""Frame-level detectors used by the game reconstruction pipeline."""

from __future__ import annotations

from PIL import Image
import numpy as np

from app.models import CardColor
from app.ocr import OCRBackend, collapse_whitespace, detect_text_with_fallback, pick_best_detection, prepare_ocr_image
from app.roi_config import ROIConfig, crop_roi


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

    return "START GAME" in top_text or "GAME SETTINGS" in top_text


def setup_screen_visible(frame: np.ndarray, roi_config: ROIConfig, ocr_backend: OCRBackend) -> bool:
    """Return whether the setup/lobby screen is visible."""

    return is_setup_screen_text(parse_top_banner_text(frame, roi_config, ocr_backend))


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
