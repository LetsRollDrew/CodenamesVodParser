from __future__ import annotations

import numpy as np

from app.frame_detectors import (
    detect_assassin_in_events,
    has_play_next_game,
    make_board_fingerprint,
    parse_counter_value,
    parse_game_counters,
    setup_screen_visible,
)
from app.models import CardColor, ImageBoundingBox, OCRDetection
from app.roi_config import ROIConfig


class FakeOCRBackend:
    def __init__(self, responses: dict[str, list[OCRDetection]]) -> None:
        self.responses = responses

    def detect_text(self, image: np.ndarray, *, hint: str | None = None) -> list[OCRDetection]:
        assert image.size > 0
        return list(self.responses.get(hint or "", []))


def make_roi_config() -> ROIConfig:
    return ROIConfig.from_raw(
        {
            "left_team_panel": {"x": 0.0, "y": 0.0, "width": 0.18, "height": 1.0},
            "right_team_panel": {"x": 0.82, "y": 0.0, "width": 0.18, "height": 1.0},
            "board_region": {"x": 0.18, "y": 0.10, "width": 0.60, "height": 0.72},
            "game_log_region": {"x": 0.80, "y": 0.50, "width": 0.18, "height": 0.40},
            "left_counter_region": {"x": 0.05, "y": 0.35, "width": 0.05, "height": 0.08},
            "right_counter_region": {"x": 0.90, "y": 0.35, "width": 0.05, "height": 0.08},
            "top_banner_region": {"x": 0.18, "y": 0.0, "width": 0.60, "height": 0.10},
            "end_banner_region": {"x": 0.18, "y": 0.80, "width": 0.64, "height": 0.16},
        }
    )


def test_parse_counter_value_extracts_digits() -> None:
    counter = np.zeros((40, 40, 3), dtype=np.uint8)
    value = parse_counter_value(
        counter,
        FakeOCRBackend(
            {
                "left_counter": [
                    OCRDetection(
                        text=" 9 ",
                        confidence=0.95,
                        box=ImageBoundingBox(left=1, top=1, right=10, bottom=10),
                    )
                ]
            }
        ),
        hint="left_counter",
    )

    assert value == 9


def test_parse_game_counters_reads_both_sides() -> None:
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    roi_config = make_roi_config()
    ocr_backend = FakeOCRBackend(
        {
            "left_counter": [
                OCRDetection(
                    text="0",
                    confidence=0.9,
                    box=ImageBoundingBox(left=1, top=1, right=8, bottom=10),
                )
            ],
            "right_counter": [
                OCRDetection(
                    text="1",
                    confidence=0.9,
                    box=ImageBoundingBox(left=1, top=1, right=8, bottom=10),
                )
            ],
        }
    )

    assert parse_game_counters(frame, roi_config, ocr_backend) == (0, 1)


def test_banner_detectors_use_targeted_ocr_regions() -> None:
    frame = np.ones((600, 1000, 3), dtype=np.uint8) * 255
    roi_config = make_roi_config()
    ocr_backend = FakeOCRBackend(
        {
            "end_banner": [
                OCRDetection(
                    text="Play next game",
                    confidence=0.99,
                    box=ImageBoundingBox(left=1, top=1, right=40, bottom=10),
                )
            ],
            "top_banner": [
                OCRDetection(
                    text="Game Settings",
                    confidence=0.99,
                    box=ImageBoundingBox(left=1, top=1, right=40, bottom=10),
                )
            ],
        }
    )

    assert has_play_next_game(frame, roi_config, ocr_backend)
    assert setup_screen_visible(frame, roi_config, ocr_backend)


def test_make_board_fingerprint_changes_with_board_pixels() -> None:
    roi_config = make_roi_config()
    first = np.zeros((600, 1000, 3), dtype=np.uint8)
    second = first.copy()
    second[120:320, 220:620] = (255, 255, 255)

    assert make_board_fingerprint(first, roi_config) != make_board_fingerprint(second, roi_config)


def test_detect_assassin_in_events_returns_true_for_black() -> None:
    assert detect_assassin_in_events([CardColor.BLUE, CardColor.BLACK])
    assert not detect_assassin_in_events([CardColor.BLUE, CardColor.RED])
