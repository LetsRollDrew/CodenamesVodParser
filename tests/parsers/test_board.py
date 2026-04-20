from __future__ import annotations

import numpy as np

from app.parsers.board import (
    board_dictionary,
    estimate_board_cell_boxes,
    normalize_board_word,
    parse_board_words,
    snap_word_to_board,
)
from app.core.models import ImageBoundingBox, OCRDetection
from app.infra.roi_config import ROIConfig


class FakeOCRBackend:
    def __init__(self, responses: dict[str, list[OCRDetection]]) -> None:
        self.responses = responses

    def detect_text(self, image: np.ndarray, *, hint: str | None = None) -> list[OCRDetection]:
        assert image.size > 0
        return list(self.responses.get(hint or "", []))


def make_roi_config() -> ROIConfig:
    return ROIConfig.from_raw(
        {
            "left_team_panel": {"x": 0.0, "y": 0.0, "width": 0.15, "height": 1.0},
            "right_team_panel": {"x": 0.85, "y": 0.0, "width": 0.15, "height": 1.0},
            "board_region": {"x": 0.20, "y": 0.10, "width": 0.60, "height": 0.70},
            "game_log_region": {"x": 0.80, "y": 0.40, "width": 0.15, "height": 0.40},
            "left_counter_region": {"x": 0.04, "y": 0.40, "width": 0.08, "height": 0.08},
            "right_counter_region": {"x": 0.88, "y": 0.40, "width": 0.08, "height": 0.08},
            "top_banner_region": {"x": 0.20, "y": 0.00, "width": 0.60, "height": 0.10},
            "end_banner_region": {"x": 0.20, "y": 0.80, "width": 0.60, "height": 0.15},
        }
    )


def test_estimate_board_cell_boxes_returns_row_major_grid() -> None:
    board_frame = np.zeros((500, 700, 3), dtype=np.uint8)

    boxes = estimate_board_cell_boxes(board_frame)

    assert len(boxes) == 25
    assert boxes[0].left < boxes[1].left
    assert boxes[0].top == boxes[1].top
    assert boxes[0].top < boxes[5].top
    assert all(box.width > 0 and box.height > 0 for box in boxes)


def test_normalize_board_word_uppercases_and_preserves_periods() -> None:
    assert normalize_board_word("  St.   Patrick \n") == "ST. PATRICK"


def test_parse_board_words_returns_row_major_cells() -> None:
    frame = np.zeros((800, 1200, 3), dtype=np.uint8)
    roi_config = make_roi_config()
    responses: dict[str, list[OCRDetection]] = {}

    words = [
        "GENIUS",
        "HAIR",
        "STAFF",
        "SCARECROW",
        "BOWL",
        "LIGHTHOUSE",
        "PUMPKIN",
        "PIN",
        "SPRAY",
        "CLOAK",
        "BARN",
        "TAP",
        "AUSTRALIA",
        "ICELAND",
        "MUD",
        "MAP",
        "ANTARCTICA",
        "BULB",
        "WITCH",
        "ST. PATRICK",
        "LOG",
        "CROW",
        "NIGHT",
        "PENNY",
        "SOLDIER",
    ]

    for index, word in enumerate(words):
        row = index // 5
        col = index % 5
        responses[f"board:{row}:{col}"] = [
            OCRDetection(
                text=word,
                confidence=0.95,
                box=ImageBoundingBox(left=5, top=5, right=40, bottom=20),
            )
        ]

    board_state = parse_board_words(frame, roi_config, FakeOCRBackend(responses))

    assert board_state.words == words
    assert board_state.cells[16].row == 3
    assert board_state.cells[16].col == 1
    assert board_state.cells[16].word == "ANTARCTICA"


def test_snap_word_to_board_uses_known_board_dictionary() -> None:
    board_state = parse_board_words(
        np.zeros((800, 1200, 3), dtype=np.uint8),
        make_roi_config(),
        FakeOCRBackend(
            {
                **{
                    f"board:{row}:{col}": [
                        OCRDetection(
                            text=f"WORD{row}{col}",
                            confidence=0.9,
                            box=ImageBoundingBox(left=1, top=1, right=10, bottom=10),
                        )
                    ]
                    for row in range(5)
                    for col in range(5)
                },
                "board:3:1": [
                    OCRDetection(
                        text="ANTARCTICA",
                        confidence=0.99,
                        box=ImageBoundingBox(left=1, top=1, right=10, bottom=10),
                    )
                ],
            }
        ),
    )

    assert snap_word_to_board("Antarctlca", board_state) == "ANTARCTICA"
    assert "ANTARCTICA" in board_dictionary(board_state)


def test_parse_board_words_keeps_dictionary_correction_metadata() -> None:
    frame = np.zeros((800, 1200, 3), dtype=np.uint8)
    roi_config = make_roi_config()
    responses = {
        "board:0:0": [
            OCRDetection(
                text="ANTARCTIC",
                confidence=0.93,
                box=ImageBoundingBox(left=5, top=5, right=40, bottom=20),
            )
        ]
    }

    board_state = parse_board_words(frame, roi_config, FakeOCRBackend(responses))
    corrected_cell = board_state.cells[0]

    assert corrected_cell.raw_ocr_text == "ANTARCTIC"
    assert corrected_cell.word == "ANTARCTICA"
    assert corrected_cell.corrected_by_default_dictionary
    assert corrected_cell.dictionary_similarity > 0.0
