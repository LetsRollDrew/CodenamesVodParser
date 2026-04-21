from __future__ import annotations

import numpy as np

from app.parsers.gamelog import ClueEvent, GuessEvent
from app.runtime.smoke.board_reveals import (
    _capture_board_reveal_baseline,
    _detect_board_reveal_guess_events,
    _detect_revealed_card_color,
    _sync_board_reveals_from_history,
)
from app.core.models import BoardCell, BoardState, CardColor, ImageBoundingBox, TeamColor
from app.infra.roi_config import ROIConfig


def _make_test_roi_config() -> ROIConfig:
    return ROIConfig.from_raw(
        {
            "left_team_panel": {"x": 0.0, "y": 0.0, "width": 0.1, "height": 0.1},
            "right_team_panel": {"x": 0.9, "y": 0.0, "width": 0.1, "height": 0.1},
            "board_region": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0},
            "game_log_region": {"x": 0.0, "y": 0.9, "width": 0.1, "height": 0.1},
            "left_counter_region": {"x": 0.0, "y": 0.1, "width": 0.1, "height": 0.1},
            "right_counter_region": {"x": 0.9, "y": 0.1, "width": 0.1, "height": 0.1},
            "top_banner_region": {"x": 0.0, "y": 0.0, "width": 0.2, "height": 0.1},
            "end_banner_region": {"x": 0.0, "y": 0.8, "width": 0.2, "height": 0.2},
        }
    )


def _make_board_state() -> BoardState:
    return BoardState(
        cells=[
            BoardCell(
                row=0,
                col=0,
                word="SUIT",
                confidence=1.0,
                box=ImageBoundingBox(left=10, top=10, right=60, bottom=60),
            )
        ]
    )


def _make_multi_cell_board_state() -> BoardState:
    cells = []
    for index in range(4):
        cells.append(
            BoardCell(
                row=0,
                col=index,
                word=f"WORD{index}",
                confidence=1.0,
                box=ImageBoundingBox(
                    left=5 + (index * 22),
                    top=10,
                    right=25 + (index * 22),
                    bottom=40,
                ),
            )
        )
    return BoardState(cells=cells)


def _make_clue_history() -> list[ClueEvent]:
    return [
        ClueEvent(
            team_color=TeamColor.RED,
            spymaster_name="shadowsn",
            clue_text="PROM",
            clue_count="4",
            timestamp_sec=1.0,
            sequence_index=0,
            confidence=1.0,
        )
    ]


def test_detect_board_reveal_guess_events_requires_consecutive_confirmations() -> None:
    roi_config = _make_test_roi_config()
    board_state = _make_board_state()
    baseline_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    reveal_frame = baseline_frame.copy()
    reveal_frame[10:60, 10:60] = (0, 0, 255)
    baseline = _capture_board_reveal_baseline(baseline_frame, roi_config, board_state)
    previous_reveals = {(0, 0): None}

    first_events, first_reveals, first_pending = _detect_board_reveal_guess_events(
        reveal_frame,
        roi_config=roi_config,
        board_state=board_state,
        baseline=baseline,
        previous_reveals=previous_reveals,
        pending_reveals={},
        history=_make_clue_history(),
        timestamp_sec=2.0,
        allow_board_reveal_scan=True,
    )

    assert first_events == []
    assert first_reveals == previous_reveals
    assert first_pending == {(0, 0): (CardColor.RED, 1)}

    second_events, second_reveals, second_pending = _detect_board_reveal_guess_events(
        reveal_frame,
        roi_config=roi_config,
        board_state=board_state,
        baseline=baseline,
        previous_reveals=first_reveals,
        pending_reveals=first_pending,
        history=_make_clue_history(),
        timestamp_sec=2.5,
        allow_board_reveal_scan=True,
    )

    assert len(second_events) == 1
    assert second_events[0].word == "SUIT"
    assert second_events[0].card_color is CardColor.RED
    assert second_reveals == {(0, 0): CardColor.RED}
    assert second_pending == {}


def test_detect_board_reveal_guess_events_clears_pending_when_scan_is_disallowed() -> None:
    roi_config = _make_test_roi_config()
    board_state = _make_board_state()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    baseline = _capture_board_reveal_baseline(frame, roi_config, board_state)

    events, reveals, pending = _detect_board_reveal_guess_events(
        frame,
        roi_config=roi_config,
        board_state=board_state,
        baseline=baseline,
        previous_reveals={(0, 0): None},
        pending_reveals={(0, 0): (CardColor.RED, 1)},
        history=_make_clue_history(),
        timestamp_sec=2.0,
        allow_board_reveal_scan=False,
    )

    assert events == []
    assert reveals == {(0, 0): None}
    assert pending == {}


def test_detect_board_reveal_guess_events_drops_implausible_multi_reveal_bursts() -> None:
    roi_config = _make_test_roi_config()
    board_state = _make_multi_cell_board_state()
    baseline_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    reveal_frame = baseline_frame.copy()
    for index in range(4):
        left = 5 + (index * 22)
        right = 25 + (index * 22)
        reveal_frame[10:40, left:right] = (0, 0, 255)
    baseline = _capture_board_reveal_baseline(baseline_frame, roi_config, board_state)

    events, reveals, pending = _detect_board_reveal_guess_events(
        reveal_frame,
        roi_config=roi_config,
        board_state=board_state,
        baseline=baseline,
        previous_reveals={(0, index): None for index in range(4)},
        pending_reveals={},
        history=_make_clue_history(),
        timestamp_sec=2.0,
        allow_board_reveal_scan=True,
        minimum_confirmations=1,
    )

    assert events == []
    assert reveals == {(0, index): None for index in range(4)}
    assert pending == {}


def test_detect_revealed_card_color_prefers_dark_saturated_team_color_over_black() -> None:
    baseline_sample = np.zeros((20, 20, 3), dtype=np.uint8)
    dark_red_sample = np.full((20, 20, 3), (20, 20, 120), dtype=np.uint8)
    baseline_stats = (
        baseline_sample.reshape(-1, 3).mean(axis=0).astype(np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )

    detected = _detect_revealed_card_color(dark_red_sample, baseline_stats)

    assert detected is CardColor.RED


def test_sync_board_reveals_from_history_marks_logged_guesses_as_already_revealed() -> None:
    board_state = _make_board_state()
    reveals, pending = _sync_board_reveals_from_history(
        previous_reveals={(0, 0): None},
        pending_reveals={(0, 0): (CardColor.RED, 1)},
        board_state=board_state,
        history=[
            GuessEvent(
                player_name="unknown",
                word="SUIT",
                card_color=CardColor.RED,
                timestamp_sec=2.0,
                sequence_index=0,
                confidence=0.9,
            )
        ],
    )

    assert reveals == {(0, 0): CardColor.RED}
    assert pending == {}
