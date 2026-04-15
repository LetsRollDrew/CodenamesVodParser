"""Board-reveal detection helpers for smoke parsing."""

from __future__ import annotations

import numpy as np

from app.gamelog_parser import ClueEvent, GuessEvent
from app.models import BoardCell, BoardState, CardColor
from app.roi_config import ROIConfig, crop_roi
from app.smoke_banner import _last_clue_event


def _capture_board_reveal_baseline(
    frame: np.ndarray,
    roi_config: ROIConfig,
    board_state: BoardState,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    board_frame = crop_roi(frame, roi_config.require("board_region"))
    baseline: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for cell in board_state.cells:
        sample = _sample_board_cell_surface(board_frame, cell.box)
        baseline[(cell.row, cell.col)] = _cell_color_statistics(sample)
    return baseline


def _detect_board_reveal_guess_events(
    frame: np.ndarray,
    *,
    roi_config: ROIConfig,
    board_state: BoardState,
    baseline: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    previous_reveals: dict[tuple[int, int], CardColor | None],
    history: list[ClueEvent | GuessEvent],
    timestamp_sec: float,
) -> tuple[list[GuessEvent], dict[tuple[int, int], CardColor | None]]:
    current_turn = _last_clue_event(history)
    if current_turn is None:
        return [], previous_reveals

    board_frame = crop_roi(frame, roi_config.require("board_region"))
    current_reveals = dict(previous_reveals)
    new_events: list[GuessEvent] = []

    for cell in board_state.cells:
        key = (cell.row, cell.col)
        baseline_stats = baseline.get(key)
        if baseline_stats is None:
            continue

        sample = _sample_board_cell_surface(board_frame, cell.box)
        current_color = _detect_revealed_card_color(sample, baseline_stats)
        current_reveals[key] = current_color
        if previous_reveals.get(key) is not None or current_color is None:
            continue

        new_events.append(
            GuessEvent(
                player_name="unknown",
                word=cell.word,
                card_color=current_color,
                timestamp_sec=timestamp_sec,
                sequence_index=(cell.row * 5) + cell.col,
                confidence=0.88,
            )
        )

    return new_events, current_reveals


def _sample_board_cell_surface(board_frame: np.ndarray, box: BoardCell | object) -> np.ndarray:
    cell_frame = board_frame[box.top : box.bottom, box.left : box.right]
    height, width = cell_frame.shape[:2]
    if height <= 0 or width <= 0:
        return cell_frame.copy()
    top = max(int(round(height * 0.16)), 0)
    bottom = max(int(round(height * 0.48)), top + 1)
    left = max(int(round(width * 0.18)), 0)
    right = max(int(round(width * 0.82)), left + 1)
    return cell_frame[top:bottom, left:right].copy()


def _cell_color_statistics(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if sample.size == 0:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    mean_bgr = sample.reshape(-1, sample.shape[-1]).mean(axis=0).astype(np.float32)
    hsv = np.asarray(detect_hsv(sample), dtype=np.float32)
    return mean_bgr, hsv


def detect_hsv(sample: np.ndarray) -> np.ndarray:
    import cv2

    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    return hsv.reshape(-1, hsv.shape[-1]).mean(axis=0)


def _detect_revealed_card_color(
    sample: np.ndarray,
    baseline_stats: tuple[np.ndarray, np.ndarray],
) -> CardColor | None:
    if sample.size == 0:
        return None

    mean_bgr, mean_hsv = _cell_color_statistics(sample)
    baseline_bgr, baseline_hsv = baseline_stats
    delta_bgr = float(np.linalg.norm(mean_bgr - baseline_bgr))
    delta_hsv = float(np.linalg.norm(mean_hsv - baseline_hsv))
    if delta_bgr < 48.0 and delta_hsv < 44.0:
        return None

    blue, green, red = map(float, mean_bgr[:3])
    hue, saturation, value = map(float, mean_hsv[:3])
    if value < 75.0:
        return CardColor.BLACK
    if red > blue + 40.0 and red > green + 25.0 and saturation > 110.0:
        return CardColor.RED
    if blue > red + 28.0 and blue > green + 15.0 and 75.0 <= hue <= 135.0 and saturation > 95.0:
        return CardColor.BLUE
    return CardColor.NEUTRAL
