from __future__ import annotations

from typing import Any

import numpy as np

from app.core.models import ImageBoundingBox, OCRDetection
from app.infra.roi_config import ROIConfig
from app.infra.vod_source import FrameSample


class FakeVodSource:
    def __init__(self, frames: list[FrameSample]) -> None:
        self.frames = frames
        self.yielded = 0
        self.calls: list[tuple[float, float, float]] = []

    def iter_window_frames(self, vod_url: str, start_sec: float, duration_sec: float, fps: float):
        del vod_url
        self.calls.append((start_sec, duration_sec, fps))
        window_end_sec = start_sec + duration_sec
        for frame in self.frames:
            if start_sec <= frame.timestamp_sec < window_end_sec:
                self.yielded += 1
                yield frame


class SequencedOCRBackend:
    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.indices: dict[str, int] = {}

    def detect_text(self, image, *, hint: str | None = None):
        assert image.size > 0
        key = hint or ""
        value = self.responses.get(key, [])
        if value and isinstance(value[0], OCRDetection):
            return list(value)
        sequence = value if isinstance(value, list) else []
        index = self.indices.get(key, 0)
        self.indices[key] = index + 1
        if not sequence:
            return []
        return list(sequence[min(index, len(sequence) - 1)])


def make_roi_config() -> ROIConfig:
    return ROIConfig.from_raw(
        {
            "left_team_panel": {"x": 0.0, "y": 0.0, "width": 0.18, "height": 1.0},
            "right_team_panel": {"x": 0.82, "y": 0.0, "width": 0.18, "height": 1.0},
            "board_region": {"x": 0.18, "y": 0.10, "width": 0.60, "height": 0.72},
            "game_log_region": {"x": 0.80, "y": 0.50, "width": 0.18, "height": 0.40},
            "game_log_content_region": {"x": 0.82, "y": 0.60, "width": 0.14, "height": 0.24},
            "left_counter_region": {"x": 0.05, "y": 0.35, "width": 0.05, "height": 0.08},
            "right_counter_region": {"x": 0.90, "y": 0.35, "width": 0.05, "height": 0.08},
            "top_banner_region": {"x": 0.18, "y": 0.0, "width": 0.60, "height": 0.10},
            "end_banner_region": {"x": 0.18, "y": 0.80, "width": 0.64, "height": 0.16},
        }
    )


def make_frame(fill: int) -> np.ndarray:
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    frame[300:540, 800:980] = fill
    return frame


def make_board_responses(anchor_word: str) -> dict[str, list[OCRDetection]]:
    responses: dict[str, list[OCRDetection]] = {}
    for row in range(5):
        for col in range(5):
            detections = [
                OCRDetection(
                    text=anchor_word if (row, col) == (0, 0) else f"WORD{chr(65 + row)}{chr(65 + col)}",
                    confidence=0.95,
                    box=ImageBoundingBox(left=1, top=1, right=20, bottom=10),
                )
            ]
            for variant_index in range(5):
                responses[f"board:{row}:{col}:v{variant_index}"] = list(detections)
    return responses
