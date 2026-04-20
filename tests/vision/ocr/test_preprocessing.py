from __future__ import annotations

import numpy as np

from app.vision.ocr.preprocessing import detect_text_across_variants, detect_text_with_fallback, prepare_ocr_image
from app.core.models import ImageBoundingBox, OCRDetection


class RecordingOCRBackend:
    def __init__(self) -> None:
        self.hints: list[str | None] = []

    def detect_text(self, image: np.ndarray, *, hint: str | None = None) -> list[OCRDetection]:
        self.hints.append(hint)
        return [
            OCRDetection(
                text="TEST",
                confidence=0.9,
                box=ImageBoundingBox(left=1, top=1, right=10, bottom=10),
            )
        ]


def test_detect_text_with_fallback_keeps_hint_stable_and_preserves_raw_boxes() -> None:
    backend = RecordingOCRBackend()
    raw = np.zeros((10, 10, 3), dtype=np.uint8)
    prepared = np.zeros((50, 50, 3), dtype=np.uint8)

    detections = detect_text_with_fallback(
        backend,
        raw,
        prepared_image=prepared,
        hint="game_log",
    )

    assert backend.hints == ["game_log", "game_log"]
    assert detections[0].box.right == 10
    assert detections[0].box.bottom == 10


class PaddleLikeBackend:
    def __init__(self) -> None:
        self.hints: list[str | None] = []
        self.seen_shapes: list[tuple[int, int]] = []

    def detect_text(self, image: np.ndarray, *, hint: str | None = None) -> list[OCRDetection]:
        self.hints.append(hint)
        self.seen_shapes.append(image.shape[:2])
        box = ImageBoundingBox(left=1, top=1, right=10, bottom=10)
        if image.shape[0] <= 10:
            return [OCRDetection(text="RAW", confidence=0.9, box=box)]
        return [OCRDetection(text="PREPARED", confidence=0.9, box=ImageBoundingBox(left=5, top=5, right=50, bottom=50))]


def test_detect_text_with_fallback_keeps_paddle_raw_first_order() -> None:
    backend = PaddleLikeBackend()
    raw = np.zeros((10, 10, 3), dtype=np.uint8)
    prepared = np.zeros((50, 50, 3), dtype=np.uint8)

    detections = detect_text_with_fallback(
        backend,
        raw,
        prepared_image=prepared,
        hint="board:0:0",
    )

    assert backend.hints == ["board:0:0", "board:0:0"]
    assert backend.seen_shapes == [(10, 10), (50, 50)]
    assert detections[0].box.right <= raw.shape[1]


class VariantRecordingBackend:
    def __init__(self) -> None:
        self.seen_shapes: list[tuple[int, int]] = []

    def detect_text(self, image: np.ndarray, *, hint: str | None = None) -> list[OCRDetection]:
        del hint
        self.seen_shapes.append(image.shape[:2])
        if image.shape[0] <= 10:
            return [
                OCRDetection(
                    text="11 22 33",
                    confidence=0.85,
                    box=ImageBoundingBox(left=1, top=1, right=9, bottom=9),
                )
            ]
        return [
            OCRDetection(
                text="2",
                confidence=0.88,
                box=ImageBoundingBox(left=5, top=5, right=25, bottom=25),
            )
        ]


def test_detect_text_across_variants_prefers_task_plausibility_over_fragment_noise() -> None:
    backend = VariantRecordingBackend()
    raw = np.zeros((10, 10, 3), dtype=np.uint8)

    attempt = detect_text_across_variants(
        backend,
        raw,
        hint="center_clue_count",
    )

    assert len(backend.seen_shapes) >= 2
    assert attempt.variant_label != "raw"
    assert attempt.detections[0].text == "2"
    assert attempt.detections[0].box.right <= raw.shape[1]


def test_prepare_ocr_image_supports_legacy_profile_names() -> None:
    image = np.zeros((12, 18, 3), dtype=np.uint8)

    for profile in ("generic", "name_strip", "counter", "board_word", "banner", "game_log"):
        prepared = prepare_ocr_image(image, profile=profile)
        assert prepared.size > 0
        assert prepared.shape[0] >= image.shape[0]
        assert prepared.shape[1] >= image.shape[1]
