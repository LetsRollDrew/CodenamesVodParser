"""Concrete OCR backend adapters with lazy optional imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from app.models import ImageBoundingBox, OCRDetection
from app.ocr import OCRBackend


class OCRBackendError(RuntimeError):
    """Raised when an OCR backend cannot be created or used."""


def create_ocr_backend(
    backend_name: Literal["paddle", "easyocr"] = "paddle",
    **kwargs: Any,
) -> OCRBackend:
    """Create a concrete OCR backend by name."""

    if backend_name == "paddle":
        return PaddleOCRBackend(**kwargs)
    if backend_name == "easyocr":
        return EasyOCRBackend(**kwargs)
    raise ValueError(f"Unsupported OCR backend: {backend_name}")


@dataclass
class PaddleOCRBackend:
    """Adapter around `paddleocr.PaddleOCR`."""

    language: str = "en"
    use_angle_cls: bool = False
    _client: Any | None = None

    def __post_init__(self) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise OCRBackendError(
                "PaddleOCR is not installed. Install `paddleocr` to use the paddle backend."
            ) from exc
        self._client = PaddleOCR(lang=self.language, use_angle_cls=self.use_angle_cls)

    def detect_text(
        self,
        image: np.ndarray,
        *,
        hint: str | None = None,
    ) -> list[OCRDetection]:
        del hint
        if self._client is None:
            raise OCRBackendError("PaddleOCR backend was not initialized")
        rgb_image = image[:, :, ::-1] if image.ndim == 3 and image.shape[2] == 3 else image
        raw_results = self._client.ocr(rgb_image, cls=self.use_angle_cls)
        detections: list[OCRDetection] = []
        for line in raw_results or []:
            for item in line or []:
                points, text_conf = item
                text, confidence = text_conf
                xs = [int(round(point[0])) for point in points]
                ys = [int(round(point[1])) for point in points]
                detections.append(
                    OCRDetection(
                        text=str(text),
                        confidence=float(confidence),
                        box=ImageBoundingBox(
                            left=min(xs),
                            top=min(ys),
                            right=max(xs),
                            bottom=max(ys),
                        ),
                    )
                )
        return detections


@dataclass
class EasyOCRBackend:
    """Adapter around `easyocr.Reader`."""

    language: str = "en"
    gpu: bool = False
    _reader: Any | None = None

    def __post_init__(self) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise OCRBackendError(
                "EasyOCR is not installed. Install `easyocr` to use the easyocr backend."
            ) from exc
        self._reader = easyocr.Reader([self.language], gpu=self.gpu)

    def detect_text(
        self,
        image: np.ndarray,
        *,
        hint: str | None = None,
    ) -> list[OCRDetection]:
        del hint
        if self._reader is None:
            raise OCRBackendError("EasyOCR backend was not initialized")
        rgb_image = image[:, :, ::-1] if image.ndim == 3 and image.shape[2] == 3 else image
        raw_results = self._reader.readtext(rgb_image, detail=1)
        detections: list[OCRDetection] = []
        for item in raw_results or []:
            points, text, confidence = item
            xs = [int(round(point[0])) for point in points]
            ys = [int(round(point[1])) for point in points]
            detections.append(
                OCRDetection(
                    text=str(text),
                    confidence=float(confidence),
                    box=ImageBoundingBox(
                        left=min(xs),
                        top=min(ys),
                        right=max(xs),
                        bottom=max(ys),
                    ),
                )
            )
        return detections
