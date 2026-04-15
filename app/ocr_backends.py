"""Concrete OCR backend adapters with lazy optional imports."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np

from app.models import ImageBoundingBox, OCRDetection
from app.ocr import OCRBackend


class OCRBackendError(RuntimeError):
    """Raised when an OCR backend cannot be created or used."""


def create_ocr_backend(
    backend_name: Literal["paddle", "easyocr"] = "easyocr",
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
    device: str | None = None
    _client: Any | None = None

    def __post_init__(self) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise OCRBackendError(
                "PaddleOCR is not installed. Install `paddleocr` to use the paddle backend."
            ) from exc
        kwargs: dict[str, Any] = {
            "lang": self.language,
            "use_angle_cls": self.use_angle_cls,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        self._client = PaddleOCR(**kwargs)

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
        if hasattr(self._client, "predict"):
            raw_results = self._client.predict(
                rgb_image,
                use_textline_orientation=self.use_angle_cls,
            )
            return self._normalize_predict_results(raw_results)

        raw_results = self._client.ocr(rgb_image, cls=self.use_angle_cls)
        return self._normalize_legacy_results(raw_results)

    def _normalize_predict_results(self, raw_results: Any) -> list[OCRDetection]:
        detections: list[OCRDetection] = []
        for item in raw_results or []:
            payload = self._coerce_payload(item)
            if payload is None:
                continue

            texts = self._first_present(payload.get("rec_text"), payload.get("rec_texts"), [])
            scores = self._first_present(payload.get("rec_score"), payload.get("rec_scores"), [])
            polygons = self._first_present(payload.get("dt_polys"), payload.get("rec_polys"), [])

            text_items = self._ensure_list(texts)
            score_items = self._ensure_list(scores)
            polygon_items = self._ensure_list(polygons)
            count = min(len(text_items), len(score_items), len(polygon_items))

            for index in range(count):
                points = polygon_items[index]
                if points is None:
                    continue
                if hasattr(points, "size"):
                    if int(points.size) == 0:
                        continue
                elif len(points) == 0:
                    continue
                xs = [int(round(point[0])) for point in points]
                ys = [int(round(point[1])) for point in points]
                detections.append(
                    OCRDetection(
                        text=str(text_items[index]),
                        confidence=self._normalize_confidence(score_items[index]),
                        box=ImageBoundingBox(
                            left=min(xs),
                            top=min(ys),
                            right=max(xs),
                            bottom=max(ys),
                        ),
                    )
                )
        return detections

    def _normalize_legacy_results(self, raw_results: Any) -> list[OCRDetection]:
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
                        confidence=self._normalize_confidence(confidence),
                        box=ImageBoundingBox(
                            left=min(xs),
                            top=min(ys),
                            right=max(xs),
                            bottom=max(ys),
                        ),
                    )
                )
        return detections

    @staticmethod
    def _coerce_payload(item: Any) -> dict[str, Any] | None:
        if isinstance(item, dict):
            payload = item.get("res", item)
            return payload if isinstance(payload, dict) else None
        if hasattr(item, "res"):
            payload = getattr(item, "res")
            return payload if isinstance(payload, dict) else None
        if hasattr(item, "to_dict"):
            converted = item.to_dict()
            if isinstance(converted, dict):
                payload = converted.get("res", converted)
                return payload if isinstance(payload, dict) else None
        return None

    @staticmethod
    def _ensure_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            converted = value.tolist()
            if isinstance(converted, list):
                return converted
            return [converted]
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @staticmethod
    def _first_present(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    @staticmethod
    def _normalize_confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0

        if not math.isfinite(confidence):
            return 0.0
        if 0.0 <= confidence <= 1.0:
            return confidence
        if 1.0 < confidence <= 100.0:
            return confidence / 100.0
        return 0.0


@dataclass
class EasyOCRBackend:
    """Adapter around `easyocr.Reader`."""

    language: str = "en"
    gpu: bool = False
    _reader: Any | None = None
    _easyocr_module: Any | None = None

    def __post_init__(self) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise OCRBackendError(
                "EasyOCR is not installed. Install `easyocr` to use the easyocr backend."
            ) from exc
        self._easyocr_module = easyocr
        self._reader = self._create_reader(gpu=self.gpu)

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
        try:
            raw_results = self._reader.readtext(rgb_image, detail=1)
        except RuntimeError as exc:
            if not self._should_fallback_to_cpu(exc):
                raise
            self.gpu = False
            self._reader = self._create_reader(gpu=False)
            raw_results = self._reader.readtext(rgb_image, detail=1)
        detections: list[OCRDetection] = []
        for item in raw_results or []:
            points, text, confidence = item
            xs = [max(0, int(round(point[0]))) for point in points]
            ys = [max(0, int(round(point[1]))) for point in points]
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

    def _create_reader(self, *, gpu: bool) -> Any:
        if self._easyocr_module is None:
            raise OCRBackendError("EasyOCR backend was not initialized")
        return self._easyocr_module.Reader([self.language], gpu=gpu)

    @staticmethod
    def _should_fallback_to_cpu(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "cuda error" in message or "illegal memory access" in message
