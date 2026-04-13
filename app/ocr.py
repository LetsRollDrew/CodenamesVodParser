"""Shared OCR backend protocol and OCR preprocessing helpers."""

from __future__ import annotations

import re
from typing import Iterable, Literal, Protocol, Sequence

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from app.models import ImageBoundingBox, OCRDetection

_WHITESPACE_RE = re.compile(r"\s+")


class OCRBackend(Protocol):
    """Protocol for OCR backends used by parser modules."""

    def detect_text(
        self,
        image: np.ndarray,
        *,
        hint: str | None = None,
        ) -> Sequence[OCRDetection]:
        """Return OCR detections for the supplied image crop."""


OCRPreprocessProfile = Literal["generic", "name_strip", "counter", "board_word", "banner", "game_log"]


def prepare_ocr_image(
    image: np.ndarray,
    *,
    profile: OCRPreprocessProfile = "generic",
) -> np.ndarray:
    """Apply lightweight profile-specific preprocessing before OCR."""

    if image.size == 0:
        return image

    if image.ndim == 3 and image.shape[2] >= 3:
        pil_image = Image.fromarray(image[:, :, ::-1]).convert("L")
    else:
        pil_image = Image.fromarray(image).convert("L")

    scale = {
        "generic": 2,
        "name_strip": 4,
        "counter": 5,
        "board_word": 4,
        "banner": 3,
        "game_log": 3,
    }[profile]
    pil_image = pil_image.resize(
        (max(1, pil_image.width * scale), max(1, pil_image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    pil_image = ImageOps.autocontrast(pil_image)

    if profile in {"name_strip", "counter", "board_word"}:
        pil_image = pil_image.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)

    grayscale = np.asarray(pil_image, dtype=np.uint8)
    return np.stack([grayscale, grayscale, grayscale], axis=-1)


def detect_text_with_fallback(
    ocr_backend: OCRBackend,
    raw_image: np.ndarray,
    *,
    prepared_image: np.ndarray | None = None,
    hint: str | None = None,
) -> list[OCRDetection]:
    """Try OCR on raw and preprocessed variants with backend-aware ordering."""

    prepared = prepared_image if prepared_image is not None else raw_image
    backend_name = type(ocr_backend).__name__.casefold()
    prefer_raw_first = "paddle" in backend_name
    variants = (
        (("raw", raw_image), ("prepared", prepared))
        if prefer_raw_first
        else (("prepared", prepared), ("raw", raw_image))
    )

    for label, variant_image in variants:
        detections = list(
            ocr_backend.detect_text(
                variant_image,
                hint=f"{hint}:{label}" if hint else label,
            )
        )
        if detections:
            if label == "prepared" and prepared.shape[:2] != raw_image.shape[:2]:
                detections = _map_detections_to_raw_shape(
                    detections,
                    source_shape=prepared.shape,
                    target_shape=raw_image.shape,
                )
            return detections
    return []


def _map_detections_to_raw_shape(
    detections: list[OCRDetection],
    *,
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> list[OCRDetection]:
    source_height, source_width = source_shape[:2]
    target_height, target_width = target_shape[:2]
    if source_height <= 0 or source_width <= 0:
        return detections

    scale_x = target_width / source_width
    scale_y = target_height / source_height
    mapped: list[OCRDetection] = []
    for detection in detections:
        box = detection.box
        left = int(round(box.left * scale_x))
        top = int(round(box.top * scale_y))
        right = int(round(box.right * scale_x))
        bottom = int(round(box.bottom * scale_y))
        mapped.append(
            detection.model_copy(
                update={
                    "box": ImageBoundingBox(
                        left=max(left, 0),
                        top=max(top, 0),
                        right=max(right, left + 1),
                        bottom=max(bottom, top + 1),
                    )
                }
            )
        )
    return mapped


def collapse_whitespace(text: str) -> str:
    """Normalize repeated whitespace and trim the ends."""

    return _WHITESPACE_RE.sub(" ", text).strip()


def pick_best_detection(
    detections: Iterable[OCRDetection],
    *,
    excluded_texts: set[str] | None = None,
) -> OCRDetection | None:
    """Pick the most useful OCR detection after lightweight filtering."""

    excluded = {value.casefold() for value in excluded_texts or set()}
    filtered: list[OCRDetection] = []
    for detection in detections:
        normalized = collapse_whitespace(detection.text)
        if not normalized:
            continue
        if normalized.casefold() in excluded:
            continue
        filtered.append(detection.model_copy(update={"text": normalized}))

    if not filtered:
        return None

    return max(filtered, key=lambda item: (item.confidence, len(item.text)))
