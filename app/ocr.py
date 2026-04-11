"""Shared OCR backend protocol and small text helpers."""

from __future__ import annotations

import re
from typing import Iterable, Protocol, Sequence

import numpy as np

from app.models import OCRDetection

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
