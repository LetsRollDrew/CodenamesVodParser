"""Helpers for routing low-confidence parser outputs into the review queue."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models import ReviewEntityType, ReviewQueueItem


class ReviewFlag(BaseModel):
    """A review-worthy parser finding before ORM materialization."""

    entity_type: ReviewEntityType
    entity_id: str
    reason: str
    confidence: float = Field(ge=0, le=1)
    timestamp_sec: float | None = Field(default=None, ge=0)
    snapshot_path: str | None = None


def should_enqueue_review(confidence: float, *, threshold: float = 0.8) -> bool:
    """Return whether a confidence score should enter the review queue."""

    return confidence < threshold


def maybe_flag_review(
    flags: list[ReviewFlag],
    *,
    entity_type: ReviewEntityType,
    entity_id: str,
    confidence: float,
    threshold: float,
    reason: str,
    timestamp_sec: float | None = None,
    snapshot_path: str | None = None,
) -> None:
    """Append a review flag when the confidence is below the configured threshold."""

    if not should_enqueue_review(confidence, threshold=threshold):
        return

    flags.append(
        ReviewFlag(
            entity_type=entity_type,
            entity_id=entity_id,
            reason=reason,
            confidence=confidence,
            timestamp_sec=timestamp_sec,
            snapshot_path=snapshot_path,
        )
    )


def materialize_review_items(flags: list[ReviewFlag]) -> list[ReviewQueueItem]:
    """Convert review flags into ORM review queue rows."""

    return [
        ReviewQueueItem(
            entity_type=flag.entity_type,
            entity_id=flag.entity_id,
            reason=flag.reason,
            timestamp_sec=flag.timestamp_sec,
            snapshot_path=flag.snapshot_path,
            confidence=flag.confidence,
        )
        for flag in flags
    ]
