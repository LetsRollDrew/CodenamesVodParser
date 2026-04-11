from __future__ import annotations

from app.models import ReviewEntityType
from app.review_queue import materialize_review_items, maybe_flag_review, should_enqueue_review


def test_should_enqueue_review_respects_threshold() -> None:
    assert should_enqueue_review(0.5, threshold=0.8)
    assert not should_enqueue_review(0.9, threshold=0.8)


def test_materialize_review_items_preserves_fields() -> None:
    flags = []
    maybe_flag_review(
        flags,
        entity_type=ReviewEntityType.GUESS,
        entity_id="guess:1:0",
        confidence=0.45,
        threshold=0.8,
        reason="Low-confidence guess parsing",
        timestamp_sec=123.0,
        snapshot_path="debug_snapshots/guess-1.png",
    )

    items = materialize_review_items(flags)

    assert len(items) == 1
    assert items[0].entity_type is ReviewEntityType.GUESS
    assert items[0].entity_id == "guess:1:0"
    assert items[0].snapshot_path == "debug_snapshots/guess-1.png"
