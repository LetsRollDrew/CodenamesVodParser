from __future__ import annotations

from app.reconstruction.workflows.full import _drop_weaker_repeated_clue_candidates


def test_drop_weaker_repeated_clue_candidates_prefers_guess_supported_segment() -> None:
    candidates = [
        {
            "clue_text": "PROM",
            "clue_count": "4",
            "clue_timestamp_sec": 92.0,
            "visible_guess_count": 0,
            "clue_matches": [],
            "segment": [object()],
        },
        {
            "clue_text": "PROM",
            "clue_count": "4",
            "clue_timestamp_sec": 148.0,
            "visible_guess_count": 3,
            "clue_matches": [object()],
            "segment": [object(), object()],
        },
        {
            "clue_text": "ANIMAL",
            "clue_count": "2",
            "clue_timestamp_sec": 260.0,
            "visible_guess_count": 1,
            "clue_matches": [object()],
            "segment": [object()],
        },
    ]

    filtered = _drop_weaker_repeated_clue_candidates(candidates)

    assert len(filtered) == 2
    assert filtered[0]["clue_text"] == "PROM"
    assert filtered[0]["clue_timestamp_sec"] == 148.0
    assert filtered[1]["clue_text"] == "ANIMAL"
