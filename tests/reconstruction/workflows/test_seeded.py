from __future__ import annotations

import numpy as np
import pytest

from app.parsers.gamelog import ClueEvent, GuessEvent
from tests.support.smoke import FakeVodSource

import app.reconstruction.workflows.seeded.dense_board as seeded_dense_board
import app.reconstruction.workflows.seeded as seeded_reconstruction

from app.core.models import BoardCell, BoardState, CardColor, ImageBoundingBox, TeamColor
from app.infra.roi_config import ROIConfig
from app.infra.vod_source import FrameSample

_aggregate_turn_visible_guess_candidates = seeded_reconstruction._aggregate_turn_visible_guess_candidates
_augment_analysis_with_dense_board_rescans = seeded_reconstruction._augment_analysis_with_dense_board_rescans
_merge_turn_guesses_from_visible_rescan = seeded_reconstruction._merge_turn_guesses_from_visible_rescan
_restore_turn_stoppers_from_event_history = seeded_reconstruction._restore_turn_stoppers_from_event_history
_scan_dense_board_reveal_events = seeded_reconstruction._scan_dense_board_reveal_events
_stabilize_turn_guess_sequences = seeded_reconstruction._stabilize_turn_guess_sequences
_turn_needs_second_pass = seeded_reconstruction._turn_needs_second_pass


def _make_test_roi_config() -> ROIConfig:
    return ROIConfig.from_raw(
        {
            "left_team_panel": {"x": 0.0, "y": 0.0, "width": 0.1, "height": 0.1},
            "right_team_panel": {"x": 0.9, "y": 0.0, "width": 0.1, "height": 0.1},
            "board_region": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0},
            "game_log_region": {"x": 0.0, "y": 0.9, "width": 0.1, "height": 0.1},
            "left_counter_region": {"x": 0.0, "y": 0.1, "width": 0.1, "height": 0.1},
            "right_counter_region": {"x": 0.9, "y": 0.1, "width": 0.1, "height": 0.1},
            "top_banner_region": {"x": 0.0, "y": 0.0, "width": 0.2, "height": 0.1},
            "end_banner_region": {"x": 0.0, "y": 0.8, "width": 0.2, "height": 0.2},
        }
    )


def _make_board_state() -> BoardState:
    return BoardState(
        cells=[
            BoardCell(
                row=0,
                col=0,
                word="SUIT",
                confidence=1.0,
                box=ImageBoundingBox(left=10, top=10, right=60, bottom=60),
            )
        ]
    )


def _make_multi_cell_board_state() -> BoardState:
    cells = []
    for index in range(4):
        cells.append(
            BoardCell(
                row=0,
                col=index,
                word=f"WORD{index}",
                confidence=1.0,
                box=ImageBoundingBox(
                    left=5 + (index * 22),
                    top=10,
                    right=25 + (index * 22),
                    bottom=40,
                ),
            )
        )
    return BoardState(cells=cells)


def test_scan_dense_board_reveal_events_upgrades_neutral_to_team_color(monkeypatch: pytest.MonkeyPatch) -> None:
    roi_config = _make_test_roi_config()
    board_state = _make_board_state()
    baseline_stats = {(0, 0): (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))}
    frames = [
        FrameSample(timestamp_sec=1.0, frame_bgr=np.full((10, 10, 3), fill_value=1, dtype=np.uint8), frame_index_in_window=0),
        FrameSample(timestamp_sec=1.5, frame_bgr=np.full((10, 10, 3), fill_value=2, dtype=np.uint8), frame_index_in_window=1),
        FrameSample(timestamp_sec=2.0, frame_bgr=np.full((10, 10, 3), fill_value=3, dtype=np.uint8), frame_index_in_window=2),
        FrameSample(timestamp_sec=2.5, frame_bgr=np.full((10, 10, 3), fill_value=4, dtype=np.uint8), frame_index_in_window=3),
    ]
    vod_source = FakeVodSource(frames)
    color_by_marker = {
        1: CardColor.NEUTRAL,
        2: CardColor.NEUTRAL,
        3: CardColor.RED,
        4: CardColor.RED,
    }

    monkeypatch.setattr(
        seeded_dense_board,
        "_sample_board_cell_surface",
        lambda board_frame, box: board_frame,
    )
    monkeypatch.setattr(
        seeded_dense_board,
        "_detect_revealed_card_color",
        lambda sample, stats: color_by_marker[int(sample[0, 0, 0])],
    )

    events = _scan_dense_board_reveal_events(
        vod_source=vod_source,
        vod_url="local.mp4",
        start_sec=1.0,
        duration_sec=2.0,
        dense_fps=2.0,
        minimum_confirmations=2,
        roi_config=roi_config,
        board_state=board_state,
        board_reveal_baseline=baseline_stats,
        ignored_words=set(),
    )

    assert len(events) == 1
    assert events[0].word == "SUIT"
    assert events[0].card_color is CardColor.RED
    assert events[0].timestamp_sec == 2.0


def test_dense_backfill_allows_close_same_turn_candidates_before_latest_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = {
        "stop_sec": 20.0,
        "turns": [
            {
                "turn_index": 0,
                "team_color": "red",
                "clue_text": "LEATHER",
                "clue_count": "2",
                "timestamp_sec": 0.0,
                "guesses": [
                    {
                        "word": "GLOVE",
                        "card_color": "red",
                        "result": "correct",
                        "timestamp_sec": 10.5,
                        "player_name": "near",
                        "confidence": 0.8,
                    }
                ],
            }
        ],
    }
    board_state = _make_board_state()
    monkeypatch.setattr(
        seeded_dense_board,
        "_scan_dense_board_reveal_events",
        lambda **kwargs: [
            GuessEvent(
                player_name="unknown",
                word="CRAFT",
                card_color=CardColor.RED,
                timestamp_sec=9.5,
                sequence_index=0,
                confidence=0.86,
            )
        ],
    )

    _augment_analysis_with_dense_board_rescans(
        analysis=analysis,
        vod_source=FakeVodSource([]),
        vod_url="local.mp4",
        roi_config=_make_test_roi_config(),
        board_state=board_state,
        board_reveal_baseline={(0, 0): (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))},
    )

    assert [guess["word"] for guess in analysis["turns"][0]["guesses"]] == ["GLOVE", "CRAFT"]


def test_dense_backfill_allows_slightly_earlier_same_turn_candidate_near_slack_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = {
        "stop_sec": 20.0,
        "turns": [
            {
                "turn_index": 0,
                "team_color": "blue",
                "clue_text": "PENETRATE",
                "clue_count": "2",
                "timestamp_sec": 0.0,
                "guesses": [
                    {
                        "word": "RIFLE",
                        "card_color": "blue",
                        "result": "correct",
                        "timestamp_sec": 12.0,
                        "player_name": "unknown",
                        "confidence": 0.33,
                    },
                    {
                        "word": "THORN",
                        "card_color": "blue",
                        "result": "correct",
                        "timestamp_sec": 15.0,
                        "player_name": "unknown",
                        "confidence": 0.86,
                    },
                ],
            }
        ],
    }
    monkeypatch.setattr(
        seeded_dense_board,
        "_scan_dense_board_reveal_events",
        lambda **kwargs: [
            GuessEvent(
                player_name="unknown",
                word="RAIL",
                card_color=CardColor.BLUE,
                timestamp_sec=9.75,
                sequence_index=0,
                confidence=0.86,
            )
        ],
    )

    _augment_analysis_with_dense_board_rescans(
        analysis=analysis,
        vod_source=FakeVodSource([]),
        vod_url="local.mp4",
        roi_config=_make_test_roi_config(),
        board_state=_make_multi_cell_board_state(),
        board_reveal_baseline={(0, 0): (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))},
    )

    assert [guess["word"] for guess in analysis["turns"][0]["guesses"]] == ["RIFLE", "THORN", "RAIL"]


def test_dense_backfill_replaces_weak_trailing_infinity_guess_with_closer_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = {
        "stop_sec": 30.0,
        "turns": [
            {
                "turn_index": 0,
                "team_color": "red",
                "clue_text": "HELMET",
                "clue_count": "infinity",
                "timestamp_sec": 0.0,
                "guesses": [
                    {
                        "word": "SCUBA DIVER",
                        "card_color": "red",
                        "result": "correct",
                        "timestamp_sec": 1.0,
                        "player_name": "unknown",
                        "confidence": 0.88,
                    },
                    {
                        "word": "HEART",
                        "card_color": "red",
                        "result": "correct",
                        "timestamp_sec": 10.0,
                        "player_name": "near",
                        "confidence": 0.88,
                    },
                    {
                        "word": "FOREST",
                        "card_color": "red",
                        "result": "correct",
                        "timestamp_sec": 12.0,
                        "player_name": "unknown",
                        "confidence": 0.86,
                        "selector_source": "turn_continuity",
                    },
                ],
            }
        ],
    }
    monkeypatch.setattr(
        seeded_dense_board,
        "_scan_dense_board_reveal_events",
        lambda **kwargs: [
            GuessEvent(
                player_name="unknown",
                word="ROUND",
                card_color=CardColor.RED,
                timestamp_sec=10.5,
                sequence_index=0,
                confidence=0.86,
            )
        ],
    )

    _augment_analysis_with_dense_board_rescans(
        analysis=analysis,
        vod_source=FakeVodSource([]),
        vod_url="local.mp4",
        roi_config=_make_test_roi_config(),
        board_state=_make_board_state(),
        board_reveal_baseline={(0, 0): (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))},
    )

    assert [guess["word"] for guess in analysis["turns"][0]["guesses"]] == ["SCUBA DIVER", "HEART", "ROUND"]


def test_dense_backfill_allows_clue_count_plus_one_total_guesses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = {
        "stop_sec": 40.0,
        "turns": [
            {
                "turn_index": 0,
                "team_color": "red",
                "clue_text": "SPIDER",
                "clue_count": "2",
                "timestamp_sec": 0.0,
                "guesses": [
                    {
                        "word": "ANGEL",
                        "card_color": "red",
                        "result": "correct",
                        "timestamp_sec": 8.0,
                        "player_name": "near",
                        "confidence": 0.62,
                    }
                ],
            }
        ],
    }
    monkeypatch.setattr(
        seeded_dense_board,
        "_scan_dense_board_reveal_events",
        lambda **kwargs: [
            GuessEvent(
                player_name="unknown",
                word="WEB",
                card_color=CardColor.RED,
                timestamp_sec=7.0,
                sequence_index=0,
                confidence=0.86,
            ),
            GuessEvent(
                player_name="unknown",
                word="GREENHOUSE",
                card_color=CardColor.RED,
                timestamp_sec=9.0,
                sequence_index=1,
                confidence=0.86,
            ),
        ],
    )

    _augment_analysis_with_dense_board_rescans(
        analysis=analysis,
        vod_source=FakeVodSource([]),
        vod_url="local.mp4",
        roi_config=_make_test_roi_config(),
        board_state=_make_multi_cell_board_state(),
        board_reveal_baseline={(0, 0): (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))},
    )

    assert {guess["word"] for guess in analysis["turns"][0]["guesses"]} == {"WEB", "ANGEL", "GREENHOUSE"}


def test_aggregate_turn_visible_guess_candidates_preserves_first_seen_order() -> None:
    turn = {
        "team_color": "red",
        "spymaster_name": "Cherry",
        "clue_text": "WOMEN",
    }
    clue = ClueEvent(
        team_color=TeamColor.RED,
        spymaster_name="Cherry",
        clue_text="WOMEN",
        clue_count="3",
        timestamp_sec=1.0,
        sequence_index=0,
        confidence=0.99,
    )
    visible_frames = [
        [
            clue.model_copy(update={"timestamp_sec": 1.0}),
            GuessEvent(
                player_name="near",
                word="PURSE",
                card_color=CardColor.RED,
                timestamp_sec=1.0,
                sequence_index=1,
                confidence=0.88,
            ),
        ],
        [
            clue.model_copy(update={"timestamp_sec": 2.0}),
            GuessEvent(
                player_name="near",
                word="PURSE",
                card_color=CardColor.RED,
                timestamp_sec=2.0,
                sequence_index=1,
                confidence=0.88,
            ),
            GuessEvent(
                player_name="near",
                word="GOLDILOCKS",
                card_color=CardColor.RED,
                timestamp_sec=2.0,
                sequence_index=2,
                confidence=0.87,
            ),
        ],
        [
            clue.model_copy(update={"timestamp_sec": 3.0}),
            GuessEvent(
                player_name="revel",
                word="PURSE",
                card_color=CardColor.RED,
                timestamp_sec=3.0,
                sequence_index=1,
                confidence=0.88,
            ),
            GuessEvent(
                player_name="revel",
                word="GOLDILOCKS",
                card_color=CardColor.RED,
                timestamp_sec=3.0,
                sequence_index=2,
                confidence=0.87,
            ),
            GuessEvent(
                player_name="revel",
                word="JEWELER",
                card_color=CardColor.BLUE,
                timestamp_sec=3.0,
                sequence_index=3,
                confidence=0.82,
            ),
        ],
    ]

    guesses = _aggregate_turn_visible_guess_candidates(visible_frames, turn)

    assert [guess["word"] for guess in guesses] == ["PURSE", "GOLDILOCKS", "JEWELER"]


def test_aggregate_turn_visible_guess_candidates_accepts_active_turn_guess_only_frames() -> None:
    turn = {
        "team_color": "blue",
        "spymaster_name": "luke",
        "clue_text": "IRELAND",
    }
    clue = ClueEvent(
        team_color=TeamColor.BLUE,
        spymaster_name="luke",
        clue_text="IRELAND",
        clue_count="2",
        timestamp_sec=1.0,
        sequence_index=0,
        confidence=0.99,
    )
    visible_frames = [
        [
            clue,
            GuessEvent(
                player_name="ty",
                word="LEPRECHAUN",
                card_color=CardColor.BLUE,
                timestamp_sec=1.0,
                sequence_index=1,
                confidence=0.86,
            ),
        ],
        [
            GuessEvent(
                player_name="ty",
                word="LEPRECHAUN",
                card_color=CardColor.BLUE,
                timestamp_sec=2.0,
                sequence_index=0,
                confidence=0.86,
            ),
            GuessEvent(
                player_name="ty",
                word="EUROPE",
                card_color=CardColor.BLUE,
                timestamp_sec=2.0,
                sequence_index=1,
                confidence=0.84,
            ),
            GuessEvent(
                player_name="Drew",
                word="PIT",
                card_color=CardColor.NEUTRAL,
                timestamp_sec=2.0,
                sequence_index=2,
                confidence=0.82,
            ),
        ],
    ]

    guesses = _aggregate_turn_visible_guess_candidates(visible_frames, turn)

    assert [guess["word"] for guess in guesses] == ["LEPRECHAUN", "EUROPE", "PIT"]


def test_merge_turn_guesses_from_visible_rescan_prefers_visible_order_and_drops_weak_non_visible_guess() -> None:
    turn = {
        "team_color": "blue",
        "guesses": [
            {
                "word": "RIFLE",
                "card_color": "blue",
                "result": "correct",
                "timestamp_sec": 10.0,
                "confidence": 0.32,
                "source": "log",
                "observation_count": 1,
            },
            {
                "word": "THORN",
                "card_color": "blue",
                "result": "correct",
                "timestamp_sec": 12.0,
                "confidence": 0.86,
                "source": "log",
                "observation_count": 2,
            },
        ],
    }
    visible_guesses = [
        {
            "word": "RAIL",
            "timestamp_sec": 10.5,
            "card_color": "blue",
            "confidence": 0.83,
            "observation_count": 1,
            "source": "log_rescan",
            "_visible_order": 0,
            "_visible_position": 0,
        },
        {
            "word": "THORN",
            "timestamp_sec": 12.0,
            "card_color": "blue",
            "confidence": 0.86,
            "observation_count": 2,
            "source": "log_rescan",
            "_visible_order": 1,
            "_visible_position": 1,
        },
    ]

    _merge_turn_guesses_from_visible_rescan(turn, visible_guesses)

    assert [guess["word"] for guess in turn["guesses"]] == ["RAIL", "THORN"]


def test_merge_turn_guesses_from_visible_rescan_inserts_strong_dense_guess_without_reordering_visible_words() -> None:
    turn = {
        "team_color": "red",
        "guesses": [
            {
                "word": "ANGEL",
                "card_color": "red",
                "result": "correct",
                "timestamp_sec": 8.0,
                "confidence": 0.63,
                "source": "log",
                "observation_count": 1,
            },
            {
                "word": "WEB",
                "card_color": "red",
                "result": "correct",
                "timestamp_sec": 7.0,
                "confidence": 0.86,
                "source": "board_dense",
                "observation_count": 0,
                "board_evidence": True,
                "board_color": "red",
            },
            {
                "word": "GREENHOUSE",
                "card_color": "red",
                "result": "correct",
                "timestamp_sec": 9.0,
                "confidence": 0.86,
                "source": "board_dense",
                "observation_count": 0,
                "board_evidence": True,
                "board_color": "red",
            },
        ],
    }
    visible_guesses = [
        {
            "word": "ANGEL",
            "timestamp_sec": 8.0,
            "card_color": "red",
            "confidence": 0.63,
            "observation_count": 1,
            "source": "log_rescan",
            "_visible_order": 0,
            "_visible_position": 0,
        },
        {
            "word": "GREENHOUSE",
            "timestamp_sec": 9.0,
            "card_color": "red",
            "confidence": 0.86,
            "observation_count": 1,
            "source": "log_rescan",
            "_visible_order": 1,
            "_visible_position": 1,
        },
    ]

    _merge_turn_guesses_from_visible_rescan(turn, visible_guesses)

    assert [guess["word"] for guess in turn["guesses"]] == ["WEB", "ANGEL", "GREENHOUSE"]


def test_merge_turn_guesses_from_visible_rescan_keeps_first_visible_guess_anchored_before_legacy_extra() -> None:
    turn = {
        "team_color": "red",
        "guesses": [
            {
                "word": "GOLDILOCKS",
                "card_color": "red",
                "result": "correct",
                "timestamp_sec": 2.0,
                "confidence": 0.88,
                "source": "log",
                "observation_count": 1,
            },
            {
                "word": "PURSE",
                "card_color": "red",
                "result": "correct",
                "timestamp_sec": 3.0,
                "confidence": 0.88,
                "source": "log",
                "observation_count": 1,
            },
            {
                "word": "JEWELER",
                "card_color": "blue",
                "result": "enemy",
                "timestamp_sec": 6.0,
                "confidence": 0.42,
                "source": "log",
                "observation_count": 1,
            },
        ],
    }
    visible_guesses = [
        {
            "word": "PURSE",
            "timestamp_sec": 3.0,
            "card_color": "red",
            "confidence": 0.88,
            "observation_count": 1,
            "source": "log_rescan",
            "_visible_order": 0,
            "_visible_position": 0,
        },
        {
            "word": "JEWELER",
            "timestamp_sec": 6.0,
            "card_color": "blue",
            "confidence": 0.42,
            "observation_count": 1,
            "source": "log_rescan",
            "_visible_order": 1,
            "_visible_position": 1,
        },
    ]

    _merge_turn_guesses_from_visible_rescan(turn, visible_guesses)

    assert [guess["word"] for guess in turn["guesses"]] == ["PURSE", "GOLDILOCKS", "JEWELER"]


def test_merge_turn_guesses_from_visible_rescan_restores_clearly_earlier_legacy_guess() -> None:
    turn = {
        "team_color": "red",
        "guesses": [
            {
                "word": "SIGN",
                "card_color": "red",
                "result": "correct",
                "timestamp_sec": 10.0,
                "confidence": 0.88,
                "source": "log",
                "observation_count": 1,
            },
            {
                "word": "CODE",
                "card_color": "red",
                "result": "correct",
                "timestamp_sec": 34.0,
                "confidence": 0.65,
                "source": "log",
                "observation_count": 1,
            },
            {
                "word": "OPERA",
                "card_color": "blue",
                "result": "enemy",
                "timestamp_sec": 58.0,
                "confidence": 0.67,
                "source": "log",
                "observation_count": 1,
            },
        ],
    }
    visible_guesses = [
        {
            "word": "CODE",
            "timestamp_sec": 34.0,
            "card_color": "red",
            "confidence": 0.65,
            "observation_count": 1,
            "source": "log_rescan",
            "_visible_order": 0,
            "_visible_position": 0,
        },
    ]

    _merge_turn_guesses_from_visible_rescan(turn, visible_guesses)

    assert [guess["word"] for guess in turn["guesses"]] == ["SIGN", "CODE", "OPERA"]


def test_merge_turn_guesses_from_visible_rescan_ignores_low_confidence_one_off_new_visible_guess() -> None:
    turn = {
        "team_color": "red",
        "guesses": [
            {
                "word": "PURSE",
                "card_color": "red",
                "result": "correct",
                "timestamp_sec": 3.0,
                "confidence": 0.88,
                "source": "log",
                "observation_count": 1,
            }
        ],
    }
    visible_guesses = [
        {
            "word": "PURSE",
            "timestamp_sec": 3.0,
            "card_color": "red",
            "confidence": 0.52,
            "observation_count": 1,
            "source": "log_rescan",
            "_visible_order": 0,
            "_visible_position": 0,
        },
        {
            "word": "FEVER",
            "timestamp_sec": 6.0,
            "card_color": "blue",
            "confidence": 0.35,
            "observation_count": 1,
            "source": "log_rescan",
            "_visible_order": 1,
            "_visible_position": 0,
        },
    ]

    _merge_turn_guesses_from_visible_rescan(turn, visible_guesses)

    assert [guess["word"] for guess in turn["guesses"]] == ["PURSE"]


def test_merge_turn_guesses_from_visible_rescan_preserves_earliest_raw_stopper_over_weak_rescan_swap() -> None:
    turn = {
        "team_color": "red",
        "guesses": [
            {
                "word": "PURSE",
                "card_color": "red",
                "result": "correct",
                "timestamp_sec": 3.0,
                "confidence": 0.95,
                "source": "log",
            },
            {
                "word": "GOLDILOCKS",
                "card_color": "red",
                "result": "correct",
                "timestamp_sec": 3.0,
                "confidence": 0.88,
                "source": "log",
            },
            {
                "word": "JEWELER",
                "card_color": "blue",
                "result": "enemy",
                "timestamp_sec": 6.0,
                "confidence": 0.42,
                "source": "log",
            },
        ],
    }
    visible_guesses = [
        {
            "word": "PURSE",
            "timestamp_sec": 3.0,
            "card_color": "red",
            "confidence": 0.95,
            "observation_count": 4,
            "source": "log_rescan",
            "_visible_order": 0,
            "_visible_position": 0,
        },
        {
            "word": "GOLDILOCKS",
            "timestamp_sec": 3.0,
            "card_color": "red",
            "confidence": 0.88,
            "observation_count": 1,
            "source": "log_rescan",
            "_visible_order": 1,
            "_visible_position": 1,
        },
        {
            "word": "RIFLE",
            "timestamp_sec": 6.5,
            "card_color": "blue",
            "confidence": 0.5,
            "observation_count": 3,
            "source": "log_rescan",
            "_visible_order": 2,
            "_visible_position": 2,
        },
    ]

    _merge_turn_guesses_from_visible_rescan(turn, visible_guesses)

    assert [guess["word"] for guess in turn["guesses"]] == ["PURSE", "GOLDILOCKS", "JEWELER"]


def test_stabilize_turn_guess_sequences_drops_weak_non_board_same_team_extra_above_direct_clue_count() -> None:
    analysis = {
        "turns": [
            {
                "turn_index": 0,
                "team_color": "blue",
                "clue_text": "PENETRATE",
                "clue_count": "2",
                "timestamp_sec": 0.0,
                "guesses": [
                    {
                        "word": "RAIL",
                        "card_color": "blue",
                        "result": "correct",
                        "timestamp_sec": 1.0,
                        "confidence": 0.86,
                        "source": "board_dense",
                        "board_evidence": True,
                    },
                    {
                        "word": "RIFLE",
                        "card_color": "blue",
                        "result": "correct",
                        "timestamp_sec": 2.0,
                        "confidence": 0.33,
                        "source": "log_rescan",
                        "observation_count": 2,
                    },
                    {
                        "word": "THORN",
                        "card_color": "blue",
                        "result": "correct",
                        "timestamp_sec": 3.0,
                        "confidence": 0.88,
                        "source": "log",
                        "observation_count": 1,
                    },
                ],
            }
        ]
    }

    _stabilize_turn_guess_sequences(analysis)

    assert [guess["word"] for guess in analysis["turns"][0]["guesses"]] == ["RAIL", "THORN"]


def test_stabilize_turn_guess_sequences_moves_stopper_last_and_reorders_tied_same_team_cluster() -> None:
    analysis = {
        "turns": [
            {
                "turn_index": 0,
                "team_color": "blue",
                "clue_text": "IRELAND",
                "clue_count": "2",
                "timestamp_sec": 0.0,
                "guesses": [
                    {
                        "word": "EUROPE",
                        "card_color": "blue",
                        "result": "correct",
                        "timestamp_sec": 10.0,
                        "confidence": 0.66,
                        "observation_count": 1,
                    },
                    {
                        "word": "PIT",
                        "card_color": "neutral",
                        "result": "neutral",
                        "timestamp_sec": 12.0,
                        "confidence": 0.79,
                    },
                    {
                        "word": "LEPRECHAUN",
                        "card_color": "blue",
                        "result": "correct",
                        "timestamp_sec": 10.0,
                        "confidence": 0.45,
                        "observation_count": 2,
                    },
                ],
            }
        ]
    }

    _stabilize_turn_guess_sequences(analysis)

    assert [guess["word"] for guess in analysis["turns"][0]["guesses"]] == ["LEPRECHAUN", "EUROPE", "PIT"]


def test_restore_turn_stoppers_from_event_history_prefers_raw_stopper_over_weak_rescan_swap() -> None:
    analysis = {
        "turns": [
            {
                "turn_index": 0,
                "team_color": "red",
                "clue_text": "WOMEN",
                "clue_count": "3",
                "timestamp_sec": 1.0,
                "guesses": [
                    {
                        "word": "PURSE",
                        "card_color": "red",
                        "result": "correct",
                        "timestamp_sec": 2.0,
                        "confidence": 0.95,
                        "source": "log_rescan",
                    },
                    {
                        "word": "GOLDILOCKS",
                        "card_color": "red",
                        "result": "correct",
                        "timestamp_sec": 2.0,
                        "confidence": 0.88,
                    },
                    {
                        "word": "RIFLE",
                        "card_color": "blue",
                        "result": "enemy",
                        "timestamp_sec": 6.5,
                        "confidence": 0.5,
                        "source": "log_rescan",
                    },
                ],
            }
        ]
    }
    event_history = [
        ClueEvent(
            team_color=TeamColor.RED,
            spymaster_name="Cherry",
            clue_text="WOMEN",
            clue_count="3",
            timestamp_sec=1.0,
            sequence_index=0,
            confidence=0.9,
        ),
        GuessEvent(
            player_name="unknown",
            word="PURSE",
            card_color=CardColor.RED,
            timestamp_sec=2.0,
            sequence_index=1,
            confidence=0.95,
        ),
        GuessEvent(
            player_name="unknown",
            word="GOLDILOCKS",
            card_color=CardColor.RED,
            timestamp_sec=2.0,
            sequence_index=2,
            confidence=0.88,
        ),
        GuessEvent(
            player_name="unknown",
            word="JEWELER",
            card_color=CardColor.BLUE,
            timestamp_sec=6.0,
            sequence_index=3,
            confidence=0.42,
        ),
    ]

    _restore_turn_stoppers_from_event_history(analysis, event_history)

    assert [guess["word"] for guess in analysis["turns"][0]["guesses"]] == ["PURSE", "GOLDILOCKS", "JEWELER"]


def test_turn_needs_second_pass_for_all_correct_cluster_at_direct_clue_count() -> None:
    turn = {
        "team_color": "red",
        "clue_count": "2",
        "guesses": [
            {"word": "WEB", "card_color": "red", "result": "correct"},
            {"word": "ANGEL", "card_color": "red", "result": "correct"},
        ],
    }

    assert _turn_needs_second_pass(turn)


def test_turn_does_not_need_second_pass_once_stopper_is_present() -> None:
    turn = {
        "team_color": "blue",
        "clue_count": "2",
        "guesses": [
            {"word": "LEPRECHAUN", "card_color": "blue", "result": "correct"},
            {"word": "EUROPE", "card_color": "blue", "result": "correct"},
            {"word": "PIT", "card_color": "neutral", "result": "neutral"},
        ],
    }

    assert not _turn_needs_second_pass(turn)
