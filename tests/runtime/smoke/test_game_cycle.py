from __future__ import annotations

from pathlib import Path

from app.parsers.gamelog import ClueEvent, GuessEvent
from app.runtime.smoke.cycle.clue_window import (
    _observe_clue_candidate,
    _observe_pending_assassin_guess,
    _update_active_clue_window,
)
from app.runtime.smoke.cycle.history_merge import _should_replace_locked_board_state
from app.runtime.smoke.cycle.visible_log import (
    _commit_visible_log_events,
    _filter_visible_log_events_for_runtime,
    _should_probe_visible_log_from_state,
)
from app.runtime.smoke.runtime_models import ActiveClueWindow, BannerObservation, SmokeRuntimeState
from tests.support.smoke import (
    FakeVodSource,
    SequencedOCRBackend,
    make_board_responses,
    make_frame,
    make_roi_config,
)
from app.cli.smoke import smoke_parse_game_cycle
from app.core.models import BoardCell, BoardState, CardColor, ImageBoundingBox, OCRDetection, TeamColor
from app.infra.vod_source import FrameSample
from app.vision.detection.frame_detectors import CounterObservation


def test_smoke_parse_game_cycle_stops_on_winner_banner_and_emits_log_changes() -> None:
    frame_a = make_frame(10)
    frame_b = make_frame(80)
    frame_c = make_frame(80)
    frame_d = make_frame(150)
    for frame in (frame_a, frame_b, frame_c, frame_d):
        frame[60:492, 180:780] = 120
    vod_source = FakeVodSource(
        [
            FrameSample(timestamp_sec=10.0, frame_bgr=frame_a, frame_index_in_window=0),
            FrameSample(timestamp_sec=11.0, frame_bgr=frame_b, frame_index_in_window=1),
            FrameSample(timestamp_sec=12.0, frame_bgr=frame_c, frame_index_in_window=2),
            FrameSample(timestamp_sec=13.0, frame_bgr=frame_d, frame_index_in_window=3),
        ]
    )

    backend = SequencedOCRBackend(
        {
            "blue:operative:in_game": [
                OCRDetection(text="Drew", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=70, bottom=40))
            ],
            "blue:spymaster:in_game": [
                OCRDetection(text="ty", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=40, bottom=40))
            ],
            "red:operative:in_game": [
                OCRDetection(text="near", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=60, bottom=40))
            ],
            "red:spymaster:in_game": [
                OCRDetection(text="shadowsn", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=90, bottom=40))
            ],
            **make_board_responses("ANTARCTICA"),
            "game_log": [
                [
                    OCRDetection(text="ty", confidence=0.98, box=ImageBoundingBox(left=8, top=16, right=30, bottom=34)),
                    OCRDetection(text="THROW", confidence=0.99, box=ImageBoundingBox(left=50, top=14, right=110, bottom=36)),
                    OCRDetection(text="2", confidence=0.97, box=ImageBoundingBox(left=142, top=14, right=156, bottom=36)),
                ],
                [
                    OCRDetection(text="ty", confidence=0.98, box=ImageBoundingBox(left=8, top=16, right=30, bottom=34)),
                    OCRDetection(text="THROW", confidence=0.99, box=ImageBoundingBox(left=50, top=14, right=110, bottom=36)),
                    OCRDetection(text="2", confidence=0.97, box=ImageBoundingBox(left=142, top=14, right=156, bottom=36)),
                ],
                [
                    OCRDetection(text="ty", confidence=0.98, box=ImageBoundingBox(left=8, top=16, right=30, bottom=34)),
                    OCRDetection(text="THROW", confidence=0.99, box=ImageBoundingBox(left=50, top=14, right=110, bottom=36)),
                    OCRDetection(text="2", confidence=0.97, box=ImageBoundingBox(left=142, top=14, right=156, bottom=36)),
                    OCRDetection(text="Drew", confidence=0.92, box=ImageBoundingBox(left=8, top=68, right=40, bottom=86)),
                    OCRDetection(text="ANTARCTICA", confidence=0.91, box=ImageBoundingBox(left=50, top=66, right=130, bottom=88)),
                ],
                [
                    OCRDetection(text="ty", confidence=0.98, box=ImageBoundingBox(left=8, top=16, right=30, bottom=34)),
                    OCRDetection(text="THROW", confidence=0.99, box=ImageBoundingBox(left=50, top=14, right=110, bottom=36)),
                    OCRDetection(text="2", confidence=0.97, box=ImageBoundingBox(left=142, top=14, right=156, bottom=36)),
                    OCRDetection(text="Drew", confidence=0.92, box=ImageBoundingBox(left=8, top=68, right=40, bottom=86)),
                    OCRDetection(text="ANTARCTICA", confidence=0.91, box=ImageBoundingBox(left=50, top=66, right=130, bottom=88)),
                ],
                [
                    OCRDetection(text="ty", confidence=0.98, box=ImageBoundingBox(left=8, top=16, right=30, bottom=34)),
                    OCRDetection(text="THROW", confidence=0.99, box=ImageBoundingBox(left=50, top=14, right=110, bottom=36)),
                    OCRDetection(text="2", confidence=0.97, box=ImageBoundingBox(left=142, top=14, right=156, bottom=36)),
                    OCRDetection(text="Drew", confidence=0.92, box=ImageBoundingBox(left=8, top=68, right=40, bottom=86)),
                    OCRDetection(text="ANTARCTICA", confidence=0.91, box=ImageBoundingBox(left=50, top=66, right=130, bottom=88)),
                ],
            ],
            "left_counter": [
                [OCRDetection(text="8", confidence=0.95, box=ImageBoundingBox(left=1, top=1, right=10, bottom=10))],
                [OCRDetection(text="8", confidence=0.95, box=ImageBoundingBox(left=1, top=1, right=10, bottom=10))],
                [OCRDetection(text="8", confidence=0.95, box=ImageBoundingBox(left=1, top=1, right=10, bottom=10))],
            ],
            "right_counter": [
                [OCRDetection(text="9", confidence=0.95, box=ImageBoundingBox(left=1, top=1, right=10, bottom=10))],
                [OCRDetection(text="9", confidence=0.95, box=ImageBoundingBox(left=1, top=1, right=10, bottom=10))],
                [OCRDetection(text="9", confidence=0.95, box=ImageBoundingBox(left=1, top=1, right=10, bottom=10))],
            ],
            "top_banner": [
                [],
                [],
                [],
                [OCRDetection(text="YOUR TEAM WINS!", confidence=0.98, box=ImageBoundingBox(left=1, top=1, right=90, bottom=12))],
            ],
            "end_banner": [[], [], [], []],
        }
    )

    progress_messages: list[str] = []
    snapshot_dir = Path("build/test-temp/smoke-semantic-events")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    result = smoke_parse_game_cycle(
        vod_source=vod_source,
        vod_url="https://www.twitch.tv/videos/2745360291",
        start_sec=10.0,
        duration_sec=30.0,
        chunk_duration_sec=2.0,
        roi_config=make_roi_config(),
        ocr_backend=backend,
        vod_id="2745360291",
        snapshot_dir=snapshot_dir,
        progress_callback=progress_messages.append,
    )

    assert result.stop_reason in {"winner_banner", "assassin"}
    assert result.stop_sec in {11.0, 13.0}
    assert result.game_record is not None
    assert len(result.change_events) == 2
    assert result.change_events[0].new_events[0]["event_type"] == "clue"
    assert result.change_events[1].new_events[0]["event_type"] == "guess"
    assert result.first_frame_snapshot_path is not None
    assert Path(result.first_frame_snapshot_path).exists()
    assert result.start_snapshot_path is not None
    assert Path(result.start_snapshot_path).exists()
    assert result.roi_probe_overlay_path is not None
    assert Path(result.roi_probe_overlay_path).exists()
    assert result.roi_probe_crop_paths
    assert all(Path(path).exists() for path in result.roi_probe_crop_paths.values())
    assert all(event.snapshot_path is not None for event in result.change_events)
    assert all(Path(event.snapshot_path).exists() for event in result.change_events if event.snapshot_path is not None)
    assert vod_source.yielded >= 2
    assert vod_source.calls
    assert any(message.startswith("[chunk]") for message in progress_messages)
    assert any(message.startswith("[frame] 10.000s") for message in progress_messages)
    assert any(message.startswith("[log-change]") for message in progress_messages)
    assert any(message.startswith("[events]") for message in progress_messages)
    assert any(message.startswith("[stop]") for message in progress_messages)


def test_smoke_parse_game_cycle_emits_persistent_banner_clue_once(monkeypatch) -> None:
    frame_a = make_frame(60)
    frame_b = make_frame(60)
    frame_c = make_frame(60)
    for frame in (frame_a, frame_b, frame_c):
        frame[60:492, 180:780] = 120
    vod_source = FakeVodSource(
        [
            FrameSample(timestamp_sec=10.0, frame_bgr=frame_a, frame_index_in_window=0),
            FrameSample(timestamp_sec=11.0, frame_bgr=frame_b, frame_index_in_window=1),
            FrameSample(timestamp_sec=12.0, frame_bgr=frame_c, frame_index_in_window=2),
        ]
    )

    backend = SequencedOCRBackend(
        {
            "blue:operative:in_game": [
                OCRDetection(text="Drew", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=70, bottom=40))
            ],
            "blue:spymaster:in_game": [
                OCRDetection(text="ty", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=40, bottom=40))
            ],
            "red:operative:in_game": [
                OCRDetection(text="near", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=60, bottom=40))
            ],
            "red:spymaster:in_game": [
                OCRDetection(text="shadowsn", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=90, bottom=40))
            ],
            **make_board_responses("ANTARCTICA"),
        }
    )

    monkeypatch.setattr(
        "app.runtime.smoke.cycle.runner.parse_top_banner_text",
        lambda frame, roi_config, ocr_backend: "WAIT FOR YOUR TURN, Drew ARE GUESSING",
    )
    monkeypatch.setattr(
        "app.runtime.smoke.cycle.runner.parse_game_counters",
        lambda frame, roi_config, ocr_backend: (8, 9),
    )
    monkeypatch.setattr(
        "app.runtime.smoke.cycle.runner._parse_center_clue_banner",
        lambda frame, *, roi_config, ocr_backend: ("ANIMAL", "2"),
    )
    monkeypatch.setattr(
        "app.runtime.smoke.cycle.runner.parse_game_log",
        lambda frame, roi_config, ocr_backend, roster, board_state, timestamp_sec: [],
    )

    result = smoke_parse_game_cycle(
        vod_source=vod_source,
        vod_url="https://www.twitch.tv/videos/2745360291",
        start_sec=10.0,
        duration_sec=5.0,
        chunk_duration_sec=5.0,
        roi_config=make_roi_config(),
        ocr_backend=backend,
        vod_id="2745360291",
    )

    assert result.stop_reason == "window_exhausted"
    assert len(result.change_events) == 1
    assert len(result.change_events[0].new_events) == 1
    assert result.change_events[0].new_events[0]["event_type"] == "clue"
    assert result.change_events[0].new_events[0]["clue_text"] == "ANIMAL"


def test_should_probe_visible_log_from_state_skips_stable_quiet_banner_slice() -> None:
    state = SmokeRuntimeState()
    stable_clue = ClueEvent(
        team_color=TeamColor.BLUE,
        spymaster_name="ty",
        clue_text="ANIMAL",
        clue_count="2",
        timestamp_sec=100.0,
        sequence_index=0,
        confidence=0.9,
    )
    stable_guess = GuessEvent(
        player_name="Drew",
        word="DRAGON",
        card_color=CardColor.BLUE,
        timestamp_sec=101.0,
        sequence_index=1,
        confidence=0.84,
    )
    state.visible_log.previous_visible_events = [stable_clue, stable_guess]
    state.visible_log.quiet_log_frame_count = 6
    state.active_clue_window = ActiveClueWindow(
        clue_text="ANIMAL",
        clue_count="2",
        team_color=TeamColor.BLUE,
        first_seen_ts=100.0,
        last_seen_ts=106.0,
        sources={"banner", "visible_log"},
    )
    banner = BannerObservation(
        timestamp_sec=106.0,
        top_banner_text="WAIT FOR YOUR TURN, Drew ARE GUESSING",
        clue_text="ANIMAL",
        clue_count="2",
    )

    assert not _should_probe_visible_log_from_state(
        state,
        log_changed=False,
        banner_observation=banner,
    )


def test_filter_visible_log_events_for_runtime_drops_conflicting_unconfirmed_slice() -> None:
    state = SmokeRuntimeState()
    state.active_clue_window = ActiveClueWindow(
        clue_text="ANIMAL",
        clue_count="2",
        team_color=TeamColor.BLUE,
        first_seen_ts=100.0,
        last_seen_ts=104.0,
        sources={"banner"},
    )
    banner = BannerObservation(
        timestamp_sec=104.0,
        top_banner_text="WAIT FOR YOUR TURN, Drew ARE GUESSING",
        clue_text="ANIMAL",
        clue_count="2",
    )
    conflicting_slice = [
        ClueEvent(
            team_color=TeamColor.BLUE,
            spymaster_name="ty",
            clue_text="OE",
            clue_count="2",
            timestamp_sec=104.0,
            sequence_index=0,
            confidence=0.08,
        )
    ]

    assert _filter_visible_log_events_for_runtime(
        state,
        conflicting_slice,
        banner_observation=banner,
    ) == []


def test_same_team_low_confidence_fragment_does_not_replace_active_clue() -> None:
    state = SmokeRuntimeState()
    state.active_clue_window = ActiveClueWindow(
        clue_text="WOMEN",
        clue_count="3",
        team_color=TeamColor.RED,
        first_seen_ts=62.0,
        last_seen_ts=68.0,
        confidence=0.97,
        sources={"banner", "visible_log"},
    )
    banner = BannerObservation(
        timestamp_sec=68.0,
        top_banner_text="WAIT FOR YOUR TURN, Cherry IS GIVING A CLUE",
        clue_text="WOMEN",
        clue_count="3",
    )
    fragment = ClueEvent(
        team_color=TeamColor.RED,
        spymaster_name="Cherry",
        clue_text="MEN",
        clue_count="4",
        timestamp_sec=68.0,
        sequence_index=0,
        confidence=0.26467961968735987,
    )

    assert _filter_visible_log_events_for_runtime(
        state,
        [fragment],
        banner_observation=banner,
    ) == []
    assert state.active_clue_window.clue_text == "WOMEN"
    assert state.active_clue_window.clue_count == "3"


def test_fragment_candidate_becomes_review_only_not_committed_turn() -> None:
    state = SmokeRuntimeState(
        history=[
            ClueEvent(
                team_color=TeamColor.RED,
                spymaster_name="Cherry",
                clue_text="WOMEN",
                clue_count="3",
                timestamp_sec=62.0,
                sequence_index=0,
                confidence=0.97,
            )
        ]
    )
    state.active_clue_window = ActiveClueWindow(
        clue_text="WOMEN",
        clue_count="3",
        team_color=TeamColor.RED,
        first_seen_ts=62.0,
        last_seen_ts=68.0,
        confidence=0.97,
        sources={"banner", "visible_log"},
    )
    banner = BannerObservation(
        timestamp_sec=68.0,
        top_banner_text="WAIT FOR YOUR TURN, Cherry IS GIVING A CLUE",
        clue_text="WOMEN",
        clue_count="3",
    )
    fragment = ClueEvent(
        team_color=TeamColor.RED,
        spymaster_name="Cherry",
        clue_text="MEN",
        clue_count="4",
        timestamp_sec=68.0,
        sequence_index=0,
        confidence=0.26467961968735987,
    )

    filtered = _filter_visible_log_events_for_runtime(
        state,
        [fragment],
        banner_observation=banner,
    )
    committed = _commit_visible_log_events(state, filtered, timestamp_sec=68.0)

    assert committed == []
    assert len([event for event in state.history if isinstance(event, ClueEvent)]) == 1
    assert state.history[0].clue_text == "WOMEN"
    assert state.history[0].clue_count == "3"


def test_later_guesses_attach_to_original_active_clue_after_fragment_suppression() -> None:
    clue_event = ClueEvent(
        team_color=TeamColor.RED,
        spymaster_name="Cherry",
        clue_text="WOMEN",
        clue_count="3",
        timestamp_sec=62.0,
        sequence_index=0,
        confidence=0.97,
    )
    state = SmokeRuntimeState(history=[clue_event])
    state.active_clue_window = ActiveClueWindow(
        clue_text="WOMEN",
        clue_count="3",
        team_color=TeamColor.RED,
        first_seen_ts=62.0,
        last_seen_ts=68.0,
        confidence=0.97,
        sources={"banner", "visible_log"},
    )
    banner = BannerObservation(
        timestamp_sec=83.0,
        top_banner_text="WAIT FOR YOUR TURN, near ARE GUESSING",
        clue_text="WOMEN",
        clue_count="3",
    )
    fragment = ClueEvent(
        team_color=TeamColor.RED,
        spymaster_name="Cherry",
        clue_text="MEN",
        clue_count="4",
        timestamp_sec=68.0,
        sequence_index=0,
        confidence=0.26467961968735987,
    )
    guess = GuessEvent(
        player_name="unknown",
        word="GOLDILOCK",
        card_color=CardColor.RED,
        timestamp_sec=83.0,
        sequence_index=14,
        confidence=0.88,
    )

    filtered = _filter_visible_log_events_for_runtime(
        state,
        [fragment, guess],
        banner_observation=banner,
    )
    committed = _commit_visible_log_events(state, filtered, timestamp_sec=83.0)
    _update_active_clue_window(state, source="visible_log", events=filtered, timestamp_sec=83.0)

    assert filtered == [guess]
    assert committed == [guess]
    assert len([event for event in state.history if isinstance(event, ClueEvent)]) == 1
    assert state.history[0].clue_text == "WOMEN"
    assert state.active_clue_window is not None
    assert state.active_clue_window.clue_text == "WOMEN"
    assert state.active_clue_window.clue_count == "3"


def test_banner_supported_active_clue_beats_conflicting_visible_log_candidate() -> None:
    state = SmokeRuntimeState()
    state.active_clue_window = ActiveClueWindow(
        clue_text="WOMEN",
        clue_count="3",
        team_color=TeamColor.RED,
        first_seen_ts=62.0,
        last_seen_ts=68.0,
        confidence=0.97,
        sources={"banner", "visible_log"},
    )
    banner = BannerObservation(
        timestamp_sec=68.0,
        top_banner_text="WAIT FOR YOUR TURN, Cherry IS GIVING A CLUE",
        clue_text="WOMEN",
        clue_count="3",
    )
    conflicting_visible = [
        ClueEvent(
            team_color=TeamColor.RED,
            spymaster_name="Cherry",
            clue_text="MEN",
            clue_count="4",
            timestamp_sec=68.0,
            sequence_index=0,
            confidence=0.26467961968735987,
        )
    ]

    assert _filter_visible_log_events_for_runtime(
        state,
        conflicting_visible,
        banner_observation=banner,
    ) == []


def test_no_new_clue_commit_during_guessing_phase() -> None:
    state = SmokeRuntimeState()
    state.active_clue_window = ActiveClueWindow(
        clue_text="WOMEN",
        clue_count="3",
        team_color=TeamColor.RED,
        first_seen_ts=62.0,
        last_seen_ts=115.0,
        confidence=0.97,
        sources={"banner", "visible_log"},
    )
    banner = BannerObservation(
        timestamp_sec=120.5,
        top_banner_text="TAP ON CARDS YOU THINK MATCH THE CLUE",
        clue_text="WOMEN",
        clue_count="3",
    )
    stale_candidate = ClueEvent(
        team_color=TeamColor.RED,
        spymaster_name="Cherry",
        clue_text="CHERRY",
        clue_count="1",
        timestamp_sec=120.5,
        sequence_index=0,
        confidence=0.74,
    )

    assert _filter_visible_log_events_for_runtime(
        state,
        [stale_candidate],
        banner_observation=banner,
    ) == []


def test_stale_log_and_header_text_do_not_create_clues() -> None:
    state = SmokeRuntimeState()
    state.active_clue_window = ActiveClueWindow(
        clue_text="WOMEN",
        clue_count="3",
        team_color=TeamColor.RED,
        first_seen_ts=62.0,
        last_seen_ts=120.0,
        confidence=0.97,
        sources={"banner", "visible_log"},
    )
    banner = BannerObservation(
        timestamp_sec=124.5,
        top_banner_text="WAIT FOR LUKE TO GIVE YOU A CLUE",
        clue_text=None,
        clue_count=None,
    )
    contaminated_candidate = ClueEvent(
        team_color=TeamColor.BLUE,
        spymaster_name="luke",
        clue_text="JLEIILERGAMELOGWOMENESDTE",
        clue_count="3",
        timestamp_sec=124.5,
        sequence_index=0,
        confidence=0.72,
    )

    assert _filter_visible_log_events_for_runtime(
        state,
        [contaminated_candidate],
        banner_observation=banner,
    ) == []


def test_same_team_same_text_same_count_merges_despite_spymaster_drift() -> None:
    state = SmokeRuntimeState(
        history=[
            ClueEvent(
                team_color=TeamColor.BLUE,
                spymaster_name="luke",
                clue_text="IRELAND",
                clue_count="2",
                timestamp_sec=239.0,
                sequence_index=0,
                confidence=0.92,
            )
        ]
    )
    ireland_repeat = ClueEvent(
        team_color=TeamColor.BLUE,
        spymaster_name="Iy",
        clue_text="IRELAND",
        clue_count="2",
        timestamp_sec=243.5,
        sequence_index=0,
        confidence=0.81,
    )

    committed = _commit_visible_log_events(state, [ireland_repeat], timestamp_sec=243.5)

    assert committed == []
    clue_history = [event for event in state.history if isinstance(event, ClueEvent)]
    assert len(clue_history) == 1
    assert clue_history[0].clue_text == "IRELAND"
    assert clue_history[0].clue_count == "2"


def test_counter_zero_does_not_stop_on_single_bad_read(monkeypatch) -> None:
    frames = [
        FrameSample(timestamp_sec=0.0, frame_bgr=make_frame(60), frame_index_in_window=0),
        FrameSample(timestamp_sec=1.0, frame_bgr=make_frame(60), frame_index_in_window=1),
        FrameSample(timestamp_sec=2.0, frame_bgr=make_frame(60), frame_index_in_window=2),
    ]
    for frame in frames:
        frame.frame_bgr[60:492, 180:780] = 120
    vod_source = FakeVodSource(frames)
    backend = SequencedOCRBackend(
        {
            "blue:operative:in_game": [
                OCRDetection(text="Drew", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=70, bottom=40))
            ],
            "blue:spymaster:in_game": [
                OCRDetection(text="ty", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=40, bottom=40))
            ],
            "red:operative:in_game": [
                OCRDetection(text="near", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=60, bottom=40))
            ],
            "red:spymaster:in_game": [
                OCRDetection(text="shadowsn", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=90, bottom=40))
            ],
            **make_board_responses("ANTARCTICA"),
        }
    )

    observations = iter(
        [
            (
                CounterObservation(value=8, confidence=0.95, raw_text="8", variant_name="raw", score=4.0),
                CounterObservation(value=9, confidence=0.95, raw_text="9", variant_name="raw", score=4.0),
            ),
            (
                CounterObservation(value=0, confidence=0.95, raw_text="0", variant_name="adaptive_threshold", score=3.9),
                CounterObservation(value=9, confidence=0.95, raw_text="9", variant_name="raw", score=4.0),
            ),
            (
                CounterObservation(value=8, confidence=0.95, raw_text="8", variant_name="raw", score=4.0),
                CounterObservation(value=9, confidence=0.95, raw_text="9", variant_name="raw", score=4.0),
            ),
        ]
    )

    monkeypatch.setattr("app.runtime.smoke.cycle.runner.parse_game_counter_observations", lambda *args, **kwargs: next(observations))
    monkeypatch.setattr("app.runtime.smoke.cycle.runner.parse_top_banner_text", lambda *args, **kwargs: "")
    monkeypatch.setattr("app.runtime.smoke.cycle.runner._parse_center_clue_banner", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr("app.runtime.smoke.cycle.runner.parse_game_log", lambda *args, **kwargs: [])
    monkeypatch.setattr("app.runtime.smoke.cycle.runner.has_play_next_game", lambda *args, **kwargs: False)

    result = smoke_parse_game_cycle(
        vod_source=vod_source,
        vod_url="debug_snapshots/clips/game3-full.ts",
        start_sec=0.0,
        duration_sec=3.0,
        chunk_duration_sec=3.0,
        roi_config=make_roi_config(),
        ocr_backend=backend,
        vod_id="2745360291",
    )

    assert result.stop_reason == "window_exhausted"
    assert result.stop_sec == 2.0
    assert result.game_record is not None


def test_banner_candidate_requires_repeat_before_promotion_after_first_turn() -> None:
    state = SmokeRuntimeState(
        history=[
            ClueEvent(
                team_color=TeamColor.RED,
                spymaster_name="Cherry",
                clue_text="WOMEN",
                clue_count="3",
                timestamp_sec=62.0,
                sequence_index=0,
                confidence=0.97,
            ),
            GuessEvent(
                player_name="near",
                word="PURSE",
                card_color=CardColor.RED,
                timestamp_sec=65.0,
                sequence_index=1,
                confidence=0.84,
            ),
        ]
    )
    state.active_clue_window = ActiveClueWindow(
        clue_text="WOMEN",
        clue_count="3",
        team_color=TeamColor.RED,
        first_seen_ts=62.0,
        last_seen_ts=65.0,
        confidence=0.97,
        sources={"banner", "visible_log"},
    )
    banner = BannerObservation(
        timestamp_sec=120.0,
        top_banner_text="WAIT FOR YOUR TURN, ty IS GIVING A CLUE",
        clue_text="ANIMAL",
        clue_count="2",
    )
    candidate = ClueEvent(
        team_color=TeamColor.BLUE,
        spymaster_name="ty",
        clue_text="ANIMAL",
        clue_count="2",
        timestamp_sec=120.0,
        sequence_index=0,
        confidence=0.97,
    )

    first = _observe_clue_candidate(
        state,
        candidate,
        source="banner",
        frame_phase="spymaster_entering_clue",
        banner_observation=banner,
        timestamp_sec=120.0,
        saw_same_frame_guesses=False,
    )
    second = _observe_clue_candidate(
        state,
        candidate.model_copy(update={"timestamp_sec": 121.0}),
        source="banner",
        frame_phase="spymaster_entering_clue",
        banner_observation=BannerObservation(
            timestamp_sec=121.0,
            top_banner_text=banner.top_banner_text,
            clue_text=banner.clue_text,
            clue_count=banner.clue_count,
        ),
        timestamp_sec=121.0,
        saw_same_frame_guesses=False,
    )

    assert first is None
    assert second is not None
    assert second.clue_text == "ANIMAL"
    assert second.clue_count == "2"


def test_cross_team_contaminated_candidate_does_not_open_new_turn() -> None:
    state = SmokeRuntimeState(
        history=[
            ClueEvent(
                team_color=TeamColor.RED,
                spymaster_name="Cherry",
                clue_text="WOMEN",
                clue_count="3",
                timestamp_sec=62.0,
                sequence_index=0,
                confidence=0.97,
            ),
            GuessEvent(
                player_name="near",
                word="PURSE",
                card_color=CardColor.RED,
                timestamp_sec=70.0,
                sequence_index=1,
                confidence=0.84,
            ),
        ]
    )
    state.active_clue_window = ActiveClueWindow(
        clue_text="WOMEN",
        clue_count="3",
        team_color=TeamColor.RED,
        first_seen_ts=62.0,
        last_seen_ts=118.0,
        confidence=0.97,
        sources={"banner", "visible_log"},
    )
    banner = BannerObservation(
        timestamp_sec=120.0,
        top_banner_text="WAIT FOR LUKE TO GIVE YOU A CLUE",
        clue_text=None,
        clue_count=None,
    )
    candidate = ClueEvent(
        team_color=TeamColor.BLUE,
        spymaster_name="ty",
        clue_text="WIWOMEN",
        clue_count="3",
        timestamp_sec=120.0,
        sequence_index=0,
        confidence=0.88,
    )

    promoted = _observe_clue_candidate(
        state,
        candidate,
        source="visible_log",
        frame_phase="between_turns",
        banner_observation=banner,
        timestamp_sec=120.0,
        saw_same_frame_guesses=False,
    )

    assert promoted is None
    assert state.pending_clue is not None
    assert "cross_team_overlap" in state.pending_clue.contamination_flags


def test_first_clean_visible_log_clue_promotes_on_single_strong_frame() -> None:
    state = SmokeRuntimeState()
    candidate = ClueEvent(
        team_color=TeamColor.RED,
        spymaster_name="Cherry",
        clue_text="WOMEN",
        clue_count="3",
        timestamp_sec=62.0,
        sequence_index=0,
        confidence=0.92,
    )

    promoted = _observe_clue_candidate(
        state,
        candidate,
        source="visible_log",
        frame_phase="spymaster_entering_clue",
        banner_observation=BannerObservation(
            timestamp_sec=62.0,
            top_banner_text="WAIT FOR YOUR TURN, Cherry IS GIVING A CLUE",
            clue_text=None,
            clue_count=None,
        ),
        timestamp_sec=62.0,
        saw_same_frame_guesses=False,
    )

    assert promoted is not None
    assert promoted.clue_text == "WOMEN"


def test_low_confidence_visible_log_clue_with_same_frame_guesses_stays_uncommitted() -> None:
    state = SmokeRuntimeState()
    candidate = ClueEvent(
        team_color=TeamColor.BLUE,
        spymaster_name="unknown",
        clue_text="ANLUONTRE",
        clue_count="4",
        timestamp_sec=268.5,
        sequence_index=0,
        confidence=0.12,
    )
    banner = BannerObservation(
        timestamp_sec=268.5,
        top_banner_text="WAIT FOR YOUR TURN, near ARE GUESSING",
        clue_text=None,
        clue_count=None,
    )

    first = _observe_clue_candidate(
        state,
        candidate,
        source="visible_log",
        frame_phase="between_turns",
        banner_observation=banner,
        timestamp_sec=268.5,
        saw_same_frame_guesses=True,
    )
    second = _observe_clue_candidate(
        state,
        candidate.model_copy(update={"timestamp_sec": 269.0}),
        source="visible_log",
        frame_phase="between_turns",
        banner_observation=BannerObservation(
            timestamp_sec=269.0,
            top_banner_text=banner.top_banner_text,
            clue_text=None,
            clue_count=None,
        ),
        timestamp_sec=269.0,
        saw_same_frame_guesses=True,
    )

    assert first is None
    assert second is None
    assert state.pending_clue is not None


def test_black_guess_becomes_pending_not_stop_on_first_frame() -> None:
    state = SmokeRuntimeState()
    guess = GuessEvent(
        player_name="unknown",
        word="LEPRECHAUN",
        card_color=CardColor.BLACK,
        timestamp_sec=276.0,
        sequence_index=0,
        confidence=0.2,
    )

    first = _observe_pending_assassin_guess(
        state,
        evidence_events=[guess],
        frame_phase="guess_confirm",
        winner_visible=False,
        timestamp_sec=276.0,
    )
    second = _observe_pending_assassin_guess(
        state,
        evidence_events=[guess.model_copy(update={"timestamp_sec": 276.5})],
        frame_phase="guess_confirm",
        winner_visible=False,
        timestamp_sec=276.5,
    )

    assert first is None
    assert second is None
    assert state.pending_assassin_guess is not None
    assert state.pending_assassin_guess.event.word == "LEPRECHAUN"


def test_black_guess_confirms_after_repeated_high_confidence_evidence() -> None:
    state = SmokeRuntimeState()
    guess = GuessEvent(
        player_name="unknown",
        word="ASSASSIN",
        card_color=CardColor.BLACK,
        timestamp_sec=276.0,
        sequence_index=0,
        confidence=0.72,
    )

    first = _observe_pending_assassin_guess(
        state,
        evidence_events=[guess],
        frame_phase="guess_confirm",
        winner_visible=False,
        timestamp_sec=276.0,
    )
    second = _observe_pending_assassin_guess(
        state,
        evidence_events=[guess.model_copy(update={"timestamp_sec": 276.5})],
        frame_phase="guess_confirm",
        winner_visible=False,
        timestamp_sec=276.5,
    )

    assert first is None
    assert second is not None
    assert second.word == "ASSASSIN"


def test_locked_board_does_not_replace_on_small_or_equal_quality_change() -> None:
    current_board = BoardState(
        cells=[
            BoardCell(
                row=0,
                col=0,
                word="EUROPE",
                confidence=0.95,
                box=ImageBoundingBox(left=0, top=0, right=10, bottom=10),
            )
        ]
    )
    candidate_board = BoardState(
        cells=[
            BoardCell(
                row=0,
                col=0,
                word="DEW",
                confidence=0.96,
                box=ImageBoundingBox(left=0, top=0, right=10, bottom=10),
            )
        ]
    )

    assert not _should_replace_locked_board_state(
        current_board_state=current_board,
        candidate_board_state=candidate_board,
        board_locked=True,
    )
