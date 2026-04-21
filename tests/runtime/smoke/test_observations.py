from __future__ import annotations

from pathlib import Path

from app.runtime.smoke.observations import _materialize_roster_observations, _observe_roster
from tests.support.smoke import (
    FakeVodSource,
    SequencedOCRBackend,
    make_board_responses,
    make_frame,
    make_roi_config,
)
from app.core.models import ImageBoundingBox, OCRDetection, PlayerRole, PlayerRosterEntry, TeamColor

from app.cli.smoke import smoke_parse_game_cycle

from app.infra.vod_source import FrameSample


def test_smoke_waits_for_usable_roster_before_locking_game() -> None:
    frame_a = make_frame(10)
    frame_b = make_frame(80)
    frame_c = make_frame(150)
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
                [OCRDetection(text="Ã¹", confidence=0.8, box=ImageBoundingBox(left=20, top=20, right=40, bottom=40))],
                [OCRDetection(text="Drew", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=70, bottom=40))],
            ],
            "blue:spymaster:in_game": [
                [OCRDetection(text="Ã¹", confidence=0.8, box=ImageBoundingBox(left=20, top=20, right=40, bottom=40))],
                [OCRDetection(text="ty", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=40, bottom=40))],
            ],
            "red:operative:in_game": [
                [OCRDetection(text="Ã¹", confidence=0.8, box=ImageBoundingBox(left=20, top=20, right=40, bottom=40))],
                [OCRDetection(text="near", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=60, bottom=40))],
            ],
            "red:spymaster:in_game": [
                [OCRDetection(text="Ã¹", confidence=0.8, box=ImageBoundingBox(left=20, top=20, right=40, bottom=40))],
                [OCRDetection(text="shadowsn", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=90, bottom=40))],
            ],
            **make_board_responses("ANTARCTICA"),
            "game_log": [[], [], []],
            "left_counter": [[], [], []],
            "right_counter": [[], [], []],
            "top_banner": [
                [],
                [],
                [OCRDetection(text="YOUR TEAM WINS!", confidence=0.98, box=ImageBoundingBox(left=1, top=1, right=90, bottom=12))],
            ],
            "end_banner": [[], [], []],
        }
    )

    result = smoke_parse_game_cycle(
        vod_source=vod_source,
        vod_url="https://www.twitch.tv/videos/2745360291",
        start_sec=10.0,
        duration_sec=10.0,
        fps=1.0,
        chunk_duration_sec=10.0,
        roi_config=make_roi_config(),
        ocr_backend=backend,
        vod_id="2745360291",
        snapshot_dir=None,
    )

    assert result.game_record is not None
    assert [player.display_name for player in result.game_record.players] == ["Drew", "ty", "near", "shadowsn"]


def test_smoke_parse_game_cycle_waits_for_non_empty_roster_and_saves_snapshots() -> None:
    frame_a = make_frame(10)
    frame_b = make_frame(80)
    frame_c = make_frame(120)
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
                [],
                [OCRDetection(text="ty", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=40, bottom=40))],
            ],
            "blue:spymaster:in_game": [
                [],
                [OCRDetection(text="rush", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=60, bottom=40))],
            ],
            "red:operative:in_game": [
                [],
                [OCRDetection(text="Cherry", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=80, bottom=40))],
            ],
            "red:spymaster:in_game": [
                [],
                [OCRDetection(text="arker", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=70, bottom=40))],
            ],
            **make_board_responses("SAIL"),
            "game_log": [
                [],
                [],
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
            ],
            "end_banner": [[], [], []],
        }
    )

    snapshot_dir = Path("build/test-temp/smoke-snapshots")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    result = smoke_parse_game_cycle(
        vod_source=vod_source,
        vod_url="https://www.twitch.tv/videos/2745360291",
        start_sec=10.0,
        duration_sec=3.0,
        fps=1.0,
        chunk_duration_sec=2.0,
        roi_config=make_roi_config(),
        ocr_backend=backend,
        vod_id="2745360291",
        snapshot_dir=snapshot_dir,
    )

    assert result.stop_reason == "window_exhausted"
    assert result.game_record is not None
    assert [player.display_name for player in result.game_record.players] == ["ty", "rush", "Cherry", "arker"]
    assert result.start_snapshot_path is not None
    assert Path(result.start_snapshot_path).exists()
    assert result.first_frame_snapshot_path is not None
    assert Path(result.first_frame_snapshot_path).exists()
    assert result.first_preready_snapshot_path is not None
    assert Path(result.first_preready_snapshot_path).exists()
    assert result.roi_probe_overlay_path is not None
    assert Path(result.roi_probe_overlay_path).exists()
    assert result.roi_probe_crop_paths
    assert all(Path(path).exists() for path in result.roi_probe_crop_paths.values())
    assert result.change_events == []


def test_materialize_roster_observations_caps_team_roles_to_valid_sizes() -> None:
    observations = {}
    weighted_entries = [
        (6, PlayerRosterEntry(player_name="ty", display_name="ty", team_color=TeamColor.BLUE, role=PlayerRole.OPERATIVE, avatar_hash="aa")),
        (5, PlayerRosterEntry(player_name="drew", display_name="Drew", team_color=TeamColor.BLUE, role=PlayerRole.OPERATIVE, avatar_hash="bb")),
        (4, PlayerRosterEntry(player_name="luke", display_name="luke", team_color=TeamColor.BLUE, role=PlayerRole.OPERATIVE, avatar_hash="cc")),
        (1, PlayerRosterEntry(player_name="noise", display_name="noise", team_color=TeamColor.BLUE, role=PlayerRole.OPERATIVE, avatar_hash="dd")),
        (4, PlayerRosterEntry(player_name="rush", display_name="rush", team_color=TeamColor.BLUE, role=PlayerRole.SPYMASTER, avatar_hash="ee")),
        (1, PlayerRosterEntry(player_name="alt", display_name="alt", team_color=TeamColor.BLUE, role=PlayerRole.SPYMASTER, avatar_hash="ff")),
    ]
    for count, entry in weighted_entries:
        for _ in range(count):
            _observe_roster(observations, [entry])

    roster = _materialize_roster_observations(observations)

    assert [player.display_name for player in roster] == ["Drew", "luke", "ty", "rush"]
