from __future__ import annotations

import app.reconstruction.game_reconstruction as game_assembler

from app.core.models import CardColor, GuessResult, ImageBoundingBox, OCRDetection, PlayerRole, PlayerRosterEntry, TeamColor, WinReason

from app.parsers.gamelog import ClueEvent, GuessEvent
from app.review.queue import materialize_review_items
from app.infra.roi_config import ROIConfig
from app.infra.vod_source import FrameSample
from app.vision.ocr.preprocessing import generate_ocr_variants

GameBoundarySignal = game_assembler.GameBoundarySignal
build_game_record = game_assembler.build_game_record
classify_guess_result = game_assembler.classify_guess_result
derive_game_windows = game_assembler.derive_game_windows
parse_game_window = game_assembler.parse_game_window
parse_segment_games = game_assembler.parse_segment_games
reconstruct_game = game_assembler.reconstruct_game
split_game_windows = game_assembler.split_game_windows
_merge_visible_event_history = game_assembler._merge_visible_event_history


def make_roster() -> list[PlayerRosterEntry]:
    return [
        PlayerRosterEntry(
            player_name="fembluca",
            display_name="fembluca",
            team_color=TeamColor.BLUE,
            role=PlayerRole.SPYMASTER,
            avatar_hash="a",
        ),
        PlayerRosterEntry(
            player_name="†",
            display_name="†",
            team_color=TeamColor.BLUE,
            role=PlayerRole.OPERATIVE,
            avatar_hash="b",
        ),
        PlayerRosterEntry(
            player_name="Cherry",
            display_name="Cherry",
            team_color=TeamColor.RED,
            role=PlayerRole.OPERATIVE,
            avatar_hash="c",
        ),
        PlayerRosterEntry(
            player_name="Zek 67",
            display_name="Zek 67",
            team_color=TeamColor.RED,
            role=PlayerRole.SPYMASTER,
            avatar_hash="d",
        ),
    ]


class SequencedOCRBackend:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.indices: dict[str, int] = {}

    def detect_text(self, image, *, hint: str | None = None):
        assert image.size > 0
        key = hint or ""
        value = self.responses.get(key, [])
        if value and isinstance(value[0], OCRDetection):
            return list(value)
        sequence = value if isinstance(value, list) else []
        index = self.indices.get(key, 0)
        self.indices[key] = index + 1
        if not sequence:
            return []
        return list(sequence[min(index, len(sequence) - 1)])


def make_roi_config() -> ROIConfig:
    return ROIConfig.from_raw(
        {
            "left_team_panel": {"x": 0.0, "y": 0.0, "width": 0.18, "height": 1.0},
            "right_team_panel": {"x": 0.82, "y": 0.0, "width": 0.18, "height": 1.0},
            "board_region": {"x": 0.18, "y": 0.10, "width": 0.60, "height": 0.72},
            "game_log_region": {"x": 0.80, "y": 0.50, "width": 0.18, "height": 0.40},
            "left_counter_region": {"x": 0.05, "y": 0.35, "width": 0.05, "height": 0.08},
            "right_counter_region": {"x": 0.90, "y": 0.35, "width": 0.05, "height": 0.08},
            "top_banner_region": {"x": 0.18, "y": 0.0, "width": 0.60, "height": 0.10},
            "end_banner_region": {"x": 0.18, "y": 0.80, "width": 0.64, "height": 0.16},
        }
    )


def paint_region(frame, box, color):
    frame[box.top : box.bottom, box.left : box.right] = color


def make_base_frame() -> object:
    return __import__("numpy").zeros((600, 1000, 3), dtype=__import__("numpy").uint8)


def repeat_counter_sequence(sequence: list[list[OCRDetection]]) -> list[list[OCRDetection]]:
    # Counter OCR now probes several image variants per frame; keep each scripted frame
    # reading stable across that fan-out so tests model frame-level behavior.
    variant_reads = max(
        1,
        len(
            generate_ocr_variants(
                make_base_frame()[:20, :20],
                profile="counter",
                hint="left_counter",
            )
        ),
    )
    return [list(entry) for entry in sequence for _ in range(variant_reads)]


def test_classify_guess_result_covers_blue_and_red_teams() -> None:
    assert classify_guess_result(TeamColor.BLUE, CardColor.BLUE) is GuessResult.CORRECT
    assert classify_guess_result(TeamColor.BLUE, CardColor.RED) is GuessResult.ENEMY
    assert classify_guess_result(TeamColor.BLUE, CardColor.NEUTRAL) is GuessResult.NEUTRAL
    assert classify_guess_result(TeamColor.BLUE, CardColor.BLACK) is GuessResult.ASSASSIN
    assert classify_guess_result(TeamColor.RED, CardColor.RED) is GuessResult.CORRECT
    assert classify_guess_result(TeamColor.RED, CardColor.BLUE) is GuessResult.ENEMY


def test_merge_visible_event_history_uses_same_duplicate_guard() -> None:
    initial = GuessEvent(
        player_name="unknown",
        word="COLLAR",
        card_color=CardColor.RED,
        timestamp_sec=169.0,
        sequence_index=0,
        confidence=0.52,
    )
    improved = GuessEvent(
        player_name="near",
        word="COLLAR",
        card_color=CardColor.RED,
        timestamp_sec=183.0,
        sequence_index=0,
        confidence=0.61,
    )

    history, previous_visible = _merge_visible_event_history([], [], [initial])
    history, previous_visible = _merge_visible_event_history(history, previous_visible, [])
    history, previous_visible = _merge_visible_event_history(history, previous_visible, [improved])

    assert len(history) == 1
    assert history[0].player_name == "near"
    assert history[0].confidence == 0.61
    assert previous_visible == [improved]


def test_reconstruct_game_builds_turns_and_detects_counter_zero_win() -> None:
    events = [
        ClueEvent(
            team_color=TeamColor.BLUE,
            spymaster_name="fembluca",
            clue_text="COLD",
            clue_count="2",
            timestamp_sec=10.0,
            sequence_index=0,
            confidence=0.96,
        ),
        GuessEvent(
            player_name="†",
            word="ANTARCTICA",
            card_color=CardColor.BLUE,
            timestamp_sec=11.0,
            sequence_index=1,
            confidence=0.93,
        ),
        GuessEvent(
            player_name="†",
            word="NIGHT",
            card_color=CardColor.RED,
            timestamp_sec=12.0,
            sequence_index=2,
            confidence=0.90,
        ),
        ClueEvent(
            team_color=TeamColor.RED,
            spymaster_name="Zek 67",
            clue_text="JAX",
            clue_count="3",
            timestamp_sec=20.0,
            sequence_index=3,
            confidence=0.97,
        ),
        GuessEvent(
            player_name="Cherry",
            word="STAFF",
            card_color=CardColor.RED,
            timestamp_sec=21.0,
            sequence_index=4,
            confidence=0.92,
        ),
    ]

    reconstruction = reconstruct_game(events, make_roster(), left_counter=0, right_counter=1)

    assert len(reconstruction.turns) == 2
    assert reconstruction.turns[0].guesses[0].result is GuessResult.CORRECT
    assert reconstruction.turns[0].guesses[1].result is GuessResult.ENEMY
    assert reconstruction.winner_team is TeamColor.BLUE
    assert reconstruction.win_reason is WinReason.COUNTER_ZERO


def test_reconstruct_game_handles_assassin_loss() -> None:
    reconstruction = reconstruct_game(
        [
            ClueEvent(
                team_color=TeamColor.BLUE,
                spymaster_name="fembluca",
                clue_text="FOOD",
                clue_count="2",
                timestamp_sec=10.0,
                sequence_index=0,
                confidence=0.95,
            ),
            GuessEvent(
                player_name="†",
                word="BANK",
                card_color=CardColor.BLACK,
                timestamp_sec=12.0,
                sequence_index=1,
                confidence=0.94,
            ),
        ],
        make_roster(),
    )

    assert reconstruction.winner_team is TeamColor.RED
    assert reconstruction.win_reason is WinReason.ASSASSIN


def test_reconstruct_game_inferrs_winner_from_reached_starting_team_target() -> None:
    events = [
        ClueEvent(
            team_color=TeamColor.RED,
            spymaster_name="Zek 67",
            clue_text="PROM",
            clue_count="4",
            timestamp_sec=10.0,
            sequence_index=0,
            confidence=0.95,
        )
    ]
    for index in range(9):
        events.append(
            GuessEvent(
                player_name="Cherry",
                word=f"WORD{index}",
                card_color=CardColor.RED,
                timestamp_sec=11.0 + index,
                sequence_index=index + 1,
                confidence=0.9,
            )
        )

    reconstruction = reconstruct_game(events, make_roster())

    assert reconstruction.winner_team is TeamColor.RED
    assert reconstruction.win_reason is WinReason.COUNTER_ZERO


def test_reconstruct_game_routes_low_confidence_events_to_review_queue() -> None:
    reconstruction = reconstruct_game(
        [
            ClueEvent(
                team_color=TeamColor.BLUE,
                spymaster_name="fembluca",
                clue_text="HUNT",
                clue_count="2",
                timestamp_sec=10.0,
                sequence_index=0,
                confidence=0.55,
            ),
            GuessEvent(
                player_name="†",
                word="WITCH",
                card_color=CardColor.BLUE,
                timestamp_sec=11.0,
                sequence_index=1,
                confidence=0.51,
            ),
        ],
        make_roster(),
        review_threshold=0.8,
    )

    review_items = materialize_review_items(reconstruction.review_flags)

    assert len(review_items) >= 2
    assert any(item.entity_type.value == "turn" for item in review_items)
    assert any(item.entity_type.value == "guess" for item in review_items)


def test_build_game_record_uses_reconstruction_output() -> None:
    reconstruction = reconstruct_game([], make_roster(), left_counter=0, right_counter=1)

    record = build_game_record(
        vod_id="vod-1",
        game_index=0,
        start_sec=100.0,
        end_sec=200.0,
        players=make_roster(),
        reconstruction=reconstruction,
        starting_team=TeamColor.BLUE,
    )

    assert record.vod_id == "vod-1"
    assert record.winner_team is TeamColor.BLUE
    assert record.starting_team is TeamColor.BLUE


def test_split_game_windows_splits_on_end_state_and_new_board() -> None:
    windows = split_game_windows(
        [
            GameBoundarySignal(timestamp_sec=100.0, board_fingerprint="a"),
            GameBoundarySignal(timestamp_sec=200.0, board_fingerprint="a"),
            GameBoundarySignal(timestamp_sec=300.0, board_fingerprint="a", play_next_visible=True),
            GameBoundarySignal(timestamp_sec=350.0, setup_visible=True),
            GameBoundarySignal(timestamp_sec=400.0, board_fingerprint="b"),
            GameBoundarySignal(timestamp_sec=500.0, board_fingerprint="c"),
            GameBoundarySignal(timestamp_sec=520.0, board_fingerprint="c", assassin_revealed=True),
        ]
    )

    assert windows == [(100.0, 300.0), (400.0, 500.0), (500.0, 520.0)]


def test_parse_game_window_reconstructs_one_game_from_frame_samples() -> None:
    frame_a = make_base_frame()
    frame_b = make_base_frame()
    log_box = ImageBoundingBox(left=800, top=300, right=980, bottom=540)
    paint_region(frame_a, log_box, (10, 10, 10))
    paint_region(frame_b, log_box, (60, 60, 60))

    backend = SequencedOCRBackend(
        {
            "blue:operative:in_game": [
                OCRDetection(text="†", confidence=0.92, box=ImageBoundingBox(left=20, top=20, right=40, bottom=40))
            ],
            "blue:spymaster:in_game": [
                OCRDetection(text="fembluca", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=90, bottom=40))
            ],
            "board:0:0": [
                OCRDetection(text="ANTARCTICA", confidence=0.97, box=ImageBoundingBox(left=1, top=1, right=10, bottom=10))
            ],
            "game_log": [
                [
                    OCRDetection(text="fembluca", confidence=0.98, box=ImageBoundingBox(left=8, top=16, right=70, bottom=34)),
                    OCRDetection(text="COLD", confidence=0.99, box=ImageBoundingBox(left=78, top=14, right=132, bottom=36)),
                    OCRDetection(text="2", confidence=0.97, box=ImageBoundingBox(left=142, top=14, right=156, bottom=36)),
                ],
                [
                    OCRDetection(text="fembluca", confidence=0.98, box=ImageBoundingBox(left=8, top=16, right=70, bottom=34)),
                    OCRDetection(text="COLD", confidence=0.99, box=ImageBoundingBox(left=78, top=14, right=132, bottom=36)),
                    OCRDetection(text="2", confidence=0.97, box=ImageBoundingBox(left=142, top=14, right=156, bottom=36)),
                    OCRDetection(text="†", confidence=0.82, box=ImageBoundingBox(left=8, top=68, right=28, bottom=86)),
                    OCRDetection(text="ANTARCTlCA", confidence=0.88, box=ImageBoundingBox(left=38, top=66, right=130, bottom=88)),
                ],
            ],
            "left_counter": [
                *repeat_counter_sequence(
                    [
                        [
                            OCRDetection(
                                text="9",
                                confidence=0.95,
                                box=ImageBoundingBox(left=1, top=1, right=10, bottom=10),
                            )
                        ],
                        [
                            OCRDetection(
                                text="0",
                                confidence=0.95,
                                box=ImageBoundingBox(left=1, top=1, right=10, bottom=10),
                            )
                        ],
                    ]
                ),
            ],
            "right_counter": [
                *repeat_counter_sequence(
                    [
                        [
                            OCRDetection(
                                text="8",
                                confidence=0.95,
                                box=ImageBoundingBox(left=1, top=1, right=10, bottom=10),
                            )
                        ],
                        [
                            OCRDetection(
                                text="1",
                                confidence=0.95,
                                box=ImageBoundingBox(left=1, top=1, right=10, bottom=10),
                            )
                        ],
                    ]
                ),
            ],
        }
    )

    record, review_items = parse_game_window(
        vod_id="vod-1",
        game_index=0,
        frame_samples=[
            FrameSample(timestamp_sec=10.0, frame_bgr=frame_a, frame_index_in_window=0),
            FrameSample(timestamp_sec=11.0, frame_bgr=frame_b, frame_index_in_window=1),
        ],
        roi_config=make_roi_config(),
        ocr_backend=backend,
    )

    assert record.vod_id == "vod-1"
    assert len(record.turns) == 1
    assert len(record.turns[0].guesses) == 1
    assert record.turns[0].guesses[0].word == "ANTARCTICA"
    assert record.winner_team is TeamColor.BLUE
    assert review_items == []


def test_parse_segment_games_slices_multiple_windows() -> None:
    frame_a = make_base_frame()
    frame_b = make_base_frame()
    frame_c = make_base_frame()
    frame_d = make_base_frame()

    backend = SequencedOCRBackend(
        {
            "blue:operative:in_game": [
                OCRDetection(text="†", confidence=0.92, box=ImageBoundingBox(left=20, top=20, right=40, bottom=40))
            ],
            "blue:spymaster:in_game": [
                OCRDetection(text="fembluca", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=90, bottom=40))
            ],
            "board:0:0": [
                OCRDetection(text="ANTARCTICA", confidence=0.97, box=ImageBoundingBox(left=1, top=1, right=10, bottom=10))
            ],
            "left_counter": [
                *repeat_counter_sequence(
                    [
                        [
                            OCRDetection(
                                text="0",
                                confidence=0.95,
                                box=ImageBoundingBox(left=1, top=1, right=10, bottom=10),
                            )
                        ],
                        [
                            OCRDetection(
                                text="0",
                                confidence=0.95,
                                box=ImageBoundingBox(left=1, top=1, right=10, bottom=10),
                            )
                        ],
                        [
                            OCRDetection(
                                text="2",
                                confidence=0.95,
                                box=ImageBoundingBox(left=1, top=1, right=10, bottom=10),
                            )
                        ],
                        [
                            OCRDetection(
                                text="2",
                                confidence=0.95,
                                box=ImageBoundingBox(left=1, top=1, right=10, bottom=10),
                            )
                        ],
                    ]
                ),
            ],
            "right_counter": [
                *repeat_counter_sequence(
                    [
                        [
                            OCRDetection(
                                text="1",
                                confidence=0.95,
                                box=ImageBoundingBox(left=1, top=1, right=10, bottom=10),
                            )
                        ],
                        [
                            OCRDetection(
                                text="1",
                                confidence=0.95,
                                box=ImageBoundingBox(left=1, top=1, right=10, bottom=10),
                            )
                        ],
                        [
                            OCRDetection(
                                text="0",
                                confidence=0.95,
                                box=ImageBoundingBox(left=1, top=1, right=10, bottom=10),
                            )
                        ],
                        [
                            OCRDetection(
                                text="0",
                                confidence=0.95,
                                box=ImageBoundingBox(left=1, top=1, right=10, bottom=10),
                            )
                        ],
                    ]
                ),
            ],
        }
    )

    records, review_items = parse_segment_games(
        vod_id="vod-2",
        frame_samples=[
            FrameSample(timestamp_sec=10.0, frame_bgr=frame_a, frame_index_in_window=0),
            FrameSample(timestamp_sec=11.0, frame_bgr=frame_b, frame_index_in_window=1),
            FrameSample(timestamp_sec=20.0, frame_bgr=frame_c, frame_index_in_window=2),
            FrameSample(timestamp_sec=21.0, frame_bgr=frame_d, frame_index_in_window=3),
        ],
        roi_config=make_roi_config(),
        ocr_backend=backend,
        game_windows=[(10.0, 11.0), (20.0, 21.0)],
    )

    assert len(records) == 2
    assert records[0].winner_team is TeamColor.BLUE
    assert records[1].winner_team is TeamColor.RED
    assert review_items == []


def test_derive_game_windows_skips_setup_and_finds_end_state() -> None:
    setup_frame = make_base_frame()
    game_frame = make_base_frame()
    end_frame = make_base_frame()
    game_frame[100:300, 200:600] = 40
    end_frame[100:300, 200:600] = 180

    backend = SequencedOCRBackend(
        {
            "top_banner": [
                [OCRDetection(text="GAME SETTINGS", confidence=0.95, box=ImageBoundingBox(left=1, top=1, right=40, bottom=12))],
                [],
                [OCRDetection(text="YOUR TEAM WINS!", confidence=0.95, box=ImageBoundingBox(left=1, top=1, right=70, bottom=12))],
            ],
            "end_banner": [
                [],
                [],
                [],
            ],
            "left_counter": [
                *repeat_counter_sequence(
                    [
                        [],
                        [
                            OCRDetection(
                                text="2",
                                confidence=0.95,
                                box=ImageBoundingBox(left=1, top=1, right=10, bottom=10),
                            )
                        ],
                        [
                            OCRDetection(
                                text="0",
                                confidence=0.95,
                                box=ImageBoundingBox(left=1, top=1, right=10, bottom=10),
                            )
                        ],
                    ]
                ),
            ],
            "right_counter": [
                *repeat_counter_sequence(
                    [
                        [],
                        [
                            OCRDetection(
                                text="3",
                                confidence=0.95,
                                box=ImageBoundingBox(left=1, top=1, right=10, bottom=10),
                            )
                        ],
                        [
                            OCRDetection(
                                text="1",
                                confidence=0.95,
                                box=ImageBoundingBox(left=1, top=1, right=10, bottom=10),
                            )
                        ],
                    ]
                ),
            ],
            "blue:operative:in_game": [
                OCRDetection(text="â€ ", confidence=0.92, box=ImageBoundingBox(left=20, top=20, right=40, bottom=40))
            ],
            "blue:spymaster:in_game": [
                OCRDetection(text="fembluca", confidence=0.95, box=ImageBoundingBox(left=20, top=20, right=90, bottom=40))
            ],
            "board:0:0": [
                OCRDetection(text="ANTARCTICA", confidence=0.97, box=ImageBoundingBox(left=1, top=1, right=10, bottom=10))
            ],
            "game_log": [
                [],
                [],
            ],
        }
    )

    windows = derive_game_windows(
        [
            FrameSample(timestamp_sec=10.0, frame_bgr=setup_frame, frame_index_in_window=0),
            FrameSample(timestamp_sec=20.0, frame_bgr=game_frame, frame_index_in_window=1),
            FrameSample(timestamp_sec=30.0, frame_bgr=end_frame, frame_index_in_window=2),
        ],
        roi_config=make_roi_config(),
        ocr_backend=backend,
    )

    assert windows == [(20.0, 30.0)]
