from __future__ import annotations

from app.game_parser import (
    GameBoundarySignal,
    build_game_record,
    classify_guess_result,
    reconstruct_game,
    split_game_windows,
)
from app.gamelog_parser import ClueEvent, GuessEvent
from app.models import CardColor, GuessResult, PlayerRole, PlayerRosterEntry, TeamColor, WinReason
from app.review_queue import materialize_review_items


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


def test_classify_guess_result_covers_blue_and_red_teams() -> None:
    assert classify_guess_result(TeamColor.BLUE, CardColor.BLUE) is GuessResult.CORRECT
    assert classify_guess_result(TeamColor.BLUE, CardColor.RED) is GuessResult.ENEMY
    assert classify_guess_result(TeamColor.BLUE, CardColor.NEUTRAL) is GuessResult.NEUTRAL
    assert classify_guess_result(TeamColor.BLUE, CardColor.BLACK) is GuessResult.ASSASSIN
    assert classify_guess_result(TeamColor.RED, CardColor.RED) is GuessResult.CORRECT
    assert classify_guess_result(TeamColor.RED, CardColor.BLUE) is GuessResult.ENEMY


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
