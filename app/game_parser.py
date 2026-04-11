"""Turn reconstruction, winner detection, and game-level assembly helpers."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from app.gamelog_parser import ClueEvent, GuessEvent
from app.models import (
    CardColor,
    GameRecord,
    GuessRecord,
    GuessResult,
    PlayerRosterEntry,
    ReviewEntityType,
    TeamColor,
    TurnRecord,
    WinReason,
)
from app.review_queue import ReviewFlag, maybe_flag_review


class GameReconstructionResult(BaseModel):
    turns: list[TurnRecord] = Field(default_factory=list)
    winner_team: TeamColor | None = None
    win_reason: WinReason | None = None
    winner_confidence: float = Field(default=0.0, ge=0, le=1)
    review_flags: list[ReviewFlag] = Field(default_factory=list)


class GameBoundarySignal(BaseModel):
    timestamp_sec: float = Field(ge=0)
    board_fingerprint: str | None = None
    setup_visible: bool = False
    play_next_visible: bool = False
    left_counter: int | None = Field(default=None, ge=0)
    right_counter: int | None = Field(default=None, ge=0)
    assassin_revealed: bool = False


def classify_guess_result(team_color: TeamColor, card_color: CardColor) -> GuessResult:
    """Map an actual revealed card color into the team's outcome label."""

    if card_color is CardColor.BLACK:
        return GuessResult.ASSASSIN
    if card_color is CardColor.NEUTRAL:
        return GuessResult.NEUTRAL
    if team_color is TeamColor.BLUE:
        return GuessResult.CORRECT if card_color is CardColor.BLUE else GuessResult.ENEMY
    return GuessResult.CORRECT if card_color is CardColor.RED else GuessResult.ENEMY


def reconstruct_game(
    events: Sequence[ClueEvent | GuessEvent],
    roster: Sequence[PlayerRosterEntry],
    *,
    left_counter: int | None = None,
    right_counter: int | None = None,
    review_threshold: float = 0.8,
) -> GameReconstructionResult:
    """Convert ordered log events into turns, winner data, and review flags."""

    ordered_events = sorted(events, key=lambda item: (item.timestamp_sec, item.sequence_index))
    roster_lookup = {player.display_name: player for player in roster}

    turns: list[TurnRecord] = []
    review_flags: list[ReviewFlag] = []
    current_turn: TurnRecord | None = None

    for event in ordered_events:
        if isinstance(event, ClueEvent):
            current_turn = TurnRecord(
                turn_index=len(turns),
                team_color=event.team_color,
                spymaster_name=event.spymaster_name,
                clue_text=event.clue_text,
                clue_count=event.clue_count,
                timestamp_sec=event.timestamp_sec,
            )
            turns.append(current_turn)
            maybe_flag_review(
                review_flags,
                entity_type=ReviewEntityType.TURN,
                entity_id=f"turn:{current_turn.turn_index}",
                confidence=event.confidence,
                threshold=review_threshold,
                reason="Low-confidence clue parsing",
                timestamp_sec=event.timestamp_sec,
            )
            continue

        if current_turn is None:
            maybe_flag_review(
                review_flags,
                entity_type=ReviewEntityType.GUESS,
                entity_id=f"orphan-guess:{event.sequence_index}",
                confidence=0.0,
                threshold=1.0,
                reason="Guess was seen before any clue event",
                timestamp_sec=event.timestamp_sec,
            )
            continue

        player = roster_lookup.get(event.player_name)
        if player is None:
            maybe_flag_review(
                review_flags,
                entity_type=ReviewEntityType.GUESS,
                entity_id=f"guess:{current_turn.turn_index}:{len(current_turn.guesses)}",
                confidence=0.0,
                threshold=1.0,
                reason=f"Unknown player '{event.player_name}' in guess event",
                timestamp_sec=event.timestamp_sec,
            )
            continue

        result = classify_guess_result(player.team_color, event.card_color)
        current_turn.guesses.append(
            GuessRecord(
                player_name=event.player_name,
                word=event.word,
                card_color=event.card_color,
                result=result,
                timestamp_sec=event.timestamp_sec,
                confidence=event.confidence,
            )
        )
        maybe_flag_review(
            review_flags,
            entity_type=ReviewEntityType.GUESS,
            entity_id=f"guess:{current_turn.turn_index}:{len(current_turn.guesses) - 1}",
            confidence=event.confidence,
            threshold=review_threshold,
            reason="Low-confidence guess parsing",
            timestamp_sec=event.timestamp_sec,
        )

    winner_team, win_reason, winner_confidence = detect_winner(
        turns,
        roster,
        left_counter=left_counter,
        right_counter=right_counter,
    )
    if winner_team is None or win_reason is None:
        maybe_flag_review(
            review_flags,
            entity_type=ReviewEntityType.GAME,
            entity_id="game:winner",
            confidence=winner_confidence,
            threshold=1.0,
            reason="Winner could not be determined confidently",
        )

    return GameReconstructionResult(
        turns=turns,
        winner_team=winner_team,
        win_reason=win_reason,
        winner_confidence=winner_confidence,
        review_flags=review_flags,
    )


def detect_winner(
    turns: Sequence[TurnRecord],
    roster: Sequence[PlayerRosterEntry],
    *,
    left_counter: int | None = None,
    right_counter: int | None = None,
) -> tuple[TeamColor | None, WinReason | None, float]:
    """Determine the winner from assassin picks or score counters."""

    roster_lookup = {player.display_name: player for player in roster}
    for turn in turns:
        for guess in turn.guesses:
            if guess.result is not GuessResult.ASSASSIN:
                continue
            player = roster_lookup.get(guess.player_name)
            if player is None:
                return None, None, 0.0
            winner_team = TeamColor.RED if player.team_color is TeamColor.BLUE else TeamColor.BLUE
            return winner_team, WinReason.ASSASSIN, min(1.0, guess.confidence + 0.05)

    if left_counter == 0 and right_counter is not None:
        return TeamColor.BLUE, WinReason.COUNTER_ZERO, 0.98
    if right_counter == 0 and left_counter is not None:
        return TeamColor.RED, WinReason.COUNTER_ZERO, 0.98

    return None, None, 0.0


def build_game_record(
    *,
    vod_id: str,
    game_index: int,
    start_sec: float,
    end_sec: float,
    players: Sequence[PlayerRosterEntry],
    reconstruction: GameReconstructionResult,
    starting_team: TeamColor | None = None,
) -> GameRecord:
    """Build a `GameRecord` from reconstructed turns and winner metadata."""

    confidence_values = [reconstruction.winner_confidence]
    for turn in reconstruction.turns:
        confidence_values.extend(guess.confidence for guess in turn.guesses)
    positive_confidences = [value for value in confidence_values if value > 0]
    parse_confidence = (
        sum(positive_confidences) / len(positive_confidences) if positive_confidences else 0.0
    )

    return GameRecord(
        vod_id=vod_id,
        game_index=game_index,
        start_sec=start_sec,
        end_sec=end_sec,
        winner_team=reconstruction.winner_team,
        win_reason=reconstruction.win_reason,
        starting_team=starting_team,
        parse_confidence=parse_confidence,
        players=list(players),
        turns=reconstruction.turns,
    )


def split_game_windows(signals: Sequence[GameBoundarySignal]) -> list[tuple[float, float]]:
    """Split a sequence of boundary signals into approximate game windows."""

    ordered = sorted(signals, key=lambda item: item.timestamp_sec)
    windows: list[tuple[float, float]] = []
    current_start: float | None = None
    previous_fingerprint: str | None = None

    for signal in ordered:
        if current_start is None:
            if signal.board_fingerprint is not None and not signal.setup_visible:
                current_start = signal.timestamp_sec
                previous_fingerprint = signal.board_fingerprint
            continue

        board_changed = (
            signal.board_fingerprint is not None
            and previous_fingerprint is not None
            and signal.board_fingerprint != previous_fingerprint
        )
        game_over = (
            signal.assassin_revealed
            or signal.play_next_visible
            or signal.left_counter == 0
            or signal.right_counter == 0
        )
        if game_over:
            windows.append((current_start, signal.timestamp_sec))
            current_start = None
            previous_fingerprint = None
            continue
        if board_changed:
            windows.append((current_start, signal.timestamp_sec))
            current_start = signal.timestamp_sec
        previous_fingerprint = signal.board_fingerprint or previous_fingerprint

    return windows
