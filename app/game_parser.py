"""Turn reconstruction, winner detection, and game-level assembly helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, Field

from app.board_parser import parse_board_words
from app.frame_detectors import (
    detect_assassin_in_events,
    has_play_next_game,
    make_board_fingerprint,
    parse_game_counters,
    setup_screen_visible,
)
from app.gamelog_parser import ClueEvent, GuessEvent, log_has_changed, parse_game_log
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
from app.ocr import OCRBackend
from app.review_queue import ReviewFlag, materialize_review_items, maybe_flag_review
from app.roi_config import ROIConfig, crop_roi
from app.roster_parser import parse_rosters
from app.vod_source import FrameSample


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


def parse_game_window(
    *,
    vod_id: str,
    game_index: int,
    frame_samples: Sequence[FrameSample],
    roi_config: ROIConfig,
    ocr_backend: OCRBackend,
    review_threshold: float = 0.8,
    starting_team: TeamColor | None = None,
    snapshot_dir: str | Path | None = None,
) -> tuple[GameRecord, list]:
    """Parse one game's frame window into a `GameRecord` and review queue rows."""

    if not frame_samples:
        raise ValueError("frame_samples must not be empty")

    ordered_frames = sorted(frame_samples, key=lambda item: item.timestamp_sec)
    first_frame = ordered_frames[0].frame_bgr
    players = parse_rosters(first_frame, roi_config, ocr_backend)
    board_state = parse_board_words(first_frame, roi_config, ocr_backend)

    events: list[ClueEvent | GuessEvent] = []
    previous_visible_events: list[ClueEvent | GuessEvent] = []
    previous_log_frame = None
    last_left_counter: int | None = None
    last_right_counter: int | None = None

    for sample in ordered_frames:
        frame = sample.frame_bgr
        log_frame = crop_roi(frame, roi_config.require("game_log_region"))
        if log_has_changed(previous_log_frame, log_frame):
            parsed_events = parse_game_log(
                frame,
                roi_config,
                ocr_backend,
                players,
                board_state,
                timestamp_sec=sample.timestamp_sec,
            )
            events, previous_visible_events = _merge_visible_event_history(
                events,
                previous_visible_events,
                parsed_events,
            )
        previous_log_frame = log_frame

        left_counter, right_counter = parse_game_counters(frame, roi_config, ocr_backend)
        if left_counter is not None:
            last_left_counter = left_counter
        if right_counter is not None:
            last_right_counter = right_counter

    reconstruction = reconstruct_game(
        events,
        players,
        left_counter=last_left_counter,
        right_counter=last_right_counter,
        review_threshold=review_threshold,
    )
    record = build_game_record(
        vod_id=vod_id,
        game_index=game_index,
        start_sec=ordered_frames[0].timestamp_sec,
        end_sec=ordered_frames[-1].timestamp_sec,
        players=players,
        reconstruction=reconstruction,
        starting_team=starting_team,
    )
    review_items = materialize_review_items(reconstruction.review_flags)
    if review_items and snapshot_dir is not None:
        snapshot_path = _save_game_log_snapshot(
            ordered_frames[-1].frame_bgr,
            roi_config,
            snapshot_dir,
            f"game-{game_index}-log",
        )
        for item in review_items:
            if item.snapshot_path is None:
                item.snapshot_path = snapshot_path
    return record, review_items


def parse_segment_games(
    *,
    vod_id: str,
    frame_samples: Sequence[FrameSample],
    roi_config: ROIConfig,
    ocr_backend: OCRBackend,
    game_windows: Sequence[tuple[float, float]] | None = None,
    review_threshold: float = 0.8,
    snapshot_dir: str | Path | None = None,
) -> tuple[list[GameRecord], list]:
    """Parse one segment's frame samples into one or more `GameRecord`s."""

    if not frame_samples:
        return [], []

    ordered_frames = sorted(frame_samples, key=lambda item: item.timestamp_sec)
    windows = (
        list(game_windows)
        if game_windows is not None
        else derive_game_windows(
            ordered_frames,
            roi_config=roi_config,
            ocr_backend=ocr_backend,
        )
    )
    if not windows:
        windows = [(ordered_frames[0].timestamp_sec, ordered_frames[-1].timestamp_sec)]

    records: list[GameRecord] = []
    review_items = []
    for index, (start_sec, end_sec) in enumerate(windows):
        window_frames = [
            sample
            for sample in ordered_frames
            if start_sec <= sample.timestamp_sec <= end_sec
        ]
        if not window_frames:
            continue
        record, items = parse_game_window(
            vod_id=vod_id,
            game_index=index,
            frame_samples=window_frames,
            roi_config=roi_config,
            ocr_backend=ocr_backend,
            review_threshold=review_threshold,
            snapshot_dir=snapshot_dir,
        )
        records.append(record)
        review_items.extend(items)

    return records, review_items


def _event_signature(event: ClueEvent | GuessEvent) -> tuple[str, str, str, str]:
    if isinstance(event, ClueEvent):
        return ("clue", event.team_color.value, event.spymaster_name, f"{event.clue_text}:{event.clue_count}")
    return ("guess", event.player_name, event.word, event.card_color.value)


def _merge_visible_event_history(
    history: list[ClueEvent | GuessEvent],
    previous_visible: Sequence[ClueEvent | GuessEvent],
    current_visible: Sequence[ClueEvent | GuessEvent],
) -> tuple[list[ClueEvent | GuessEvent], list[ClueEvent | GuessEvent]]:
    current_signatures = [_event_signature(event) for event in current_visible]
    previous_signatures = [_event_signature(event) for event in previous_visible]

    overlap = 0
    max_overlap = min(len(previous_signatures), len(current_signatures))
    for candidate in range(max_overlap, -1, -1):
        if previous_signatures[-candidate:] == current_signatures[:candidate]:
            overlap = candidate
            break

    appended = list(current_visible[overlap:])
    return history + appended, list(current_visible)


def derive_game_windows(
    frame_samples: Sequence[FrameSample],
    *,
    roi_config: ROIConfig,
    ocr_backend: OCRBackend,
) -> list[tuple[float, float]]:
    """Derive approximate game windows directly from a segment's frame sequence."""

    ordered_frames = sorted(frame_samples, key=lambda item: item.timestamp_sec)
    signals: list[GameBoundarySignal] = []
    current_players: Sequence[PlayerRosterEntry] | None = None
    current_board_state = None
    previous_log_frame = None

    for sample in ordered_frames:
        frame = sample.frame_bgr
        setup_visible = setup_screen_visible(frame, roi_config, ocr_backend)
        board_fingerprint = make_board_fingerprint(frame, roi_config)
        left_counter, right_counter = parse_game_counters(frame, roi_config, ocr_backend)
        play_next_visible = has_play_next_game(frame, roi_config, ocr_backend)
        assassin_revealed = False

        if not setup_visible:
            if current_players is None or current_board_state is None:
                current_players = parse_rosters(frame, roi_config, ocr_backend)
                current_board_state = parse_board_words(frame, roi_config, ocr_backend)
                previous_log_frame = None

            log_frame = crop_roi(frame, roi_config.require("game_log_region"))
            if log_has_changed(previous_log_frame, log_frame):
                parsed_events = parse_game_log(
                    frame,
                    roi_config,
                    ocr_backend,
                    current_players,
                    current_board_state,
                    timestamp_sec=sample.timestamp_sec,
                )
                assassin_revealed = detect_assassin_in_events(
                    [
                        event.card_color
                        for event in parsed_events
                        if isinstance(event, GuessEvent)
                    ]
                )
            previous_log_frame = log_frame

        signals.append(
            GameBoundarySignal(
                timestamp_sec=sample.timestamp_sec,
                board_fingerprint=board_fingerprint,
                setup_visible=setup_visible,
                play_next_visible=play_next_visible,
                left_counter=left_counter,
                right_counter=right_counter,
                assassin_revealed=assassin_revealed,
            )
        )

        if play_next_visible or left_counter == 0 or right_counter == 0 or assassin_revealed:
            current_players = None
            current_board_state = None
            previous_log_frame = None

    return split_game_windows(signals)


def _save_game_log_snapshot(
    frame: object,
    roi_config: ROIConfig,
    snapshot_dir: str | Path,
    stem: str,
) -> str:
    directory = Path(snapshot_dir)
    directory.mkdir(parents=True, exist_ok=True)
    log_crop = crop_roi(frame, roi_config.require("game_log_region"))
    path = directory / f"{stem}.png"
    Image.fromarray(log_crop[:, :, ::-1]).save(path)
    return str(path)
