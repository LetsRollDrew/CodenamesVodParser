"""Shared helpers for seeded reconstruction workflows"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.models import BoardCell, BoardState, CardColor, ImageBoundingBox, PlayerRosterEntry, TeamColor
from app.infra.roi_config import crop_roi
from app.parsers.board import estimate_board_cell_boxes
from app.parsers.gamelog import ClueEvent, GuessEvent
from app.reconstruction.game_reconstruction import classify_guess_result
from app.vision.ocr.preprocessing import collapse_whitespace


def _parse_time_offset(value: str) -> float:
    stripped = value.strip()
    if ":" not in stripped:
        return float(stripped)
    hours, minutes, seconds = stripped.split(":")
    return (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)


def _normalize_device_is_gpu(device: str | None) -> bool:
    if device is None:
        return False
    return device.strip().lower().startswith("gpu")


def _load_seed(seed_path: Path) -> tuple[list[PlayerRosterEntry], list[str]]:
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    roster = [
        PlayerRosterEntry(
            player_name=collapse_whitespace(str(item.get("player_name") or item["display_name"])).casefold(),
            display_name=str(item["display_name"]),
            team_color=item["team_color"],
            role=item["role"],
            avatar_hash=None,
        )
        for item in payload["roster"]
    ]
    board_words = [str(word).strip().upper() for word in payload["board_words"]]
    if len(board_words) != 25:
        raise ValueError("seed board_words must contain exactly 25 entries")
    return roster, board_words


def _build_seeded_board_state(frame, roi_config, board_words: list[str]) -> BoardState:
    board_frame = crop_roi(frame, roi_config.require("board_region"))
    boxes = estimate_board_cell_boxes(board_frame)
    cells = [
        BoardCell(
            row=index // 5,
            col=index % 5,
            word=board_words[index],
            confidence=1.0,
            box=box,
        )
        for index, box in enumerate(boxes)
    ]
    return BoardState(cells=cells)


def _clue_text_key(text: str) -> str:
    return "".join(character for character in collapse_whitespace(text).upper() if character.isalpha())


def _normalized_word(value: object) -> str:
    return collapse_whitespace(str(value or "")).upper()


def _clue_guess_limit(clue_count: str) -> int | None:
    normalized = str(clue_count or "").casefold()
    if normalized == "infinity":
        return None
    if normalized.isdigit():
        return int(normalized) + 1
    return None


def _direct_clue_guess_target(clue_count: str) -> int | None:
    normalized = str(clue_count or "").casefold()
    if normalized == "infinity":
        return None
    if normalized.isdigit():
        return int(normalized)
    return None


def _guess_color_rank(team_color: str, card_color: str) -> int:
    if card_color == team_color:
        return 3
    if card_color == "neutral":
        return 2
    if card_color == "black":
        return 0
    return 1


def _reveal_color_strength(card_color: CardColor | None) -> int:
    if card_color is None:
        return 0
    if card_color in {CardColor.RED, CardColor.BLUE, CardColor.BLACK}:
        return 3
    if card_color is CardColor.NEUTRAL:
        return 1
    return 0


def _guess_result_value(
    *,
    turn_team_color: str,
    card_color: str,
    fallback: str | None = None,
) -> str | None:
    normalized_team = collapse_whitespace(turn_team_color).casefold()
    normalized_color = collapse_whitespace(card_color).casefold()
    if normalized_team not in {"red", "blue"}:
        return fallback
    try:
        return classify_guess_result(TeamColor(normalized_team), CardColor(normalized_color)).value
    except ValueError:
        return fallback


def _clue_matches_turn(event: ClueEvent, turn: dict[str, object]) -> bool:
    if _clue_text_key(event.clue_text) != _clue_text_key(str(turn.get("clue_text") or "")):
        return False
    turn_team_color = collapse_whitespace(str(turn.get("team_color") or "")).casefold()
    turn_spymaster_name = collapse_whitespace(str(turn.get("spymaster_name") or "")).casefold()
    if not turn_team_color and not turn_spymaster_name:
        return True
    if turn_team_color and event.team_color.value == turn_team_color:
        return True
    return bool(turn_spymaster_name) and collapse_whitespace(event.spymaster_name).casefold() == turn_spymaster_name


def _visible_guesses_for_turn(
    events: list[ClueEvent | GuessEvent],
    turn: dict[str, object],
) -> list[GuessEvent]:
    active = False
    guesses: list[GuessEvent] = []
    for event in events:
        if isinstance(event, ClueEvent):
            if active:
                break
            active = _clue_matches_turn(event, turn)
            continue
        if active:
            guesses.append(event)
    return guesses


def _apply_board_guess_update(
    guess: dict[str, object],
    event: GuessEvent,
    *,
    turn_team_color: str,
) -> None:
    existing_card_color = str(guess.get("card_color") or "")
    candidate_card_color = event.card_color.value
    if event.timestamp_sec < float(guess.get("timestamp_sec") or 0.0):
        guess["timestamp_sec"] = event.timestamp_sec
    if _guess_color_rank(turn_team_color, candidate_card_color) > _guess_color_rank(turn_team_color, existing_card_color):
        guess["card_color"] = candidate_card_color
        guess["result"] = classify_guess_result(
            TeamColor(turn_team_color),
            event.card_color,
        ).value
