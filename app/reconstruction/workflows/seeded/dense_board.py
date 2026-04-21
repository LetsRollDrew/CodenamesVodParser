"""Analysis backfill helpers for seeded reconstruction workflows"""

from __future__ import annotations

from app.core.models import BoardState, CardColor, TeamColor
from app.infra.roi_config import crop_roi
from app.infra.vod_source import VodSource
from app.parsers.gamelog import GuessEvent
from app.runtime.smoke.board_reveals import _detect_revealed_card_color, _sample_board_cell_surface

from .shared import (
    _apply_board_guess_update,
    _clue_guess_limit,
    _guess_result_value,
    _reveal_color_strength,
)


def _augment_analysis_with_dense_board_rescans(
    *,
    analysis: dict[str, object],
    vod_source: VodSource,
    vod_url: str,
    roi_config,
    board_state: BoardState,
    board_reveal_baseline,
    dense_fps: float = 4.0,
    minimum_confirmations: int = 2,
    turn_indexes: set[int] | None = None,
) -> None:
    same_turn_cluster_slack_sec = 2.5
    infinity_replacement_window_sec = 4.0
    turns = analysis.get("turns", [])
    if not isinstance(turns, list) or not turns:
        return

    previous_guessed_words: set[str] = set()
    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        turn_team_color = str(turn.get("team_color") or "")
        current_guesses = turn.setdefault("guesses", [])
        if not isinstance(current_guesses, list):
            continue
        current_guess_by_word = {
            str(guess.get("word")): guess
            for guess in current_guesses
            if isinstance(guess, dict) and guess.get("word")
        }
        if turn_indexes is not None and turn_index not in turn_indexes:
            previous_guessed_words.update(current_guess_by_word)
            previous_guessed_words.update(
                str(guess.get("word"))
                for guess in current_guesses
                if isinstance(guess, dict) and guess.get("word")
            )
            continue
        next_turn_sec = (
            float(turns[turn_index + 1].get("timestamp_sec") or analysis.get("stop_sec") or 0.0)
            if turn_index + 1 < len(turns)
            else float(analysis.get("stop_sec") or turn.get("timestamp_sec") or 0.0)
        )
        turn_start_sec = float(turn.get("timestamp_sec") or 0.0)
        window_duration = max(0.0, next_turn_sec - turn_start_sec)
        if window_duration < 1.0:
            previous_guessed_words.update(current_guess_by_word)
            continue

        dense_events = _scan_dense_board_reveal_events(
            vod_source=vod_source,
            vod_url=vod_url,
            start_sec=turn_start_sec,
            duration_sec=window_duration,
            dense_fps=dense_fps,
            minimum_confirmations=minimum_confirmations,
            roi_config=roi_config,
            board_state=board_state,
            board_reveal_baseline=board_reveal_baseline,
            ignored_words=previous_guessed_words,
        )
        candidate_by_word: dict[str, GuessEvent] = {}
        same_team_guesses = [
            guess
            for guess in current_guesses
            if isinstance(guess, dict) and str(guess.get("card_color") or "") == turn_team_color
        ]
        latest_same_team_guess_ts = max(
            (float(guess.get("timestamp_sec") or 0.0) for guess in same_team_guesses),
            default=turn_start_sec,
        )
        weak_trailing_guess: dict[str, object] | None = None
        if same_team_guesses:
            latest_same_team_guess = max(
                same_team_guesses,
                key=lambda guess: float(guess.get("timestamp_sec") or 0.0),
            )
            if (
                str(latest_same_team_guess.get("player_name") or "").strip().casefold() in {"", "unknown"}
                and not latest_same_team_guess.get("selector_crop_path")
                and not latest_same_team_guess.get("selector_crop_x4_path")
                and not latest_same_team_guess.get("selector_candidates")
                and str(latest_same_team_guess.get("selector_source") or "") not in {"selector_crop", "guess_player"}
            ):
                weak_trailing_guess = latest_same_team_guess
        anchor_same_team_guess_ts = latest_same_team_guess_ts
        if weak_trailing_guess is not None:
            stable_same_team_guess_timestamps = [
                float(guess.get("timestamp_sec") or 0.0)
                for guess in same_team_guesses
                if guess is not weak_trailing_guess
            ]
            if stable_same_team_guess_timestamps:
                anchor_same_team_guess_ts = max(stable_same_team_guess_timestamps)
        for event in dense_events:
            if event.word in current_guess_by_word:
                if event.card_color.value != turn_team_color:
                    continue
                _apply_board_guess_update(
                    current_guess_by_word[event.word],
                    event,
                    turn_team_color=turn_team_color,
                )
                continue
            if event.card_color.value != turn_team_color:
                continue
            if event.timestamp_sec < (anchor_same_team_guess_ts - same_turn_cluster_slack_sec):
                continue
            existing = candidate_by_word.get(event.word)
            if existing is None or event.timestamp_sec < existing.timestamp_sec:
                candidate_by_word[event.word] = event

        clue_guess_limit = _clue_guess_limit(str(turn.get("clue_count") or ""))
        remaining_slots = None if clue_guess_limit is None else max(0, clue_guess_limit - len(current_guesses))
        ordered_candidates = sorted(candidate_by_word.values(), key=lambda item: item.timestamp_sec)
        if clue_guess_limit is None and ordered_candidates:
            close_candidates = [
                item
                for item in ordered_candidates
                if abs(item.timestamp_sec - anchor_same_team_guess_ts) <= 6.0
            ]
            if close_candidates:
                ordered_candidates = sorted(
                    close_candidates,
                    key=lambda item: (abs(item.timestamp_sec - anchor_same_team_guess_ts), item.timestamp_sec),
                )[:1]
        if clue_guess_limit is None and weak_trailing_guess is not None and ordered_candidates:
            replacement_event = ordered_candidates[0]
            weak_trailing_ts = float(weak_trailing_guess.get("timestamp_sec") or 0.0)
            if (
                replacement_event.timestamp_sec < weak_trailing_ts
                and (weak_trailing_ts - replacement_event.timestamp_sec) <= infinity_replacement_window_sec
            ):
                old_word = str(weak_trailing_guess.get("word") or "")
                weak_trailing_guess["word"] = replacement_event.word
                weak_trailing_guess["card_color"] = replacement_event.card_color.value
                weak_trailing_guess["result"] = _guess_result_value(
                    turn_team_color=turn_team_color,
                    card_color=replacement_event.card_color.value,
                )
                weak_trailing_guess["timestamp_sec"] = replacement_event.timestamp_sec
                weak_trailing_guess["confidence"] = replacement_event.confidence
                weak_trailing_guess["player_name"] = replacement_event.player_name
                weak_trailing_guess["source"] = "board_dense"
                weak_trailing_guess["observation_count"] = 0
                weak_trailing_guess["board_evidence"] = True
                weak_trailing_guess["board_color"] = replacement_event.card_color.value
                for field_name in (
                    "selector_crop_path",
                    "selector_crop_x4_path",
                    "selector_candidates",
                    "resolved_selector_candidates",
                    "selector_name",
                    "selector_source",
                    "selector_confidence",
                    "selector_ocr_text",
                ):
                    weak_trailing_guess.pop(field_name, None)
                if old_word:
                    current_guess_by_word.pop(old_word, None)
                current_guess_by_word[replacement_event.word] = weak_trailing_guess
                ordered_candidates = ordered_candidates[1:]
        additions = 0
        for event in ordered_candidates:
            if remaining_slots is not None and additions >= remaining_slots:
                break
            current_guesses.append(
                {
                    "player_name": event.player_name,
                    "word": event.word,
                    "card_color": event.card_color.value,
                    "result": _guess_result_value(
                        turn_team_color=turn_team_color,
                        card_color=event.card_color.value,
                    ),
                    "timestamp_sec": event.timestamp_sec,
                    "confidence": event.confidence,
                    "source": "board_dense",
                    "observation_count": 0,
                    "board_evidence": True,
                    "board_color": event.card_color.value,
                }
            )
            current_guess_by_word[event.word] = current_guesses[-1]
            additions += 1

        previous_guessed_words.update(current_guess_by_word)
        previous_guessed_words.update(
            str(guess.get("word"))
            for guess in current_guesses
            if isinstance(guess, dict) and guess.get("word")
        )


def _scan_dense_board_reveal_events(
    *,
    vod_source: VodSource,
    vod_url: str,
    start_sec: float,
    duration_sec: float,
    dense_fps: float,
    minimum_confirmations: int,
    roi_config,
    board_state: BoardState,
    board_reveal_baseline,
    ignored_words: set[str],
) -> list[GuessEvent]:
    if duration_sec <= 0:
        return []

    board_region = roi_config.require("board_region")
    tracked_cells = [
        cell
        for cell in board_state.cells
        if cell.word not in ignored_words
    ]
    if not tracked_cells:
        return []

    streaks: dict[tuple[int, int], tuple[CardColor | None, int, float]] = {}
    emitted: dict[tuple[int, int], GuessEvent] = {}
    for sample in vod_source.iter_window_frames(vod_url, start_sec, duration_sec, dense_fps):
        board_frame = crop_roi(sample.frame_bgr, board_region)
        for cell in tracked_cells:
            key = (cell.row, cell.col)
            baseline_stats = board_reveal_baseline.get(key)
            if baseline_stats is None:
                continue
            surface = _sample_board_cell_surface(board_frame, cell.box)
            current_color = _detect_revealed_card_color(surface, baseline_stats)
            previous_color, previous_count, first_seen_ts = streaks.get(key, (None, 0, sample.timestamp_sec))
            if current_color is None:
                streaks[key] = (None, 0, sample.timestamp_sec)
                continue
            if current_color == previous_color:
                current_count = previous_count + 1
                current_first_seen_ts = first_seen_ts
            else:
                current_count = 1
                current_first_seen_ts = sample.timestamp_sec
            streaks[key] = (current_color, current_count, current_first_seen_ts)
            if current_count < minimum_confirmations:
                continue

            existing_event = emitted.get(key)
            candidate_event = GuessEvent(
                player_name="unknown",
                word=cell.word,
                card_color=current_color,
                timestamp_sec=current_first_seen_ts,
                sequence_index=(cell.row * 5) + cell.col,
                confidence=0.86,
            )
            if existing_event is None:
                emitted[key] = candidate_event
                continue
            existing_strength = _reveal_color_strength(existing_event.card_color)
            candidate_strength = _reveal_color_strength(candidate_event.card_color)
            if candidate_strength > existing_strength:
                emitted[key] = candidate_event
                continue
            if (
                candidate_strength == existing_strength
                and candidate_event.card_color == existing_event.card_color
                and candidate_event.timestamp_sec < existing_event.timestamp_sec
            ):
                emitted[key] = candidate_event
    return sorted(emitted.values(), key=lambda event: event.timestamp_sec)


def _refresh_analysis_winner_fields(analysis: dict[str, object]) -> None:
    final_counters = analysis.get("final_counters", {})
    if isinstance(final_counters, dict):
        left_counter = final_counters.get("left")
        right_counter = final_counters.get("right")
        if left_counter == 0 and right_counter is not None:
            analysis["winner_team"] = "blue"
            analysis["win_reason"] = "counter_zero"
            return
        if right_counter == 0 and left_counter is not None:
            analysis["winner_team"] = "red"
            analysis["win_reason"] = "counter_zero"
            return

    turns = analysis.get("turns", [])
    if not isinstance(turns, list) or not turns:
        return
    starting_team = str(turns[0].get("team_color") or "").strip().casefold()
    if starting_team not in {"red", "blue"}:
        return
    opposing_team = "red" if starting_team == "blue" else "blue"
    target_counts = {
        starting_team: 9,
        opposing_team: 8,
    }
    correct_words_by_team: dict[str, set[str]] = {"blue": set(), "red": set()}
    for turn in turns:
        team_color = str(turn.get("team_color") or "").strip().casefold()
        if team_color not in correct_words_by_team:
            continue
        for guess in turn.get("guesses", []):
            if not isinstance(guess, dict):
                continue
            if str(guess.get("result") or "").strip().casefold() != "correct":
                continue
            if str(guess.get("card_color") or "").strip().casefold() != team_color:
                continue
            word = str(guess.get("word") or "").strip().upper()
            if word:
                correct_words_by_team[team_color].add(word)
    reached_targets = [
        team_color
        for team_color, target_count in target_counts.items()
        if len(correct_words_by_team[team_color]) >= target_count
    ]
    if len(reached_targets) == 1:
        analysis["winner_team"] = reached_targets[0]
        analysis["win_reason"] = "counter_zero"
