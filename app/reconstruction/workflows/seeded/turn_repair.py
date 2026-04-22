"""Turn guess merging and sequence stabilization for seeded workflows"""

from __future__ import annotations

from app.parsers.gamelog import ClueEvent, GuessEvent
from app.vision.ocr.preprocessing import collapse_whitespace

from .shared import _direct_clue_guess_target, _guess_color_rank, _guess_result_value, _normalized_word
from .visible_frames import _aggregate_turn_visible_guess_candidates, _collect_turn_visible_log_frames


def _should_keep_non_visible_guess(guess: dict[str, object], *, turn_team_color: str) -> bool:
    word = _normalized_word(guess.get("word"))
    if not word:
        return False

    source = collapse_whitespace(str(guess.get("source") or "")).casefold()
    confidence = float(guess.get("confidence") or 0.0)
    observation_count = int(guess.get("observation_count") or 0)
    if guess.get("board_evidence") or source in {"board", "board_dense", "board_boundary"}:
        return True
    if observation_count >= 2:
        return True
    if confidence >= 0.82:
        return True

    card_color = collapse_whitespace(str(guess.get("card_color") or "")).casefold()
    if _guess_color_rank(turn_team_color, card_color) <= 1:
        return confidence >= 0.6
    return confidence >= 0.55


def _should_adopt_new_visible_guess(guess: dict[str, object], *, turn_team_color: str) -> bool:
    confidence = float(guess.get("confidence") or 0.0)
    observation_count = int(guess.get("observation_count") or 0)
    if confidence >= 0.55:
        return True

    card_color = collapse_whitespace(str(guess.get("card_color") or "")).casefold()
    if observation_count < 2:
        return False
    if _guess_color_rank(turn_team_color, card_color) >= 2:
        return confidence >= 0.4
    return confidence >= 0.5


def _guess_timestamp_sec(guess: dict[str, object]) -> float:
    return float(guess.get("timestamp_sec") or 0.0)


def _guess_word_length(guess: dict[str, object]) -> int:
    return len(_normalized_word(guess.get("word")))


def _guess_is_same_team_correct(guess: dict[str, object], *, turn_team_color: str) -> bool:
    return collapse_whitespace(str(guess.get("card_color") or "")).casefold() == turn_team_color


def _guess_is_stopper(guess: dict[str, object], *, turn_team_color: str) -> bool:
    card_color = collapse_whitespace(str(guess.get("card_color") or "")).casefold()
    return bool(card_color) and card_color != turn_team_color


def _select_raw_stopper_candidate(
    candidates: list[dict[str, object]],
    *,
    turn_team_color: str,
    merged: list[dict[str, object]],
) -> dict[str, object] | None:
    latest_same_team_timestamp = max(
        (
            _guess_timestamp_sec(guess)
            for guess in merged
            if _guess_is_same_team_correct(guess, turn_team_color=turn_team_color)
        ),
        default=-1.0,
    )
    raw_candidates = [
        dict(guess)
        for guess in candidates
        if _guess_is_stopper(guess, turn_team_color=turn_team_color)
        and not guess.get("board_evidence")
        and collapse_whitespace(str(guess.get("source") or "")).casefold()
        not in {"log_rescan", "board", "board_dense", "board_boundary"}
        and _guess_timestamp_sec(guess) >= latest_same_team_timestamp - 0.01
    ]
    if not raw_candidates:
        return None
    raw_candidates.sort(
        key=lambda guess: (
            _guess_timestamp_sec(guess),
            -float(guess.get("confidence") or 0.0),
            _normalized_word(guess.get("word")),
        )
    )
    return raw_candidates[0]


def _insert_guess_in_timestamp_order(
    merged: list[dict[str, object]],
    guess: dict[str, object],
) -> list[dict[str, object]]:
    guess_timestamp = _guess_timestamp_sec(guess)
    insert_index = next(
        (
            index
            for index, existing in enumerate(merged)
            if guess_timestamp < _guess_timestamp_sec(existing) - 0.01
        ),
        None,
    )
    if insert_index is None:
        return [*merged, guess]
    return [*merged[:insert_index], guess, *merged[insert_index:]]


def _prefer_raw_stopper_candidate(
    merged: list[dict[str, object]],
    *,
    raw_stopper_candidate: dict[str, object] | None,
    turn_team_color: str,
) -> list[dict[str, object]]:
    if raw_stopper_candidate is None:
        return merged

    raw_word = _normalized_word(raw_stopper_candidate.get("word"))
    if not raw_word:
        return merged
    if any(_normalized_word(guess.get("word")) == raw_word for guess in merged):
        return merged

    updated = list(merged)
    raw_timestamp = _guess_timestamp_sec(raw_stopper_candidate)
    removable_indexes = [
        index
        for index, guess in enumerate(updated)
        if _guess_is_stopper(guess, turn_team_color=turn_team_color)
        and collapse_whitespace(str(guess.get("source") or "")).casefold() == "log_rescan"
        and not guess.get("board_evidence")
        and float(guess.get("confidence") or 0.0) < 0.6
        and _guess_timestamp_sec(guess) >= raw_timestamp - 0.01
    ]
    if removable_indexes:
        updated.pop(removable_indexes[0])

    return _insert_guess_in_timestamp_order(updated, raw_stopper_candidate)


def _event_history_guess_slices(
    events: list[ClueEvent | GuessEvent],
) -> list[list[GuessEvent]]:
    ordered_events = sorted(events, key=lambda item: (item.timestamp_sec, item.sequence_index))
    guess_slices: list[list[GuessEvent]] = []
    current_guesses: list[GuessEvent] | None = None
    for event in ordered_events:
        if isinstance(event, ClueEvent):
            current_guesses = []
            guess_slices.append(current_guesses)
            continue
        if current_guesses is None:
            continue
        current_guesses.append(event)
    return guess_slices


def _select_event_history_stopper_candidate(
    raw_guesses: list[GuessEvent],
    *,
    turn_team_color: str,
) -> dict[str, object] | None:
    latest_same_team_timestamp = max(
        (
            event.timestamp_sec
            for event in raw_guesses
            if event.card_color.value == turn_team_color
        ),
        default=-1.0,
    )
    stopper_candidates = [
        event
        for event in raw_guesses
        if event.card_color.value != turn_team_color
        and event.timestamp_sec >= latest_same_team_timestamp - 0.01
    ]
    if not stopper_candidates:
        return None
    stopper_candidates.sort(key=lambda event: (event.timestamp_sec, -event.confidence, event.word))
    event = stopper_candidates[0]
    return {
        "word": event.word,
        "card_color": event.card_color.value,
        "result": _guess_result_value(
            turn_team_color=turn_team_color,
            card_color=event.card_color.value,
        ),
        "timestamp_sec": event.timestamp_sec,
        "confidence": event.confidence,
        "source": "event_history",
    }


def _restore_turn_stoppers_from_event_history(
    analysis: dict[str, object],
    event_history: list[ClueEvent | GuessEvent],
) -> None:
    turns = analysis.get("turns", [])
    if not isinstance(turns, list) or not turns:
        return

    guess_slices = _event_history_guess_slices(event_history)
    for turn, raw_guesses in zip(turns, guess_slices):
        if not isinstance(turn, dict):
            continue
        guesses = turn.get("guesses")
        if not isinstance(guesses, list) or not guesses:
            continue
        team_color = collapse_whitespace(str(turn.get("team_color") or "")).casefold()
        raw_stopper_candidate = _select_event_history_stopper_candidate(
            raw_guesses,
            turn_team_color=team_color,
        )
        if raw_stopper_candidate is None:
            continue
        turn["guesses"] = _prefer_raw_stopper_candidate(
            [dict(guess) for guess in guesses if isinstance(guess, dict)],
            raw_stopper_candidate=raw_stopper_candidate,
            turn_team_color=team_color,
        )


def _turn_needs_second_pass(turn: dict[str, object]) -> bool:
    guesses = turn.get("guesses")
    if not isinstance(guesses, list):
        return False
    clue_count = _direct_clue_guess_target(str(turn.get("clue_count") or ""))
    if clue_count is None:
        return False
    team_color = collapse_whitespace(str(turn.get("team_color") or "")).casefold()
    if not team_color:
        return False
    if len(guesses) < clue_count:
        return True
    if not guesses:
        return False
    all_same_team = all(
        _guess_is_same_team_correct(guess, turn_team_color=team_color)
        for guess in guesses
        if isinstance(guess, dict)
    )
    if not all_same_team:
        return False
    return len(guesses) <= clue_count


def _merge_turn_guesses_from_visible_rescan(
    turn: dict[str, object],
    visible_guesses: list[dict[str, object]],
) -> None:
    guesses = turn.get("guesses")
    if not isinstance(guesses, list) or not visible_guesses:
        return

    team_color = collapse_whitespace(str(turn.get("team_color") or "")).casefold()
    current_by_word = {
        _normalized_word(guess.get("word")): dict(guess)
        for guess in guesses
        if isinstance(guess, dict) and _normalized_word(guess.get("word"))
    }

    merged: list[dict[str, object]] = []
    visible_words: set[str] = set()
    for visible_guess in visible_guesses:
        word_key = _normalized_word(visible_guess.get("word"))
        if not word_key:
            continue
        visible_words.add(word_key)
        current_guess = current_by_word.pop(word_key, None)
        if current_guess is None:
            if not _should_adopt_new_visible_guess(visible_guess, turn_team_color=team_color):
                continue
            merged_guess = dict(visible_guess)
            merged_guess["result"] = _guess_result_value(
                turn_team_color=team_color,
                card_color=str(merged_guess.get("card_color") or ""),
                fallback=str(merged_guess.get("result") or ""),
            )
            merged.append(merged_guess)
            continue

        merged_guess = dict(current_guess)
        visible_color = str(visible_guess.get("card_color") or "")
        current_color = str(current_guess.get("card_color") or "")
        chosen_color = visible_color
        if current_color and _guess_color_rank(team_color, current_color) > _guess_color_rank(team_color, visible_color):
            chosen_color = current_color
        merged_guess.update(visible_guess)
        merged_guess["word"] = word_key
        merged_guess["timestamp_sec"] = round(
            min(
                float(current_guess.get("timestamp_sec") or visible_guess.get("timestamp_sec") or 0.0),
                float(visible_guess.get("timestamp_sec") or 0.0),
            ),
            3,
        )
        merged_guess["confidence"] = max(
            float(current_guess.get("confidence") or 0.0),
            float(visible_guess.get("confidence") or 0.0),
        )
        merged_guess["observation_count"] = max(
            int(current_guess.get("observation_count") or 0),
            int(visible_guess.get("observation_count") or 0),
        )
        merged_guess["source"] = "log_rescan"
        merged_guess["card_color"] = chosen_color or current_color
        merged_guess["result"] = _guess_result_value(
            turn_team_color=team_color,
            card_color=str(merged_guess.get("card_color") or ""),
            fallback=str(current_guess.get("result") or ""),
        )
        merged.append(merged_guess)

    raw_stopper_candidate = _select_raw_stopper_candidate(
        list(current_by_word.values()),
        turn_team_color=team_color,
        merged=merged,
    )
    extras = [
        dict(guess)
        for word_key, guess in current_by_word.items()
        if word_key not in visible_words and _should_keep_non_visible_guess(guess, turn_team_color=team_color)
    ]
    extras.sort(
        key=lambda item: (
            float(item.get("timestamp_sec") or 0.0),
            -float(item.get("confidence") or 0.0),
            _normalized_word(item.get("word")),
        )
    )
    board_backed_extras: list[dict[str, object]] = []
    trailing_extras: list[dict[str, object]] = []
    for extra_guess in extras:
        source = collapse_whitespace(str(extra_guess.get("source") or "")).casefold()
        if extra_guess.get("board_evidence") or source in {"board", "board_dense", "board_boundary"}:
            board_backed_extras.append(extra_guess)
            continue
        trailing_extras.append(extra_guess)

    for extra_guess in board_backed_extras:
        extra_timestamp = float(extra_guess.get("timestamp_sec") or 0.0)
        insert_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if extra_timestamp < float(existing.get("timestamp_sec") or 0.0) - 0.01
            ),
            None,
        )
        if insert_index is None:
            merged.append(extra_guess)
            continue
        merged.insert(insert_index, extra_guess)

    allow_leading_non_visible_margin_sec = 1.5
    for extra_guess in trailing_extras:
        if not merged:
            merged.append(extra_guess)
            continue
        extra_timestamp = float(extra_guess.get("timestamp_sec") or 0.0)
        first_visible_timestamp = float(merged[0].get("timestamp_sec") or 0.0)
        if extra_timestamp <= first_visible_timestamp - allow_leading_non_visible_margin_sec:
            insert_index = next(
                (
                    index
                    for index, existing in enumerate(merged)
                    if extra_timestamp < float(existing.get("timestamp_sec") or 0.0) - 0.01
                ),
                None,
            )
        else:
            insert_index = next(
                (
                    index
                    for index, existing in enumerate(merged[1:], start=1)
                    if extra_timestamp < float(existing.get("timestamp_sec") or 0.0) - 0.01
                ),
                None,
            )
        if insert_index is None:
            merged.append(extra_guess)
            continue
        merged.insert(insert_index, extra_guess)

    merged = _prefer_raw_stopper_candidate(
        merged,
        raw_stopper_candidate=raw_stopper_candidate,
        turn_team_color=team_color,
    )
    for guess in merged:
        guess.pop("_visible_order", None)
        guess.pop("_visible_position", None)
    turn["guesses"] = merged


def _refine_turn_guesses_with_visible_log_rescan(
    *,
    analysis: dict[str, object],
    vod_source,
    vod_url: str,
    roi_config,
    ocr_backend,
    roster,
    board_state,
    fps: float = 1.0,
    pre_roll_sec: float = 4.0,
    turn_indexes: set[int] | None = None,
) -> None:
    turns = analysis.get("turns", [])
    if not isinstance(turns, list) or not turns:
        return

    clip_start_sec = float(analysis.get("start_sec") or 0.0)
    clip_end_sec = float(analysis.get("stop_sec") or 0.0)
    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        if turn_indexes is not None and turn_index not in turn_indexes:
            continue
        turn_start_sec = float(turn.get("timestamp_sec") or 0.0)
        next_turn_sec = (
            float(turns[turn_index + 1].get("timestamp_sec") or clip_end_sec)
            if turn_index + 1 < len(turns)
            else clip_end_sec
        )
        search_start_sec = max(clip_start_sec, turn_start_sec - pre_roll_sec)
        search_end_sec = max(turn_start_sec + 0.5, next_turn_sec)
        visible_frames = _collect_turn_visible_log_frames(
            vod_source=vod_source,
            vod_url=vod_url,
            start_sec=search_start_sec,
            end_sec=search_end_sec,
            fps=fps,
            roi_config=roi_config,
            ocr_backend=ocr_backend,
            roster=roster,
            board_state=board_state,
            turn=turn,
        )
        visible_guesses = _aggregate_turn_visible_guess_candidates(visible_frames, turn)
        if not visible_guesses:
            continue
        _merge_turn_guesses_from_visible_rescan(turn, visible_guesses)


def _stabilize_turn_guess_sequences(analysis: dict[str, object]) -> None:
    turns = analysis.get("turns", [])
    if not isinstance(turns, list):
        return

    previous_turn_words: set[str] = set()
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        guesses = turn.get("guesses")
        if not isinstance(guesses, list) or not guesses:
            previous_turn_words = set()
            continue

        team_color = str(turn.get("team_color") or "")
        unique_words = {
            collapse_whitespace(str(guess.get("word") or "")).upper()
            for guess in guesses
            if isinstance(guess, dict)
        }
        if len(unique_words) > 1:
            deduped: list[dict[str, object]] = []
            seen_by_word: dict[str, int] = {}
            for guess in guesses:
                if not isinstance(guess, dict):
                    continue
                word = collapse_whitespace(str(guess.get("word") or "")).upper()
                if not word:
                    deduped.append(guess)
                    continue
                existing_index = seen_by_word.get(word)
                if existing_index is None:
                    seen_by_word[word] = len(deduped)
                    deduped.append(guess)
                    continue
                existing_guess = deduped[existing_index]
                if (
                    float(guess.get("confidence") or 0.0) > float(existing_guess.get("confidence") or 0.0) + 0.08
                    and _guess_color_rank(team_color, str(guess.get("card_color") or ""))
                    >= _guess_color_rank(team_color, str(existing_guess.get("card_color") or ""))
                ):
                    replacement = dict(existing_guess)
                    replacement.update(guess)
                    replacement["timestamp_sec"] = min(
                        float(existing_guess.get("timestamp_sec") or 0.0),
                        float(guess.get("timestamp_sec") or 0.0),
                    )
                    deduped[existing_index] = replacement
            guesses = deduped

        if len(guesses) >= 3 and previous_turn_words:
            first_guess = guesses[0]
            first_word = collapse_whitespace(str(first_guess.get("word") or "")).upper()
            turn_timestamp_sec = float(turn.get("timestamp_sec") or 0.0)
            first_timestamp_sec = float(first_guess.get("timestamp_sec") or 0.0)
            if (
                first_word
                and first_word in previous_turn_words
                and abs(first_timestamp_sec - turn_timestamp_sec) <= 4.0
            ):
                other_confidences = [
                    float(guess.get("confidence") or 0.0)
                    for guess in guesses[1:]
                    if isinstance(guess, dict)
                ]
                if other_confidences and max(other_confidences) >= (float(first_guess.get("confidence") or 0.0) + 0.05):
                    guesses = guesses[1:]

        max_guesses = _direct_clue_guess_target(str(turn.get("clue_count") or ""))
        if max_guesses is not None:
            max_guesses += 1
        if max_guesses is not None and len(guesses) > max_guesses:
            protected_indices = {0}
            while len(guesses) > max_guesses:
                removable_indexes = [
                    index
                    for index in range(len(guesses))
                    if index not in protected_indices
                ] or list(range(len(guesses)))
                drop_index = min(
                    removable_indexes,
                    key=lambda index: (
                        float(guesses[index].get("confidence") or 0.0),
                        float(guesses[index].get("timestamp_sec") or 0.0),
                    ),
                )
                guesses.pop(drop_index)
                protected_indices = {index for index in protected_indices if index < len(guesses)}

        direct_guess_target = _direct_clue_guess_target(str(turn.get("clue_count") or ""))
        if direct_guess_target is not None:
            same_team_indexes = [
                index
                for index, guess in enumerate(guesses)
                if collapse_whitespace(str(guess.get("card_color") or "")).casefold() == team_color
            ]
            while len(same_team_indexes) > direct_guess_target:
                removable_same_team_indexes = [
                    index
                    for index in same_team_indexes
                    if not guesses[index].get("board_evidence")
                    and collapse_whitespace(str(guesses[index].get("source") or "")).casefold()
                    not in {"board", "board_dense", "board_boundary"}
                    and float(guesses[index].get("confidence") or 0.0) < 0.4
                ]
                if not removable_same_team_indexes:
                    break
                drop_index = min(
                    removable_same_team_indexes,
                    key=lambda index: (
                        float(guesses[index].get("confidence") or 0.0),
                        float(guesses[index].get("timestamp_sec") or 0.0),
                    ),
                )
                guesses.pop(drop_index)
                same_team_indexes = [
                    index
                    for index, guess in enumerate(guesses)
                    if collapse_whitespace(str(guess.get("card_color") or "")).casefold() == team_color
                ]

        if len(guesses) >= 3:
            stopper_indexes = [
                index
                for index, guess in enumerate(guesses)
                if _guess_is_stopper(guess, turn_team_color=team_color)
            ]
            if len(stopper_indexes) == 1:
                stopper_index = stopper_indexes[0]
                stopper_guess = guesses[stopper_index]
                if any(
                    _guess_is_same_team_correct(guess, turn_team_color=team_color)
                    for guess in guesses[stopper_index + 1 :]
                ):
                    guesses = [guess for index, guess in enumerate(guesses) if index != stopper_index]
                    guesses.append(stopper_guess)

            stabilized_guesses = list(guesses)
            index = 0
            timestamp_tie_tolerance_sec = 0.01
            while index < len(stabilized_guesses):
                guess = stabilized_guesses[index]
                if not _guess_is_same_team_correct(guess, turn_team_color=team_color):
                    index += 1
                    continue
                group_end = index + 1
                while group_end < len(stabilized_guesses):
                    candidate = stabilized_guesses[group_end]
                    if not _guess_is_same_team_correct(candidate, turn_team_color=team_color):
                        break
                    if abs(_guess_timestamp_sec(candidate) - _guess_timestamp_sec(guess)) > timestamp_tie_tolerance_sec:
                        break
                    group_end += 1
                if group_end - index > 1:
                    stabilized_guesses[index:group_end] = sorted(
                        stabilized_guesses[index:group_end],
                        key=lambda item: (
                            -int(item.get("observation_count") or 0),
                            -_guess_word_length(item),
                            -float(item.get("confidence") or 0.0),
                            _normalized_word(item.get("word")),
                        ),
                    )
                index = group_end
            guesses = stabilized_guesses

        turn["guesses"] = guesses
        previous_turn_words = {
            collapse_whitespace(str(guess.get("word") or "")).upper()
            for guess in guesses
            if isinstance(guess, dict) and guess.get("word")
        }
