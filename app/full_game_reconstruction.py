"""Automatic full-game reconstruction for one Codenames clip or VOD window."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.board_parser import parse_board_words
from app.frame_detectors import parse_game_counters, parse_top_banner_text
from app.gamelog_parser import ClueEvent, GuessEvent, log_has_changed, parse_game_log
from app.models import BoardState, CardColor, PlayerRosterEntry, TeamColor
from app.ocr_backends import create_ocr_backend
from app.roi_config import crop_roi, load_roi_config
from app.roster_parser import parse_rosters
from app.selector_attribution import attribute_selectors_from_analysis
from app.smoke import (
    _capture_board_reveal_baseline,
    _detect_revealed_card_color,
    _infer_active_guessing_team,
    _materialize_board_observations,
    _materialize_roster_observations,
    _observe_board_state,
    _observe_roster,
    _parse_center_clue_banner,
    _sample_board_cell_surface,
)
from app.vod_source import VodSource


@dataclass(frozen=True, slots=True)
class ClueObservation:
    timestamp_sec: float
    clue_text: str
    clue_count: str
    top_banner_text: str
    left_counter: int | None
    right_counter: int | None


@dataclass(frozen=True, slots=True)
class VisibleEventFrame:
    timestamp_sec: float
    events: list[ClueEvent | GuessEvent]


@dataclass(frozen=True, slots=True)
class BoardReveal:
    timestamp_sec: float
    word: str
    card_color: CardColor


@dataclass(frozen=True, slots=True)
class CounterSample:
    timestamp_sec: float
    left_counter: int | None
    right_counter: int | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-url", required=True)
    parser.add_argument("--start", default="0")
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--roi-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ocr-backend", default="easyocr")
    parser.add_argument("--ocr-device", default="gpu:0")
    parser.add_argument("--ffmpeg-path", default="ffmpeg")
    parser.add_argument("--scan-fps", type=float, default=1.0)
    parser.add_argument("--selector-window-sec", type=float, default=1.5)
    parser.add_argument("--selector-fps", type=float, default=4.0)
    parser.add_argument("--process-timeout", type=float, default=180.0)
    return parser.parse_args()


def _normalize_device_is_gpu(device: str | None) -> bool:
    if device is None:
        return False
    return device.strip().lower().startswith("gpu")


def _parse_time_offset(value: str) -> float:
    stripped = value.strip()
    if ":" not in stripped:
        return float(stripped)
    hours, minutes, seconds = stripped.split(":")
    return (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)


def _clue_text_key(text: str) -> str:
    return "".join(character for character in text.upper() if character.isalpha())


def _mode(values: list[str]) -> str | None:
    if not values:
        return None
    counter = Counter(values)
    return max(counter.items(), key=lambda item: (item[1], len(item[0])))[0]


def _infer_starting_team(observations: list[ClueObservation]) -> TeamColor | None:
    for observation in observations:
        if observation.left_counter is None or observation.right_counter is None:
            continue
        if observation.left_counter == observation.right_counter:
            continue
        return TeamColor.BLUE if observation.left_counter > observation.right_counter else TeamColor.RED
    return None


def _infer_turn_team(observations: list[ClueObservation], roster: list[PlayerRosterEntry]) -> TeamColor | None:
    votes: Counter[TeamColor] = Counter()
    roster_by_name = {player.display_name.upper(): player for player in roster}
    for observation in observations:
        inferred = _infer_active_guessing_team(observation.top_banner_text, roster)
        if inferred is not None:
            votes[inferred] += 2
        normalized = observation.top_banner_text.upper()
        if "GIVING A CLUE" in normalized:
            for name, player in roster_by_name.items():
                if player.role.value != "spymaster":
                    continue
                if name and name in normalized:
                    votes[player.team_color] += 3
        if "GUESSING" in normalized:
            for name, player in roster_by_name.items():
                if player.role.value != "operative":
                    continue
                if name and name in normalized:
                    votes[player.team_color] += 1
    if not votes:
        return None
    return max(votes.items(), key=lambda item: item[1])[0]


def _build_turn_schedule(
    observations: list[ClueObservation],
    roster: list[PlayerRosterEntry],
    *,
    clip_end_sec: float,
    visible_frames: list[VisibleEventFrame],
) -> list[dict[str, Any]]:
    filtered = [
        observation
        for observation in observations
        if observation.clue_text
        and observation.clue_count
        and observation.clue_count.isdigit()
        and len(observation.clue_count) == 1
        and observation.clue_count != "0"
    ]
    if not filtered:
        return []

    segments: list[list[ClueObservation]] = []
    current: list[ClueObservation] = []
    current_key: str | None = None
    for observation in filtered:
        observation_key = _clue_text_key(observation.clue_text)
        if not observation_key:
            continue
        if current_key is None or observation_key == current_key:
            current.append(observation)
            current_key = observation_key
            continue
        segments.append(current)
        current = [observation]
        current_key = observation_key
    if current:
        segments.append(current)

    merged_segments: list[list[ClueObservation]] = []
    for segment in segments:
        if (
            merged_segments
            and _clue_text_key(merged_segments[-1][0].clue_text) == _clue_text_key(segment[0].clue_text)
            and segment[0].timestamp_sec - merged_segments[-1][-1].timestamp_sec <= 8.0
        ):
            merged_segments[-1].extend(segment)
            continue
        merged_segments.append(segment)

    raw_segments: list[dict[str, Any]] = []
    for turn_index, segment in enumerate(merged_segments):
        chosen_text = _mode([_clue_text_key(item.clue_text) for item in segment]) or _clue_text_key(segment[0].clue_text)
        chosen_count = _mode([item.clue_count for item in segment if item.clue_count.isdigit()]) or segment[0].clue_count
        clue_timestamp_sec = next(
            (
                item.timestamp_sec
                for item in segment
                if _clue_text_key(item.clue_text) == chosen_text and item.clue_count == chosen_count
            ),
            segment[0].timestamp_sec,
        )
        next_segment_start = clip_end_sec if turn_index + 1 >= len(merged_segments) else merged_segments[turn_index + 1][0].timestamp_sec
        clue_matches = [
            event
            for record in visible_frames
            if clue_timestamp_sec <= record.timestamp_sec <= next_segment_start
            for event in record.events
            if isinstance(event, ClueEvent) and _clue_matches_turn(event, {"clue_text": chosen_text, "team_color": "", "spymaster_name": ""})
        ]
        visible_guess_count = sum(
            len(_visible_guesses_for_turn(record.events, {"clue_text": chosen_text, "team_color": "", "spymaster_name": ""}))
            for record in visible_frames
            if clue_timestamp_sec <= record.timestamp_sec <= next_segment_start
        )
        raw_segments.append(
            {
                "segment": segment,
                "clue_text": chosen_text,
                "clue_count": chosen_count,
                "clue_timestamp_sec": clue_timestamp_sec,
                "next_segment_start": next_segment_start,
                "clue_matches": clue_matches,
                "visible_guess_count": visible_guess_count,
                "duration_sec": segment[-1].timestamp_sec - segment[0].timestamp_sec,
            }
        )

    filtered_segments: list[dict[str, Any]] = []
    for candidate in raw_segments:
        support = len(candidate["segment"])
        has_clue_matches = bool(candidate["clue_matches"])
        has_visible_guesses = candidate["visible_guess_count"] > 0
        if not has_clue_matches and not has_visible_guesses and (support == 1 or candidate["duration_sec"] < 10.0):
            continue
        filtered_segments.append(candidate)

    coalesced_segments: list[dict[str, Any]] = []
    for candidate in filtered_segments:
        if (
            coalesced_segments
            and candidate["clue_text"] == coalesced_segments[-1]["clue_text"]
            and candidate["clue_count"] == coalesced_segments[-1]["clue_count"]
        ):
            coalesced_segments[-1]["segment"].extend(candidate["segment"])
            coalesced_segments[-1]["clue_matches"].extend(candidate["clue_matches"])
            coalesced_segments[-1]["visible_guess_count"] += candidate["visible_guess_count"]
            coalesced_segments[-1]["next_segment_start"] = candidate["next_segment_start"]
            continue
        coalesced_segments.append(candidate)

    starting_team = _infer_starting_team(observations)
    if starting_team is None and coalesced_segments:
        starting_team = _infer_turn_team(coalesced_segments[0]["segment"], roster)
    if starting_team is None:
        starting_team = TeamColor.BLUE

    turns: list[dict[str, Any]] = []
    for turn_index, candidate in enumerate(coalesced_segments):
        if starting_team is TeamColor.BLUE:
            team_color = TeamColor.BLUE if turn_index % 2 == 0 else TeamColor.RED
        else:
            team_color = TeamColor.RED if turn_index % 2 == 0 else TeamColor.BLUE
        spymaster_name = next(
            (
                player.display_name
                for player in roster
                if player.team_color == team_color and player.role.value == "spymaster"
            ),
            "unknown",
        )
        turns.append(
            {
                "turn_index": turn_index,
                "team_color": team_color.value,
                "spymaster_name": spymaster_name,
                "clue_text": candidate["clue_text"],
                "clue_count": candidate["clue_count"],
                "clue_timestamp_sec": round(candidate["clue_timestamp_sec"], 3),
            }
        )

    for index, turn in enumerate(turns):
        next_start = clip_end_sec if index + 1 >= len(turns) else float(turns[index + 1]["clue_timestamp_sec"])
        turn["window_end_sec"] = round(next_start, 3)
        turn["guesses"] = []

    return turns


def _clue_matches_turn(event: ClueEvent, turn: dict[str, Any]) -> bool:
    if _clue_text_key(event.clue_text) != _clue_text_key(turn["clue_text"]):
        return False
    if not turn.get("team_color") and not turn.get("spymaster_name"):
        return True
    if event.team_color.value == turn["team_color"]:
        return True
    return event.spymaster_name == turn["spymaster_name"]


def _visible_guesses_for_turn(
    events: list[ClueEvent | GuessEvent],
    turn: dict[str, Any],
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


def _aggregate_guess_candidates(
    visible_frames: list[VisibleEventFrame],
    turn: dict[str, Any],
) -> list[dict[str, Any]]:
    by_word: dict[str, list[GuessEvent]] = defaultdict(list)
    window_start = float(turn["clue_timestamp_sec"])
    window_end = float(turn["window_end_sec"])
    for record in visible_frames:
        if not (window_start <= record.timestamp_sec <= window_end):
            continue
        for guess_event in _visible_guesses_for_turn(record.events, turn):
            by_word[guess_event.word].append(guess_event)

    guesses: list[dict[str, Any]] = []
    for word, events in by_word.items():
        first_timestamp = min(event.timestamp_sec for event in events)
        color_counter = Counter(event.card_color.value for event in events)
        chosen_color = max(color_counter.items(), key=lambda item: item[1])[0]
        confidence = max(event.confidence for event in events)
        guesses.append(
            {
                "timestamp_sec": round(first_timestamp, 3),
                "word": word,
                "card_color": chosen_color,
                "source": "log",
                "confidence": round(confidence, 4),
                "observation_count": len(events),
            }
        )
    return sorted(guesses, key=lambda item: item["timestamp_sec"])


def _scan_board_reveals(
    *,
    vod_source: VodSource,
    vod_url: str,
    start_sec: float,
    duration_sec: float,
    fps: float,
    roi_config_path: Path,
    board_state: BoardState,
) -> list[BoardReveal]:
    roi_config = load_roi_config(roi_config_path)
    iterator = vod_source.iter_window_frames(vod_url, start_sec, duration_sec, fps)
    first_sample = next(iterator, None)
    if first_sample is None:
        return []
    baseline = _capture_board_reveal_baseline(first_sample.frame_bgr, roi_config, board_state)
    previous_reveals: dict[tuple[int, int], CardColor | None] = {}
    reveals: list[BoardReveal] = []

    def process_sample(timestamp_sec: float, frame) -> None:
        board_frame = crop_roi(frame, roi_config.require("board_region"))
        for cell in board_state.cells:
            key = (cell.row, cell.col)
            sample = _sample_board_cell_surface(board_frame, cell.box)
            detected_color = _detect_revealed_card_color(sample, baseline[key])
            if detected_color is None:
                continue
            if previous_reveals.get(key) is None:
                previous_reveals[key] = detected_color
                reveals.append(
                    BoardReveal(
                        timestamp_sec=timestamp_sec,
                        word=cell.word,
                        card_color=detected_color,
                    )
                )

    process_sample(first_sample.timestamp_sec, first_sample.frame_bgr)
    for sample in iterator:
        process_sample(sample.timestamp_sec, sample.frame_bgr)
    return reveals


def _collect_visible_event_frames_window(
    *,
    vod_source: VodSource,
    vod_url: str,
    start_sec: float,
    end_sec: float,
    fps: float,
    roi_config,
    ocr_backend,
    roster: list[PlayerRosterEntry],
    board_state: BoardState,
) -> list[VisibleEventFrame]:
    previous_log_frame = None
    frames: list[VisibleEventFrame] = []
    duration = max(0.0, end_sec - start_sec)
    for sample in vod_source.iter_window_frames(vod_url, start_sec, duration, fps):
        log_frame = crop_roi(sample.frame_bgr, roi_config.require("game_log_region"))
        if previous_log_frame is not None and not log_has_changed(previous_log_frame, log_frame):
            continue
        previous_log_frame = log_frame
        parsed_events = parse_game_log(
            sample.frame_bgr,
            roi_config,
            ocr_backend,
            roster,
            board_state,
            timestamp_sec=sample.timestamp_sec,
        )
        if parsed_events:
            frames.append(VisibleEventFrame(timestamp_sec=sample.timestamp_sec, events=parsed_events))
    return frames


def _fetch_frame_at(
    *,
    vod_source: VodSource,
    vod_url: str,
    timestamp_sec: float,
):
    iterator = vod_source.iter_window_frames(vod_url, max(0.0, timestamp_sec), 1.0, 1.0)
    sample = next(iterator, None)
    if sample is None:
        raise RuntimeError(f"Could not decode frame at {timestamp_sec:.3f}s")
    return sample.frame_bgr


def _board_state_snapshot_at(
    *,
    vod_source: VodSource,
    vod_url: str,
    timestamp_sec: float,
    roi_config,
    board_state: BoardState,
    baseline: dict[tuple[int, int], Any],
    cache: dict[float, dict[str, CardColor | None]],
) -> dict[str, CardColor | None]:
    key = round(timestamp_sec, 3)
    cached = cache.get(key)
    if cached is not None:
        return cached
    frame = _fetch_frame_at(vod_source=vod_source, vod_url=vod_url, timestamp_sec=timestamp_sec)
    board_frame = crop_roi(frame, roi_config.require("board_region"))
    snapshot: dict[str, CardColor | None] = {}
    for cell in board_state.cells:
        sample = _sample_board_cell_surface(board_frame, cell.box)
        snapshot[cell.word] = _detect_revealed_card_color(sample, baseline[(cell.row, cell.col)])
    cache[key] = snapshot
    return snapshot


def _selected_boundary_words(
    *,
    changed: list[tuple[str, str]],
    team_color: str,
    own_delta: int,
    opponent_delta: int,
    clue_count: int,
) -> list[tuple[str, str]]:
    opponent_color = "red" if team_color == "blue" else "blue"
    own_words = [(word, color) for word, color in changed if color == team_color]
    opponent_words = [(word, color) for word, color in changed if color == opponent_color]
    neutral_words = [(word, color) for word, color in changed if color == "neutral"]

    selected: list[tuple[str, str]] = []
    selected.extend(own_words[:own_delta])
    selected.extend(opponent_words[:opponent_delta])
    target_total = max(len(selected), min(clue_count, len(selected) + len(neutral_words)))
    for item in neutral_words:
        if len(selected) >= target_total:
            break
        selected.append(item)
    return selected


def _refine_turn_guesses_with_boundary_board_diff(
    *,
    turns: list[dict[str, Any]],
    vod_source: VodSource,
    vod_url: str,
    roi_config,
    board_state: BoardState,
    baseline: dict[tuple[int, int], Any],
    observations: list[ClueObservation],
    final_left_counter: int | None,
    final_right_counter: int | None,
    clip_end_sec: float,
) -> None:
    snapshot_cache: dict[float, dict[str, CardColor | None]] = {}
    for index, turn in enumerate(turns):
        start_snapshot = _nearest_counter_snapshot(observations, float(turn["clue_timestamp_sec"]))
        if start_snapshot is None:
            continue
        if index + 1 < len(turns):
            end_snapshot = _nearest_counter_snapshot(observations, float(turns[index + 1]["clue_timestamp_sec"]))
        elif final_left_counter is not None and final_right_counter is not None:
            end_snapshot = (final_left_counter, final_right_counter)
        else:
            end_snapshot = start_snapshot
        if end_snapshot is None:
            continue

        start_left, start_right = start_snapshot
        end_left, end_right = end_snapshot
        blue_delta = max(0, start_left - end_left)
        red_delta = max(0, start_right - end_right)
        own_color = turn["team_color"]
        opponent_color = "red" if own_color == "blue" else "blue"
        own_delta = blue_delta if own_color == "blue" else red_delta
        opponent_delta = red_delta if own_color == "blue" else blue_delta
        if own_delta == 0 and opponent_delta == 0:
            continue

        start_time = float(turn["clue_timestamp_sec"]) + 0.5
        candidate_times = {
            max(start_time + 1.0, min(float(turn["window_end_sec"]) - 1.0, clip_end_sec - 1.0))
        }
        for offset in (20.0, 40.0, 60.0, 80.0):
            candidate_time = start_time + offset
            if candidate_time < clip_end_sec - 1.0:
                candidate_times.add(candidate_time)

        start_state = _board_state_snapshot_at(
            vod_source=vod_source,
            vod_url=vod_url,
            timestamp_sec=start_time,
            roi_config=roi_config,
            board_state=board_state,
            baseline=baseline,
            cache=snapshot_cache,
        )

        best_selected: list[tuple[str, str]] = []
        best_score: float | None = None
        clue_count = int(turn["clue_count"]) if str(turn["clue_count"]).isdigit() else 0
        for candidate_time in sorted(candidate_times):
            end_state = _board_state_snapshot_at(
                vod_source=vod_source,
                vod_url=vod_url,
                timestamp_sec=candidate_time,
                roi_config=roi_config,
                board_state=board_state,
                baseline=baseline,
                cache=snapshot_cache,
            )
            changed = [
                (word, color.value)
                for word, color in end_state.items()
                if start_state.get(word) is None and color is not None and color is not CardColor.BLACK
            ]
            selected = _selected_boundary_words(
                changed=changed,
                team_color=own_color,
                own_delta=own_delta,
                opponent_delta=opponent_delta,
                clue_count=clue_count,
            )
            own_matches = sum(1 for _, color in selected if color == own_color)
            opp_matches = sum(1 for _, color in selected if color == opponent_color)
            extras = max(0, len(changed) - len(selected))
            score = own_matches * 5 + opp_matches * 4 + len(selected) - extras * 2 - candidate_time / 10000.0
            if own_matches < own_delta or opp_matches < opponent_delta:
                score -= 10
            if best_score is None or score > best_score:
                best_score = score
                best_selected = selected

        if not best_selected:
            continue

        merged = {guess["word"]: dict(guess) for guess in turn["guesses"]}
        selected_words = {word for word, _ in best_selected}
        for word, color in best_selected:
            current = merged.get(word)
            if current is None:
                merged[word] = {
                    "timestamp_sec": round(float(turn["window_end_sec"]) - 1.0, 3),
                    "word": word,
                    "card_color": color,
                    "source": "board_boundary",
                    "confidence": 0.86,
                    "observation_count": 0,
                    "board_evidence": True,
                    "board_color": color,
                }
                continue
            current["board_evidence"] = True
            current["board_color"] = color
            if current.get("card_color") == "neutral" and color != "neutral":
                current["card_color"] = color
            merged[word] = current

        turn["guesses"] = sorted(
            [
                guess
                for guess in merged.values()
                if guess["word"] in selected_words or guess.get("source") != "log"
            ],
            key=lambda item: item["timestamp_sec"],
        )


def _merge_turn_guesses(
    *,
    turn: dict[str, Any],
    visible_guesses: list[dict[str, Any]],
    board_reveals: list[BoardReveal],
) -> list[dict[str, Any]]:
    guesses = {guess["word"]: dict(guess) for guess in visible_guesses}
    window_start = float(turn["clue_timestamp_sec"])
    window_end = float(turn["window_end_sec"])
    for reveal in board_reveals:
        if not (window_start <= reveal.timestamp_sec <= window_end):
            continue
        if reveal.card_color is CardColor.BLACK:
            continue
        if reveal.word in guesses:
            guesses[reveal.word]["board_evidence"] = True
            guesses[reveal.word]["board_color"] = reveal.card_color.value
            continue
        guesses[reveal.word] = {
            "timestamp_sec": round(reveal.timestamp_sec, 3),
            "word": reveal.word,
            "card_color": reveal.card_color.value,
            "source": "board",
            "confidence": 0.88,
            "observation_count": 0,
            "board_evidence": True,
            "board_color": reveal.card_color.value,
        }
    return sorted(guesses.values(), key=lambda item: item["timestamp_sec"])


def _refine_turn_guesses_with_log_rescan(
    *,
    turns: list[dict[str, Any]],
    vod_source: VodSource,
    vod_url: str,
    roi_config,
    ocr_backend,
    roster: list[PlayerRosterEntry],
    board_state: BoardState,
    fps: float,
    clip_end_sec: float,
) -> None:
    for index, turn in enumerate(turns):
        if index + 2 < len(turns):
            search_end = float(turns[index + 2]["clue_timestamp_sec"])
        else:
            search_end = clip_end_sec
        search_start = float(turn["clue_timestamp_sec"])
        if search_end <= search_start:
            continue
        rescanned_frames = _collect_visible_event_frames_window(
            vod_source=vod_source,
            vod_url=vod_url,
            start_sec=search_start,
            end_sec=search_end,
            fps=fps,
            roi_config=roi_config,
            ocr_backend=ocr_backend,
            roster=roster,
            board_state=board_state,
        )
        rescanned_guesses = _aggregate_guess_candidates(rescanned_frames, turn)
        if not rescanned_guesses:
            continue
        merged: dict[str, dict[str, Any]] = {guess["word"]: dict(guess) for guess in turn["guesses"]}
        for guess in rescanned_guesses:
            current = merged.get(guess["word"])
            if current is None:
                merged[guess["word"]] = guess
                continue
            board_evidence = current.get("board_evidence") or guess.get("board_evidence")
            board_color = current.get("board_color") or guess.get("board_color")
            if current.get("source") == "board" or guess.get("confidence", 0.0) >= current.get("confidence", 0.0):
                updated = dict(current)
                updated.update(guess)
            else:
                updated = dict(current)
                updated["observation_count"] = max(current.get("observation_count", 0), guess.get("observation_count", 0))
                updated["confidence"] = max(current.get("confidence", 0.0), guess.get("confidence", 0.0))
            if board_evidence:
                updated["board_evidence"] = True
            if board_color:
                updated["board_color"] = board_color
            merged[guess["word"]] = updated
        turn["guesses"] = sorted(merged.values(), key=lambda item: item["timestamp_sec"])


def _nearest_counter_snapshot(
    observations: list[ClueObservation],
    timestamp_sec: float,
) -> tuple[int, int] | None:
    candidates = [
        observation
        for observation in observations
        if observation.left_counter is not None and observation.right_counter is not None
    ]
    if not candidates:
        return None
    chosen = min(candidates, key=lambda item: abs(item.timestamp_sec - timestamp_sec))
    return chosen.left_counter, chosen.right_counter


def _apply_counter_consistency(
    turns: list[dict[str, Any]],
    observations: list[ClueObservation],
    *,
    final_left_counter: int | None,
    final_right_counter: int | None,
) -> None:
    for index, turn in enumerate(turns):
        start_snapshot = _nearest_counter_snapshot(observations, float(turn["clue_timestamp_sec"]))
        if start_snapshot is None:
            continue
        if index + 1 < len(turns):
            end_snapshot = _nearest_counter_snapshot(observations, float(turns[index + 1]["clue_timestamp_sec"]))
        elif final_left_counter is not None and final_right_counter is not None:
            end_snapshot = (final_left_counter, final_right_counter)
        else:
            end_snapshot = start_snapshot
        if end_snapshot is None:
            continue

        start_left, start_right = start_snapshot
        end_left, end_right = end_snapshot
        blue_delta = max(0, start_left - end_left)
        red_delta = max(0, start_right - end_right)
        own_color = turn["team_color"]
        opponent_color = "red" if own_color == "blue" else "blue"
        own_delta = blue_delta if own_color == "blue" else red_delta
        opponent_delta = red_delta if own_color == "blue" else blue_delta
        guesses = turn["guesses"]
        if not guesses:
            continue

        if len(guesses) == 1:
            if own_delta == 1 and opponent_delta == 0:
                guesses[0]["card_color"] = own_color
            elif own_delta == 0 and opponent_delta == 1:
                guesses[0]["card_color"] = opponent_color
            elif own_delta == 0 and opponent_delta == 0:
                guesses[0]["card_color"] = "neutral"
            continue

        if own_delta == len(guesses) and opponent_delta == 0:
            for guess in guesses:
                guess["card_color"] = own_color
            continue
        if opponent_delta == len(guesses) and own_delta == 0:
            for guess in guesses:
                guess["card_color"] = opponent_color
            continue
        if own_delta == 0 and opponent_delta == 0:
            for guess in guesses:
                guess["card_color"] = "neutral"
            continue
        if len(guesses) == 2 and own_delta == 1 and opponent_delta == 0:
            own_guess = next((guess for guess in guesses if guess.get("card_color") == own_color), None)
            neutral_guess = next((guess for guess in guesses if guess.get("card_color") == "neutral"), None)
            if own_guess is not None and neutral_guess is not None:
                continue
        if len(guesses) == 2 and own_delta == 1 and opponent_delta == 1:
            own_guess = next((guess for guess in guesses if guess.get("card_color") == own_color), None)
            opp_guess = next((guess for guess in guesses if guess.get("card_color") == opponent_color), None)
            if own_guess is not None and opp_guess is not None:
                continue


def _prune_turn_guesses(turns: list[dict[str, Any]]) -> None:
    seen_words: set[str] = set()
    for turn in turns:
        filtered: list[dict[str, Any]] = []
        for guess in turn["guesses"]:
            word = guess["word"]
            if word in seen_words:
                continue
            filtered.append(guess)
            seen_words.add(word)

        clue_count = int(turn["clue_count"]) if str(turn["clue_count"]).isdigit() else 0
        strong = [
            guess
            for guess in filtered
            if guess.get("board_evidence") or guess.get("source") == "board" or guess.get("observation_count", 0) >= 2
        ]
        weak = [
            guess
            for guess in filtered
            if guess not in strong
        ]
        weak.sort(key=lambda item: (item.get("confidence", 0.0), item.get("observation_count", 0), -item["timestamp_sec"]))
        while len(filtered) > max(clue_count + 1, len(strong)) and weak:
            drop = weak.pop(0)
            filtered = [guess for guess in filtered if guess is not drop]

        if len(filtered) > clue_count and clue_count > 0:
            filtered = [
                guess
                for guess in filtered
                if not (
                    guess.get("source") == "log"
                    and guess.get("observation_count", 0) <= 1
                    and guess.get("confidence", 0.0) < 0.55
                    and guess.get("board_evidence")
                    and guess.get("board_color")
                    and guess.get("board_color") != guess.get("card_color")
                )
            ]

        turn["guesses"] = sorted(filtered, key=lambda item: item["timestamp_sec"])


def _selector_payload(turns: list[dict[str, Any]], clip_path: str, roster: list[PlayerRosterEntry], board_words: list[str]) -> dict[str, Any]:
    return {
        "clip_path": clip_path,
        "roster": [player.model_dump(mode="json") for player in roster],
        "board_words": board_words,
        "turns": turns,
    }


def reconstruct_game_segment(
    *,
    vod_url: str,
    start_sec: float,
    duration_sec: float,
    roi_config_path: Path,
    ffmpeg_path: str,
    process_timeout_sec: float,
    ocr_backend_name: str,
    ocr_device: str,
    scan_fps: float,
    selector_output_dir: Path,
    selector_window_sec: float,
    selector_fps: float,
) -> dict[str, Any]:
    roi_config = load_roi_config(roi_config_path)
    backend_kwargs = {"gpu": _normalize_device_is_gpu(ocr_device)}
    ocr_backend = create_ocr_backend(ocr_backend_name, **backend_kwargs)
    vod_source = VodSource(
        ffmpeg_path=ffmpeg_path,
        process_timeout_sec=process_timeout_sec,
    )

    roster_observations: dict[tuple[str, str, str], dict[str, tuple[int, int, PlayerRosterEntry]]] = {}
    board_observations: dict[tuple[int, int], dict[str, tuple[int, int, Any]]] = {}
    best_roster: list[PlayerRosterEntry] = []
    best_board: BoardState | None = None
    clue_observations: list[ClueObservation] = []
    counter_samples: list[CounterSample] = []
    visible_frames: list[VisibleEventFrame] = []
    last_left_counter: int | None = None
    last_right_counter: int | None = None
    previous_log_frame = None

    clip_path = vod_url
    for sample in vod_source.iter_window_frames(vod_url, start_sec, duration_sec, scan_fps):
        frame = sample.frame_bgr

        if sample.timestamp_sec <= start_sec + 60.0:
            parsed_roster = parse_rosters(frame, roi_config, ocr_backend)
            if parsed_roster:
                _observe_roster(roster_observations, parsed_roster)
                materialized_roster = _materialize_roster_observations(roster_observations)
                if len(materialized_roster) >= len(best_roster):
                    best_roster = materialized_roster

            parsed_board = parse_board_words(frame, roi_config, ocr_backend)
            if parsed_board is not None:
                _observe_board_state(board_observations, parsed_board)
                materialized_board = _materialize_board_observations(board_observations)
                if materialized_board is not None:
                    best_board = materialized_board

        top_banner_text = parse_top_banner_text(frame, roi_config, ocr_backend)
        clue_text, clue_count = _parse_center_clue_banner(
            frame,
            roi_config=roi_config,
            ocr_backend=ocr_backend,
        )
        left_counter, right_counter = parse_game_counters(frame, roi_config, ocr_backend)
        counter_samples.append(
            CounterSample(
                timestamp_sec=sample.timestamp_sec,
                left_counter=left_counter,
                right_counter=right_counter,
            )
        )
        if left_counter is not None:
            last_left_counter = left_counter
        if right_counter is not None:
            last_right_counter = right_counter
        if clue_text and clue_count and clue_count.isdigit():
            clue_observations.append(
                ClueObservation(
                    timestamp_sec=sample.timestamp_sec,
                    clue_text=clue_text,
                    clue_count=clue_count,
                    top_banner_text=top_banner_text,
                    left_counter=left_counter,
                    right_counter=right_counter,
                )
            )

        if best_roster and best_board is not None:
            log_frame = crop_roi(frame, roi_config.require("game_log_region"))
            if log_has_changed(previous_log_frame, log_frame):
                parsed_events = parse_game_log(
                    frame,
                    roi_config,
                    ocr_backend,
                    best_roster,
                    best_board,
                    timestamp_sec=sample.timestamp_sec,
                )
                if parsed_events:
                    visible_frames.append(
                        VisibleEventFrame(timestamp_sec=sample.timestamp_sec, events=parsed_events)
                    )
            previous_log_frame = log_frame

    if not best_roster or best_board is None:
        raise RuntimeError("Could not materialize roster and board state from the clip")

    effective_clip_end_sec = start_sec + duration_sec
    for counter_sample in counter_samples:
        if counter_sample.left_counter == 0 or counter_sample.right_counter == 0:
            effective_clip_end_sec = min(effective_clip_end_sec, counter_sample.timestamp_sec + 2.0)
            break

    turns = _build_turn_schedule(
        clue_observations,
        best_roster,
        clip_end_sec=effective_clip_end_sec,
        visible_frames=visible_frames,
    )
    board_reveals = _scan_board_reveals(
        vod_source=vod_source,
        vod_url=vod_url,
        start_sec=start_sec,
        duration_sec=max(0.0, effective_clip_end_sec - start_sec),
        fps=scan_fps,
        roi_config_path=roi_config_path,
        board_state=best_board,
    )
    for turn in turns:
        visible_guesses = _aggregate_guess_candidates(visible_frames, turn)
        turn["guesses"] = _merge_turn_guesses(
            turn=turn,
            visible_guesses=visible_guesses,
            board_reveals=board_reveals,
        )
    _refine_turn_guesses_with_log_rescan(
        turns=turns,
        vod_source=vod_source,
        vod_url=vod_url,
        roi_config=roi_config,
        ocr_backend=ocr_backend,
        roster=best_roster,
        board_state=best_board,
        fps=max(2.0, scan_fps * 2.0),
        clip_end_sec=effective_clip_end_sec,
    )
    baseline_frame = _fetch_frame_at(vod_source=vod_source, vod_url=vod_url, timestamp_sec=start_sec)
    board_baseline = _capture_board_reveal_baseline(baseline_frame, roi_config, best_board)
    _refine_turn_guesses_with_boundary_board_diff(
        turns=turns,
        vod_source=vod_source,
        vod_url=vod_url,
        roi_config=roi_config,
        board_state=best_board,
        baseline=board_baseline,
        observations=clue_observations,
        final_left_counter=last_left_counter,
        final_right_counter=last_right_counter,
        clip_end_sec=effective_clip_end_sec,
    )
    _prune_turn_guesses(turns)
    _apply_counter_consistency(
        turns,
        clue_observations,
        final_left_counter=last_left_counter,
        final_right_counter=last_right_counter,
    )

    winner_team = None
    win_reason = None
    if last_left_counter == 0 and last_right_counter is not None:
        winner_team = "blue"
        win_reason = "counter_zero"
    elif last_right_counter == 0 and last_left_counter is not None:
        winner_team = "red"
        win_reason = "counter_zero"

    analysis = {
        "clip_path": clip_path,
        "roster": [player.model_dump(mode="json") for player in best_roster],
        "board_words": best_board.words,
        "turns": turns,
        "final_counters": {
            "left": last_left_counter,
            "right": last_right_counter,
        },
        "winner_team": winner_team,
        "win_reason": win_reason,
    }
    try:
        selector_results = attribute_selectors_from_analysis(
            analysis=_selector_payload(turns, clip_path, best_roster, best_board.words),
            analysis_reference_path=None,
            roi_config_path=roi_config_path,
            output_dir=selector_output_dir,
            ffmpeg_path=ffmpeg_path,
            ocr_backend_name=ocr_backend_name,
            ocr_device=ocr_device,
            window_sec=selector_window_sec,
            fps=selector_fps,
        )
        selector_lookup = {
            (item["turn_index"], item["guess_index"]): item
            for item in selector_results["guesses"]
        }
        for turn in analysis["turns"]:
            for guess_index, guess in enumerate(turn["guesses"]):
                selector_item = selector_lookup.get((turn["turn_index"], guess_index))
                if selector_item is None:
                    continue
                guess["selector_crop_path"] = selector_item["crop_path"]
                guess["selector_crop_x4_path"] = selector_item["crop_x4_path"]
                guess["selector_candidates"] = selector_item["candidate_scores"]
    except Exception as error:
        analysis["selector_attribution_error"] = str(error)

    return analysis


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = reconstruct_game_segment(
        vod_url=args.vod_url,
        start_sec=_parse_time_offset(args.start),
        duration_sec=args.duration,
        roi_config_path=Path(args.roi_config),
        ffmpeg_path=args.ffmpeg_path,
        process_timeout_sec=args.process_timeout,
        ocr_backend_name=args.ocr_backend,
        ocr_device=args.ocr_device,
        scan_fps=args.scan_fps,
        selector_output_dir=output_dir / "selector_attribution",
        selector_window_sec=args.selector_window_sec,
        selector_fps=args.selector_fps,
    )
    output_path = output_dir / "result.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
