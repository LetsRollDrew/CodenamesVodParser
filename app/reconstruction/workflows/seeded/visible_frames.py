"""Visible-log frame collection and candidate aggregation for seeded workflows"""

from __future__ import annotations

from collections import Counter

from app.core.models import BoardState, PlayerRosterEntry
from app.infra.vod_source import VodSource
from app.parsers.banner import (
    _focus_visible_events_on_latest_turn,
    _parse_center_clue_banner,
    _refine_visible_events_from_banner,
)
from app.parsers.gamelog import ClueEvent, GuessEvent, parse_game_log
from app.runtime.smoke.cycle.clue_window import _update_active_clue_window
from app.runtime.smoke.cycle.visible_log import _filter_visible_log_events_for_runtime
from app.runtime.smoke.runtime_models import BannerObservation, SmokeRuntimeState
from app.vision.detection.frame_detectors import parse_top_banner_text

from .shared import _clue_matches_turn, _guess_color_rank, _normalized_word, _visible_guesses_for_turn


def _collect_turn_visible_log_frames(
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
    turn: dict[str, object],
) -> list[list[ClueEvent | GuessEvent]]:
    if end_sec <= start_sec:
        return []

    frames: list[list[ClueEvent | GuessEvent]] = []
    runtime_state = SmokeRuntimeState()
    active_turn_visible = False
    turn_start_timestamp = float(turn.get("timestamp_sec") or start_sec)
    guess_only_activation_delay_sec = 2.0
    for sample in vod_source.iter_window_frames(vod_url, start_sec, end_sec - start_sec, fps):
        top_banner_text = parse_top_banner_text(sample.frame_bgr, roi_config, ocr_backend)
        clue_banner_text, clue_banner_count = _parse_center_clue_banner(
            sample.frame_bgr,
            roi_config=roi_config,
            ocr_backend=ocr_backend,
        )
        banner_observation = BannerObservation(
            timestamp_sec=sample.timestamp_sec,
            top_banner_text=top_banner_text,
            clue_text=clue_banner_text,
            clue_count=clue_banner_count,
        )
        current_visible_events = parse_game_log(
            sample.frame_bgr,
            roi_config,
            ocr_backend,
            roster,
            board_state,
            timestamp_sec=sample.timestamp_sec,
        )
        if not current_visible_events:
            continue
        current_visible_events = _focus_visible_events_on_latest_turn(current_visible_events)
        current_visible_events = _refine_visible_events_from_banner(
            current_visible_events,
            top_banner_text=top_banner_text,
            roster=roster,
            clue_banner_text=clue_banner_text,
            clue_banner_count=clue_banner_count,
        )
        current_visible_events = _filter_visible_log_events_for_runtime(
            runtime_state,
            current_visible_events,
            banner_observation=banner_observation,
        )
        if not current_visible_events:
            continue
        clue_event = next((event for event in current_visible_events if isinstance(event, ClueEvent)), None)
        if clue_event is not None:
            if _clue_matches_turn(clue_event, turn):
                active_turn_visible = True
            elif active_turn_visible:
                break
            else:
                continue
        elif not active_turn_visible:
            if sample.timestamp_sec < turn_start_timestamp + guess_only_activation_delay_sec:
                continue
            guess_only_events = [event for event in current_visible_events if isinstance(event, GuessEvent)]
            if not guess_only_events:
                continue
            active_turn_visible = True
        _update_active_clue_window(
            runtime_state,
            source="visible_log",
            events=current_visible_events,
            timestamp_sec=sample.timestamp_sec,
        )
        frame_guesses = _visible_guesses_for_turn(current_visible_events, turn)
        if not frame_guesses and clue_event is None:
            frame_guesses = [event for event in current_visible_events if isinstance(event, GuessEvent)]
        if not frame_guesses:
            continue
        frames.append(current_visible_events)
    return frames


def _aggregate_turn_visible_guess_candidates(
    visible_frames: list[list[ClueEvent | GuessEvent]],
    turn: dict[str, object],
) -> list[dict[str, object]]:
    team_color = str(turn.get("team_color") or "").strip().casefold()
    by_word: dict[str, dict[str, object]] = {}
    next_visible_order = 0

    for events in visible_frames:
        frame_guesses = _visible_guesses_for_turn(events, turn)
        if not frame_guesses and not any(isinstance(event, ClueEvent) for event in events):
            frame_guesses = [event for event in events if isinstance(event, GuessEvent)]
        for position, guess_event in enumerate(frame_guesses):
            word = _normalized_word(guess_event.word)
            if not word:
                continue
            entry = by_word.get(word)
            if entry is None:
                entry = {
                    "word": word,
                    "timestamp_sec": guess_event.timestamp_sec,
                    "confidence": guess_event.confidence,
                    "observation_count": 0,
                    "_visible_order": next_visible_order,
                    "_visible_position": position,
                    "_color_votes": Counter(),
                }
                by_word[word] = entry
                next_visible_order += 1
            entry["timestamp_sec"] = min(float(entry["timestamp_sec"]), guess_event.timestamp_sec)
            entry["confidence"] = max(float(entry["confidence"]), guess_event.confidence)
            entry["observation_count"] = int(entry["observation_count"]) + 1
            color_votes = entry["_color_votes"]
            assert isinstance(color_votes, Counter)
            color_votes[guess_event.card_color.value] += 1

    aggregated: list[dict[str, object]] = []
    for entry in by_word.values():
        color_votes = entry.pop("_color_votes")
        assert isinstance(color_votes, Counter)
        chosen_color = max(
            color_votes.items(),
            key=lambda item: (
                item[1],
                _guess_color_rank(team_color, item[0]),
                -len(item[0]),
            ),
        )[0]
        aggregated.append(
            {
                "word": entry["word"],
                "timestamp_sec": round(float(entry["timestamp_sec"]), 3),
                "card_color": chosen_color,
                "confidence": round(float(entry["confidence"]), 4),
                "observation_count": int(entry["observation_count"]),
                "source": "log_rescan",
                "_visible_order": entry["_visible_order"],
                "_visible_position": entry["_visible_position"],
            }
        )
    return sorted(
        aggregated,
        key=lambda item: (int(item["_visible_order"]), float(item["timestamp_sec"]), int(item["_visible_position"])),
    )
