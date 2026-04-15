"""Index full-VOD Codenames game windows from pregame/start/end anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.board_parser import parse_board_words
from app.frame_detectors import (
    has_play_next_game,
    is_setup_panel_text,
    is_winner_banner_text,
    parse_game_counters,
    parse_setup_panel_text,
    parse_top_banner_text,
    start_game_button_visible,
)
from app.ocr_backends import create_ocr_backend
from app.roi_config import load_roi_config
from app.vod_source import VodSource


class IndexedGameWindow(BaseModel):
    game_index: int = Field(ge=0)
    pregame_sec: float | None = Field(default=None, ge=0)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    end_reason: Literal["winner_banner", "play_next", "counter_zero"]
    left_counter: int | None = Field(default=None, ge=0)
    right_counter: int | None = Field(default=None, ge=0)


class VODIndexResult(BaseModel):
    vod_url: str
    requested_start_sec: float = Field(ge=0)
    requested_duration_sec: float = Field(gt=0)
    scan_fps: float = Field(gt=0)
    games: list[IndexedGameWindow] = Field(default_factory=list)


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


def index_vod_games(
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
    chunk_duration_sec: float,
) -> VODIndexResult:
    roi_config = load_roi_config(roi_config_path)
    backend_kwargs = {"gpu": _normalize_device_is_gpu(ocr_device)}
    ocr_backend = create_ocr_backend(ocr_backend_name, **backend_kwargs)
    vod_source = VodSource(
        ffmpeg_path=ffmpeg_path,
        process_timeout_sec=process_timeout_sec,
    )

    windows: list[IndexedGameWindow] = []
    pending_pregame_sec: float | None = None
    active_start_sec: float | None = None
    last_left_counter: int | None = None
    last_right_counter: int | None = None
    state: Literal["seeking_pregame", "pregame", "active", "cooldown"] = "seeking_pregame"
    consecutive_winner_frames = 0
    consecutive_zero_frames = 0
    consecutive_setup_frames = 0
    max_pregame_gap_sec = 120.0

    requested_end_sec = start_sec + duration_sec
    window_start_sec = start_sec

    while window_start_sec < requested_end_sec:
        window_duration = min(chunk_duration_sec, requested_end_sec - window_start_sec)
        for sample in vod_source.iter_window_frames(vod_url, window_start_sec, window_duration, scan_fps):
            frame = sample.frame_bgr
            timestamp_sec = sample.timestamp_sec
            left_counter, right_counter = parse_game_counters(frame, roi_config, ocr_backend)
            counters_visible = left_counter is not None and right_counter is not None
            if left_counter is not None:
                last_left_counter = left_counter
            if right_counter is not None:
                last_right_counter = right_counter

            setup_visible = False
            if not counters_visible and start_game_button_visible(frame):
                setup_visible = is_setup_panel_text(parse_setup_panel_text(frame, ocr_backend))

            if state in {"seeking_pregame", "cooldown"}:
                if setup_visible:
                    pending_pregame_sec = timestamp_sec
                    state = "pregame"
                continue

            if (
                state == "pregame"
                and pending_pregame_sec is not None
                and timestamp_sec - pending_pregame_sec > max_pregame_gap_sec
            ):
                pending_pregame_sec = None
                state = "seeking_pregame"
                continue

            play_next_visible = False
            winner_visible = False
            if state == "pregame" and setup_visible:
                pending_pregame_sec = timestamp_sec
            if state == "pregame" and counters_visible:
                board_state = parse_board_words(frame, roi_config, ocr_backend)
                if board_state is not None and len(board_state.words) >= 25:
                    active_start_sec = timestamp_sec
                    consecutive_winner_frames = 0
                    consecutive_zero_frames = 0
                    consecutive_setup_frames = 0
                    state = "active"
                    continue

            if state == "active":
                play_next_visible = has_play_next_game(frame, roi_config, ocr_backend)
                if not play_next_visible:
                    winner_visible = is_winner_banner_text(parse_top_banner_text(frame, roi_config, ocr_backend))
                consecutive_winner_frames = consecutive_winner_frames + 1 if winner_visible else 0
                zero_visible = left_counter == 0 or right_counter == 0
                consecutive_zero_frames = consecutive_zero_frames + 1 if zero_visible else 0
                consecutive_setup_frames = consecutive_setup_frames + 1 if setup_visible else 0

            if state == "active" and (
                consecutive_setup_frames >= 2
                or
                play_next_visible
                or consecutive_winner_frames >= 2
            ):
                if consecutive_setup_frames >= 2:
                    end_reason: Literal["winner_banner", "play_next", "counter_zero"] = "play_next"
                elif winner_visible:
                    end_reason: Literal["winner_banner", "play_next", "counter_zero"] = "winner_banner"
                elif play_next_visible:
                    end_reason = "play_next"
                else:
                    end_reason = "counter_zero"
                windows.append(
                    IndexedGameWindow(
                        game_index=len(windows),
                        pregame_sec=pending_pregame_sec,
                        start_sec=active_start_sec,
                        end_sec=timestamp_sec,
                        end_reason=end_reason,
                        left_counter=left_counter if left_counter is not None else last_left_counter,
                        right_counter=right_counter if right_counter is not None else last_right_counter,
                    )
                )
                pending_pregame_sec = None
                active_start_sec = None
                last_left_counter = left_counter
                last_right_counter = right_counter
                consecutive_winner_frames = 0
                consecutive_zero_frames = 0
                consecutive_setup_frames = 0
                state = "cooldown"

        window_start_sec += window_duration

    return VODIndexResult(
        vod_url=vod_url,
        requested_start_sec=start_sec,
        requested_duration_sec=duration_sec,
        scan_fps=scan_fps,
        games=windows,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-url", required=True)
    parser.add_argument("--start", default="0")
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--roi-config", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ocr-backend", default="easyocr")
    parser.add_argument("--ocr-device", default="gpu:0")
    parser.add_argument("--ffmpeg-path", default="ffmpeg")
    parser.add_argument("--scan-fps", type=float, default=0.5)
    parser.add_argument("--chunk-duration", type=float, default=300.0)
    parser.add_argument("--process-timeout", type=float, default=900.0)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    result = index_vod_games(
        vod_url=args.vod_url,
        start_sec=_parse_time_offset(args.start),
        duration_sec=args.duration,
        roi_config_path=Path(args.roi_config),
        ffmpeg_path=args.ffmpeg_path,
        process_timeout_sec=args.process_timeout,
        ocr_backend_name=args.ocr_backend,
        ocr_device=args.ocr_device,
        scan_fps=args.scan_fps,
        chunk_duration_sec=args.chunk_duration,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
