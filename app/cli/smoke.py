"""Smoke-parse one bounded Codenames game window from a VOD"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.core.settings import get_settings
from app.parsers.banner import (
    _build_banner_clue_event,
    _infer_active_guessing_team,
    _last_clue_event,
    _opposite_team,
    _parse_center_clue_banner,
    _parse_center_clue_count_bubble,
    _refine_visible_events_from_banner,
)
from app.infra.roi_config import load_roi_config
from app.runtime.smoke.board_reveals import (
    _capture_board_reveal_baseline,
    _cell_color_statistics,
    _detect_board_reveal_guess_events,
    _detect_revealed_card_color,
    _sample_board_cell_surface,
    detect_hsv,
)
from app.runtime.smoke.cycle.runner import smoke_parse_game_cycle
from app.runtime.smoke.observations import (
    _avatar_hash_distance,
    _board_cell_quality,
    _board_is_usable,
    _board_quality,
    _board_signature,
    _find_matching_roster_identity,
    _materialize_board_observations,
    _materialize_roster_observations,
    _merge_board_states,
    _merge_rosters,
    _observe_board_state,
    _observe_roster,
    _player_name_quality,
    _roster_identity,
    _roster_is_usable,
    _roster_quality,
    _roster_signature,
    _should_replace_roster_name,
)
from app.runtime.smoke.runtime_models import SmokeChangeEvent, SmokeParseResult
from app.vision.ocr.backends import create_ocr_backend
from app.infra.vod_source import VodSource


def parse_time_offset(value: str) -> float:
    """Parse either raw seconds or `HH:MM:SS` timestamps."""

    stripped = value.strip()
    if ":" not in stripped:
        return float(stripped)

    parts = stripped.split(":")
    if len(parts) != 3:
        raise ValueError("Timestamp must be raw seconds or HH:MM:SS")
    hours, minutes, seconds = parts
    return (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the smoke module CLI parser."""

    settings = get_settings()
    parser = argparse.ArgumentParser(description="Smoke-parse one Codenames game cycle from a VOD.")
    parser.add_argument("--vod-url", required=True)
    parser.add_argument("--start", required=True, help="Raw seconds or HH:MM:SS")
    parser.add_argument("--duration", type=float, default=900.0)
    parser.add_argument("--chunk-duration", type=float, default=120.0)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional fixed fps override. Omit to use adaptive idle/active sampling.",
    )
    parser.add_argument("--idle-fps", type=float, default=1.0)
    parser.add_argument("--active-fps", type=float, default=2.0)
    parser.add_argument("--active-window", type=float, default=12.0)
    parser.add_argument("--roi-config", default="config/rois.example.json")
    parser.add_argument("--ocr-backend", choices=("paddle", "easyocr"), default="easyocr")
    parser.add_argument("--ocr-device", default=None, help="OCR device, e.g. cpu or gpu:0")
    parser.add_argument("--snapshot-dir", default="debug_snapshots/smoke")
    parser.add_argument("--vod-id", default="smoke")
    parser.add_argument("--game-index", type=int, default=0)
    parser.add_argument("--stream-quality", default="best")
    parser.add_argument("--ffmpeg-path", default=settings.ffmpeg_path)
    parser.add_argument("--process-timeout", type=float, default=180.0)
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--output-json", default=None, help="Optional path to write the final smoke result JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the smoke parser as a module command."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    roi_config = load_roi_config(args.roi_config)
    backend_kwargs: dict[str, object] = {}
    if args.ocr_backend == "paddle" and args.ocr_device is not None:
        backend_kwargs["device"] = args.ocr_device
    if args.ocr_backend == "easyocr" and args.ocr_device is not None:
        normalized_device = str(args.ocr_device).lower()
        backend_kwargs["gpu"] = normalized_device.startswith("gpu")
    ocr_backend = create_ocr_backend(args.ocr_backend, **backend_kwargs)
    progress_callback = None if args.quiet_progress else lambda message: print(message, file=sys.stderr)
    vod_source = VodSource(
        ffmpeg_path=args.ffmpeg_path,
        stream_quality=args.stream_quality,
        process_timeout_sec=args.process_timeout,
    )
    result = smoke_parse_game_cycle(
        vod_source=vod_source,
        vod_url=args.vod_url,
        start_sec=parse_time_offset(args.start),
        duration_sec=args.duration,
        fps=args.fps,
        chunk_duration_sec=args.chunk_duration,
        idle_fps=args.idle_fps,
        active_fps=args.active_fps,
        active_window_sec=args.active_window,
        roi_config=roi_config,
        ocr_backend=ocr_backend,
        vod_id=args.vod_id,
        game_index=args.game_index,
        snapshot_dir=args.snapshot_dir,
        progress_callback=progress_callback,
    )
    result_json = json.dumps(result.model_dump(mode="json"), indent=2)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result_json, encoding="utf-8")
    print(result_json)
    return 0


__all__ = [
    "SmokeChangeEvent",
    "SmokeParseResult",
    "build_arg_parser",
    "detect_hsv",
    "main",
    "parse_time_offset",
    "smoke_parse_game_cycle",
    "_avatar_hash_distance",
    "_board_cell_quality",
    "_board_is_usable",
    "_board_quality",
    "_board_signature",
    "_build_banner_clue_event",
    "_capture_board_reveal_baseline",
    "_cell_color_statistics",
    "_detect_board_reveal_guess_events",
    "_detect_revealed_card_color",
    "_find_matching_roster_identity",
    "_infer_active_guessing_team",
    "_last_clue_event",
    "_materialize_board_observations",
    "_materialize_roster_observations",
    "_merge_board_states",
    "_merge_rosters",
    "_observe_board_state",
    "_observe_roster",
    "_opposite_team",
    "_parse_center_clue_banner",
    "_parse_center_clue_count_bubble",
    "_player_name_quality",
    "_refine_visible_events_from_banner",
    "_roster_identity",
    "_roster_is_usable",
    "_roster_quality",
    "_roster_signature",
    "_sample_board_cell_surface",
    "_should_replace_roster_name",
]


if __name__ == "__main__":
    raise SystemExit(main())
