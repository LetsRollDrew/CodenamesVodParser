"""CLI entrypoint for seeded reconstruction workflows"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .shared import _parse_time_offset
from .workflow import reconstruct_seeded_game


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-url", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--roi-config", required=True)
    parser.add_argument("--seed-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ocr-backend", default="easyocr")
    parser.add_argument("--ocr-device", default="gpu:0")
    parser.add_argument("--ffmpeg-path", default="ffmpeg")
    parser.add_argument("--fps", type=float, default=0.5)
    parser.add_argument("--selector-window-sec", type=float, default=1.5)
    parser.add_argument("--selector-fps", type=float, default=4.0)
    parser.add_argument("--skip-selector-attribution", action="store_true")
    parser.add_argument("--enable-dense-board-backfill", action="store_true")
    parser.add_argument("--process-timeout", type=float, default=900.0)
    parser.add_argument("--game-index", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = reconstruct_seeded_game(
        vod_url=args.vod_url,
        start_sec=_parse_time_offset(args.start),
        duration_sec=args.duration,
        roi_config_path=Path(args.roi_config),
        seed_json_path=Path(args.seed_json),
        ffmpeg_path=args.ffmpeg_path,
        process_timeout_sec=args.process_timeout,
        ocr_backend_name=args.ocr_backend,
        ocr_device=args.ocr_device,
        fps=args.fps,
        game_index=args.game_index,
        selector_output_dir=Path(args.output_json).parent / "selector_attribution",
        selector_window_sec=args.selector_window_sec,
        selector_fps=args.selector_fps,
        enable_selector_attribution=not args.skip_selector_attribution,
        enable_dense_board_backfill=args.enable_dense_board_backfill,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
