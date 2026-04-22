"""Repair an existing runtime analysis using dense board and visible-log rescans"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.models import BoardCell, BoardState, ImageBoundingBox, PlayerRole, PlayerRosterEntry, TeamColor
from app.infra.roi_config import crop_roi, load_roi_config
from app.infra.vod_source import VodSource
from app.reconstruction.workflows.seeded import (
    _normalize_device_is_gpu,
    _parse_time_offset,
    _augment_analysis_with_dense_board_rescans,
    _refine_turn_guesses_with_visible_log_rescan,
    _refresh_analysis_winner_fields,
    _stabilize_turn_guess_sequences,
)
from app.parsers.board import estimate_board_cell_boxes
from app.runtime.smoke.board_reveals import _capture_board_reveal_baseline
from app.vision.ocr.backends import create_ocr_backend

DEFAULT_VISIBLE_RESCAN_FPS = 2.0
DEFAULT_VISIBLE_RESCAN_PRE_ROLL_SEC = 6.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--vod-url")
    parser.add_argument("--start")
    parser.add_argument("--stop")
    parser.add_argument("--roi-config", required=True)
    parser.add_argument("--ocr-backend", default="easyocr")
    parser.add_argument("--ocr-device", default="gpu:0")
    parser.add_argument("--ffmpeg-path", default="ffmpeg")
    parser.add_argument("--process-timeout", type=float, default=900.0)
    parser.add_argument("--turn-index", action="append", type=int, dest="turn_indexes")
    return parser.parse_args()


def _load_analysis(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("analysis-json must contain an object")
    return payload


def _load_roster(analysis: dict[str, object]) -> list[PlayerRosterEntry]:
    roster_entries = analysis.get("roster", [])
    if not isinstance(roster_entries, list):
        raise TypeError("analysis roster must be a list")
    roster: list[PlayerRosterEntry] = []
    for item in roster_entries:
        if not isinstance(item, dict):
            continue
        roster.append(
            PlayerRosterEntry(
                player_name=str(item.get("player_name") or item["display_name"]).casefold(),
                display_name=str(item["display_name"]),
                team_color=TeamColor(str(item["team_color"])),
                role=PlayerRole(str(item["role"])),
                avatar_hash=item.get("avatar_hash"),
            )
        )
    return roster


def _materialize_board_state(
    *,
    frame,
    roi_config,
    board_words: list[str],
) -> BoardState:
    board_frame = crop_roi(frame, roi_config.require("board_region"))
    boxes = estimate_board_cell_boxes(board_frame)
    return BoardState(
        cells=[
            BoardCell(
                row=index // 5,
                col=index % 5,
                word=board_words[index],
                confidence=1.0,
                box=box if isinstance(box, ImageBoundingBox) else ImageBoundingBox.model_validate(box),
            )
            for index, box in enumerate(boxes)
        ]
    )


def repair_analysis(
    *,
    analysis_json_path: Path,
    output_json_path: Path,
    vod_url_override: str | None,
    start_override: float | None,
    stop_override: float | None,
    roi_config_path: Path,
    ffmpeg_path: str,
    process_timeout_sec: float,
    ocr_backend_name: str,
    ocr_device: str,
    turn_indexes: set[int] | None = None,
) -> dict[str, object]:
    analysis = _load_analysis(analysis_json_path)
    roi_config = load_roi_config(roi_config_path)
    roster = _load_roster(analysis)
    board_words = [str(word).strip().upper() for word in analysis.get("board_words", [])]
    if len(board_words) != 25:
        raise ValueError("analysis must contain 25 board_words")

    vod_url = vod_url_override or str(analysis.get("vod_url") or analysis.get("clip_path") or "")
    if not vod_url:
        raise ValueError("vod_url missing from analysis and no override was provided")
    start_sec = start_override if start_override is not None else float(analysis.get("start_sec") or 0.0)
    stop_sec = stop_override if stop_override is not None else float(analysis.get("stop_sec") or 0.0)
    if stop_sec <= start_sec:
        raise ValueError("analysis start/stop window is invalid")

    ocr_backend = create_ocr_backend(ocr_backend_name, gpu=_normalize_device_is_gpu(ocr_device))
    vod_source = VodSource(ffmpeg_path=ffmpeg_path, process_timeout_sec=process_timeout_sec)
    first_sample = next(vod_source.iter_window_frames(vod_url, start_sec, 1.0, 1.0), None)
    if first_sample is None:
        raise RuntimeError("Could not decode the first frame for analysis repair")

    board_state = _materialize_board_state(
        frame=first_sample.frame_bgr,
        roi_config=roi_config,
        board_words=board_words,
    )
    board_reveal_baseline = _capture_board_reveal_baseline(first_sample.frame_bgr, roi_config, board_state)

    analysis["vod_url"] = vod_url
    analysis["clip_path"] = vod_url
    analysis["start_sec"] = start_sec
    analysis["stop_sec"] = stop_sec

    _augment_analysis_with_dense_board_rescans(
        analysis=analysis,
        vod_source=vod_source,
        vod_url=vod_url,
        roi_config=roi_config,
        board_state=board_state,
        board_reveal_baseline=board_reveal_baseline,
        turn_indexes=turn_indexes,
    )
    _refine_turn_guesses_with_visible_log_rescan(
        analysis=analysis,
        vod_source=vod_source,
        vod_url=vod_url,
        roi_config=roi_config,
        ocr_backend=ocr_backend,
        roster=roster,
        board_state=board_state,
        fps=DEFAULT_VISIBLE_RESCAN_FPS,
        pre_roll_sec=DEFAULT_VISIBLE_RESCAN_PRE_ROLL_SEC,
        turn_indexes=turn_indexes,
    )
    _stabilize_turn_guess_sequences(analysis)
    _refresh_analysis_winner_fields(analysis)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    return analysis


def main() -> int:
    args = _parse_args()
    result = repair_analysis(
        analysis_json_path=Path(args.analysis_json),
        output_json_path=Path(args.output_json),
        vod_url_override=args.vod_url,
        start_override=None if args.start is None else _parse_time_offset(args.start),
        stop_override=None if args.stop is None else _parse_time_offset(args.stop),
        roi_config_path=Path(args.roi_config),
        ffmpeg_path=args.ffmpeg_path,
        process_timeout_sec=args.process_timeout,
        ocr_backend_name=args.ocr_backend,
        ocr_device=args.ocr_device,
        turn_indexes=None if not args.turn_indexes else set(args.turn_indexes),
    )
    print(
        json.dumps(
            {
                "output_json": str(Path(args.output_json)),
                "turn_count": len(result.get("turns", [])),
                "winner_team": result.get("winner_team"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
