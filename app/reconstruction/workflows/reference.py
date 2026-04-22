"""Reconstruct a game using screenshot-confirmed roster and board references"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.core.models import BoardCell, BoardState, ImageBoundingBox, PlayerRole, PlayerRosterEntry, TeamColor
from app.infra.roi_config import crop_roi, load_roi_config
from app.infra.vod_source import VodSource
from app.reconstruction.game_reconstruction import reconstruct_game
from app.reconstruction.workflows.seeded import (
    _augment_analysis_with_dense_board_rescans,
    _refine_turn_guesses_with_visible_log_rescan,
    _refresh_analysis_winner_fields,
    _stabilize_turn_guess_sequences,
)
from app.parsers.banner import (
    _focus_visible_events_on_latest_turn,
    _parse_center_clue_banner,
    _refine_visible_events_from_banner,
)
from app.parsers.board import estimate_board_cell_boxes
from app.parsers.gamelog import log_has_changed, parse_game_log
from app.parsers.selectors.attribution import attribute_selectors_from_analysis
from app.parsers.selectors.resolution import merge_selector_results_into_analysis, resolve_selector_fields_inplace
from app.runtime.smoke.board_reveals import _capture_board_reveal_baseline
from app.runtime.smoke.cycle.clue_window import _update_active_clue_window
from app.runtime.smoke.cycle.visible_log import (
    _commit_visible_log_events,
    _filter_visible_log_events_for_runtime,
)
from app.runtime.smoke.runtime_models import BannerObservation, SmokeRuntimeState
from app.vision.detection.frame_detectors import parse_game_counters, parse_top_banner_text
from app.vision.ocr.backends import create_ocr_backend


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-url", required=True)
    parser.add_argument("--start", default="0")
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--roi-config", required=True)
    parser.add_argument("--reference-json", required=True)
    parser.add_argument("--reference-key", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ocr-backend", default="easyocr")
    parser.add_argument("--ocr-device", default="gpu:0")
    parser.add_argument("--ffmpeg-path", default="ffmpeg")
    parser.add_argument("--scan-fps", type=float, default=0.5)
    parser.add_argument("--selector-window-sec", type=float, default=1.5)
    parser.add_argument("--selector-fps", type=float, default=4.0)
    parser.add_argument("--skip-selector-attribution", action="store_true")
    parser.add_argument("--process-timeout", type=float, default=900.0)
    return parser.parse_args()


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


def _load_reference(reference_json_path: Path, reference_key: str) -> dict[str, Any]:
    references = json.loads(reference_json_path.read_text(encoding="utf-8"))
    try:
        reference = references[reference_key]
    except KeyError as exc:
        raise KeyError(f"Unknown reference key {reference_key!r}") from exc
    if not isinstance(reference, dict):
        raise TypeError(f"Reference {reference_key!r} must be an object")
    return reference


def _build_roster(reference: dict[str, Any]) -> list[PlayerRosterEntry]:
    roster_entries = reference.get("roster", [])
    roster: list[PlayerRosterEntry] = []
    for item in roster_entries:
        display_name = str(item["display_name"]).strip()
        roster.append(
            PlayerRosterEntry(
                player_name=display_name.casefold(),
                display_name=display_name,
                team_color=TeamColor(str(item["team_color"])),
                role=PlayerRole(str(item["role"])),
            )
        )
    return roster


def _build_board_state(reference: dict[str, Any]) -> BoardState:
    board_words = [str(word).strip().upper() for word in reference.get("board_words", [])]
    cells = [
        BoardCell(
            row=index // 5,
            col=index % 5,
            word=word,
            confidence=1.0,
            box=ImageBoundingBox(left=0, top=0, right=1, bottom=1),
        )
        for index, word in enumerate(board_words)
    ]
    return BoardState(cells=cells)


def _materialize_board_state_boxes(
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
                box=box,
            )
            for index, box in enumerate(boxes)
        ]
    )


def reconstruct_reference_game(
    *,
    vod_url: str,
    start_sec: float,
    duration_sec: float,
    roi_config_path: Path,
    reference_json_path: Path,
    reference_key: str,
    ffmpeg_path: str,
    process_timeout_sec: float,
    ocr_backend_name: str,
    ocr_device: str,
    scan_fps: float,
    selector_output_dir: Path,
    selector_window_sec: float,
    selector_fps: float,
    enable_selector_attribution: bool = True,
) -> dict[str, Any]:
    reference = _load_reference(reference_json_path, reference_key)
    roster = _build_roster(reference)
    board_state = _build_board_state(reference)

    roi_config = load_roi_config(roi_config_path)
    backend_kwargs = {"gpu": _normalize_device_is_gpu(ocr_device)}
    ocr_backend = create_ocr_backend(ocr_backend_name, **backend_kwargs)
    vod_source = VodSource(
        ffmpeg_path=ffmpeg_path,
        process_timeout_sec=process_timeout_sec,
    )

    log_roi = roi_config.require("game_log_region")
    runtime_state = SmokeRuntimeState()
    last_left_counter: int | None = None
    last_right_counter: int | None = None
    previous_zero_side: str | None = None
    previous_zero_timestamp: float | None = None
    last_frame = None
    board_reveal_baseline = None

    for sample in vod_source.iter_window_frames(vod_url, start_sec, duration_sec, scan_fps):
        frame = sample.frame_bgr
        last_frame = frame
        if board_reveal_baseline is None:
            board_state = _materialize_board_state_boxes(
                frame=frame,
                roi_config=roi_config,
                board_words=board_state.words,
            )
            board_reveal_baseline = _capture_board_reveal_baseline(frame, roi_config, board_state)

        clue_text, clue_count = _parse_center_clue_banner(
            frame,
            roi_config=roi_config,
            ocr_backend=ocr_backend,
        )
        left_counter, right_counter = parse_game_counters(frame, roi_config, ocr_backend)
        if left_counter is not None:
            last_left_counter = left_counter
        if right_counter is not None:
            last_right_counter = right_counter

        log_frame = crop_roi(frame, log_roi)
        log_changed = log_has_changed(runtime_state.visible_log.previous_log_frame, log_frame, minimum_change_ratio=0.02)

        if log_changed:
            top_banner_text = parse_top_banner_text(frame, roi_config, ocr_backend)
            banner_observation = BannerObservation(
                timestamp_sec=sample.timestamp_sec,
                top_banner_text=top_banner_text,
                clue_text=clue_text,
                clue_count=clue_count,
            )
            parsed_events = parse_game_log(
                frame,
                roi_config,
                ocr_backend,
                roster,
                board_state,
                timestamp_sec=sample.timestamp_sec,
            )
            parsed_events = _focus_visible_events_on_latest_turn(parsed_events)
            parsed_events = _refine_visible_events_from_banner(
                parsed_events,
                top_banner_text=top_banner_text,
                roster=roster,
                clue_banner_text=clue_text,
                clue_banner_count=clue_count,
            )
            parsed_events = _filter_visible_log_events_for_runtime(
                runtime_state,
                parsed_events,
                banner_observation=banner_observation,
            )
            _commit_visible_log_events(
                runtime_state,
                parsed_events,
                timestamp_sec=sample.timestamp_sec,
            )
            _update_active_clue_window(
                runtime_state,
                source="visible_log",
                events=parsed_events,
                timestamp_sec=sample.timestamp_sec,
            )
        runtime_state.visible_log.previous_log_frame = log_frame

        if "YOUR TEAM WINS" in parse_top_banner_text(frame, roi_config, ocr_backend):
            break
        zero_side: str | None = None
        if last_left_counter == 0:
            zero_side = "left"
        elif last_right_counter == 0:
            zero_side = "right"
        if zero_side is not None and sample.timestamp_sec < start_sec + 120.0:
            previous_zero_side = None
            previous_zero_timestamp = None
            continue
        if zero_side is None:
            previous_zero_side = None
            previous_zero_timestamp = None
            continue
        if (
            previous_zero_side == zero_side
            and previous_zero_timestamp is not None
            and sample.timestamp_sec - previous_zero_timestamp <= max(3.0, 1.5 / max(scan_fps, 0.1))
        ):
            break
        previous_zero_side = zero_side
        previous_zero_timestamp = sample.timestamp_sec

    reconstruction = reconstruct_game(
        runtime_state.history,
        roster,
        left_counter=last_left_counter,
        right_counter=last_right_counter,
    )

    winner_team = None if reconstruction.winner_team is None else reconstruction.winner_team.value
    win_reason = None if reconstruction.win_reason is None else reconstruction.win_reason.value
    final_banner_text = ""
    if last_frame is not None:
        final_banner_text = parse_top_banner_text(last_frame, roi_config, ocr_backend)
    if winner_team is None:
        if "YOUR TEAM WINS" in final_banner_text:
            winner_team = "blue"
            win_reason = "winner_banner"
        elif "OPPOSING TEAM WINS" in final_banner_text:
            winner_team = "red"
            win_reason = "winner_banner"
    if winner_team is None and reference.get("winner_team"):
        winner_team = str(reference["winner_team"])
        win_reason = win_reason or "reference"

    analysis = {
        "vod_url": vod_url,
        "clip_path": vod_url,
        "reference_frame_sec": start_sec + min(5.0, duration_sec / 4.0),
        "reference_key": reference_key,
        "start_sec": start_sec,
        "stop_sec": start_sec + duration_sec,
        "roster": [player.model_dump(mode="json") for player in roster],
        "board_words": board_state.words,
        "turns": [turn.model_dump(mode="json") for turn in reconstruction.turns],
        "final_counters": {
            "left": last_left_counter,
            "right": last_right_counter,
        },
        "winner_team": winner_team,
        "win_reason": win_reason,
        "event_count": len(runtime_state.history),
        "final_banner_text": final_banner_text,
    }
    if board_reveal_baseline is not None:
        _augment_analysis_with_dense_board_rescans(
            analysis=analysis,
            vod_source=vod_source,
            vod_url=vod_url,
            roi_config=roi_config,
            board_state=board_state,
            board_reveal_baseline=board_reveal_baseline,
        )
    _refine_turn_guesses_with_visible_log_rescan(
        analysis=analysis,
        vod_source=vod_source,
        vod_url=vod_url,
        roi_config=roi_config,
        ocr_backend=ocr_backend,
        roster=roster,
        board_state=board_state,
    )
    _stabilize_turn_guess_sequences(analysis)
    _refresh_analysis_winner_fields(analysis)
    if enable_selector_attribution:
        try:
            selector_results = attribute_selectors_from_analysis(
                analysis=analysis,
                analysis_reference_path=None,
                roi_config_path=roi_config_path,
                output_dir=selector_output_dir,
                ffmpeg_path=ffmpeg_path,
                process_timeout_sec=process_timeout_sec,
                ocr_backend_name=ocr_backend_name,
                ocr_device=ocr_device,
                window_sec=selector_window_sec,
                fps=selector_fps,
            )
            merge_selector_results_into_analysis(analysis, selector_results)
            resolve_selector_fields_inplace(analysis, ocr_backend=ocr_backend)
        except Exception as error:
            analysis["selector_attribution_error"] = str(error)
    return analysis


def main() -> int:
    args = _parse_args()
    result = reconstruct_reference_game(
        vod_url=args.vod_url,
        start_sec=_parse_time_offset(args.start),
        duration_sec=args.duration,
        roi_config_path=Path(args.roi_config),
        reference_json_path=Path(args.reference_json),
        reference_key=args.reference_key,
        ffmpeg_path=args.ffmpeg_path,
        process_timeout_sec=args.process_timeout,
        ocr_backend_name=args.ocr_backend,
        ocr_device=args.ocr_device,
        scan_fps=args.scan_fps,
        selector_output_dir=Path(args.output_json).parent / "selector_attribution",
        selector_window_sec=args.selector_window_sec,
        selector_fps=args.selector_fps,
        enable_selector_attribution=not args.skip_selector_attribution,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
