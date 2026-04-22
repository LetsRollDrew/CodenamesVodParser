"""Seeded reconstruction runner"""

from __future__ import annotations

from pathlib import Path

from app.core.models import BoardState, CardColor
from app.infra.roi_config import crop_roi, load_roi_config
from app.infra.vod_source import VodSource
from app.parsers.banner import (
    _banner_signatures_equivalent,
    _build_banner_clue_event,
    _focus_visible_events_on_latest_turn,
    _parse_center_clue_banner,
    _refine_visible_events_from_banner,
)
from app.parsers.gamelog import log_has_changed, parse_game_log
from app.parsers.selectors.attribution import attribute_selectors_from_analysis
from app.parsers.selectors.resolution import merge_selector_results_into_analysis, resolve_selector_fields_inplace
from app.reconstruction.game_reconstruction import build_game_record, reconstruct_game
from app.review.queue import materialize_review_items
from app.runtime.smoke.board_reveals import (
    _capture_board_reveal_baseline,
    _detect_board_reveal_guess_events,
    _sync_board_reveals_from_history,
)
from app.runtime.smoke.cycle.clue_window import _update_active_clue_window
from app.runtime.smoke.cycle.history_merge import (
    _append_new_visible_events,
    _find_matching_clue_index_anywhere,
    _is_stale_banner_residue,
)
from app.runtime.smoke.cycle.visible_log import (
    _commit_visible_log_events,
    _filter_visible_log_events_for_runtime,
    _should_probe_visible_log_from_state,
)
from app.runtime.smoke.runtime_models import BannerObservation, SmokeRuntimeState
from app.vision.detection.frame_detectors import (
    has_play_next_game,
    is_winner_banner_text,
    parse_game_counters,
    parse_top_banner_text,
)
from app.vision.ocr.backends import create_ocr_backend

from .dense_board import _augment_analysis_with_dense_board_rescans, _refresh_analysis_winner_fields
from .shared import _build_seeded_board_state, _load_seed, _normalize_device_is_gpu
from .turn_repair import (
    _refine_turn_guesses_with_visible_log_rescan,
    _restore_turn_stoppers_from_event_history,
    _stabilize_turn_guess_sequences,
    _turn_needs_second_pass,
)

REPAIR_VISIBLE_RESCAN_FPS = 2.0
REPAIR_VISIBLE_RESCAN_PRE_ROLL_SEC = 6.0


def reconstruct_seeded_game(
    *,
    vod_url: str,
    start_sec: float,
    duration_sec: float,
    roi_config_path: Path,
    seed_json_path: Path,
    ffmpeg_path: str,
    process_timeout_sec: float,
    ocr_backend_name: str,
    ocr_device: str,
    fps: float,
    game_index: int,
    selector_output_dir: Path,
    selector_window_sec: float,
    selector_fps: float,
    enable_selector_attribution: bool = True,
    enable_dense_board_backfill: bool = False,
) -> dict[str, object]:
    roi_config = load_roi_config(roi_config_path)
    roster, board_words = _load_seed(seed_json_path)
    ocr_backend = create_ocr_backend(ocr_backend_name, gpu=_normalize_device_is_gpu(ocr_device))
    vod_source = VodSource(ffmpeg_path=ffmpeg_path, process_timeout_sec=process_timeout_sec)

    runtime_state = SmokeRuntimeState()
    board_state: BoardState | None = None
    board_reveal_baseline = None
    previous_board_reveals: dict[tuple[int, int], CardColor | None] = {}
    pending_board_reveals: dict[tuple[int, int], tuple[CardColor, int]] = {}
    last_left_counter: int | None = None
    last_right_counter: int | None = None
    stop_reason = "window_exhausted"
    stop_sec = start_sec + duration_sec
    log_roi = roi_config.require("game_log_region")

    for sample in vod_source.iter_window_frames(vod_url, start_sec, duration_sec, fps):
        frame = sample.frame_bgr
        stop_sec = sample.timestamp_sec
        if board_state is None:
            board_state = _build_seeded_board_state(frame, roi_config, board_words)
            board_reveal_baseline = _capture_board_reveal_baseline(frame, roi_config, board_state)
            previous_board_reveals = {(cell.row, cell.col): None for cell in board_state.cells}
            pending_board_reveals = {}
            runtime_state.visible_log.quiet_log_frame_count = 0

        top_banner_text = parse_top_banner_text(frame, roi_config, ocr_backend)
        clue_banner_text, clue_banner_count = _parse_center_clue_banner(
            frame,
            roi_config=roi_config,
            ocr_backend=ocr_backend,
        )
        banner_observation = BannerObservation(
            timestamp_sec=sample.timestamp_sec,
            top_banner_text=top_banner_text,
            clue_text=clue_banner_text,
            clue_count=clue_banner_count,
        )
        left_counter, right_counter = parse_game_counters(frame, roi_config, ocr_backend)
        if left_counter is not None:
            last_left_counter = left_counter
        if right_counter is not None:
            last_right_counter = right_counter

        frame_new_events = []

        log_frame = crop_roi(frame, log_roi)
        log_changed = log_has_changed(
            runtime_state.visible_log.previous_log_frame,
            log_frame,
            minimum_change_ratio=0.02,
        )
        saw_visible_clue = False
        if log_changed:
            runtime_state.visible_log.quiet_log_frame_count = 0
        else:
            runtime_state.visible_log.quiet_log_frame_count += 1

        should_parse_visible_log = _should_probe_visible_log_from_state(
            runtime_state,
            log_changed=log_changed,
            banner_observation=banner_observation,
        )
        if should_parse_visible_log:
            current_visible_events = parse_game_log(
                frame,
                roi_config,
                ocr_backend,
                roster,
                board_state,
                timestamp_sec=sample.timestamp_sec,
            )
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
            saw_visible_clue = any(event.event_type == "clue" for event in current_visible_events)
            new_events = _commit_visible_log_events(
                runtime_state,
                current_visible_events,
                timestamp_sec=sample.timestamp_sec,
            )
            frame_new_events.extend(new_events)
            _update_active_clue_window(
                runtime_state,
                source="visible_log",
                events=current_visible_events,
                timestamp_sec=sample.timestamp_sec,
            )
            if log_changed:
                runtime_state.visible_log.quiet_log_frame_count = 0
        runtime_state.visible_log.previous_log_frame = log_frame

        if board_reveal_baseline is not None:
            previous_board_reveals, pending_board_reveals = _sync_board_reveals_from_history(
                previous_reveals=previous_board_reveals,
                pending_reveals=pending_board_reveals,
                board_state=board_state,
                history=runtime_state.history,
            )

        banner_clue_event = _build_banner_clue_event(
            clue_text=clue_banner_text,
            clue_count=clue_banner_count,
            top_banner_text=top_banner_text,
            roster=roster,
            history=runtime_state.history,
            left_counter=last_left_counter,
            right_counter=last_right_counter,
            timestamp_sec=sample.timestamp_sec,
        )
        current_banner_signature = None if banner_clue_event is None else (
            banner_clue_event.clue_text,
            banner_clue_event.clue_count,
        )
        if (
            banner_clue_event is not None
            and not saw_visible_clue
            and not _banner_signatures_equivalent(runtime_state.banner_signature, current_banner_signature)
            and _find_matching_clue_index_anywhere(runtime_state.history, banner_clue_event) is None
            and not _is_stale_banner_residue(runtime_state.history, banner_clue_event)
        ):
            runtime_state.history.append(banner_clue_event)
            frame_new_events.append(banner_clue_event)
            _update_active_clue_window(
                runtime_state,
                source="banner",
                events=[banner_clue_event],
                timestamp_sec=sample.timestamp_sec,
            )
        if current_banner_signature is not None:
            runtime_state.banner_signature = current_banner_signature

        if board_reveal_baseline is not None:
            new_board_guess_events, previous_board_reveals, pending_board_reveals = _detect_board_reveal_guess_events(
                frame,
                roi_config=roi_config,
                board_state=board_state,
                baseline=board_reveal_baseline,
                previous_reveals=previous_board_reveals,
                pending_reveals=pending_board_reveals,
                history=runtime_state.history,
                timestamp_sec=sample.timestamp_sec,
                allow_board_reveal_scan=runtime_state.visible_log.quiet_log_frame_count >= 1,
            )
            runtime_state.history, _, appended_board_events = _append_new_visible_events(
                runtime_state.history,
                [],
                new_board_guess_events,
            )
            frame_new_events.extend(appended_board_events)
            if any(event.card_color is CardColor.BLACK for event in appended_board_events):
                stop_reason = "assassin"
                break

        if is_winner_banner_text(top_banner_text):
            stop_reason = "winner_banner"
            break
        if has_play_next_game(frame, roi_config, ocr_backend):
            stop_reason = "play_next"
            break

    if board_state is None:
        raise RuntimeError("No frames decoded for seeded reconstruction")

    reconstruction = reconstruct_game(
        runtime_state.history,
        roster,
        left_counter=last_left_counter,
        right_counter=last_right_counter,
    )
    game_record = build_game_record(
        vod_id="seeded",
        game_index=game_index,
        start_sec=start_sec,
        end_sec=stop_sec,
        players=roster,
        reconstruction=reconstruction,
    )
    review_items = [
        {
            "entity_type": item.entity_type.value,
            "entity_id": item.entity_id,
            "reason": item.reason,
            "timestamp_sec": item.timestamp_sec,
            "snapshot_path": item.snapshot_path,
            "confidence": item.confidence,
        }
        for item in materialize_review_items(reconstruction.review_flags)
    ]
    analysis = {
        "vod_url": vod_url,
        "clip_path": vod_url,
        "reference_frame_sec": start_sec + min(5.0, duration_sec / 4.0),
        "start_sec": start_sec,
        "stop_sec": stop_sec,
        "stop_reason": stop_reason,
        "roster": [player.model_dump(mode="json") for player in roster],
        "board_words": board_state.words,
        "turns": [turn.model_dump(mode="json") for turn in game_record.turns],
        "winner_team": None if game_record.winner_team is None else game_record.winner_team.value,
        "win_reason": None if game_record.win_reason is None else game_record.win_reason.value,
        "final_counters": {
            "left": last_left_counter,
            "right": last_right_counter,
        },
        "event_count": len(runtime_state.history),
        "review_items": review_items,
    }
    if enable_dense_board_backfill:
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
    _restore_turn_stoppers_from_event_history(analysis, runtime_state.history)
    _stabilize_turn_guess_sequences(analysis)
    second_pass_turn_indexes = {
        turn_index
        for turn_index, turn in enumerate(analysis.get("turns", []))
        if isinstance(turn, dict) and _turn_needs_second_pass(turn)
    }
    if second_pass_turn_indexes:
        if enable_dense_board_backfill:
            _augment_analysis_with_dense_board_rescans(
                analysis=analysis,
                vod_source=vod_source,
                vod_url=vod_url,
                roi_config=roi_config,
                board_state=board_state,
                board_reveal_baseline=board_reveal_baseline,
                turn_indexes=second_pass_turn_indexes,
            )
        _refine_turn_guesses_with_visible_log_rescan(
            analysis=analysis,
            vod_source=vod_source,
            vod_url=vod_url,
            roi_config=roi_config,
            ocr_backend=ocr_backend,
            roster=roster,
            board_state=board_state,
            fps=REPAIR_VISIBLE_RESCAN_FPS,
            pre_roll_sec=REPAIR_VISIBLE_RESCAN_PRE_ROLL_SEC,
            turn_indexes=second_pass_turn_indexes,
        )
        _stabilize_turn_guess_sequences(analysis)
        _restore_turn_stoppers_from_event_history(analysis, runtime_state.history)
        _stabilize_turn_guess_sequences(analysis)
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
    _refresh_analysis_winner_fields(analysis)
    return analysis
