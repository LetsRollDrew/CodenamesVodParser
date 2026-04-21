"""Runtime loop for the bounded smoke parser"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import numpy as np
from PIL import Image

from app.core.models import BoardCell, CardColor, PlayerRosterEntry
from app.infra.roi_config import ROIConfig, crop_roi
from app.infra.vod_source import VodSource
from app.parsers.banner import (
    _build_banner_clue_event,
    _focus_visible_events_on_latest_turn,
    _parse_center_clue_banner,
    _refine_visible_events_from_banner,
)
from app.parsers.board import parse_board_words
from app.parsers.gamelog import ClueEvent, GuessEvent, log_has_changed, parse_game_log
from app.parsers.roster import parse_rosters
from app.reconstruction.game_reconstruction import build_game_record, reconstruct_game
from app.review.queue import materialize_review_items
from app.runtime.smoke.board_reveals import (
    _capture_board_reveal_baseline,
    _detect_board_reveal_guess_events,
    _sync_board_reveals_from_history,
)
from app.runtime.smoke.observations import (
    _board_is_usable,
    _board_quality,
    _board_signature,
    _materialize_board_observations,
    _materialize_roster_observations,
    _observe_board_state,
    _observe_roster,
    _roster_is_usable,
    _roster_quality,
    _roster_signature,
)
from app.runtime.smoke.runtime_models import (
    BannerObservation,
    CounterRuntimeState,
    FrameObservations,
    SmokeChangeEvent,
    SmokeParseResult,
    SmokeRuntimeState,
)
from app.vision.debug.debug_visuals import ROIProbeArtifacts, save_annotated_log_event_snapshot, save_roi_probe
from app.vision.detection.frame_detectors import (
    has_play_next_game,
    is_setup_screen_text,
    is_winner_banner_text,
    parse_game_counter_observations,
    parse_game_counters,
    parse_top_banner_text,
)
from app.vision.ocr.preprocessing import OCRBackend

from .clue_window import (
    _banner_supports_active_window_strict,
    _classify_frame_phase,
    _clear_stale_pending_clue,
    _observe_clue_candidate,
    _observe_pending_assassin_guess,
    _phase_allows_board_reveal_scan,
    _phase_allows_new_clue_commit,
    _split_visible_clue_and_trailing_events,
    _update_active_clue_window,
    _visible_clue_requires_pending_promotion,
)
from .history_merge import (
    _append_new_visible_events,
    _clue_events_equivalent,
    _find_matching_clue_index_anywhere,
    _is_stale_banner_residue,
    _should_replace_locked_board_state,
)
from .visible_log import (
    _commit_visible_log_events,
    _filter_visible_log_events_for_runtime,
    _should_probe_visible_log_from_state,
)


def smoke_parse_game_cycle(
    *,
    vod_source: VodSource,
    vod_url: str,
    start_sec: float,
    duration_sec: float,
    fps: float | None = None,
    chunk_duration_sec: float = 120.0,
    idle_fps: float = 1.0,
    active_fps: float = 2.0,
    active_window_sec: float = 12.0,
    roi_config: ROIConfig,
    ocr_backend: OCRBackend,
    vod_id: str = "smoke",
    game_index: int = 0,
    snapshot_dir: str | Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> SmokeParseResult:
    """Parse one real VOD game cycle from a bounded window until a stop condition is met"""

    if chunk_duration_sec <= 0:
        raise ValueError("chunk_duration_sec must be > 0")
    if fps is not None and fps <= 0:
        raise ValueError("fps must be > 0")
    if idle_fps <= 0 or active_fps <= 0:
        raise ValueError("idle_fps and active_fps must be > 0")
    if active_window_sec <= 0:
        raise ValueError("active_window_sec must be > 0")

    candidate_roster: list[PlayerRosterEntry] = []
    candidate_board_state = None
    roster: list[PlayerRosterEntry] | None = None
    board_state = None
    roster_observations: dict[tuple[str, str, str], dict[str, tuple[int, int, PlayerRosterEntry]]] = {}
    board_observations: dict[tuple[int, int], dict[str, tuple[int, int, BoardCell]]] = {}
    game_start_sec: float | None = None
    runtime_state = SmokeRuntimeState()
    change_events: list[SmokeChangeEvent] = []
    board_reveal_baseline: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None = None
    previous_board_reveals: dict[tuple[int, int], CardColor | None] = {}
    pending_board_reveals: dict[tuple[int, int], tuple[CardColor, int]] = {}
    stop_reason: Literal["counter_zero", "assassin", "play_next", "winner_banner", "window_exhausted"] = "window_exhausted"
    stop_sec = start_sec
    active_until_sec = start_sec

    snapshots_path: Path | None = None
    first_frame_snapshot_path: str | None = None
    first_preready_snapshot_path: str | None = None
    roi_probe_artifacts = ROIProbeArtifacts(overlay_path=None, crop_paths={})
    start_snapshot_path: str | None = None
    if snapshot_dir is not None:
        snapshots_path = Path(snapshot_dir)
        snapshots_path.mkdir(parents=True, exist_ok=True)

    requested_end_sec = start_sec + duration_sec
    window_start_sec = start_sec
    should_stop = False
    log_roi = roi_config.require("game_log_region")

    while window_start_sec < requested_end_sec and not should_stop:
        window_duration = min(chunk_duration_sec, requested_end_sec - window_start_sec)
        yielded_any_frames = False
        current_fps = fps if fps is not None else (active_fps if window_start_sec < active_until_sec else idle_fps)
        _emit_progress(
            progress_callback,
            f"[chunk] {window_start_sec:.3f}s -> {window_start_sec + window_duration:.3f}s @ {current_fps:.3f}fps",
        )

        for sample in vod_source.iter_window_frames(vod_url, window_start_sec, window_duration, current_fps):
            yielded_any_frames = True
            frame = sample.frame_bgr
            stop_sec = sample.timestamp_sec
            if (
                snapshots_path is not None
                and first_preready_snapshot_path is None
                and (roster is None or board_state is None)
            ):
                first_preready_snapshot_path = str(
                    _save_snapshot(
                        frame,
                        snapshots_path,
                        timestamp_sec=sample.timestamp_sec,
                        label="preready",
                    )
                )
            if snapshots_path is not None and first_frame_snapshot_path is None:
                first_frame_snapshot_path = str(
                    _save_snapshot(
                        frame,
                        snapshots_path,
                        timestamp_sec=sample.timestamp_sec,
                        label="first",
                    )
                )
                roi_probe_artifacts = save_roi_probe(
                    frame,
                    roi_config,
                    ocr_backend,
                    snapshots_path,
                    timestamp_sec=sample.timestamp_sec,
                )
            _emit_progress(progress_callback, f"[frame] {sample.timestamp_sec:.3f}s")
            top_banner_text = parse_top_banner_text(frame, roi_config, ocr_backend)
            setup_visible = is_setup_screen_text(top_banner_text)
            winner_visible = is_winner_banner_text(top_banner_text)
            left_counter_observation, right_counter_observation = parse_game_counter_observations(
                frame,
                roi_config,
                ocr_backend,
            )
            left_counter = _update_counter_runtime_state(
                runtime_state.left_counter_state,
                left_counter_observation,
                supporting_end_signal=winner_visible,
            )
            right_counter = _update_counter_runtime_state(
                runtime_state.right_counter_state,
                right_counter_observation,
                supporting_end_signal=winner_visible,
            )
            runtime_state.last_left_counter = left_counter
            runtime_state.last_right_counter = right_counter

            parsed_roster = parse_rosters(frame, roi_config, ocr_backend)
            if parsed_roster:
                _observe_roster(roster_observations, parsed_roster)
                voted_roster = _materialize_roster_observations(roster_observations)
                if _roster_quality(voted_roster) > _roster_quality(candidate_roster):
                    candidate_roster = voted_roster
                if (
                    roster is not None
                    and _roster_signature(voted_roster) != _roster_signature(roster)
                    and _roster_quality(voted_roster) >= _roster_quality(roster)
                ):
                    roster = voted_roster
                    _emit_progress(
                        progress_callback,
                        f"[game] roster improved at {sample.timestamp_sec:.3f}s (quality={_roster_quality(roster)})",
                    )

            current_board_state = None
            if not setup_visible:
                current_board_state = parse_board_words(frame, roi_config, ocr_backend)
                if current_board_state is not None:
                    _observe_board_state(board_observations, current_board_state)
                    voted_board_state = _materialize_board_observations(board_observations)
                    if _board_quality(voted_board_state) > _board_quality(candidate_board_state):
                        candidate_board_state = voted_board_state
                    if (
                        board_state is not None
                        and _board_signature(voted_board_state) != _board_signature(board_state)
                        and _should_replace_locked_board_state(
                            current_board_state=board_state,
                            candidate_board_state=voted_board_state,
                            board_locked=runtime_state.board_locked,
                        )
                    ):
                        board_state = voted_board_state
                        _emit_progress(
                            progress_callback,
                            f"[game] board improved at {sample.timestamp_sec:.3f}s (quality={_board_quality(board_state)})",
                        )

            if roster is None and not setup_visible:
                locked_roster = (
                    parsed_roster if _roster_quality(parsed_roster) >= _roster_quality(candidate_roster) else candidate_roster
                )
                if _roster_is_usable(locked_roster):
                    roster = list(locked_roster)
                    _emit_progress(
                        progress_callback,
                        f"[game] roster locked at {sample.timestamp_sec:.3f}s (quality={_roster_quality(roster)})",
                    )

            if board_state is None and not setup_visible:
                locked_board_state = (
                    current_board_state if _board_quality(current_board_state) >= _board_quality(candidate_board_state) else candidate_board_state
                )
                if _board_is_usable(locked_board_state):
                    board_state = locked_board_state
                    runtime_state.board_locked = True
                    runtime_state.board_lock_timestamp = sample.timestamp_sec
                    _emit_progress(
                        progress_callback,
                        f"[game] board locked at {sample.timestamp_sec:.3f}s (quality={_board_quality(board_state)})",
                    )

            if game_start_sec is None and roster is not None and board_state is not None:
                game_start_sec = sample.timestamp_sec
                if board_reveal_baseline is None:
                    board_reveal_baseline = _capture_board_reveal_baseline(frame, roi_config, board_state)
                    previous_board_reveals = {
                        (cell.row, cell.col): None for cell in board_state.cells
                    }
                    pending_board_reveals = {}
                    runtime_state.visible_log.quiet_log_frame_count = 0
                if snapshots_path is not None and start_snapshot_path is None:
                    start_snapshot_path = str(
                        _save_snapshot(
                            frame,
                            snapshots_path,
                            timestamp_sec=sample.timestamp_sec,
                            label="start",
                        )
                    )
                _emit_progress(
                    progress_callback,
                    f"[game] parser ready at {sample.timestamp_sec:.3f}s",
                )

            log_frame = crop_roi(frame, log_roi)
            log_changed = log_has_changed(
                runtime_state.visible_log.previous_log_frame,
                log_frame,
                minimum_change_ratio=0.02,
            )
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
            frame_phase = _classify_frame_phase(top_banner_text)
            frame_observations = FrameObservations(
                timestamp_sec=sample.timestamp_sec,
                log_changed=log_changed,
                banner=banner_observation,
                left_counter=runtime_state.last_left_counter,
                right_counter=runtime_state.last_right_counter,
            )

            if roster is None or board_state is None:
                if snapshots_path is not None and first_preready_snapshot_path is None:
                    first_preready_snapshot_path = str(
                        _save_snapshot(
                            frame,
                            snapshots_path,
                            timestamp_sec=sample.timestamp_sec,
                            label="preready",
                        )
                    )
                if log_changed:
                    _emit_progress(
                        progress_callback,
                        f"[log-change] {sample.timestamp_sec:.3f}s before parser ready "
                        f"(roster_q={_roster_quality(roster or candidate_roster)}, "
                        f"board_q={_board_quality(board_state or candidate_board_state)})",
                    )
                runtime_state.visible_log.previous_log_frame = log_frame
                continue

            frame_new_events: list[ClueEvent | GuessEvent] = []
            current_visible_events: list[ClueEvent | GuessEvent] = []
            visible_clue_candidate: ClueEvent | None = None
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
                if log_changed:
                    _emit_progress(progress_callback, f"[log-change] {sample.timestamp_sec:.3f}s")
                else:
                    _emit_progress(progress_callback, f"[log-reprobe] {sample.timestamp_sec:.3f}s")
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
                visible_clue_candidate, current_visible_events = _split_visible_clue_and_trailing_events(
                    current_visible_events
                )
                frame_observations.visible_events = current_visible_events
                if log_changed:
                    runtime_state.visible_log.quiet_log_frame_count = 0

            if board_reveal_baseline is not None:
                previous_board_reveals, pending_board_reveals = _sync_board_reveals_from_history(
                    previous_reveals=previous_board_reveals,
                    pending_reveals=pending_board_reveals,
                    board_state=board_state,
                    history=runtime_state.history,
                )

            banner_clue_event = _build_banner_clue_event(
                clue_text=frame_observations.banner.clue_text,
                clue_count=frame_observations.banner.clue_count,
                top_banner_text=frame_observations.banner.top_banner_text,
                roster=roster,
                history=runtime_state.history,
                left_counter=frame_observations.left_counter,
                right_counter=frame_observations.right_counter,
                timestamp_sec=frame_observations.timestamp_sec,
            )
            current_banner_signature = None if banner_clue_event is None else (
                banner_clue_event.clue_text,
                banner_clue_event.clue_count,
            )
            if banner_clue_event is not None and not _phase_allows_new_clue_commit(frame_phase):
                active_window = runtime_state.active_clue_window
                if (
                    active_window is None
                    and not any(isinstance(event, ClueEvent) for event in runtime_state.history)
                ):
                    pass
                elif active_window is None or not _banner_supports_active_window_strict(active_window, banner_observation):
                    banner_clue_event = None
                    current_banner_signature = None
            promoted_clue_event: ClueEvent | None = None
            if visible_clue_candidate is not None and _visible_clue_requires_pending_promotion(
                runtime_state,
                visible_clue_candidate,
            ):
                promoted_clue_event = _observe_clue_candidate(
                    runtime_state,
                    visible_clue_candidate,
                    source="visible_log",
                    frame_phase=frame_phase,
                    banner_observation=banner_observation,
                    timestamp_sec=sample.timestamp_sec,
                    saw_same_frame_guesses=any(isinstance(event, GuessEvent) for event in current_visible_events[1:]),
                )
            elif visible_clue_candidate is None:
                _clear_stale_pending_clue(runtime_state, frame_phase=frame_phase, timestamp_sec=sample.timestamp_sec)

            if banner_clue_event is not None:
                promoted_from_banner = _observe_clue_candidate(
                    runtime_state,
                    banner_clue_event,
                    source="banner",
                    frame_phase=frame_phase,
                    banner_observation=banner_observation,
                    timestamp_sec=sample.timestamp_sec,
                    saw_same_frame_guesses=False,
                )
                if promoted_clue_event is None:
                    promoted_clue_event = promoted_from_banner

            if (
                visible_clue_candidate is not None
                and _visible_clue_requires_pending_promotion(runtime_state, visible_clue_candidate)
                and (
                    promoted_clue_event is None
                    or not _clue_events_equivalent(promoted_clue_event, visible_clue_candidate)
                )
            ):
                current_visible_events = []
                frame_observations.visible_events = []

            if (
                promoted_clue_event is not None
                and _find_matching_clue_index_anywhere(runtime_state.history, promoted_clue_event) is None
                and not _is_stale_banner_residue(runtime_state.history, promoted_clue_event)
            ):
                runtime_state.history.append(promoted_clue_event)
                frame_new_events.append(promoted_clue_event)
                _update_active_clue_window(
                    runtime_state,
                    source="banner" if banner_clue_event is not None else "visible_log",
                    events=[promoted_clue_event],
                    timestamp_sec=sample.timestamp_sec,
                )
            if current_banner_signature is not None:
                runtime_state.banner_signature = current_banner_signature

            new_events = _commit_visible_log_events(
                runtime_state,
                current_visible_events,
                timestamp_sec=sample.timestamp_sec,
            )
            frame_new_events.extend(new_events)
            if new_events:
                _update_active_clue_window(
                    runtime_state,
                    source="visible_log",
                    events=new_events,
                    timestamp_sec=sample.timestamp_sec,
                )

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
                    allow_board_reveal_scan=(
                        runtime_state.visible_log.quiet_log_frame_count >= 1
                        and _phase_allows_board_reveal_scan(frame_phase)
                        and runtime_state.active_clue_window is not None
                    ),
                )
                frame_observations.board_guess_events = new_board_guess_events
                runtime_state.history, _, appended_board_events = _append_new_visible_events(
                    runtime_state.history,
                    [],
                    new_board_guess_events,
                )
                frame_new_events.extend(appended_board_events)

            if frame_new_events:
                snapshot_path = None
                if snapshots_path is not None:
                    snapshot_path = str(
                        save_annotated_log_event_snapshot(
                            frame,
                            roi_config,
                            ocr_backend,
                            snapshots_path,
                            timestamp_sec=sample.timestamp_sec,
                        )
                    )
                change_events.append(
                    SmokeChangeEvent(
                        timestamp_sec=sample.timestamp_sec,
                        new_events=[event.model_dump(mode="json") for event in frame_new_events],
                        left_counter=runtime_state.last_left_counter,
                        right_counter=runtime_state.last_right_counter,
                        snapshot_path=snapshot_path,
                    )
                )
                if fps is None:
                    active_until_sec = max(active_until_sec, sample.timestamp_sec + active_window_sec)
                clue_count = sum(1 for event in frame_new_events if isinstance(event, ClueEvent))
                guess_count = sum(1 for event in frame_new_events if isinstance(event, GuessEvent))
                _emit_progress(
                    progress_callback,
                    f"[events] {sample.timestamp_sec:.3f}s emitted {len(frame_new_events)} new event(s) "
                    f"(clues={clue_count}, guesses={guess_count})",
                )
            elif log_changed:
                _emit_progress(
                    progress_callback,
                    f"[log-change] {sample.timestamp_sec:.3f}s no semantic delta",
                )
            runtime_state.visible_log.previous_log_frame = log_frame

            confirmed_assassin = _observe_pending_assassin_guess(
                runtime_state,
                evidence_events=[
                    event
                    for event in (
                        list(frame_observations.visible_events)
                        + list(frame_observations.board_guess_events)
                        + [event for event in frame_new_events if isinstance(event, GuessEvent)]
                    )
                    if isinstance(event, GuessEvent)
                ],
                frame_phase=frame_phase,
                winner_visible=winner_visible,
                timestamp_sec=sample.timestamp_sec,
            )
            if confirmed_assassin is not None:
                stop_reason = "assassin"
                _emit_progress(progress_callback, f"[stop] assassin at {sample.timestamp_sec:.3f}s")
                should_stop = True
                break

            if runtime_state.left_counter_state.confirmed_zero or runtime_state.right_counter_state.confirmed_zero:
                stop_reason = "counter_zero"
                _emit_progress(progress_callback, f"[stop] counter_zero at {sample.timestamp_sec:.3f}s")
                should_stop = True
                break
            if winner_visible:
                stop_reason = "winner_banner"
                _emit_progress(progress_callback, f"[stop] winner_banner at {sample.timestamp_sec:.3f}s")
                should_stop = True
                break
            if has_play_next_game(frame, roi_config, ocr_backend):
                stop_reason = "play_next"
                _emit_progress(progress_callback, f"[stop] play_next at {sample.timestamp_sec:.3f}s")
                should_stop = True
                break

        if not yielded_any_frames:
            _emit_progress(progress_callback, f"[chunk] no frames yielded at {window_start_sec:.3f}s")
            break
        window_start_sec += window_duration

    game_record = None
    review_items: list[dict[str, object]] = []
    if roster is not None and board_state is not None:
        reconstruction = reconstruct_game(
            runtime_state.history,
            roster,
            left_counter=runtime_state.last_left_counter,
            right_counter=runtime_state.last_right_counter,
        )
        game_record = build_game_record(
            vod_id=vod_id,
            game_index=game_index,
            start_sec=game_start_sec or start_sec,
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

    return SmokeParseResult(
        vod_url=vod_url,
        requested_start_sec=start_sec,
        stop_sec=stop_sec,
        stop_reason=stop_reason,
        game_record=game_record,
        review_items=review_items,
        roster=roster or candidate_roster,
        board_words=[] if (board_state or candidate_board_state) is None else (board_state or candidate_board_state).words,
        change_events=change_events,
        first_frame_snapshot_path=first_frame_snapshot_path,
        first_preready_snapshot_path=first_preready_snapshot_path,
        roi_probe_overlay_path=roi_probe_artifacts.overlay_path,
        roi_probe_crop_paths=roi_probe_artifacts.crop_paths,
        start_snapshot_path=start_snapshot_path,
    )


def _emit_progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _save_snapshot(
    frame: np.ndarray,
    output_dir: Path,
    *,
    timestamp_sec: float,
    label: str = "frame",
) -> Path:
    filename = f"smoke-{label}-{timestamp_sec:010.3f}.png"
    output_path = output_dir / filename
    Image.fromarray(frame[:, :, ::-1]).save(output_path)
    return output_path


def _update_counter_runtime_state(
    counter_state: CounterRuntimeState,
    observation,
    *,
    supporting_end_signal: bool,
) -> int | None:
    counter_state.last_observation = observation
    value = observation.value
    if value is None:
        counter_state.confirmed_zero = False
        return counter_state.stable_value

    previous_stable_value = counter_state.stable_value
    if value == 0:
        counter_state.zero_streak += 1
        strong_zero = (
            observation.raw_text.strip() == "0"
            and observation.confidence >= 0.985
            and (observation.variant_name or "") in {"raw", "grayscale_upscale"}
        )
        zero_confirmed = supporting_end_signal or counter_state.zero_streak >= 2 or (
            strong_zero and (previous_stable_value in {None, 0, 1})
        )
        counter_state.confirmed_zero = zero_confirmed
        if zero_confirmed:
            counter_state.stable_value = 0
            return 0
        return previous_stable_value

    counter_state.zero_streak = 0
    counter_state.confirmed_zero = False
    counter_state.stable_value = value
    return value
