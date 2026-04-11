"""Deterministic segment detection for finding Codenames sections in a VOD tail."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.models import SegmentBounds, VodMeta
from app.roi_config import ROIConfig
from app.vod_source import FrameSample, VodSource


@dataclass(frozen=True, slots=True)
class FrameSignal:
    """Boolean cues extracted from a frame for Codenames detection."""

    has_blue_team_text: bool = False
    has_red_team_text: bool = False
    has_game_log_text: bool = False
    has_setup_phrase: bool = False
    has_board_grid: bool = False
    has_side_panels: bool = False


@dataclass(frozen=True, slots=True)
class WindowSummary:
    """Aggregated classification summary for a sampled time window."""

    start_sec: float
    end_sec: float
    best_frame_score: float
    positive_frame_count: int
    total_frame_count: int
    threshold: float

    @property
    def is_positive(self) -> bool:
        return self.best_frame_score >= self.threshold


@dataclass(frozen=True, slots=True)
class SegmentFinderConfig:
    """Configuration for deterministic tail-backtracking segment detection."""

    tail_scan_hours: float = 3.0
    tail_padding_sec: float = 2.0
    window_duration_sec: float = 90.0
    backward_step_sec: float = 60.0
    sample_fps: float = 0.2
    positive_score_threshold: float = 0.65
    consecutive_negative_windows: int = 3
    minimum_positive_windows: int = 2
    refine_step_sec: float = 5.0
    refine_window_duration_sec: float = 30.0


class FrameSignalExtractor(Protocol):
    """Callable protocol for extracting deterministic signals from sampled frames."""

    def __call__(self, frame_sample: FrameSample, roi_config: ROIConfig) -> FrameSignal:
        ...


def score_frame_signal(signal: FrameSignal) -> float:
    """Score how strongly a frame resembles the Codenames layout."""

    score = 0.0
    if signal.has_blue_team_text:
        score += 0.18
    if signal.has_red_team_text:
        score += 0.18
    if signal.has_game_log_text:
        score += 0.18
    if signal.has_board_grid:
        score += 0.22
    if signal.has_side_panels:
        score += 0.14
    if signal.has_setup_phrase:
        score += 0.10
    if signal.has_blue_team_text and signal.has_red_team_text and signal.has_board_grid:
        score += 0.10
    return min(score, 1.0)


def classify_window(
    vod_meta: VodMeta,
    vod_source: VodSource,
    roi_config: ROIConfig,
    signal_extractor: FrameSignalExtractor,
    *,
    window_start_sec: float,
    window_duration_sec: float,
    sample_fps: float,
    positive_score_threshold: float,
) -> WindowSummary:
    """Classify a window as positive or negative for Codenames content."""

    bounded_start = max(0.0, min(window_start_sec, vod_meta.duration_seconds))
    bounded_duration = min(window_duration_sec, max(0.0, vod_meta.duration_seconds - bounded_start))

    best_frame_score = 0.0
    positive_frame_count = 0
    total_frame_count = 0

    if bounded_duration <= 0:
        return WindowSummary(
            start_sec=bounded_start,
            end_sec=bounded_start,
            best_frame_score=0.0,
            positive_frame_count=0,
            total_frame_count=0,
            threshold=positive_score_threshold,
        )

    for frame_sample in vod_source.iter_window_frames(
        vod_meta.url,
        start_sec=bounded_start,
        duration_sec=bounded_duration,
        fps=sample_fps,
    ):
        total_frame_count += 1
        score = score_frame_signal(signal_extractor(frame_sample, roi_config))
        best_frame_score = max(best_frame_score, score)
        if score >= positive_score_threshold:
            positive_frame_count += 1

    return WindowSummary(
        start_sec=bounded_start,
        end_sec=min(vod_meta.duration_seconds, bounded_start + bounded_duration),
        best_frame_score=best_frame_score,
        positive_frame_count=positive_frame_count,
        total_frame_count=total_frame_count,
        threshold=positive_score_threshold,
    )


def find_codenames_segment(
    vod_meta: VodMeta,
    vod_source: VodSource,
    roi_config: ROIConfig,
    signal_extractor: FrameSignalExtractor,
    cfg: SegmentFinderConfig | None = None,
) -> SegmentBounds | None:
    """Backtrack from the VOD tail and return the earliest Codenames segment boundary."""

    config = cfg or SegmentFinderConfig()
    scan_end = max(0.0, vod_meta.duration_seconds - config.tail_padding_sec)
    scan_start = max(0.0, scan_end - (config.tail_scan_hours * 3600.0))

    latest_positive: WindowSummary | None = None
    earliest_positive: WindowSummary | None = None
    boundary_negative: WindowSummary | None = None
    positive_window_count = 0
    max_region_score = 0.0
    negative_run = 0

    for window_start in _iter_backward_window_starts(scan_start, scan_end, config.backward_step_sec):
        summary = classify_window(
            vod_meta,
            vod_source,
            roi_config,
            signal_extractor,
            window_start_sec=window_start,
            window_duration_sec=config.window_duration_sec,
            sample_fps=config.sample_fps,
            positive_score_threshold=config.positive_score_threshold,
        )
        max_region_score = max(max_region_score, summary.best_frame_score)

        if summary.is_positive:
            latest_positive = latest_positive or summary
            earliest_positive = summary
            positive_window_count += 1
            boundary_negative = None
            negative_run = 0
            continue

        if earliest_positive is None:
            continue

        if boundary_negative is None:
            boundary_negative = summary
        negative_run += 1
        if negative_run >= config.consecutive_negative_windows:
            break

    if latest_positive is None or earliest_positive is None:
        return None
    if positive_window_count < config.minimum_positive_windows:
        return None

    refinement_floor = scan_start if boundary_negative is None else boundary_negative.start_sec
    refined_start = refine_segment_start(
        vod_meta,
        vod_source,
        roi_config,
        signal_extractor,
        low_sec=refinement_floor,
        high_sec=earliest_positive.start_sec,
        cfg=config,
    )

    return SegmentBounds(
        vod_id=vod_meta.vod_id,
        start_sec=refined_start,
        end_sec=vod_meta.duration_seconds,
        confidence=max_region_score,
    )


def refine_segment_start(
    vod_meta: VodMeta,
    vod_source: VodSource,
    roi_config: ROIConfig,
    signal_extractor: FrameSignalExtractor,
    *,
    low_sec: float,
    high_sec: float,
    cfg: SegmentFinderConfig,
) -> float:
    """Linearly refine the boundary between a negative and positive region."""

    current = max(0.0, low_sec)
    bounded_high = max(current, high_sec)

    while current <= bounded_high:
        summary = classify_window(
            vod_meta,
            vod_source,
            roi_config,
            signal_extractor,
            window_start_sec=current,
            window_duration_sec=cfg.refine_window_duration_sec,
            sample_fps=cfg.sample_fps,
            positive_score_threshold=cfg.positive_score_threshold,
        )
        if summary.is_positive:
            return current
        current += cfg.refine_step_sec

    return bounded_high


def _iter_backward_window_starts(scan_start: float, scan_end: float, step_sec: float):
    current = scan_end
    while current >= scan_start:
        yield current
        current -= step_sec
