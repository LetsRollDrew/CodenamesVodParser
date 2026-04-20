from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from app.core.models import VodMeta
from app.infra.roi_config import ROIConfig
from app.infra.vod_source import FrameSample
from app.vision.detection.segment_finder import (
    FrameSignal,
    SegmentFinderConfig,
    classify_window,
    find_codenames_segment,
    score_frame_signal,
)


def make_roi_config() -> ROIConfig:
    return ROIConfig.from_raw(
        {
            "left_team_panel": {"x": 0.00, "y": 0.05, "width": 0.18, "height": 0.60},
            "right_team_panel": {"x": 0.82, "y": 0.05, "width": 0.18, "height": 0.60},
            "board_region": {"x": 0.20, "y": 0.10, "width": 0.46, "height": 0.56},
            "game_log_region": {"x": 0.67, "y": 0.40, "width": 0.14, "height": 0.38},
            "left_counter_region": {"x": 0.05, "y": 0.01, "width": 0.06, "height": 0.06},
            "right_counter_region": {"x": 0.89, "y": 0.01, "width": 0.06, "height": 0.06},
            "top_banner_region": {"x": 0.18, "y": 0.00, "width": 0.48, "height": 0.09},
            "end_banner_region": {"x": 0.24, "y": 0.77, "width": 0.52, "height": 0.16}
        }
    )


class FakeVodSource:
    def __init__(self) -> None:
        self.calls: list[tuple[float, float, float]] = []

    def iter_window_frames(self, vod_url: str, start_sec: float, duration_sec: float, fps: float):
        del vod_url
        self.calls.append((start_sec, duration_sec, fps))
        yield FrameSample(
            timestamp_sec=start_sec,
            frame_bgr=np.zeros((1, 1, 3), dtype=np.uint8),
            frame_index_in_window=0,
        )


def make_vod_meta(duration_seconds: int = 600) -> VodMeta:
    return VodMeta(
        vod_id="vod-1",
        streamer_login="tiewhy",
        created_at=datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc),
        title="synthetic timeline",
        url="https://twitch.tv/videos/vod-1",
        duration_seconds=duration_seconds,
    )


def test_score_frame_signal_rewards_multiple_codenames_cues() -> None:
    positive_signal = FrameSignal(
        has_blue_team_text=True,
        has_red_team_text=True,
        has_game_log_text=True,
        has_board_grid=True,
        has_side_panels=True,
    )
    weak_signal = FrameSignal(has_game_log_text=True)

    assert score_frame_signal(positive_signal) > 0.8
    assert score_frame_signal(weak_signal) < 0.3


def test_classify_window_marks_positive_when_threshold_met() -> None:
    vod_source = FakeVodSource()
    roi_config = make_roi_config()
    vod_meta = make_vod_meta()

    def extractor(frame_sample: FrameSample, _roi_config: ROIConfig) -> FrameSignal:
        del frame_sample
        return FrameSignal(
            has_blue_team_text=True,
            has_red_team_text=True,
            has_game_log_text=True,
            has_board_grid=True,
        )

    summary = classify_window(
        vod_meta,
        vod_source,
        roi_config,
        extractor,
        window_start_sec=300.0,
        window_duration_sec=60.0,
        sample_fps=0.2,
        positive_score_threshold=0.65,
    )

    assert summary.is_positive is True
    assert summary.best_frame_score >= 0.65


def test_find_codenames_segment_refines_boundary_from_synthetic_timeline() -> None:
    vod_source = FakeVodSource()
    roi_config = make_roi_config()
    vod_meta = make_vod_meta(duration_seconds=600)
    cfg = SegmentFinderConfig(
        tail_scan_hours=1.0,
        tail_padding_sec=0.0,
        window_duration_sec=60.0,
        backward_step_sec=60.0,
        sample_fps=0.2,
        positive_score_threshold=0.65,
        consecutive_negative_windows=2,
        minimum_positive_windows=2,
        refine_step_sec=10.0,
        refine_window_duration_sec=10.0,
    )

    def extractor(frame_sample: FrameSample, _roi_config: ROIConfig) -> FrameSignal:
        if 240.0 <= frame_sample.timestamp_sec <= 420.0:
            return FrameSignal(
                has_blue_team_text=True,
                has_red_team_text=True,
                has_game_log_text=True,
                has_board_grid=True,
                has_side_panels=True,
            )
        return FrameSignal()

    segment = find_codenames_segment(vod_meta, vod_source, roi_config, extractor, cfg)

    assert segment is not None
    assert segment.start_sec == 240.0
    assert segment.end_sec == 600.0
    assert segment.confidence >= 0.65


def test_find_codenames_segment_rejects_single_positive_blip() -> None:
    vod_source = FakeVodSource()
    roi_config = make_roi_config()
    vod_meta = make_vod_meta(duration_seconds=600)
    cfg = SegmentFinderConfig(
        tail_scan_hours=1.0,
        tail_padding_sec=0.0,
        window_duration_sec=60.0,
        backward_step_sec=60.0,
        sample_fps=0.2,
        positive_score_threshold=0.65,
        consecutive_negative_windows=2,
        minimum_positive_windows=2,
        refine_step_sec=10.0,
        refine_window_duration_sec=10.0,
    )

    def extractor(frame_sample: FrameSample, _roi_config: ROIConfig) -> FrameSignal:
        if frame_sample.timestamp_sec == 360.0:
            return FrameSignal(
                has_blue_team_text=True,
                has_red_team_text=True,
                has_game_log_text=True,
                has_board_grid=True,
            )
        return FrameSignal()

    segment = find_codenames_segment(vod_meta, vod_source, roi_config, extractor, cfg)

    assert segment is None
