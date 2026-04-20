from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.infra.roi_config import (
    REQUIRED_ROI_NAMES,
    ROIConfig,
    NormalizedROI,
    crop_roi,
    denormalize_roi,
    load_roi_config,
    render_roi_map,
)


def make_roi_mapping() -> dict[str, dict[str, float]]:
    return {
        "left_team_panel": {"x": 0.00, "y": 0.05, "width": 0.18, "height": 0.60},
        "right_team_panel": {"x": 0.82, "y": 0.05, "width": 0.18, "height": 0.60},
        "board_region": {"x": 0.20, "y": 0.10, "width": 0.46, "height": 0.56},
        "game_log_region": {"x": 0.67, "y": 0.40, "width": 0.14, "height": 0.38},
        "left_counter_region": {"x": 0.05, "y": 0.01, "width": 0.06, "height": 0.06},
        "right_counter_region": {"x": 0.89, "y": 0.01, "width": 0.06, "height": 0.06},
        "top_banner_region": {"x": 0.18, "y": 0.00, "width": 0.48, "height": 0.09},
        "end_banner_region": {"x": 0.24, "y": 0.77, "width": 0.52, "height": 0.16},
    }


def test_load_roi_config_accepts_wrapped_regions_object() -> None:
    config_path = Path("config") / "_test_rois.json"
    try:
        config_path.write_text(
            json.dumps({"regions": make_roi_mapping()}),
            encoding="utf-8",
        )

        roi_config = load_roi_config(config_path)

        assert set(REQUIRED_ROI_NAMES).issubset(roi_config.regions)
    finally:
        if config_path.exists():
            config_path.unlink()


def test_roi_config_rejects_missing_required_region() -> None:
    incomplete = make_roi_mapping()
    incomplete.pop("game_log_region")

    with pytest.raises(ValueError, match="Missing required ROI regions"):
        ROIConfig.from_raw({"regions": incomplete})


def test_denormalize_and_crop_roi_scale_to_frame_size() -> None:
    roi = NormalizedROI(x=0.25, y=0.10, width=0.50, height=0.20)
    frame = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)

    pixel_roi = denormalize_roi(roi, frame_width=200, frame_height=100)
    cropped = crop_roi(frame, roi)

    assert (pixel_roi.left, pixel_roi.top, pixel_roi.right, pixel_roi.bottom) == (50, 10, 150, 30)
    assert cropped.shape == (20, 100, 3)


def test_render_roi_map_returns_pixel_rois_for_all_regions() -> None:
    roi_config = ROIConfig.from_raw(make_roi_mapping())

    roi_map = render_roi_map(roi_config, frame_width=1920, frame_height=1080)

    assert set(REQUIRED_ROI_NAMES).issubset(roi_map)
    assert roi_map["board_region"].width > 0
    assert roi_map["board_region"].height > 0
