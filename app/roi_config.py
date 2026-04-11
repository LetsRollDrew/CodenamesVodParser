"""Normalized ROI configuration utilities for frame-based parsers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

REQUIRED_ROI_NAMES = (
    "left_team_panel",
    "right_team_panel",
    "board_region",
    "game_log_region",
    "left_counter_region",
    "right_counter_region",
    "top_banner_region",
    "end_banner_region",
)


class NormalizedROI(BaseModel):
    """A region of interest stored as percentages of frame width and height."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "NormalizedROI":
        if self.x + self.width > 1.0:
            raise ValueError("ROI x + width must be <= 1.0")
        if self.y + self.height > 1.0:
            raise ValueError("ROI y + height must be <= 1.0")
        return self


@dataclass(frozen=True, slots=True)
class PixelROI:
    """A ROI expressed in clamped pixel coordinates."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


class ROIConfig(BaseModel):
    """Named ROI collection for a specific stream layout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    regions: dict[str, NormalizedROI]

    @model_validator(mode="after")
    def _validate_required_regions(self) -> "ROIConfig":
        missing = [name for name in REQUIRED_ROI_NAMES if name not in self.regions]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing required ROI regions: {joined}")
        return self

    @classmethod
    def from_raw(cls, raw_value: object) -> "ROIConfig":
        if isinstance(raw_value, dict) and "regions" in raw_value:
            return cls.model_validate(raw_value)
        if isinstance(raw_value, dict):
            return cls.model_validate({"regions": raw_value})
        raise TypeError("ROI config must be a mapping or a {'regions': ...} object")

    def require(self, name: str) -> NormalizedROI:
        try:
            return self.regions[name]
        except KeyError as exc:
            raise KeyError(f"Unknown ROI region: {name}") from exc


def load_roi_config(path: str | Path) -> ROIConfig:
    """Load ROI config from a JSON file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)
    return ROIConfig.from_raw(raw_data)


def denormalize_roi(
    roi: NormalizedROI,
    frame_width: int,
    frame_height: int,
    *,
    clamp: bool = True,
) -> PixelROI:
    """Convert a normalized ROI into pixel coordinates for a specific frame size."""

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width and frame_height must be > 0")

    left = int(round(roi.x * frame_width))
    top = int(round(roi.y * frame_height))
    right = int(round((roi.x + roi.width) * frame_width))
    bottom = int(round((roi.y + roi.height) * frame_height))

    if clamp:
        left = min(max(left, 0), frame_width)
        top = min(max(top, 0), frame_height)
        right = min(max(right, 0), frame_width)
        bottom = min(max(bottom, 0), frame_height)

    if right <= left:
        right = min(frame_width, left + 1)
    if bottom <= top:
        bottom = min(frame_height, top + 1)

    return PixelROI(left=left, top=top, right=right, bottom=bottom)


def crop_roi(frame: np.ndarray, roi: NormalizedROI, *, clamp: bool = True) -> np.ndarray:
    """Crop a normalized ROI from a frame and return a copy."""

    if frame.ndim < 2:
        raise ValueError("frame must have at least two dimensions")

    height, width = frame.shape[:2]
    pixel_roi = denormalize_roi(roi, width, height, clamp=clamp)
    return frame[pixel_roi.top : pixel_roi.bottom, pixel_roi.left : pixel_roi.right].copy()


def render_roi_map(
    roi_config: ROIConfig,
    *,
    frame_width: int,
    frame_height: int,
) -> dict[str, PixelROI]:
    """Convert the full ROI config into pixel coordinates for a given frame size."""

    return {
        name: denormalize_roi(region, frame_width, frame_height)
        for name, region in roi_config.regions.items()
    }
