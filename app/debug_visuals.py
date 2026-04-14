"""Debug visual artifact helpers for smoke parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.board_parser import crop_word_strip, estimate_board_cell_boxes
from app.models import ImageBoundingBox, OCRDetection
from app.ocr import OCRBackend, collapse_whitespace, prepare_ocr_image
from app.roi_config import ROIConfig, crop_roi, render_roi_map


@dataclass(frozen=True, slots=True)
class ROIProbeArtifacts:
    overlay_path: str | None
    crop_paths: dict[str, str]


_ROI_LABELS: tuple[tuple[str, str], ...] = (
    ("blue_operatives_region", "blue-operatives"),
    ("blue_spymaster_region", "blue-spymaster"),
    ("left_counter_region", "left-counter"),
    ("red_operatives_region", "red-operatives"),
    ("red_spymaster_region", "red-spymaster"),
    ("right_counter_region", "right-counter"),
    ("board_region", "board"),
    ("game_log_content_region", "game-log"),
)

_OVERLAY_REGION_OVERRIDES: dict[str, str] = {
    "blue_operatives_region": "blue_operatives_overlay_region",
    "red_operatives_region": "red_operatives_overlay_region",
    "board_region": "board_overlay_region",
}


def save_roi_probe(
    frame: np.ndarray,
    roi_config: ROIConfig,
    ocr_backend: OCRBackend,
    output_dir: Path,
    *,
    timestamp_sec: float,
) -> ROIProbeArtifacts:
    """Save one full-frame ROI overlay and OCR-ready crop artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / f"smoke-roi-overlay-{timestamp_sec:010.3f}.png"
    crop_paths: dict[str, str] = {}

    full_frame = Image.fromarray(frame[:, :, ::-1])
    draw = ImageDraw.Draw(full_frame)
    roi_map = render_roi_map(
        roi_config,
        frame_width=frame.shape[1],
        frame_height=frame.shape[0],
    )

    for region_name, label in _ROI_LABELS:
        overlay_region_name = _OVERLAY_REGION_OVERRIDES.get(region_name, region_name)
        pixel_roi = roi_map.get(overlay_region_name) or roi_map.get(region_name)
        if pixel_roi is None:
            continue
        _draw_box(draw, pixel_roi, outline="#00E5FF", label=label)
        if region_name == "board_region":
            _save_board_probe(frame, roi_config, ocr_backend, output_dir, timestamp_sec, crop_paths)
            continue

        profile, hint = _profile_for_region(region_name)
        prepared = prepare_ocr_image(crop_roi(frame, roi_config.require(region_name)), profile=profile)
        detections = list(ocr_backend.detect_text(prepared, hint=hint))
        crop_path = output_dir / f"smoke-probe-{label}-{timestamp_sec:010.3f}.png"
        _annotate_crop(prepared, detections).save(crop_path)
        crop_paths[label] = str(crop_path)

    full_frame.save(overlay_path)
    return ROIProbeArtifacts(
        overlay_path=str(overlay_path),
        crop_paths=crop_paths,
    )


def save_annotated_log_event_snapshot(
    frame: np.ndarray,
    roi_config: ROIConfig,
    ocr_backend: OCRBackend,
    output_dir: Path,
    *,
    timestamp_sec: float,
) -> Path:
    """Save a full-frame event snapshot with ROI and game-log OCR boxes."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"smoke-log-{timestamp_sec:010.3f}.png"
    full_frame = Image.fromarray(frame[:, :, ::-1])
    draw = ImageDraw.Draw(full_frame)
    roi_map = render_roi_map(
        roi_config,
        frame_width=frame.shape[1],
        frame_height=frame.shape[0],
    )

    region_name = "game_log_content_region" if "game_log_content_region" in roi_map else "game_log_region"
    log_roi = roi_map[region_name]
    _draw_box(draw, log_roi, outline="#00E5FF", label="game-log")

    raw_log_crop = crop_roi(frame, roi_config.require(region_name))
    prepared_log_crop = prepare_ocr_image(raw_log_crop, profile="game_log")
    detections = list(ocr_backend.detect_text(prepared_log_crop, hint="debug:game_log"))
    mapped_detections = _map_detections_to_frame(
        detections,
        prepared_shape=prepared_log_crop.shape,
        raw_shape=raw_log_crop.shape,
        offset_left=log_roi.left,
        offset_top=log_roi.top,
    )
    for detection in mapped_detections:
        _draw_box(draw, detection.box, outline="#00FF66", label=collapse_whitespace(detection.text))

    full_frame.save(output_path)
    return output_path


def _save_board_probe(
    frame: np.ndarray,
    roi_config: ROIConfig,
    ocr_backend: OCRBackend,
    output_dir: Path,
    timestamp_sec: float,
    crop_paths: dict[str, str],
) -> None:
    board_frame = crop_roi(frame, roi_config.require("board_region"))
    board_image = Image.fromarray(board_frame[:, :, ::-1])
    draw = ImageDraw.Draw(board_image)
    boxes = estimate_board_cell_boxes(board_frame)
    for box in boxes:
        _draw_box(draw, box, outline="#FFAA00")
    board_grid_path = output_dir / f"smoke-probe-board-grid-{timestamp_sec:010.3f}.png"
    board_image.save(board_grid_path)
    crop_paths["board-grid"] = str(board_grid_path)

    sample_indices = ((0, 0), (0, 1), (0, 2))
    for row, col in sample_indices:
        index = (row * 5) + col
        box = boxes[index]
        word_strip = crop_word_strip(board_frame, box)
        prepared = prepare_ocr_image(word_strip, profile="board_word")
        detections = list(ocr_backend.detect_text(prepared, hint=f"probe:board:{row}:{col}"))
        cell_path = output_dir / f"smoke-probe-board-cell-{row}-{col}-{timestamp_sec:010.3f}.png"
        _annotate_crop(prepared, detections).save(cell_path)
        crop_paths[f"board-cell-{row}-{col}"] = str(cell_path)


def _profile_for_region(region_name: str) -> tuple[str, str]:
    if region_name.endswith("counter_region"):
        return "counter", f"probe:{region_name.replace('_region', '')}"
    if "operatives" in region_name:
        base = "blue:operative:explicit" if region_name.startswith("blue") else "red:operative:explicit"
        return "name_strip", f"probe:{base}"
    if "spymaster" in region_name:
        base = "blue:spymaster:explicit" if region_name.startswith("blue") else "red:spymaster:explicit"
        return "name_strip", f"probe:{base}"
    if "top_banner" in region_name or "end_banner" in region_name:
        return "banner", f"probe:{region_name.replace('_region', '')}"
    if "game_log" in region_name:
        return "game_log", "probe:game_log"
    return "generic", f"probe:{region_name.replace('_region', '')}"


def _annotate_crop(image: np.ndarray, detections: list[OCRDetection]) -> Image.Image:
    pil_image = Image.fromarray(image[:, :, ::-1])
    draw = ImageDraw.Draw(pil_image)
    for detection in detections:
        _draw_box(draw, detection.box, outline="#00FF66", label=collapse_whitespace(detection.text))
    return pil_image


def _map_detections_to_frame(
    detections: list[OCRDetection],
    *,
    prepared_shape: tuple[int, ...],
    raw_shape: tuple[int, ...],
    offset_left: int,
    offset_top: int,
) -> list[OCRDetection]:
    prepared_height, prepared_width = prepared_shape[:2]
    raw_height, raw_width = raw_shape[:2]
    if prepared_height <= 0 or prepared_width <= 0:
        return []
    scale_x = raw_width / prepared_width
    scale_y = raw_height / prepared_height
    mapped: list[OCRDetection] = []
    for detection in detections:
        mapped.append(
            detection.model_copy(
                update={
                    "box": ImageBoundingBox(
                        left=offset_left + int(round(detection.box.left * scale_x)),
                        top=offset_top + int(round(detection.box.top * scale_y)),
                        right=offset_left + int(round(detection.box.right * scale_x)),
                        bottom=offset_top + int(round(detection.box.bottom * scale_y)),
                    )
                }
            )
        )
    return mapped


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: ImageBoundingBox,
    *,
    outline: str,
    label: str | None = None,
) -> None:
    draw.rectangle((box.left, box.top, box.right, box.bottom), outline=outline, width=3)
    if label:
        text_x = box.left + 2
        text_y = max(0, box.top - 14)
        draw.text((text_x, text_y), label, fill=outline)
