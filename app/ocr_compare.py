"""One-frame OCR backend comparison for saved smoke snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from app.board_parser import crop_word_strip, estimate_board_cell_boxes
from app.ocr import prepare_ocr_image
from app.ocr_backends import OCRBackendError, create_ocr_backend
from app.roi_config import ROIConfig, crop_roi, load_roi_config


class OCRTextDetection(BaseModel):
    text: str
    confidence: float = Field(ge=0.0)


class OCRVariantResult(BaseModel):
    texts: list[OCRTextDetection] = Field(default_factory=list)


class OCRBackendRegionResult(BaseModel):
    raw: OCRVariantResult
    prepared: OCRVariantResult


class OCRRegionCompareResult(BaseModel):
    crop_path: str | None = None
    prepared_crop_path: str | None = None
    paddle: OCRBackendRegionResult | None = None
    easyocr: OCRBackendRegionResult | None = None


class OCRCompareResult(BaseModel):
    frame_path: str
    roi_config_path: str
    regions: dict[str, OCRRegionCompareResult]


def _load_frame(frame_path: str | Path) -> np.ndarray:
    image = Image.open(frame_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    return rgb[:, :, ::-1].copy()


def _crop_role_text_band(section_frame: np.ndarray, *, role_name: str) -> np.ndarray:
    height = section_frame.shape[0]
    if height <= 0:
        return section_frame.copy()

    if role_name == "operative":
        top = int(round(height * 0.56))
        bottom = int(round(height * 0.98))
    else:
        top = int(round(height * 0.54))
        bottom = int(round(height * 0.94))

    top = max(min(top, height - 1), 0)
    bottom = max(min(bottom, height), top + 1)
    return section_frame[top:bottom, :].copy()


def _save_image(image: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 3 and image.shape[2] == 3:
        Image.fromarray(image[:, :, ::-1]).save(path)
    else:
        Image.fromarray(image).save(path)
    return str(path)


def _detect_variant(backend_name: str, backend: object | None, image: np.ndarray, *, hint: str) -> OCRVariantResult:
    if backend is None:
        return OCRVariantResult()
    detections = backend.detect_text(image, hint=hint)
    return OCRVariantResult(
        texts=[
            OCRTextDetection(text=str(detection.text), confidence=float(detection.confidence))
            for detection in detections
            if str(detection.text).strip()
        ]
    )


def _compare_region(
    *,
    label: str,
    raw_crop: np.ndarray,
    prepared_crop: np.ndarray,
    output_dir: Path | None,
    paddle_backend: object | None,
    easyocr_backend: object | None,
) -> OCRRegionCompareResult:
    crop_path = None
    prepared_crop_path = None
    if output_dir is not None:
        crop_path = _save_image(raw_crop, output_dir / f"{label}-raw.png")
        prepared_crop_path = _save_image(prepared_crop, output_dir / f"{label}-prepared.png")

    return OCRRegionCompareResult(
        crop_path=crop_path,
        prepared_crop_path=prepared_crop_path,
        paddle=OCRBackendRegionResult(
            raw=_detect_variant("paddle", paddle_backend, raw_crop, hint=f"compare:{label}:raw"),
            prepared=_detect_variant("paddle", paddle_backend, prepared_crop, hint=f"compare:{label}:prepared"),
        )
        if paddle_backend is not None
        else None,
        easyocr=OCRBackendRegionResult(
            raw=_detect_variant("easyocr", easyocr_backend, raw_crop, hint=f"compare:{label}:raw"),
            prepared=_detect_variant("easyocr", easyocr_backend, prepared_crop, hint=f"compare:{label}:prepared"),
        )
        if easyocr_backend is not None
        else None,
    )


def compare_ocr_on_frame(
    *,
    frame_path: str | Path,
    roi_config: ROIConfig,
    paddle_device: str | None,
    easyocr_gpu: bool,
    output_dir: str | Path | None,
) -> OCRCompareResult:
    frame = _load_frame(frame_path)
    compare_dir = Path(output_dir) if output_dir is not None else None

    paddle_backend = None
    easyocr_backend = None
    try:
        paddle_backend = create_ocr_backend("paddle", **({"device": paddle_device} if paddle_device else {}))
    except OCRBackendError:
        paddle_backend = None
    try:
        easyocr_backend = create_ocr_backend("easyocr", gpu=easyocr_gpu)
    except OCRBackendError:
        easyocr_backend = None

    regions: dict[str, OCRRegionCompareResult] = {}

    blue_ops = crop_roi(frame, roi_config.require("blue_operatives_region"))
    blue_ops_text = _crop_role_text_band(blue_ops, role_name="operative")
    regions["blue-operatives"] = _compare_region(
        label="blue-operatives",
        raw_crop=blue_ops_text,
        prepared_crop=prepare_ocr_image(blue_ops_text, profile="name_strip"),
        output_dir=compare_dir,
        paddle_backend=paddle_backend,
        easyocr_backend=easyocr_backend,
    )

    blue_spy = crop_roi(frame, roi_config.require("blue_spymaster_region"))
    blue_spy_text = _crop_role_text_band(blue_spy, role_name="spymaster")
    regions["blue-spymaster"] = _compare_region(
        label="blue-spymaster",
        raw_crop=blue_spy_text,
        prepared_crop=prepare_ocr_image(blue_spy_text, profile="name_strip"),
        output_dir=compare_dir,
        paddle_backend=paddle_backend,
        easyocr_backend=easyocr_backend,
    )

    left_counter = crop_roi(frame, roi_config.require("left_counter_region"))
    regions["left-counter"] = _compare_region(
        label="left-counter",
        raw_crop=left_counter,
        prepared_crop=prepare_ocr_image(left_counter, profile="counter"),
        output_dir=compare_dir,
        paddle_backend=paddle_backend,
        easyocr_backend=easyocr_backend,
    )

    red_ops = crop_roi(frame, roi_config.require("red_operatives_region"))
    red_ops_text = _crop_role_text_band(red_ops, role_name="operative")
    regions["red-operatives"] = _compare_region(
        label="red-operatives",
        raw_crop=red_ops_text,
        prepared_crop=prepare_ocr_image(red_ops_text, profile="name_strip"),
        output_dir=compare_dir,
        paddle_backend=paddle_backend,
        easyocr_backend=easyocr_backend,
    )

    red_spy = crop_roi(frame, roi_config.require("red_spymaster_region"))
    red_spy_text = _crop_role_text_band(red_spy, role_name="spymaster")
    regions["red-spymaster"] = _compare_region(
        label="red-spymaster",
        raw_crop=red_spy_text,
        prepared_crop=prepare_ocr_image(red_spy_text, profile="name_strip"),
        output_dir=compare_dir,
        paddle_backend=paddle_backend,
        easyocr_backend=easyocr_backend,
    )

    right_counter = crop_roi(frame, roi_config.require("right_counter_region"))
    regions["right-counter"] = _compare_region(
        label="right-counter",
        raw_crop=right_counter,
        prepared_crop=prepare_ocr_image(right_counter, profile="counter"),
        output_dir=compare_dir,
        paddle_backend=paddle_backend,
        easyocr_backend=easyocr_backend,
    )

    board_frame = crop_roi(frame, roi_config.require("board_region"))
    board_boxes = estimate_board_cell_boxes(board_frame)
    for row, col in ((0, 0), (0, 1), (0, 2)):
        label = f"board-cell-{row}-{col}"
        word_strip = crop_word_strip(board_frame, board_boxes[(row * 5) + col])
        regions[label] = _compare_region(
            label=label,
            raw_crop=word_strip,
            prepared_crop=prepare_ocr_image(word_strip, profile="board_word"),
            output_dir=compare_dir,
            paddle_backend=paddle_backend,
            easyocr_backend=easyocr_backend,
        )

    return OCRCompareResult(
        frame_path=str(frame_path),
        roi_config_path="",
        regions=regions,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare OCR backends on a saved smoke frame.")
    parser.add_argument("--frame-path", required=True)
    parser.add_argument("--roi-config", default="config/rois.example.json")
    parser.add_argument("--output-dir", default="debug_snapshots/ocr-compare/latest")
    parser.add_argument("--paddle-device", default=None)
    parser.add_argument("--easyocr-gpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    roi_config = load_roi_config(args.roi_config)
    result = compare_ocr_on_frame(
        frame_path=args.frame_path,
        roi_config=roi_config,
        paddle_device=args.paddle_device,
        easyocr_gpu=args.easyocr_gpu,
        output_dir=args.output_dir,
    )
    payload = result.model_dump(mode="json")
    payload["roi_config_path"] = args.roi_config
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
