"""Extract and score per-guess selector chips from a scheduled game reconstruction."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.gamelog_parser import _group_detections_by_row
from app.ocr import collapse_whitespace, detect_text_with_fallback, prepare_ocr_image
from app.ocr_backends import create_ocr_backend
from app.roi_config import crop_roi, load_roi_config
from app.roster_parser import (
    _crop_role_text_band,
    _estimate_avatar_box,
    _filter_name_detections,
    _offset_detection_box,
    average_hash,
)
from app.vod_source import VodSource


@dataclass(frozen=True, slots=True)
class ReferenceIdentity:
    display_name: str
    team_color: str
    role: str
    avatar_hash: str
    avatar_image: np.ndarray
    name_image: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--roi-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ffmpeg-path", default="ffmpeg")
    parser.add_argument("--ocr-backend", default="easyocr")
    parser.add_argument("--ocr-device", default="gpu:0")
    parser.add_argument("--window-sec", type=float, default=1.5)
    parser.add_argument("--fps", type=float, default=4.0)
    return parser.parse_args()


def _normalized_device_is_gpu(device: str | None) -> bool:
    if device is None:
        return False
    return device.strip().lower().startswith("gpu")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _extract_reference_identities(
    *,
    frame: np.ndarray,
    roi_config_path: Path,
    ocr_backend,
) -> dict[str, list[ReferenceIdentity]]:
    roi_config = load_roi_config(roi_config_path)
    specs = (
        ("blue_operatives_region", "blue", "operative"),
        ("red_operatives_region", "red", "operative"),
    )
    identities: dict[str, list[ReferenceIdentity]] = {"blue": [], "red": []}
    for region_name, team_color, role in specs:
        region = roi_config.require(region_name)
        section = crop_roi(frame, region)
        text_frame, top_offset = _crop_role_text_band(
            section,
            role=__import__("app.models", fromlist=["PlayerRole"]).PlayerRole.OPERATIVE,
        )
        detections = _filter_name_detections(
            detect_text_with_fallback(
                ocr_backend,
                text_frame,
                prepared_image=prepare_ocr_image(text_frame, profile="name_strip"),
                hint=f"selector_refs:{region_name}",
            )
        )
        adjusted = [_offset_detection_box(detection, top_offset=top_offset) for detection in detections]
        for detection in sorted(adjusted, key=lambda item: item.box.center_x):
            display_name = collapse_whitespace(detection.text)
            if not display_name:
                continue
            avatar_box = _estimate_avatar_box(section.shape, detection.box)
            avatar_image = section[
                avatar_box.top : avatar_box.bottom,
                avatar_box.left : avatar_box.right,
            ].copy()
            name_image = section[
                detection.box.top : detection.box.bottom,
                detection.box.left : detection.box.right,
            ].copy()
            identities[team_color].append(
                ReferenceIdentity(
                    display_name=display_name,
                    team_color=team_color,
                    role=role,
                    avatar_hash=average_hash(avatar_image),
                    avatar_image=avatar_image,
                    name_image=name_image,
                )
            )
    return identities


def _find_matching_detection(log_frame: np.ndarray, ocr_backend, target_word: str):
    detections = [
        detection.model_copy(update={"text": collapse_whitespace(detection.text)})
        for detection in detect_text_with_fallback(
            ocr_backend,
            log_frame,
            prepared_image=prepare_ocr_image(log_frame, profile="game_log"),
            hint=f"selector_probe:{target_word}",
        )
    ]
    rows = _group_detections_by_row(sorted(detections, key=lambda item: (item.box.center_y, item.box.left)))
    best_detection = None
    best_score = 0.0
    best_row_index = -1
    for row_index, row in enumerate(rows):
        for detection in row:
            score = SequenceMatcher(
                a=collapse_whitespace(detection.text).upper(),
                b=target_word.upper(),
            ).ratio()
            if score > best_score:
                best_score = score
                best_detection = detection
                best_row_index = row_index
    return best_detection, best_score, best_row_index


def _crop_selector_chip(log_frame: np.ndarray, detection_box) -> np.ndarray:
    height, width = log_frame.shape[:2]
    chip_left = max(0, int(detection_box.left - detection_box.height * 1.8))
    chip_right = min(width, int(detection_box.right + detection_box.height * 0.2))
    chip_top = max(0, int(detection_box.top - detection_box.height * 0.25))
    chip_bottom = min(height, int(detection_box.bottom + detection_box.height * 1.3))
    return log_frame[chip_top:chip_bottom, chip_left:chip_right].copy()


def _average_crops(crops: list[np.ndarray]) -> np.ndarray:
    max_height = max(crop.shape[0] for crop in crops)
    max_width = max(crop.shape[1] for crop in crops)
    stack: list[np.ndarray] = []
    for crop in crops:
        canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        canvas[: crop.shape[0], : crop.shape[1]] = crop
        stack.append(canvas.astype(np.float32))
    return np.mean(stack, axis=0).astype(np.uint8)


def _extract_name_pill(chip_image: np.ndarray) -> np.ndarray:
    height, width = chip_image.shape[:2]
    return chip_image[int(height * 0.55) : height, 0 : max(1, int(width * 0.34))].copy()


def _extract_identity_block(chip_image: np.ndarray) -> np.ndarray:
    height, width = chip_image.shape[:2]
    return chip_image[0 : max(1, int(height * 0.9)), 0 : max(1, int(width * 0.38))].copy()


def _extract_bright_text_features(image: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 150), (180, 120, 255))
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return {
            "count": 0.0,
            "bbox_width_ratio": 0.0,
            "bbox_height_ratio": 0.0,
            "x_centroid_ratio": 0.0,
            "y_centroid_ratio": 0.0,
        }
    return {
        "count": float(len(xs)),
        "bbox_width_ratio": float(xs.max() - xs.min() + 1) / float(image.shape[1] or 1),
        "bbox_height_ratio": float(ys.max() - ys.min() + 1) / float(image.shape[0] or 1),
        "x_centroid_ratio": float(xs.mean()) / float(image.shape[1] or 1),
        "y_centroid_ratio": float(ys.mean()) / float(image.shape[0] or 1),
    }


def _feature_distance(left: dict[str, float], right: dict[str, float]) -> float:
    return (
        abs(left["bbox_width_ratio"] - right["bbox_width_ratio"]) * 2.4
        + abs(left["bbox_height_ratio"] - right["bbox_height_ratio"]) * 1.2
        + abs(left["x_centroid_ratio"] - right["x_centroid_ratio"]) * 2.0
        + abs(left["y_centroid_ratio"] - right["y_centroid_ratio"]) * 1.0
        + (abs(left["count"] - right["count"]) / max(1.0, right["count"])) * 0.5
    )


def _hamming_distance(left_hash: str, right_hash: str) -> int:
    if not left_hash or not right_hash or len(left_hash) != len(right_hash):
        return 999
    left_bits = bin(int(left_hash, 16))[2:].zfill(len(left_hash) * 4)
    right_bits = bin(int(right_hash, 16))[2:].zfill(len(right_hash) * 4)
    return sum(left_bit != right_bit for left_bit, right_bit in zip(left_bits, right_bits))


def _score_candidates(
    *,
    chip_image: np.ndarray,
    references: list[ReferenceIdentity],
) -> list[dict[str, Any]]:
    pill_image = _extract_name_pill(chip_image)
    identity_block = _extract_identity_block(chip_image)
    pill_features = _extract_bright_text_features(pill_image)
    identity_hash = average_hash(identity_block)
    scored: list[dict[str, Any]] = []
    for reference in references:
        ref_features = _extract_bright_text_features(reference.name_image)
        pill_distance = _feature_distance(pill_features, ref_features)
        avatar_distance = float(_hamming_distance(identity_hash, reference.avatar_hash))
        composite_score = (1.0 / (1.0 + pill_distance)) * 0.65 + (1.0 / (1.0 + avatar_distance)) * 0.35
        scored.append(
            {
                "display_name": reference.display_name,
                "pill_distance": round(pill_distance, 4),
                "avatar_hamming": int(avatar_distance),
                "composite_score": round(composite_score, 4),
            }
        )
    return sorted(scored, key=lambda item: item["composite_score"], reverse=True)


def attribute_selectors_from_analysis(
    *,
    analysis: dict[str, Any],
    analysis_reference_path: Path | None,
    roi_config_path: Path,
    output_dir: Path,
    ffmpeg_path: str,
    ocr_backend_name: str,
    ocr_device: str,
    window_sec: float,
    fps: float,
) -> dict[str, Any]:
    clip_reference = analysis["clip_path"]
    if isinstance(clip_reference, str) and (
        clip_reference.startswith("https://") or clip_reference.startswith("http://")
    ):
        clip_path: str | Path = clip_reference
    else:
        clip_path = Path(clip_reference)
        if not clip_path.is_absolute():
            if analysis_reference_path is not None:
                clip_path = analysis_reference_path.parent.parent.parent / clip_path
            else:
                clip_path = Path.cwd() / clip_path

    output_dir.mkdir(parents=True, exist_ok=True)
    backend_kwargs = {"gpu": _normalized_device_is_gpu(ocr_device)}
    ocr_backend = create_ocr_backend(ocr_backend_name, **backend_kwargs)
    vod_source = VodSource(ffmpeg_path=ffmpeg_path, process_timeout_sec=120)

    reference_frame = next(vod_source.iter_window_frames(str(clip_path), 5.0, 0.1, 1.0)).frame_bgr
    reference_identities = _extract_reference_identities(
        frame=reference_frame,
        roi_config_path=roi_config_path,
        ocr_backend=ocr_backend,
    )
    roi_config = load_roi_config(roi_config_path)
    log_roi = roi_config.optional("game_log_content_region") or roi_config.require("game_log_region")

    attribution_results: dict[str, Any] = {
        "analysis_json": None if analysis_reference_path is None else str(analysis_reference_path),
        "clip_path": str(clip_path),
        "guesses": [],
    }

    for turn in analysis["turns"]:
        team_color = turn["team_color"]
        operative_refs = [identity for identity in reference_identities[team_color] if identity.role == "operative"]
        for guess_index, guess in enumerate(turn["guesses"]):
            timestamp_sec = float(guess["timestamp_sec"])
            word = guess["word"]
            crops: list[np.ndarray] = []
            row_indexes: list[int] = []
            detection_scores: list[float] = []
            start_sec = max(0.0, timestamp_sec - window_sec)
            for sample in vod_source.iter_window_frames(str(clip_path), start_sec, window_sec * 2.0, fps):
                log_frame = crop_roi(sample.frame_bgr, log_roi)
                detection, similarity, row_index = _find_matching_detection(log_frame, ocr_backend, word)
                if detection is None or similarity < 0.72:
                    continue
                crops.append(_crop_selector_chip(log_frame, detection.box))
                row_indexes.append(row_index)
                detection_scores.append(float(similarity))

            if not crops:
                attribution_results["guesses"].append(
                    {
                        "turn_index": turn["turn_index"],
                        "guess_index": guess_index,
                        "word": word,
                        "timestamp_sec": timestamp_sec,
                        "team_color": team_color,
                        "crop_path": None,
                        "crop_x4_path": None,
                        "candidate_scores": [],
                    }
                )
                continue

            averaged_chip = _average_crops(crops)
            crop_stem = f"turn{turn['turn_index']}-guess{guess_index}-{word}"
            crop_path = output_dir / f"{crop_stem}.png"
            crop_x4_path = output_dir / f"{crop_stem}-x4.png"
            cv2.imwrite(str(crop_path), averaged_chip)
            cv2.imwrite(
                str(crop_x4_path),
                cv2.resize(averaged_chip, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4),
            )

            candidate_scores = _score_candidates(
                chip_image=averaged_chip,
                references=operative_refs,
            )
            attribution_results["guesses"].append(
                {
                    "turn_index": turn["turn_index"],
                    "guess_index": guess_index,
                    "word": word,
                    "timestamp_sec": timestamp_sec,
                    "team_color": team_color,
                    "crop_path": str(crop_path),
                    "crop_x4_path": str(crop_x4_path),
                    "frames_used": len(crops),
                    "row_indexes": row_indexes,
                    "detection_similarity_scores": [round(score, 4) for score in detection_scores],
                    "candidate_scores": candidate_scores,
                }
            )

    return attribution_results


def attribute_selectors(
    *,
    analysis_json_path: Path,
    roi_config_path: Path,
    output_dir: Path,
    ffmpeg_path: str,
    ocr_backend_name: str,
    ocr_device: str,
    window_sec: float,
    fps: float,
) -> dict[str, Any]:
    analysis = _load_json(analysis_json_path)
    return attribute_selectors_from_analysis(
        analysis=analysis,
        analysis_reference_path=analysis_json_path,
        roi_config_path=roi_config_path,
        output_dir=output_dir,
        ffmpeg_path=ffmpeg_path,
        ocr_backend_name=ocr_backend_name,
        ocr_device=ocr_device,
        window_sec=window_sec,
        fps=fps,
    )


def main() -> int:
    args = _parse_args()
    result = attribute_selectors(
        analysis_json_path=Path(args.analysis_json),
        roi_config_path=Path(args.roi_config),
        output_dir=Path(args.output_dir),
        ffmpeg_path=args.ffmpeg_path,
        ocr_backend_name=args.ocr_backend,
        ocr_device=args.ocr_device,
        window_sec=args.window_sec,
        fps=args.fps,
    )
    output_json = Path(args.output_dir) / "selector-attribution.json"
    _save_json(output_json, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
