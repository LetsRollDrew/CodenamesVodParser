"""Extract and score per-guess selector chips from a scheduled game reconstruction"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.parsers.gamelog import _group_detections_by_row
from app.parsers.roster import (
    _crop_role_text_band,
    _estimate_avatar_box,
    _filter_name_detections,
    _offset_detection_box,
    average_hash,
)
from app.infra.roi_config import NormalizedROI, crop_roi, load_roi_config
from app.vision.ocr.backends import create_ocr_backend
from app.vision.ocr.preprocessing import collapse_whitespace, detect_text_with_fallback, prepare_ocr_image
from app.infra.vod_source import VodSource


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
    parser.add_argument("--process-timeout", type=float, default=900.0)
    parser.add_argument("--ocr-backend", default="easyocr")
    parser.add_argument("--ocr-device", default="gpu:0")
    parser.add_argument("--window-sec", type=float, default=1.5)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--turn-index", action="append", type=int, dest="turn_indexes")
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


def _operative_display_names(analysis_roster: list[dict[str, Any]], *, team_color: str) -> list[str]:
    names: list[str] = []
    for player in analysis_roster:
        if str(player.get("team_color")) != team_color:
            continue
        if str(player.get("role")) != "operative":
            continue
        display_name = collapse_whitespace(str(player.get("display_name") or ""))
        if display_name:
            names.append(display_name)
    return names


def _reference_identities_from_detections(
    *,
    section: np.ndarray,
    detections,
    team_color: str,
    role: str,
    operative_names: list[str],
) -> list[ReferenceIdentity]:
    ordered_detections = sorted(detections, key=lambda item: item.box.center_x)
    if operative_names and len(ordered_detections) != len(operative_names):
        return []

    display_names = operative_names or [collapse_whitespace(detection.text) for detection in ordered_detections]
    identities: list[ReferenceIdentity] = []
    for display_name, detection in zip(display_names, ordered_detections):
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
        identities.append(
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


def _reference_identities_from_slots(
    *,
    section: np.ndarray,
    operative_names: list[str],
    team_color: str,
    role: str,
) -> list[ReferenceIdentity]:
    if section.size == 0 or not operative_names:
        return []

    height, width = section.shape[:2]
    slot_width = width / max(len(operative_names), 1)
    identities: list[ReferenceIdentity] = []
    for index, display_name in enumerate(operative_names):
        left = max(0, int(round(index * slot_width)))
        right = min(width, max(left + 1, int(round((index + 1) * slot_width))))
        slot = section[:, left:right].copy()
        if slot.size == 0:
            continue
        slot_height, slot_width_px = slot.shape[:2]
        avatar_left = max(0, int(slot_width_px * 0.04))
        avatar_right = min(slot_width_px, max(avatar_left + 1, int(slot_width_px * 0.96)))
        avatar_top = max(0, int(slot_height * 0.18))
        avatar_bottom = min(slot_height, max(avatar_top + 1, int(slot_height * 0.72)))
        name_top = max(0, int(slot_height * 0.58))
        name_bottom = min(slot_height, max(name_top + 1, int(slot_height * 0.99)))
        avatar_image = slot[avatar_top:avatar_bottom, avatar_left:avatar_right].copy()
        name_image = slot[name_top:name_bottom, :].copy()
        identities.append(
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


def _extract_reference_identities(
    *,
    frame: np.ndarray,
    roi_config_path: Path,
    ocr_backend,
    analysis_roster: list[dict[str, Any]] | None = None,
) -> dict[str, list[ReferenceIdentity]]:
    roi_config = load_roi_config(roi_config_path)
    identities: dict[str, list[ReferenceIdentity]] = {"blue": [], "red": []}
    active_roster = analysis_roster or []
    for team_color in ("blue", "red"):
        operative_names = _operative_display_names(active_roster, team_color=team_color)
        for region_name in (f"{team_color}_operatives_overlay_region", f"{team_color}_operatives_region"):
            region = roi_config.optional(region_name)
            if region is None:
                continue
            section = crop_roi(frame, region)
            text_frame, top_offset = _crop_role_text_band(
                section,
                role=__import__("app.core.models", fromlist=["PlayerRole"]).PlayerRole.OPERATIVE,
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
            resolved = _reference_identities_from_detections(
                section=section,
                detections=adjusted,
                team_color=team_color,
                role="operative",
                operative_names=operative_names,
            )
            if resolved:
                identities[team_color] = resolved
                break
            slot_resolved = _reference_identities_from_slots(
                section=section,
                operative_names=operative_names,
                team_color=team_color,
                role="operative",
            )
            if slot_resolved:
                identities[team_color] = slot_resolved
                break
    return identities


def _find_matching_detection(
    log_frame: np.ndarray,
    ocr_backend,
    target_word: str,
    *,
    minimum_row_index: int | None = None,
    minimum_similarity: float = 0.0,
):
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
        if minimum_row_index is not None and row_index < minimum_row_index:
            continue
        for detection in row:
            score = SequenceMatcher(
                a=collapse_whitespace(detection.text).upper(),
                b=target_word.upper(),
            ).ratio()
            if score < minimum_similarity:
                continue
            if score > best_score:
                best_score = score
                best_detection = detection
                best_row_index = row_index
    return best_detection, best_score, best_row_index


def _crop_selector_chip(log_frame: np.ndarray, detection_box) -> np.ndarray:
    height, width = log_frame.shape[:2]
    word_height = max(1.0, float(detection_box.height))
    chip_left = max(0, int(detection_box.left - word_height * 1.85))
    chip_right = min(width, int(detection_box.right + word_height * 0.3))
    chip_top = max(0, int(detection_box.top - word_height * 0.35))
    # Keep enough lower chip area to retain the selector name bar beneath the word row.
    chip_bottom = min(height, int(detection_box.bottom + word_height * 5.8))
    return log_frame[chip_top:chip_bottom, chip_left:chip_right].copy()


def _expand_roi(
    roi: NormalizedROI,
    *,
    bottom_pad: float = 0.0,
    top_pad: float = 0.0,
    left_pad: float = 0.0,
    right_pad: float = 0.0,
) -> NormalizedROI:
    left = max(0.0, roi.x - left_pad)
    top = max(0.0, roi.y - top_pad)
    right = min(1.0, roi.x + roi.width + right_pad)
    bottom = min(1.0, roi.y + roi.height + bottom_pad)
    return NormalizedROI(
        x=left,
        y=top,
        width=max(0.001, right - left),
        height=max(0.001, bottom - top),
    )


def _average_crops(crops: list[np.ndarray]) -> np.ndarray:
    max_height = max(crop.shape[0] for crop in crops)
    max_width = max(crop.shape[1] for crop in crops)
    stack: list[np.ndarray] = []
    for crop in crops:
        canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        canvas[: crop.shape[0], : crop.shape[1]] = crop
        stack.append(canvas.astype(np.float32))
    return np.mean(stack, axis=0).astype(np.uint8)


def _sharpness_score(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grayscale, cv2.CV_64F).var())


def _select_representative_crop(crops: list[np.ndarray], detection_scores: list[float]) -> np.ndarray:
    return crops[_select_representative_crop_index(crops, detection_scores)]


def _select_representative_crop_index(crops: list[np.ndarray], detection_scores: list[float]) -> int:
    if len(crops) == 1:
        return 0
    return max(
        range(len(crops)),
        key=lambda index: (_sharpness_score(crops[index]), detection_scores[index] if index < len(detection_scores) else 0.0),
    )


def _extract_name_pill(chip_image: np.ndarray) -> np.ndarray:
    height, width = chip_image.shape[:2]
    top = max(0, int(height * 0.18))
    bottom = min(height, max(top + 1, int(height * 0.40)))
    right = max(1, int(width * 0.34))
    return chip_image[top:bottom, 0:right].copy()


def _extract_identity_block(chip_image: np.ndarray) -> np.ndarray:
    height, width = chip_image.shape[:2]
    return chip_image[0 : max(1, int(height * 0.52)), 0 : max(1, int(width * 0.44))].copy()


def _extract_avatar_panel(chip_image: np.ndarray) -> np.ndarray:
    height, width = chip_image.shape[:2]
    return chip_image[0 : max(1, int(height * 0.34)), 0 : max(1, int(width * 0.34))].copy()


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


def _template_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    resized_left = cv2.resize(left, (64, 24), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    resized_right = cv2.resize(right, (64, 24), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    mse = float(np.mean((resized_left - resized_right) ** 2))
    return 1.0 / (1.0 + (mse / 4000.0))


def _score_candidates(
    *,
    chip_image: np.ndarray,
    references: list[ReferenceIdentity],
) -> list[dict[str, Any]]:
    pill_image = _extract_name_pill(chip_image)
    identity_block = _extract_identity_block(chip_image)
    avatar_panel = _extract_avatar_panel(chip_image)
    pill_features = _extract_bright_text_features(pill_image)
    identity_hash = average_hash(identity_block)
    scored: list[dict[str, Any]] = []
    for reference in references:
        ref_features = _extract_bright_text_features(reference.name_image)
        pill_distance = _feature_distance(pill_features, ref_features)
        avatar_distance = float(_hamming_distance(identity_hash, reference.avatar_hash))
        template_score = _template_similarity(pill_image, reference.name_image)
        avatar_template_score = _template_similarity(avatar_panel, reference.avatar_image)
        name_shape_score = 1.0 / (1.0 + pill_distance)
        avatar_hash_score = 1.0 / (1.0 + avatar_distance)
        composite_score = (
            name_shape_score * 0.28
            + avatar_hash_score * 0.12
            + template_score * 0.2
            + avatar_template_score * 0.4
        )
        scored.append(
            {
                "display_name": reference.display_name,
                "pill_distance": round(pill_distance, 4),
                "avatar_hamming": int(avatar_distance),
                "template_score": round(template_score, 4),
                "avatar_template_score": round(avatar_template_score, 4),
                "composite_score": round(composite_score, 4),
            }
        )
    return sorted(scored, key=lambda item: item["composite_score"], reverse=True)


def _reference_timestamp_sec(analysis: dict[str, Any]) -> float:
    explicit = analysis.get("reference_frame_sec")
    if explicit is not None:
        return max(0.0, float(explicit))
    timestamps: list[float] = []
    for turn in analysis.get("turns", []):
        for field_name in ("clue_timestamp_sec", "timestamp_sec"):
            value = turn.get(field_name)
            if value is not None:
                timestamps.append(float(value))
                break
        for guess in turn.get("guesses", []):
            value = guess.get("timestamp_sec")
            if value is not None:
                timestamps.append(float(value))
    if not timestamps:
        return 5.0
    return max(0.0, min(timestamps) - 2.0)


def attribute_selectors_from_analysis(
    *,
    analysis: dict[str, Any],
    analysis_reference_path: Path | None,
    roi_config_path: Path,
    output_dir: Path,
    ffmpeg_path: str,
    process_timeout_sec: float = 900.0,
    ocr_backend_name: str,
    ocr_device: str,
    window_sec: float,
    fps: float,
    turn_indexes: set[int] | None = None,
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
    vod_source = VodSource(ffmpeg_path=ffmpeg_path, process_timeout_sec=process_timeout_sec)

    reference_timestamp_sec = _reference_timestamp_sec(analysis)
    reference_frame = next(
        vod_source.iter_window_frames(
            str(clip_path),
            reference_timestamp_sec,
            0.1,
            1.0,
        )
    ).frame_bgr
    reference_identities = _extract_reference_identities(
        frame=reference_frame,
        roi_config_path=roi_config_path,
        ocr_backend=ocr_backend,
        analysis_roster=analysis.get("roster", []),
    )
    roi_config = load_roi_config(roi_config_path)
    log_roi = _expand_roi(roi_config.require("game_log_region"), bottom_pad=0.04)

    attribution_results: dict[str, Any] = {
        "analysis_json": None if analysis_reference_path is None else str(analysis_reference_path),
        "clip_path": str(clip_path),
        "reference_timestamp_sec": round(reference_timestamp_sec, 3),
        "guesses": [],
    }

    turns = analysis["turns"]
    stop_sec = float(analysis.get("stop_sec") or 0.0)
    for turn_index, turn in enumerate(turns):
        if turn_indexes is not None and turn_index not in turn_indexes:
            continue
        team_color = turn["team_color"]
        operative_refs = [identity for identity in reference_identities[team_color] if identity.role == "operative"]
        turn_start_sec = float(turn.get("timestamp_sec") or 0.0)
        turn_end_sec = (
            float(turns[turn_index + 1].get("timestamp_sec") or stop_sec)
            if turn_index + 1 < len(turns)
            else max(stop_sec, turn_start_sec)
        )
        previous_row_index: int | None = None
        previous_guess_timestamp: float | None = None
        for guess_index, guess in enumerate(turn["guesses"]):
            timestamp_sec = float(guess["timestamp_sec"])
            word = guess["word"]
            crops: list[np.ndarray] = []
            row_indexes: list[int] = []
            detection_scores: list[float] = []
            primary_start_sec = max(turn_start_sec, timestamp_sec - 1.0)
            if previous_guess_timestamp is not None:
                primary_start_sec = max(primary_start_sec, previous_guess_timestamp - 0.5)
            primary_end_sec = min(turn_end_sec, timestamp_sec + max(window_sec * 2.0, 4.0))
            middle_end_sec = min(turn_end_sec, timestamp_sec + max(window_sec * 5.0, 9.0))
            late_end_sec = min(turn_end_sec, timestamp_sec + max(window_sec * 9.0, 18.0))
            minimum_row_index = None if previous_row_index is None else previous_row_index + 1
            relaxed_fps = min(4.0, max(fps, 2.5))
            preferred_search_windows = [
                (primary_start_sec, primary_end_sec, fps, 0.72, minimum_row_index),
                (primary_end_sec, middle_end_sec, relaxed_fps, 0.62, minimum_row_index),
            ]
            fallback_search_windows = [
                (middle_end_sec, late_end_sec, relaxed_fps, 0.52, None),
            ]
            if guess_index > 0:
                fallback_search_windows.append((primary_start_sec, late_end_sec, relaxed_fps, 0.48, None))

            for start_sec, end_sec, search_fps, similarity_threshold, row_floor in preferred_search_windows:
                duration_sec = max(0.0, end_sec - start_sec)
                if duration_sec <= 0.0:
                    continue
                for sample in vod_source.iter_window_frames(str(clip_path), start_sec, duration_sec, search_fps):
                    log_frame = crop_roi(sample.frame_bgr, log_roi)
                    detection, similarity, row_index = _find_matching_detection(
                        log_frame,
                        ocr_backend,
                        word,
                        minimum_row_index=row_floor,
                        minimum_similarity=similarity_threshold,
                    )
                    if detection is None:
                        continue
                    crops.append(_crop_selector_chip(log_frame, detection.box))
                    row_indexes.append(row_index)
                    detection_scores.append(float(similarity))
            if not crops:
                for start_sec, end_sec, search_fps, similarity_threshold, row_floor in fallback_search_windows:
                    duration_sec = max(0.0, end_sec - start_sec)
                    if duration_sec <= 0.0:
                        continue
                    for sample in vod_source.iter_window_frames(str(clip_path), start_sec, duration_sec, search_fps):
                        log_frame = crop_roi(sample.frame_bgr, log_roi)
                        detection, similarity, row_index = _find_matching_detection(
                            log_frame,
                            ocr_backend,
                            word,
                            minimum_row_index=row_floor,
                            minimum_similarity=similarity_threshold,
                        )
                        if detection is None:
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

            preferred_row_index = max(
                set(row_indexes),
                key=lambda row_index: (
                    max(
                        (
                            score
                            for observed_row_index, score in zip(row_indexes, detection_scores)
                            if observed_row_index == row_index
                        ),
                        default=0.0,
                    ),
                    sum(
                        score
                        for observed_row_index, score in zip(row_indexes, detection_scores)
                        if observed_row_index == row_index
                    ),
                    row_indexes.count(row_index),
                    1 if previous_row_index is not None and row_index > previous_row_index else 0,
                    row_index,
                ),
            )
            filtered_observations = [
                (crop, row_index, detection_score)
                for crop, row_index, detection_score in zip(crops, row_indexes, detection_scores)
                if row_index == preferred_row_index
            ]
            crops = [crop for crop, _, _ in filtered_observations]
            row_indexes = [row_index for _, row_index, _ in filtered_observations]
            detection_scores = [score for _, _, score in filtered_observations]
            averaged_chip = _average_crops(crops)
            representative_index = _select_representative_crop_index(crops, detection_scores)
            representative_chip = crops[representative_index]
            selected_row_index = row_indexes[representative_index] if representative_index < len(row_indexes) else None
            if selected_row_index is not None:
                previous_row_index = selected_row_index
            previous_guess_timestamp = timestamp_sec
            crop_stem = f"turn{turn['turn_index']}-guess{guess_index}-{word}"
            crop_path = output_dir / f"{crop_stem}.png"
            crop_x4_path = output_dir / f"{crop_stem}-x4.png"
            cv2.imwrite(str(crop_path), representative_chip)
            cv2.imwrite(
                str(crop_x4_path),
                cv2.resize(representative_chip, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4),
            )

            candidate_scores = _score_candidates(
                chip_image=representative_chip,
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
                    "averaged_crop_path": None if len(crops) == 1 else str(output_dir / f"{crop_stem}-avg.png"),
                    "frames_used": len(crops),
                    "row_indexes": row_indexes,
                    "detection_similarity_scores": [round(score, 4) for score in detection_scores],
                    "candidate_scores": candidate_scores,
                }
            )
            if len(crops) > 1:
                cv2.imwrite(str(output_dir / f"{crop_stem}-avg.png"), averaged_chip)

    return attribution_results


def attribute_selectors(
    *,
    analysis_json_path: Path,
    roi_config_path: Path,
    output_dir: Path,
    ffmpeg_path: str,
    process_timeout_sec: float = 900.0,
    ocr_backend_name: str,
    ocr_device: str,
    window_sec: float,
    fps: float,
    turn_indexes: set[int] | None = None,
) -> dict[str, Any]:
    analysis = _load_json(analysis_json_path)
    return attribute_selectors_from_analysis(
        analysis=analysis,
        analysis_reference_path=analysis_json_path,
        roi_config_path=roi_config_path,
        output_dir=output_dir,
        ffmpeg_path=ffmpeg_path,
        process_timeout_sec=process_timeout_sec,
        ocr_backend_name=ocr_backend_name,
        ocr_device=ocr_device,
        window_sec=window_sec,
        fps=fps,
        turn_indexes=turn_indexes,
    )


def main() -> int:
    args = _parse_args()
    result = attribute_selectors(
        analysis_json_path=Path(args.analysis_json),
        roi_config_path=Path(args.roi_config),
        output_dir=Path(args.output_dir),
        ffmpeg_path=args.ffmpeg_path,
        process_timeout_sec=args.process_timeout,
        ocr_backend_name=args.ocr_backend,
        ocr_device=args.ocr_device,
        window_sec=args.window_sec,
        fps=args.fps,
        turn_indexes=None if not args.turn_indexes else set(args.turn_indexes),
    )
    output_json = Path(args.output_dir) / "selector-attribution.json"
    _save_json(output_json, result)
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "guess_count": len(result.get("guesses", [])),
                "reference_timestamp_sec": result.get("reference_timestamp_sec"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
