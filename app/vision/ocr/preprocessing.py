"""Shared OCR backend protocol and OCR preprocessing helpers"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable, Literal, Protocol, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from app.core.models import ImageBoundingBox, OCRDetection

_WHITESPACE_RE = re.compile(r"\s+")
_ALNUM_RE = re.compile(r"[A-Z0-9_]+")


class OCRBackend(Protocol):
    """Protocol for OCR backends used by parser modules."""

    def detect_text(
        self,
        image: np.ndarray,
        *,
        hint: str | None = None,
    ) -> Sequence[OCRDetection]:
        """Return OCR detections for the supplied image crop."""


OCRPreprocessProfile = Literal["generic", "name_strip", "counter", "board_word", "banner", "game_log"]
OCRTaskKind = Literal["generic", "counter", "board_word", "banner", "name_strip", "game_log"]


@dataclass(frozen=True, slots=True)
class OCRVariantSpec:
    """One OCR attempt over a concrete image variant."""

    label: str
    image: np.ndarray
    source_shape: tuple[int, ...]
    hint: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OCRAttemptResult:
    """Scored OCR attempt across multiple image variants."""

    detections: list[OCRDetection]
    score: float
    variant_label: str
    hint: str | None = None
    variant_scores: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class OCRTaskProfile:
    """Lightweight OCR task profile inferred from the hint namespace."""

    task_kind: OCRTaskKind
    preprocess_profile: OCRPreprocessProfile
    expected_token_range: tuple[int, int]
    alpha_weight: float = 0.0
    alnum_weight: float = 0.0
    digits_only: bool = False
    prefer_right_edge: bool = False


def infer_ocr_task_profile(
    *,
    hint: str | None = None,
    profile: OCRPreprocessProfile | None = None,
) -> OCRTaskProfile:
    """Infer a scoring/variant profile from the OCR hint namespace."""

    normalized_hint = (hint or "").casefold()
    normalized_profile = profile or "generic"

    if any(token in normalized_hint for token in ("left_counter", "right_counter", "center_clue_count")):
        return OCRTaskProfile(
            task_kind="counter",
            preprocess_profile="counter",
            expected_token_range=(1, 2),
            digits_only=True,
            prefer_right_edge=True,
        )

    if normalized_hint.startswith("board:") or normalized_profile == "board_word":
        return OCRTaskProfile(
            task_kind="board_word",
            preprocess_profile="board_word",
            expected_token_range=(1, 2),
            alpha_weight=1.0,
        )

    if any(
        token in normalized_hint
        for token in (
            "center_clue_banner",
            "top_banner",
            "end_banner",
            "setup_panel",
        )
    ) or normalized_profile == "banner":
        return OCRTaskProfile(
            task_kind="banner",
            preprocess_profile="banner",
            expected_token_range=(1, 4),
            alpha_weight=0.9,
        )

    if any(
        token in normalized_hint
        for token in (
            "selector_name:",
            "selector_chip:",
            "selector_refs:",
            "blue:operative:",
            "blue:spymaster:",
            "red:operative:",
            "red:spymaster:",
            "name_strip",
        )
    ) or normalized_profile == "name_strip":
        return OCRTaskProfile(
            task_kind="name_strip",
            preprocess_profile="name_strip",
            expected_token_range=(1, 3),
            alnum_weight=0.8,
        )

    if "game_log" in normalized_hint or normalized_hint.startswith("selector_probe:") or normalized_profile == "game_log":
        return OCRTaskProfile(
            task_kind="game_log",
            preprocess_profile="game_log",
            expected_token_range=(1, 5),
            alnum_weight=0.35,
            alpha_weight=0.45,
        )

    return OCRTaskProfile(
        task_kind="generic",
        preprocess_profile=normalized_profile,
        expected_token_range=(1, 4),
        alnum_weight=0.25,
        alpha_weight=0.2,
    )


def prepare_ocr_image(
    image: np.ndarray,
    *,
    profile: OCRPreprocessProfile = "generic",
) -> np.ndarray:
    """Apply lightweight profile-specific preprocessing before OCR."""

    if image.size == 0:
        return image

    return _prepare_scaled_grayscale(image, profile=profile)


def generate_ocr_variants(
    image: np.ndarray,
    *,
    profile: OCRPreprocessProfile | None = None,
    hint: str | None = None,
) -> list[OCRVariantSpec]:
    """Generate OCR variants tuned for the requested task profile."""

    if image.size == 0:
        return []

    task_profile = infer_ocr_task_profile(hint=hint, profile=profile)
    variants: list[OCRVariantSpec] = [
        OCRVariantSpec(
            label="raw",
            image=image,
            source_shape=image.shape,
            hint=hint,
        )
    ]

    def add_variant(label: str, variant_image: np.ndarray) -> None:
        if variant_image.size == 0:
            return
        if any(existing.label == label for existing in variants):
            return
        variants.append(
            OCRVariantSpec(
                label=label,
                image=variant_image,
                source_shape=variant_image.shape,
                hint=hint,
            )
        )

    preprocess_profile = task_profile.preprocess_profile
    add_variant("grayscale_upscale", _prepare_scaled_grayscale(image, profile=preprocess_profile))

    if task_profile.task_kind in {"name_strip", "board_word", "banner", "game_log"}:
        add_variant("sharpened_grayscale", _prepare_scaled_grayscale(image, profile=preprocess_profile, extra_sharpen=1))

    if task_profile.task_kind in {"counter", "banner", "name_strip", "board_word", "game_log"}:
        add_variant("adaptive_threshold", _adaptive_threshold_variant(image, profile=preprocess_profile))
        add_variant("inverted_threshold", _adaptive_threshold_variant(image, profile=preprocess_profile, invert=True))

    if task_profile.task_kind in {"counter", "banner", "game_log"}:
        add_variant("bright_text_mask", _bright_text_mask_variant(image, profile=preprocess_profile))

    if task_profile.task_kind == "game_log":
        add_variant("color_suppressed", _color_suppressed_text_variant(image, profile=preprocess_profile))

    return variants


def detect_text_across_variants(
    ocr_backend: OCRBackend,
    raw_image: np.ndarray,
    *,
    variants: Sequence[OCRVariantSpec] | None = None,
    profile: OCRPreprocessProfile | None = None,
    hint: str | None = None,
) -> OCRAttemptResult:
    """Run OCR across variants and return the best-scoring attempt."""

    task_profile = infer_ocr_task_profile(hint=hint, profile=profile)
    variant_list = list(variants or generate_ocr_variants(raw_image, profile=profile, hint=hint))
    ordered_variants = _order_variants_for_backend(ocr_backend, variant_list)

    best_detections: list[OCRDetection] = []
    best_score = float("-inf")
    best_label = "raw"
    scores: list[tuple[str, float]] = []

    for variant in ordered_variants:
        detections = list(ocr_backend.detect_text(variant.image, hint=variant.hint))
        if not detections:
            scores.append((variant.label, float("-inf")))
            continue
        if (
            variant.source_shape[:2] != raw_image.shape[:2]
            and _detections_need_rescale(detections, target_shape=raw_image.shape)
        ):
            detections = _map_detections_to_raw_shape(
                detections,
                source_shape=variant.source_shape,
                target_shape=raw_image.shape,
            )
        score = score_ocr_attempt(detections, raw_image, hint=hint, task_profile=task_profile)
        scores.append((variant.label, score))
        if score > best_score:
            best_detections = detections
            best_score = score
            best_label = variant.label

    return OCRAttemptResult(
        detections=best_detections,
        score=best_score,
        variant_label=best_label,
        hint=hint,
        variant_scores=tuple(scores),
    )


def detect_text_with_fallback(
    ocr_backend: OCRBackend,
    raw_image: np.ndarray,
    *,
    prepared_image: np.ndarray | None = None,
    hint: str | None = None,
) -> list[OCRDetection]:
    """Compatibility wrapper over the richer OCR variant engine."""

    variants: list[OCRVariantSpec] | None = None
    if prepared_image is not None:
        variants = [
            OCRVariantSpec(label="raw", image=raw_image, source_shape=raw_image.shape, hint=hint),
            OCRVariantSpec(
                label="prepared",
                image=prepared_image,
                source_shape=prepared_image.shape,
                hint=hint,
            ),
        ]

    attempt = detect_text_across_variants(
        ocr_backend,
        raw_image,
        variants=variants,
        hint=hint,
    )
    return attempt.detections


def score_ocr_attempt(
    detections: Sequence[OCRDetection],
    raw_image: np.ndarray,
    *,
    hint: str | None = None,
    task_profile: OCRTaskProfile | None = None,
) -> float:
    """Score an OCR result set using task-aware plausibility checks."""

    if not detections:
        return float("-inf")

    profile = task_profile or infer_ocr_task_profile(hint=hint)
    filtered = [
        detection.model_copy(update={"text": collapse_whitespace(detection.text)})
        for detection in detections
        if collapse_whitespace(detection.text)
    ]
    if not filtered:
        return float("-inf")

    total_confidence = sum(max(float(detection.confidence), 0.0) for detection in filtered)
    average_confidence = total_confidence / len(filtered)
    raw_height, raw_width = raw_image.shape[:2]
    score = (average_confidence * 2.0) + total_confidence

    expected_min, expected_max = profile.expected_token_range
    token_count = len(filtered)
    if token_count < expected_min:
        score -= (expected_min - token_count) * 0.9
    if token_count > expected_max:
        score -= (token_count - expected_max) * 0.55

    normalized_tokens = [item.text.upper() for item in filtered]
    alpha_ratios: list[float] = []
    alnum_ratios: list[float] = []
    for token in normalized_tokens:
        if not token:
            continue
        alpha_ratios.append(sum(character.isalpha() for character in token) / len(token))
        alnum_ratios.append(sum(character.isalnum() or character == "_" for character in token) / len(token))

    if alpha_ratios:
        score += (sum(alpha_ratios) / len(alpha_ratios)) * profile.alpha_weight
    if alnum_ratios:
        score += (sum(alnum_ratios) / len(alnum_ratios)) * profile.alnum_weight

    total_text_length = sum(len(token) for token in normalized_tokens)
    if profile.task_kind == "counter":
        valid_tokens = 0
        invalid_tokens = 0
        right_edge_scores: list[float] = []
        for detection, token in zip(filtered, normalized_tokens):
            normalized_count = normalize_count_like_token(token)
            if normalized_count is not None:
                valid_tokens += 1
                right_edge_scores.append(min(1.0, detection.box.right / max(raw_width, 1)))
                score += 1.5
                if normalized_count == "infinity":
                    score += 0.1
            else:
                invalid_tokens += 1
                score -= 0.9
        if valid_tokens == 0:
            return float("-inf")
        if profile.prefer_right_edge and right_edge_scores:
            score += max(right_edge_scores) * 0.8
        if invalid_tokens:
            score -= invalid_tokens * 0.4
        return score

    if profile.task_kind in {"board_word", "banner"}:
        alpha_lengths = [
            len("".join(character for character in token if character.isalpha()))
            for token in normalized_tokens
        ]
        if alpha_lengths:
            longest = max(alpha_lengths)
            if 3 <= longest <= 18:
                score += 0.8
            else:
                score -= 0.35
        if any(any(character.isdigit() for character in token) for token in normalized_tokens):
            score -= 0.55

    if profile.task_kind == "name_strip":
        if total_text_length <= 1:
            score -= 0.9
        if any(len(_ALNUM_RE.findall(token)) == 0 for token in normalized_tokens):
            score -= 0.35

    if profile.task_kind == "game_log":
        vertical_coverage = (
            max(detection.box.bottom for detection in filtered) - min(detection.box.top for detection in filtered)
        ) / max(raw_height, 1)
        if vertical_coverage > 0.45:
            score -= 0.4
        tiny_fragments = sum(1 for token in normalized_tokens if len(token) <= 1)
        if tiny_fragments:
            score -= tiny_fragments * 0.18

    score += min(total_text_length, 32) * 0.015
    return score


def _order_variants_for_backend(
    ocr_backend: OCRBackend,
    variants: Sequence[OCRVariantSpec],
) -> list[OCRVariantSpec]:
    backend_name = type(ocr_backend).__name__.casefold()
    prefer_raw_first = "paddle" in backend_name
    if prefer_raw_first:
        return sorted(variants, key=lambda variant: (variant.label != "raw", variant.label))
    return sorted(variants, key=lambda variant: (variant.label == "raw", variant.label))


def _profile_scale(profile: OCRPreprocessProfile) -> int:
    return {
        "generic": 2,
        "name_strip": 4,
        "counter": 5,
        "board_word": 4,
        "banner": 3,
        "game_log": 5,
    }[profile]


def _to_grayscale_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 3 and image.shape[2] >= 3:
        return Image.fromarray(image[:, :, ::-1]).convert("L")
    return Image.fromarray(image).convert("L")


def _prepare_scaled_grayscale(
    image: np.ndarray,
    *,
    profile: OCRPreprocessProfile,
    extra_sharpen: int = 0,
) -> np.ndarray:
    pil_image = _to_grayscale_pil(image)
    scale = _profile_scale(profile)
    pil_image = pil_image.resize(
        (max(1, pil_image.width * scale), max(1, pil_image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    pil_image = ImageOps.autocontrast(pil_image)

    sharpen_passes = 0
    if profile in {"name_strip", "counter", "board_word", "game_log"}:
        sharpen_passes = 2
    if profile == "banner":
        sharpen_passes = 1
    for _ in range(sharpen_passes + extra_sharpen):
        pil_image = pil_image.filter(ImageFilter.SHARPEN)

    grayscale = np.asarray(pil_image, dtype=np.uint8)
    return np.stack([grayscale, grayscale, grayscale], axis=-1)


def _adaptive_threshold_variant(
    image: np.ndarray,
    *,
    profile: OCRPreprocessProfile,
    invert: bool = False,
) -> np.ndarray:
    grayscale_triplet = _prepare_scaled_grayscale(image, profile=profile)
    grayscale = grayscale_triplet[:, :, 0]
    blurred = np.asarray(
        Image.fromarray(grayscale).filter(ImageFilter.BoxBlur(radius=max(1, _profile_scale(profile) // 2))),
        dtype=np.float32,
    )
    offset = 10.0 if profile == "counter" else 6.0
    if invert:
        mask = grayscale.astype(np.float32) <= (blurred - offset)
    else:
        mask = grayscale.astype(np.float32) >= (blurred - offset)
    thresholded = np.where(mask, 255, 0).astype(np.uint8)
    return np.stack([thresholded, thresholded, thresholded], axis=-1)


def _bright_text_mask_variant(
    image: np.ndarray,
    *,
    profile: OCRPreprocessProfile,
) -> np.ndarray:
    scaled = _prepare_scaled_grayscale(image, profile=profile)
    if image.ndim != 3 or image.shape[2] < 3:
        grayscale = scaled[:, :, 0]
        mask = np.where(grayscale >= 155, 255, 0).astype(np.uint8)
        return np.stack([mask, mask, mask], axis=-1)

    scale = _profile_scale(profile)
    pil_image = Image.fromarray(image[:, :, ::-1]).resize(
        (max(1, image.shape[1] * scale), max(1, image.shape[0] * scale)),
        Image.Resampling.LANCZOS,
    )
    rgb = np.asarray(pil_image, dtype=np.uint8)
    max_channel = rgb.max(axis=2).astype(np.int16)
    min_channel = rgb.min(axis=2).astype(np.int16)
    saturation = max_channel - min_channel
    mask = (max_channel >= 160) & (saturation <= 120)
    binary = np.where(mask, 255, 0).astype(np.uint8)
    return np.stack([binary, binary, binary], axis=-1)


def _color_suppressed_text_variant(
    image: np.ndarray,
    *,
    profile: OCRPreprocessProfile,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] < 3:
        return _prepare_scaled_grayscale(image, profile=profile)

    scale = _profile_scale(profile)
    pil_image = Image.fromarray(image[:, :, ::-1]).resize(
        (max(1, image.shape[1] * scale), max(1, image.shape[0] * scale)),
        Image.Resampling.LANCZOS,
    )
    rgb = np.asarray(pil_image, dtype=np.float32)
    luminance = (rgb[:, :, 0] * 0.299) + (rgb[:, :, 1] * 0.587) + (rgb[:, :, 2] * 0.114)
    color_penalty = np.abs(rgb[:, :, 0] - rgb[:, :, 2]) * 0.35
    suppressed = np.clip(luminance - color_penalty, 0, 255).astype(np.uint8)
    enhanced = ImageEnhance.Contrast(Image.fromarray(suppressed)).enhance(1.8)
    grayscale = np.asarray(ImageOps.autocontrast(enhanced), dtype=np.uint8)
    return np.stack([grayscale, grayscale, grayscale], axis=-1)


def normalize_count_like_token(text: str) -> str | None:
    """Normalize a small OCR token into a count-like value when plausible."""

    stripped = collapse_whitespace(text).upper().strip(".,:;()[]{}")
    if not stripped:
        return None
    if stripped in {"0", "00", "09", "90"}:
        return "infinity"
    if stripped in {"\u221e", "INF", "INFINITY"}:
        return "infinity"
    if stripped.isdigit():
        if len(stripped) == 1:
            return stripped
        if len(set(stripped)) == 1:
            return stripped[0]
        if len(stripped) <= 2:
            return stripped[-1]
        return None
    digits = "".join(character for character in stripped if character.isdigit())
    if len(digits) == 1:
        return digits
    if digits and len(set(digits)) == 1:
        return digits[0]
    if stripped in {"I", "L", "|"}:
        return "1"
    return None


def _detections_need_rescale(
    detections: Sequence[OCRDetection],
    *,
    target_shape: tuple[int, ...],
) -> bool:
    target_height, target_width = target_shape[:2]
    return any(
        detection.box.right > target_width or detection.box.bottom > target_height
        for detection in detections
    )


def _map_detections_to_raw_shape(
    detections: list[OCRDetection],
    *,
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> list[OCRDetection]:
    source_height, source_width = source_shape[:2]
    target_height, target_width = target_shape[:2]
    if source_height <= 0 or source_width <= 0:
        return detections

    scale_x = target_width / source_width
    scale_y = target_height / source_height
    mapped: list[OCRDetection] = []
    for detection in detections:
        box = detection.box
        left = int(round(box.left * scale_x))
        top = int(round(box.top * scale_y))
        right = int(round(box.right * scale_x))
        bottom = int(round(box.bottom * scale_y))
        mapped.append(
            detection.model_copy(
                update={
                    "box": ImageBoundingBox(
                        left=max(left, 0),
                        top=max(top, 0),
                        right=max(right, left + 1),
                        bottom=max(bottom, top + 1),
                    )
                }
            )
        )
    return mapped


def collapse_whitespace(text: str) -> str:
    """Normalize repeated whitespace and trim the ends."""

    return _WHITESPACE_RE.sub(" ", text).strip()


def pick_best_detection(
    detections: Iterable[OCRDetection],
    *,
    excluded_texts: set[str] | None = None,
) -> OCRDetection | None:
    """Pick the most useful OCR detection after lightweight filtering."""

    excluded = {value.casefold() for value in excluded_texts or set()}
    filtered: list[OCRDetection] = []
    for detection in detections:
        normalized = collapse_whitespace(detection.text)
        if not normalized:
            continue
        if normalized.casefold() in excluded:
            continue
        filtered.append(detection.model_copy(update={"text": normalized}))

    if not filtered:
        return None

    return max(filtered, key=lambda item: (item.confidence, len(item.text)))
