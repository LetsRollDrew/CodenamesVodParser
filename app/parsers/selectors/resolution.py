"""Resolve selector candidates into concrete operative names"""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from itertools import product
from pathlib import Path
from typing import Any

import cv2

from app.core.models import CardColor, TeamColor
from app.reconstruction.game_reconstruction import classify_guess_result
from app.vision.ocr.preprocessing import collapse_whitespace, detect_text_with_fallback, prepare_ocr_image


def _normalized_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalpha())


def _skeleton_name(value: str) -> str:
    normalized = _normalized_name(value)
    return (
        normalized.replace("1", "l")
        .replace("!", "l")
        .replace("|", "l")
        .replace("i", "l")
        .replace("v", "y")
    )


def _name_similarity(left: str, right: str) -> float:
    normalized_left = _normalized_name(left)
    normalized_right = _normalized_name(right)
    if not normalized_left or not normalized_right:
        return 0.0
    direct = SequenceMatcher(a=normalized_left, b=normalized_right).ratio()
    skeleton = SequenceMatcher(a=_skeleton_name(normalized_left), b=_skeleton_name(normalized_right)).ratio()
    return max(direct, skeleton)


def _resolve_name_from_text(raw_text: str, operative_names: list[str]) -> tuple[str | None, float]:
    scored_names: list[tuple[float, str]] = []
    for operative_name in operative_names:
        score = _name_similarity(raw_text, operative_name)
        scored_names.append((score, operative_name))
    if not scored_names:
        return None, 0.0
    scored_names.sort(key=lambda item: item[0], reverse=True)
    best_score, best_name = scored_names[0]
    runner_up_score = scored_names[1][0] if len(scored_names) > 1 else 0.0
    if best_score >= 0.7:
        return best_name, best_score
    if (
        best_score >= 0.6
        and len(_normalized_name(raw_text)) <= 4
        and (best_score - runner_up_score) >= 0.08
    ):
        return best_name, best_score
    return None, best_score


def _resolve_name_from_initial_hint(raw_text: str, operative_names: list[str]) -> tuple[str | None, float]:
    normalized_text = _normalized_name(raw_text)
    if len(normalized_text) < 1:
        return None, 0.0
    initial = normalized_text[0]
    initial_matches = [name for name in operative_names if _normalized_name(name).startswith(initial)]
    if len(initial_matches) != 1:
        return None, 0.0
    candidate_name = initial_matches[0]
    similarity = _name_similarity(raw_text, candidate_name)
    candidate_length = len(_normalized_name(candidate_name))
    if (candidate_length - len(normalized_text)) > 2:
        return None, similarity
    if len(normalized_text) == 1:
        minimum_similarity = 0.5
    elif len(normalized_text) == 2:
        minimum_similarity = 0.48
    else:
        minimum_similarity = 0.5
    if similarity >= minimum_similarity:
        return candidate_name, similarity
    return None, similarity


def _analysis_base_path(analysis_reference_path: Path | None) -> Path:
    if analysis_reference_path is None:
        return Path.cwd()
    return analysis_reference_path.parent


def _resolve_image_path(
    image_reference: str | None,
    *,
    analysis_reference_path: Path | None,
) -> Path | None:
    if not image_reference:
        return None
    image_path = Path(image_reference)
    if image_path.is_absolute():
        return image_path
    primary = (_analysis_base_path(analysis_reference_path) / image_path).resolve()
    if primary.exists():
        return primary
    fallback = (Path.cwd() / image_path).resolve()
    if fallback.exists():
        return fallback
    return primary


def _template_similarity(left, right) -> float:
    if left is None or right is None or left.size == 0 or right.size == 0:
        return 0.0
    resized_left = cv2.resize(left, (64, 24), interpolation=cv2.INTER_CUBIC).astype("float32")
    resized_right = cv2.resize(right, (64, 24), interpolation=cv2.INTER_CUBIC).astype("float32")
    mse = float(((resized_left - resized_right) ** 2).mean())
    return 1.0 / (1.0 + (mse / 4000.0))


def _selector_template_crop_variants(chip_image):
    height, width = chip_image.shape[:2]
    for top_ratio, bottom_ratio, left_ratio, right_ratio in product(
        (0.18, 0.20, 0.22, 0.24),
        (0.36, 0.40, 0.42),
        (0.00, 0.02, 0.04),
        (0.24, 0.28, 0.30),
    ):
        top = max(0, int(height * top_ratio))
        bottom = min(height, max(top + 1, int(height * bottom_ratio)))
        left = max(0, int(width * left_ratio))
        right = min(width, max(left + 1, int(width * right_ratio)))
        crop = chip_image[top:bottom, left:right].copy()
        if crop.size != 0:
            yield crop


def _explicit_reference_templates(
    *,
    analysis: dict[str, Any],
    analysis_reference_path: Path | None,
) -> dict[str, dict[str, list[Any]]]:
    templates_by_team: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for template in analysis.get("selector_reference_templates", []):
        team_color = collapse_whitespace(str(template.get("team_color") or ""))
        display_name = collapse_whitespace(str(template.get("display_name") or ""))
        image_path = _resolve_image_path(
            template.get("image_path"),
            analysis_reference_path=analysis_reference_path,
        )
        if not team_color or not display_name or image_path is None or not image_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None or image.size == 0:
            continue
        templates_by_team[team_color][display_name].append(image)
    return templates_by_team


def _extract_name_pill(chip_image) -> Any:
    height, width = chip_image.shape[:2]
    # The selector label consistently sits in a lower-left band beneath the avatar.
    # A fixed crop is more stable than trying to chase dark separators via components.
    top = max(0, int(height * 0.18))
    bottom = min(height, max(top + 1, int(height * 0.40)))
    right = max(1, int(width * 0.34))
    return chip_image[top:bottom, 0:right].copy()


def _should_preserve_existing_selector(guess: dict[str, Any]) -> bool:
    selector_name = collapse_whitespace(str(guess.get("selector_name") or ""))
    if not selector_name:
        return False
    selector_source = collapse_whitespace(str(guess.get("selector_source") or "")).casefold()
    selector_confidence = float(guess.get("selector_confidence") or 0.0)
    if selector_source in {"team_dominance", "turn_continuity", "selector_candidates"}:
        return False
    if selector_confidence >= 0.88:
        return True
    if selector_source == "guess_player" and selector_confidence >= 0.82:
        return True
    if selector_source == "selector_crop" and selector_confidence >= 0.82:
        return True
    return False


def _short_hint_conflicts_with_candidates(
    *,
    raw_text: str,
    matched_name: str,
    operative_names: list[str],
    guess: dict[str, Any],
) -> bool:
    normalized_text = _normalized_name(raw_text)
    if len(normalized_text) > 2:
        return False
    resolved_candidates = _resolved_candidate_scores(guess=guess, operative_names=operative_names)
    if not resolved_candidates:
        return False
    matched_candidate = next(
        (candidate for candidate in resolved_candidates if candidate["display_name"] == matched_name),
        None,
    )
    if matched_candidate is None:
        return True
    top_candidate = resolved_candidates[0]
    if top_candidate["display_name"] == matched_name:
        return False
    score_gap = float(top_candidate.get("composite_score") or 0.0) - float(
        matched_candidate.get("composite_score") or 0.0
    )
    return score_gap > 0.02


def _ocr_selector_name(
    *,
    guess: dict[str, Any],
    operative_names: list[str],
    ocr_backend,
    analysis_reference_path: Path | None,
) -> tuple[str | None, float, str | None]:
    if ocr_backend is None:
        return None, 0.0, None

    guess_word = collapse_whitespace(str(guess.get("word") or ""))

    for path_key in ("selector_crop_x4_path", "selector_avg_crop_path", "selector_crop_path"):
        image_path = _resolve_image_path(
            guess.get(path_key),
            analysis_reference_path=analysis_reference_path,
        )
        if image_path is None or not image_path.exists():
            continue
        chip_image = cv2.imread(str(image_path))
        if chip_image is None:
            continue
        name_pill = _extract_name_pill(chip_image)
        detections = detect_text_with_fallback(
            ocr_backend,
            name_pill,
            prepared_image=prepare_ocr_image(name_pill, profile="name_strip"),
            hint=f"selector_name:{image_path.name}",
        )
        best_name = None
        best_confidence = 0.0
        best_raw_text = None
        for detection in detections:
            raw_text = collapse_whitespace(detection.text)
            if not raw_text:
                continue
            matched_name, similarity = _resolve_name_from_text(raw_text, operative_names)
            if matched_name is None:
                matched_name, similarity = _resolve_name_from_initial_hint(raw_text, operative_names)
            if matched_name is None:
                continue
            if _short_hint_conflicts_with_candidates(
                raw_text=raw_text,
                matched_name=matched_name,
                operative_names=operative_names,
                guess=guess,
            ):
                continue
            confidence = min(0.99, 0.82 + (similarity * 0.12) + (float(detection.confidence) * 0.06))
            if confidence > best_confidence:
                best_name = matched_name
                best_confidence = confidence
                best_raw_text = raw_text
        if best_name is None:
            whole_detections = detect_text_with_fallback(
                ocr_backend,
                chip_image,
                prepared_image=prepare_ocr_image(chip_image, profile="name_strip"),
                hint=f"selector_chip:{image_path.name}",
            )
            for detection in whole_detections:
                raw_text = collapse_whitespace(detection.text)
                if not raw_text or _normalized_name(raw_text) == _normalized_name(guess_word):
                    continue
                matched_name, similarity = _resolve_name_from_text(raw_text, operative_names)
                matched_from_direct_text = matched_name is not None
                if matched_name is None:
                    matched_name, similarity = _resolve_name_from_initial_hint(raw_text, operative_names)
                    if matched_name is None:
                        continue
                if _short_hint_conflicts_with_candidates(
                    raw_text=raw_text,
                    matched_name=matched_name,
                    operative_names=operative_names,
                    guess=guess,
                ):
                    continue
                if not matched_from_direct_text:
                    confidence = min(0.91, 0.82 + (similarity * 0.12) + (float(detection.confidence) * 0.08))
                else:
                    confidence = min(0.92, 0.8 + (similarity * 0.1) + (float(detection.confidence) * 0.08))
                if confidence > best_confidence:
                    best_name = matched_name
                    best_confidence = confidence
                    best_raw_text = raw_text
        if best_name is not None:
            return best_name, round(best_confidence, 4), best_raw_text
    return None, 0.0, None


def _resolved_candidate_scores(
    *,
    guess: dict[str, Any],
    operative_names: list[str],
) -> list[dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for candidate in guess.get("selector_candidates", []):
        raw_name = str(candidate.get("display_name") or "")
        matched_name, similarity = _resolve_name_from_text(raw_name, operative_names)
        if matched_name is None:
            continue
        composite_score = float(candidate.get("composite_score") or 0.0)
        pill_distance = candidate.get("pill_distance")
        template_score = candidate.get("template_score")
        if pill_distance is not None and template_score is not None:
            try:
                name_shape_score = 1.0 / (1.0 + max(float(pill_distance), 0.0))
                candidate_score = (name_shape_score * 0.6) + (float(template_score) * 0.4)
                composite_score = max(composite_score, round(candidate_score, 4))
            except (TypeError, ValueError):
                pass
        current = aggregated.get(matched_name)
        if current is None or composite_score > current["composite_score"]:
            aggregated[matched_name] = {
                "display_name": matched_name,
                "raw_display_name": raw_name,
                "name_similarity": round(similarity, 4),
                "composite_score": round(composite_score, 4),
                "pill_distance": candidate.get("pill_distance"),
                "template_score": candidate.get("template_score"),
                "avatar_template_score": candidate.get("avatar_template_score"),
                "avatar_hamming": candidate.get("avatar_hamming"),
            }
    return sorted(
        aggregated.values(),
        key=lambda item: (item["composite_score"], item["name_similarity"]),
        reverse=True,
    )


def _set_selector_resolution(
    guess: dict[str, Any],
    *,
    selector_name: str,
    selector_source: str,
    selector_confidence: float,
    raw_text: str | None = None,
) -> None:
    guess["selector_name"] = selector_name
    guess["selector_source"] = selector_source
    guess["selector_confidence"] = round(selector_confidence, 4)
    if raw_text:
        guess["selector_ocr_text"] = raw_text
    guess["selector_provenance"] = {
        "source": selector_source,
        "confidence": round(selector_confidence, 4),
        "raw_text": raw_text,
        "evidence_type": "direct" if selector_source in {"selector_crop", "selector_templates", "selector_candidates", "guess_player"} else "heuristic",
    }


def _reference_template_candidates(
    *,
    guess: dict[str, Any],
    operative_names: list[str],
    team_templates: dict[str, list[Any]],
    analysis_reference_path: Path | None,
) -> list[dict[str, Any]]:
    if not operative_names or not team_templates:
        return []

    best_scored: list[dict[str, Any]] = []
    best_objective = float("-inf")
    for path_key in ("selector_crop_x4_path", "selector_avg_crop_path", "selector_crop_path"):
        image_path = _resolve_image_path(
            guess.get(path_key),
            analysis_reference_path=analysis_reference_path,
        )
        if image_path is None or not image_path.exists():
            continue
        chip_image = cv2.imread(str(image_path))
        if chip_image is None or chip_image.size == 0:
            continue
        for crop in _selector_template_crop_variants(chip_image):
            scored: list[dict[str, Any]] = []
            for operative_name in operative_names:
                reference_images = team_templates.get(operative_name) or []
                if not reference_images:
                    continue
                best_score = max(
                    _template_similarity(crop, reference_image)
                    for reference_image in reference_images
                )
                if best_score <= 0.0:
                    continue
                scored.append(
                    {
                        "display_name": operative_name,
                        "reference_template_score": round(best_score, 4),
                    }
                )
            if len(scored) < 2:
                continue
            scored.sort(key=lambda item: item["reference_template_score"], reverse=True)
            top_score = float(scored[0]["reference_template_score"])
            runner_up_score = float(scored[1]["reference_template_score"])
            objective = (top_score - runner_up_score) + (top_score * 0.2)
            if objective > best_objective:
                best_scored = scored
                best_objective = objective
    return best_scored


def _template_candidate_supported_by_existing_scores(
    *,
    guess: dict[str, Any],
    candidate_name: str,
    operative_names: list[str],
) -> tuple[bool, float]:
    existing_candidates = _resolved_candidate_scores(guess=guess, operative_names=operative_names)
    if not existing_candidates:
        return False, 0.0
    top_candidate = existing_candidates[0]
    matched_candidate = next(
        (candidate for candidate in existing_candidates if candidate["display_name"] == candidate_name),
        None,
    )
    if matched_candidate is None:
        return False, float("inf")
    gap = float(top_candidate.get("composite_score") or 0.0) - float(matched_candidate.get("composite_score") or 0.0)
    return True, gap


def _maybe_resolve_from_template_candidates(
    *,
    guess: dict[str, Any],
    operative_names: list[str],
    template_candidates: list[dict[str, Any]],
) -> tuple[str | None, float]:
    if len(template_candidates) < 1:
        return None, 0.0
    top = template_candidates[0]
    runner_up_score = (
        float(template_candidates[1].get("reference_template_score") or 0.0)
        if len(template_candidates) > 1
        else 0.0
    )
    top_score = float(top.get("reference_template_score") or 0.0)
    score_gap = top_score - runner_up_score
    existing_candidates = _resolved_candidate_scores(guess=guess, operative_names=operative_names)
    existing_top_name = str(existing_candidates[0].get("display_name") or "") if existing_candidates else ""
    supported_by_existing_scores, existing_gap = _template_candidate_supported_by_existing_scores(
        guess=guess,
        candidate_name=str(top.get("display_name") or ""),
        operative_names=operative_names,
    )
    if (
        existing_top_name
        and existing_top_name == str(top.get("display_name") or "")
        and top_score >= 0.52
        and score_gap >= 0.008
        and supported_by_existing_scores
        and existing_gap <= 0.01
    ):
        confidence = min(0.84, 0.7 + (top_score * 0.16) + (score_gap * 2.0))
        return str(top.get("display_name") or ""), round(confidence, 4)
    return None, 0.0


def _candidate_alternatives(
    guess: dict[str, Any],
    *,
    excluded_names: set[str],
) -> list[dict[str, Any]]:
    candidates = guess.get("reference_selector_candidates") or guess.get("resolved_selector_candidates") or []
    return [
        candidate
        for candidate in candidates
        if str(candidate.get("display_name") or "") not in excluded_names
    ]


def _resolve_from_candidates(
    *,
    guess: dict[str, Any],
    operative_names: list[str],
) -> tuple[str | None, float, list[dict[str, Any]]]:
    resolved_candidates = _resolved_candidate_scores(guess=guess, operative_names=operative_names)
    if not resolved_candidates:
        return None, 0.0, []

    if len(resolved_candidates) == 1:
        candidate = resolved_candidates[0]
        confidence = min(0.92, 0.74 + (candidate["name_similarity"] * 0.12) + (candidate["composite_score"] * 0.2))
        return candidate["display_name"], round(confidence, 4), resolved_candidates

    top = resolved_candidates[0]
    runner_up = resolved_candidates[1]
    score_gap = top["composite_score"] - runner_up["composite_score"]
    similarity_gap = top["name_similarity"] - runner_up["name_similarity"]
    if score_gap >= 0.03 or similarity_gap >= 0.2:
        confidence = min(0.9, 0.72 + (top["name_similarity"] * 0.1) + (score_gap * 0.8))
        return top["display_name"], round(confidence, 4), resolved_candidates
    uses_raw_selector_features = any(
        top.get(field_name) is not None
        for field_name in ("pill_distance", "template_score", "avatar_template_score", "avatar_hamming")
    )
    second_gap_threshold = 0.025 if uses_raw_selector_features else 0.015
    if top["composite_score"] >= 0.4 and score_gap >= second_gap_threshold:
        confidence = min(0.86, 0.69 + (top["name_similarity"] * 0.08) + (score_gap * 1.6))
        return top["display_name"], round(confidence, 4), resolved_candidates
    return None, 0.0, resolved_candidates


def _derive_selector_templates_from_resolved_guesses(
    analysis: dict[str, Any],
    *,
    analysis_reference_path: Path | None,
) -> dict[str, dict[str, list[Any]]]:
    derived: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for turn in analysis.get("turns", []):
        team_color = str(turn.get("team_color") or "")
        for guess in turn.get("guesses", []):
            selector_name = collapse_whitespace(str(guess.get("selector_name") or ""))
            selector_source = collapse_whitespace(str(guess.get("selector_source") or ""))
            selector_confidence = float(guess.get("selector_confidence") or 0.0)
            if not selector_name or selector_source not in {"selector_crop", "guess_player"}:
                continue
            if selector_confidence < 0.88:
                continue
            chip_image = None
            for path_key in ("selector_crop_x4_path", "selector_avg_crop_path", "selector_crop_path"):
                image_path = _resolve_image_path(
                    guess.get(path_key),
                    analysis_reference_path=analysis_reference_path,
                )
                if image_path is None or not image_path.exists():
                    continue
                chip_image = cv2.imread(str(image_path))
                if chip_image is not None and chip_image.size != 0:
                    break
                chip_image = None
            if chip_image is None:
                continue
            derived[team_color][selector_name].append(_extract_name_pill(chip_image))
    return derived


def merge_selector_results_into_analysis(
    analysis: dict[str, Any],
    selector_results: dict[str, Any],
) -> None:
    selector_lookup = {
        (item["turn_index"], item["guess_index"]): item
        for item in selector_results.get("guesses", [])
    }
    for turn in analysis.get("turns", []):
        for guess_index, guess in enumerate(turn.get("guesses", [])):
            selector_item = selector_lookup.get((turn.get("turn_index"), guess_index))
            if selector_item is None:
                continue
            has_fresh_selector_evidence = bool(
                selector_item.get("crop_path")
                or selector_item.get("crop_x4_path")
                or selector_item.get("averaged_crop_path")
                or selector_item.get("candidate_scores")
            )
            if has_fresh_selector_evidence:
                for field_name in (
                    "selector_name",
                    "selector_source",
                    "selector_confidence",
                    "selector_ocr_text",
                    "resolved_selector_candidates",
                    "selector_candidates",
                ):
                    guess.pop(field_name, None)
            guess["selector_crop_path"] = selector_item.get("crop_path")
            guess["selector_crop_x4_path"] = selector_item.get("crop_x4_path")
            guess["selector_avg_crop_path"] = selector_item.get("averaged_crop_path")
            guess["selector_candidates"] = selector_item.get("candidate_scores", [])


def resolve_selector_fields_inplace(
    analysis: dict[str, Any],
    *,
    ocr_backend=None,
    analysis_reference_path: Path | None = None,
    roster: list[dict[str, Any]] | None = None,
) -> None:
    operative_names_by_team: dict[str, list[str]] = defaultdict(list)
    active_roster = analysis.get("roster", []) if roster is None else roster
    for player in active_roster:
        if str(player.get("role")) != "operative":
            continue
        operative_names_by_team[str(player.get("team_color"))].append(str(player.get("display_name")))
    explicit_templates_by_team = _explicit_reference_templates(
        analysis=analysis,
        analysis_reference_path=analysis_reference_path,
    )

    for turn in analysis.get("turns", []):
        team_color = str(turn.get("team_color"))
        operative_names = operative_names_by_team.get(team_color, [])
        team_templates = explicit_templates_by_team.get(team_color, {})
        if not operative_names:
            continue

        for guess in turn.get("guesses", []):
            if _should_preserve_existing_selector(guess):
                continue
            if guess.get("selector_name"):
                for field_name in (
                    "selector_name",
                    "selector_source",
                    "selector_confidence",
                    "selector_ocr_text",
                    "resolved_selector_candidates",
                    "reference_selector_candidates",
                ):
                    guess.pop(field_name, None)
            resolved_name = None
            resolved_confidence = 0.0
            raw_text = None
            if guess.get("selector_crop_path") or guess.get("selector_crop_x4_path"):
                resolved_name, resolved_confidence, raw_text = _ocr_selector_name(
                    guess=guess,
                    operative_names=operative_names,
                    ocr_backend=ocr_backend,
                    analysis_reference_path=analysis_reference_path,
                )
            if resolved_name is not None:
                _set_selector_resolution(
                    guess,
                    selector_name=resolved_name,
                    selector_source="selector_crop",
                    selector_confidence=resolved_confidence,
                    raw_text=raw_text,
                )
                continue

            reference_candidates = _reference_template_candidates(
                guess=guess,
                operative_names=operative_names,
                team_templates=team_templates,
                analysis_reference_path=analysis_reference_path,
            )
            if reference_candidates:
                guess["reference_selector_candidates"] = reference_candidates
                resolved_name, resolved_confidence = _maybe_resolve_from_template_candidates(
                    guess=guess,
                    operative_names=operative_names,
                    template_candidates=reference_candidates,
                )
                if resolved_name is not None:
                    _set_selector_resolution(
                        guess,
                        selector_name=resolved_name,
                        selector_source="selector_templates",
                        selector_confidence=resolved_confidence,
                    )
                    continue

            resolved_name, resolved_confidence, resolved_candidates = _resolve_from_candidates(
                guess=guess,
                operative_names=operative_names,
            )
            if resolved_candidates:
                guess["resolved_selector_candidates"] = resolved_candidates
            if resolved_name is not None:
                _set_selector_resolution(
                    guess,
                    selector_name=resolved_name,
                    selector_source="selector_candidates",
                    selector_confidence=resolved_confidence,
                )
                continue

            parsed_player_name = collapse_whitespace(str(guess.get("player_name") or ""))
            if len(operative_names) <= 2 and parsed_player_name and parsed_player_name.casefold() != "unknown":
                matched_name, similarity = _resolve_name_from_text(parsed_player_name, operative_names)
                if matched_name is not None and similarity >= 0.85:
                    _set_selector_resolution(
                        guess,
                        selector_name=matched_name,
                        selector_source="guess_player",
                        selector_confidence=min(0.9, 0.8 + (similarity * 0.1)),
                    )

        strong_turn_names = {
            guess.get("selector_name")
            for guess in turn.get("guesses", [])
            if guess.get("selector_source") in {"selector_crop", "guess_player"}
            and float(guess.get("selector_confidence") or 0.0) >= 0.88
            and guess.get("selector_name")
        }
        if len(turn.get("guesses", [])) >= 3:
            final_guess = turn["guesses"][-1]
            if not final_guess.get("selector_name"):
                final_candidates = final_guess.get("resolved_selector_candidates") or []
                if len(final_candidates) >= 2:
                    top_candidate = final_candidates[0]
                    runner_up = final_candidates[1]
                    top_score = float(top_candidate.get("composite_score") or 0.0)
                    score_gap = top_score - float(runner_up.get("composite_score") or 0.0)
                    if top_score >= 0.42 and score_gap >= 0.02:
                        final_guess["selector_name"] = str(top_candidate.get("display_name") or "")
                        final_guess["selector_source"] = "last_guess_candidate_top"
                        final_guess["selector_confidence"] = round(
                            min(0.8, 0.64 + (top_score * 0.18) + (score_gap * 1.8)),
                            4,
                        )

        resolved_names = {
            guess.get("selector_name")
            for guess in turn.get("guesses", [])
            if guess.get("selector_name")
        }
        unresolved_guesses = [guess for guess in turn.get("guesses", []) if not guess.get("selector_name")]
        if len(turn.get("guesses", [])) == 3 and len(resolved_names) == 1 and len(unresolved_guesses) >= 2:
            resolved_name = next(iter(resolved_names))
            reference_alt_names: list[str] = []
            resolved_alt_names: list[str] = []
            for guess in unresolved_guesses:
                reference_alternatives = _candidate_alternatives(
                    guess,
                    excluded_names={resolved_name},
                )
                if not reference_alternatives:
                    reference_alt_names = []
                    break
                reference_alt_names.append(str(reference_alternatives[0].get("display_name") or ""))

                resolved_alternatives = [
                    candidate
                    for candidate in (guess.get("resolved_selector_candidates") or [])
                    if str(candidate.get("display_name") or "") != resolved_name
                ]
                if not resolved_alternatives:
                    resolved_alt_names = []
                    break
                resolved_alt_names.append(str(resolved_alternatives[0].get("display_name") or ""))

            if (
                reference_alt_names
                and len(set(reference_alt_names)) == 1
                and resolved_alt_names
                and len(set(resolved_alt_names)) == 1
                and reference_alt_names[0] == resolved_alt_names[0]
            ):
                shared_name = reference_alt_names[0]
                for guess in unresolved_guesses:
                    has_selector_evidence = bool(
                        guess.get("selector_crop_path")
                        or guess.get("selector_avg_crop_path")
                        or guess.get("selector_crop_x4_path")
                        or guess.get("reference_selector_candidates")
                        or guess.get("resolved_selector_candidates")
                    )
                    if not has_selector_evidence:
                        continue
                    _set_selector_resolution(
                        guess,
                        selector_name=shared_name,
                        selector_source="turn_shared_alternative",
                        selector_confidence=0.66,
                    )

        if (
            len(turn.get("guesses", [])) == 3
            and turn["guesses"][1].get("selector_name")
            and not turn["guesses"][0].get("selector_name")
            and not turn["guesses"][2].get("selector_name")
        ):
            middle_name = str(turn["guesses"][1].get("selector_name") or "")
            outer_alternative_names: list[str] = []
            for guess in (turn["guesses"][0], turn["guesses"][2]):
                alternatives = _candidate_alternatives(
                    guess,
                    excluded_names={middle_name},
                )
                if not alternatives:
                    outer_alternative_names = []
                    break
                outer_alternative_names.append(str(alternatives[0].get("display_name") or ""))
            if outer_alternative_names and len(set(outer_alternative_names)) == 1:
                shared_name = outer_alternative_names[0]
                for guess in (turn["guesses"][0], turn["guesses"][2]):
                    has_selector_evidence = bool(
                        guess.get("selector_crop_path")
                        or guess.get("selector_avg_crop_path")
                        or guess.get("selector_crop_x4_path")
                        or guess.get("reference_selector_candidates")
                        or guess.get("resolved_selector_candidates")
                    )
                    if not has_selector_evidence:
                        continue
                    _set_selector_resolution(
                        guess,
                        selector_name=shared_name,
                        selector_source="turn_middle_breakout",
                        selector_confidence=0.65,
                    )

        if len(strong_turn_names) == 1:
            strong_name = next(iter(strong_turn_names))
            first_strong_index = min(
                (
                    guess_index
                    for guess_index, guess in enumerate(turn.get("guesses", []))
                    if guess.get("selector_name") == strong_name
                    and guess.get("selector_source") in {"selector_crop", "guess_player"}
                    and float(guess.get("selector_confidence") or 0.0) >= 0.88
                ),
                default=0,
            )
            strong_confidence = max(
                (
                    float(guess.get("selector_confidence") or 0.0)
                    for guess in turn.get("guesses", [])
                    if guess.get("selector_name") == strong_name
                ),
                default=0.0,
            )
            for guess_index, guess in enumerate(turn.get("guesses", [])):
                if guess.get("selector_source") not in {"selector_candidates", "selector_templates"}:
                    continue
                resolved_candidates = (
                    guess.get("reference_selector_candidates")
                    or guess.get("resolved_selector_candidates")
                    or []
                )
                inherited_candidate = next(
                    (
                        candidate
                        for candidate in resolved_candidates
                        if candidate.get("display_name") == strong_name
                    ),
                    None,
                )
                if inherited_candidate is None:
                    continue
                score_field = (
                    "reference_template_score"
                    if "reference_template_score" in resolved_candidates[0]
                    else "composite_score"
                )
                top_score = float(resolved_candidates[0].get(score_field) or 0.0)
                inherited_score = float(inherited_candidate.get(score_field) or 0.0)
                if (top_score - inherited_score) > 0.03:
                    continue
                if len(operative_names) >= 3 and guess_index < first_strong_index:
                    continue
                _set_selector_resolution(
                    guess,
                    selector_name=strong_name,
                    selector_source="turn_continuity",
                    selector_confidence=max(0.62, strong_confidence - 0.16),
                )

        resolved_names_in_turn = {
            guess.get("selector_name")
            for guess in turn.get("guesses", [])
            if guess.get("selector_name")
        }
        if len(operative_names) >= 3:
            for guess_index, guess in enumerate(turn.get("guesses", [])):
                if guess.get("selector_name"):
                    continue
                resolved_candidates = guess.get("resolved_selector_candidates") or []
                if len(resolved_candidates) < 2:
                    continue
                top_candidate = resolved_candidates[0]
                runner_up = resolved_candidates[1]
                top_score = float(top_candidate.get("composite_score") or 0.0)
                score_gap = top_score - float(runner_up.get("composite_score") or 0.0)
                top_name = str(top_candidate.get("display_name") or "")
                if not top_name or top_score < 0.32 or score_gap < 0.003:
                    continue
                earlier_same_name = any(
                    other_guess.get("selector_name") == top_name
                    for other_guess in turn.get("guesses", [])[:guess_index]
                )
                if earlier_same_name or (len(turn.get("guesses", [])) == 2 and len(resolved_names_in_turn) == 1):
                    _set_selector_resolution(
                        guess,
                        selector_name=top_name,
                        selector_source="turn_candidate_top",
                        selector_confidence=min(0.78, 0.64 + (top_score * 0.18) + (score_gap * 2.0)),
                    )

        resolved_names = {
            guess.get("selector_name")
            for guess in turn.get("guesses", [])
            if guess.get("selector_name")
        }
        if len(resolved_names) != 1:
            continue
        inherited_name = next(iter(resolved_names))
        first_resolved_index = min(
            (
                guess_index
                for guess_index, guess in enumerate(turn.get("guesses", []))
                if guess.get("selector_name") == inherited_name
            ),
            default=0,
        )
        continuity_source_confidence = max(
            (
                float(guess.get("selector_confidence") or 0.0)
                for guess in turn.get("guesses", [])
                if guess.get("selector_name") == inherited_name
            ),
            default=0.0,
        )
        if continuity_source_confidence < 0.78:
            continue
        for guess in turn.get("guesses", []):
            if guess.get("selector_name"):
                continue
            has_selector_evidence = bool(
                guess.get("selector_crop_path")
                or guess.get("selector_avg_crop_path")
                or guess.get("selector_crop_x4_path")
                or guess.get("reference_selector_candidates")
                or guess.get("resolved_selector_candidates")
            )
            guess_index = turn.get("guesses", []).index(guess)
            if guess_index < first_resolved_index and len(turn.get("guesses", [])) >= 3:
                continuity_candidates = guess.get("resolved_selector_candidates") or guess.get("reference_selector_candidates")
            else:
                continuity_candidates = guess.get("reference_selector_candidates") or guess.get("resolved_selector_candidates")
            if continuity_candidates:
                top_candidate = continuity_candidates[0]
                inherited_candidate = next(
                    (
                        candidate
                        for candidate in continuity_candidates
                        if candidate["display_name"] == inherited_name
                    ),
                    None,
                )
                if inherited_candidate is None:
                    continue
                conflicting = [
                    candidate
                    for candidate in continuity_candidates
                    if candidate["display_name"] != inherited_name
                ]
                if conflicting:
                    if top_candidate["display_name"] == inherited_name:
                        continue
                    score_field = (
                        "reference_template_score"
                        if "reference_template_score" in top_candidate
                        else "composite_score"
                    )
                    score_gap = float(top_candidate.get(score_field) or 0.0) - float(
                        inherited_candidate.get(score_field) or 0.0
                    )
                    if score_gap > 0.03:
                        continue
            elif not has_selector_evidence and len(operative_names) > 2:
                continue
            _set_selector_resolution(
                guess,
                selector_name=inherited_name,
                selector_source="turn_continuity",
                selector_confidence=max(0.62, continuity_source_confidence - 0.16),
            )

    team_selector_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for turn in analysis.get("turns", []):
        team_color = str(turn.get("team_color") or "")
        for guess in turn.get("guesses", []):
            selector_name = str(guess.get("selector_name") or "")
            if selector_name:
                team_selector_counts[team_color][selector_name] += 1

    dominant_selector_by_team: dict[str, str] = {}
    for team_color, counts in team_selector_counts.items():
        if not counts:
            continue
        ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        dominant_name, dominant_count = ordered[0]
        runner_up_count = ordered[1][1] if len(ordered) > 1 else 0
        if dominant_count >= 2 and runner_up_count == 0:
            dominant_selector_by_team[team_color] = dominant_name

    for turn in analysis.get("turns", []):
        team_color = str(turn.get("team_color") or "")
        operative_names = operative_names_by_team.get(team_color, [])
        dominant_name = dominant_selector_by_team.get(team_color)
        if dominant_name is None or dominant_name not in operative_names:
            continue
        for guess in turn.get("guesses", []):
            if guess.get("selector_name"):
                continue
            resolved_candidates = guess.get("resolved_selector_candidates") or []
            if resolved_candidates:
                candidate_names = {candidate.get("display_name") for candidate in resolved_candidates[:2]}
                if dominant_name not in candidate_names:
                    continue
            parsed_player_name = collapse_whitespace(str(guess.get("player_name") or ""))
            if parsed_player_name and parsed_player_name.casefold() != "unknown":
                matched_name, similarity = _resolve_name_from_text(parsed_player_name, operative_names)
                if matched_name is not None and matched_name != dominant_name and similarity >= 0.85:
                    continue
            _set_selector_resolution(
                guess,
                selector_name=dominant_name,
                selector_source="team_dominance",
                selector_confidence=0.64,
            )

    analysis["selector_attribution_notes"] = {
        "method": "selector crop OCR first, normalized selector candidates second, single-selector turn continuity fallback last",
        "high_confidence_source": "selector_crop",
        "lower_confidence_source": "turn_continuity",
    }


def annotate_guess_results_inplace(analysis: dict[str, Any]) -> None:
    for turn in analysis.get("turns", []):
        team_color = TeamColor(str(turn["team_color"]))
        for guess in turn.get("guesses", []):
            if guess.get("result"):
                continue
            guess["result"] = classify_guess_result(
                team_color,
                CardColor(str(guess["card_color"])),
            ).value
