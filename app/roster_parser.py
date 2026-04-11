"""Roster parsing for the left and right team panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image

from app.models import ImageBoundingBox, OCRDetection, PlayerRole, PlayerRosterEntry, TeamColor
from app.ocr import OCRBackend, collapse_whitespace
from app.roi_config import ROIConfig, crop_roi

_EXCLUDED_NAME_TEXTS = {
    "operatives",
    "spymasters",
    "join team",
    "blue team",
    "red team",
    "codenames app",
    "play anywhere, anytime!",
}


@dataclass(frozen=True, slots=True)
class PanelLayoutProfile:
    name: str
    operative_section: tuple[float, float, float, float]
    spymaster_section: tuple[float, float, float, float]


PANEL_LAYOUTS: tuple[PanelLayoutProfile, ...] = (
    PanelLayoutProfile(
        name="in_game",
        operative_section=(0.05, 0.08, 0.90, 0.28),
        spymaster_section=(0.05, 0.52, 0.90, 0.26),
    ),
    PanelLayoutProfile(
        name="lobby",
        operative_section=(0.06, 0.14, 0.88, 0.32),
        spymaster_section=(0.06, 0.50, 0.88, 0.32),
    ),
)


def canonicalize_player_name(text: str) -> str:
    """Create a stable lookup key while preserving the display name separately."""

    return collapse_whitespace(text).casefold()


def average_hash(image: np.ndarray, *, hash_size: int = 8) -> str:
    """Compute a simple perceptual hash for avatar crops."""

    if image.size == 0:
        return ""
    grayscale = Image.fromarray(image).convert("L").resize((hash_size, hash_size))
    pixels = np.asarray(grayscale, dtype=np.float32)
    threshold = float(pixels.mean())
    bits = "".join("1" if value >= threshold else "0" for value in pixels.flatten())
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def parse_rosters(
    frame: np.ndarray,
    roi_config: ROIConfig,
    ocr_backend: OCRBackend,
) -> list[PlayerRosterEntry]:
    """Parse player rosters from the blue and red side panels."""

    players: list[PlayerRosterEntry] = []
    players.extend(
        _parse_team_panel(
            crop_roi(frame, roi_config.require("left_team_panel")),
            team_color=TeamColor.BLUE,
            ocr_backend=ocr_backend,
        )
    )
    players.extend(
        _parse_team_panel(
            crop_roi(frame, roi_config.require("right_team_panel")),
            team_color=TeamColor.RED,
            ocr_backend=ocr_backend,
        )
    )
    return players


def _parse_team_panel(
    panel_frame: np.ndarray,
    *,
    team_color: TeamColor,
    ocr_backend: OCRBackend,
) -> list[PlayerRosterEntry]:
    best_entries: list[PlayerRosterEntry] = []
    best_score = -1.0

    for layout in PANEL_LAYOUTS:
        entries: list[PlayerRosterEntry] = []
        entries.extend(
            _parse_role_section(
                _crop_relative(panel_frame, layout.operative_section),
                team_color=team_color,
                role=PlayerRole.OPERATIVE,
                ocr_backend=ocr_backend,
                layout_name=layout.name,
            )
        )
        entries.extend(
            _parse_role_section(
                _crop_relative(panel_frame, layout.spymaster_section),
                team_color=team_color,
                role=PlayerRole.SPYMASTER,
                ocr_backend=ocr_backend,
                layout_name=layout.name,
            )
        )
        score = float(len(entries)) + sum(len(entry.display_name) for entry in entries) / 100.0
        if score > best_score:
            best_score = score
            best_entries = entries

    return best_entries


def _parse_role_section(
    section_frame: np.ndarray,
    *,
    team_color: TeamColor,
    role: PlayerRole,
    ocr_backend: OCRBackend,
    layout_name: str,
) -> list[PlayerRosterEntry]:
    hint = f"{team_color.value}:{role.value}:{layout_name}"
    detections = _filter_name_detections(ocr_backend.detect_text(section_frame, hint=hint))
    entries: list[PlayerRosterEntry] = []

    for detection in sorted(detections, key=lambda item: item.box.center_x):
        display_name = collapse_whitespace(detection.text)
        if not display_name:
            continue

        avatar_box = _estimate_avatar_box(section_frame.shape, detection.box)
        avatar_crop = section_frame[avatar_box.top : avatar_box.bottom, avatar_box.left : avatar_box.right]
        entries.append(
            PlayerRosterEntry(
                player_name=canonicalize_player_name(display_name),
                display_name=display_name,
                team_color=team_color,
                role=role,
                avatar_hash=average_hash(avatar_crop),
            )
        )

    deduped: dict[tuple[str, TeamColor, PlayerRole], PlayerRosterEntry] = {}
    for entry in entries:
        deduped[(entry.player_name, entry.team_color, entry.role)] = entry
    return list(deduped.values())


def _filter_name_detections(detections: Iterable[OCRDetection]) -> list[OCRDetection]:
    filtered: list[OCRDetection] = []
    for detection in detections:
        text = collapse_whitespace(detection.text)
        if not text:
            continue
        if text.casefold() in _EXCLUDED_NAME_TEXTS:
            continue
        filtered.append(detection.model_copy(update={"text": text}))
    return filtered


def _estimate_avatar_box(
    section_shape: tuple[int, ...],
    name_box: ImageBoundingBox,
) -> ImageBoundingBox:
    height, width = section_shape[:2]
    avatar_width = max(int(name_box.width * 1.35), int(width * 0.16))
    avatar_width = min(avatar_width, max(int(width * 0.42), 1))
    avatar_height = min(max(int(height * 0.38), 1), max(int(height * 0.52), 1))
    center_x = int(round(name_box.center_x))
    gap = max(int(height * 0.03), 1)
    bottom = max(name_box.top - gap, avatar_height)
    top = max(bottom - avatar_height, 0)
    left = max(center_x - (avatar_width // 2), 0)
    right = min(left + avatar_width, width)
    if right - left < avatar_width:
        left = max(right - avatar_width, 0)
    return ImageBoundingBox(left=left, top=top, right=right, bottom=min(bottom, height))


def _crop_relative(frame: np.ndarray, relative_box: tuple[float, float, float, float]) -> np.ndarray:
    x, y, width, height = relative_box
    frame_height, frame_width = frame.shape[:2]
    left = int(round(frame_width * x))
    top = int(round(frame_height * y))
    right = int(round(frame_width * (x + width)))
    bottom = int(round(frame_height * (y + height)))
    return frame[top:bottom, left:right].copy()
