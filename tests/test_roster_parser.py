from __future__ import annotations

import numpy as np

from app.models import ImageBoundingBox, OCRDetection, PlayerRole, TeamColor
from app.roi_config import ROIConfig
from app.roster_parser import canonicalize_player_name, parse_rosters


class FakeOCRBackend:
    def __init__(self, responses: dict[str, list[OCRDetection]]) -> None:
        self.responses = responses

    def detect_text(self, image: np.ndarray, *, hint: str | None = None) -> list[OCRDetection]:
        assert image.size > 0
        return list(self.responses.get(hint or "", []))


def make_roi_config() -> ROIConfig:
    return ROIConfig.from_raw(
        {
            "left_team_panel": {"x": 0.0, "y": 0.0, "width": 0.18, "height": 1.0},
            "right_team_panel": {"x": 0.82, "y": 0.0, "width": 0.18, "height": 1.0},
            "board_region": {"x": 0.18, "y": 0.10, "width": 0.64, "height": 0.72},
            "game_log_region": {"x": 0.82, "y": 0.40, "width": 0.18, "height": 0.40},
            "left_counter_region": {"x": 0.05, "y": 0.35, "width": 0.05, "height": 0.08},
            "right_counter_region": {"x": 0.90, "y": 0.35, "width": 0.05, "height": 0.08},
            "top_banner_region": {"x": 0.18, "y": 0.0, "width": 0.64, "height": 0.10},
            "end_banner_region": {"x": 0.18, "y": 0.80, "width": 0.64, "height": 0.16},
        }
    )


def test_canonicalize_player_name_collapses_whitespace() -> None:
    assert canonicalize_player_name("  Zek   67 ") == "zek 67"
    assert canonicalize_player_name(" † ") == "†"


def test_parse_rosters_prefers_in_game_layout_and_preserves_symbol_names() -> None:
    frame = np.zeros((900, 1400, 3), dtype=np.uint8)
    roi_config = make_roi_config()
    box = ImageBoundingBox(left=40, top=90, right=110, bottom=120)
    responses = {
        "blue:operative:in_game": [
            OCRDetection(text="Urethra", confidence=0.96, box=box),
            OCRDetection(
                text="†",
                confidence=0.82,
                box=ImageBoundingBox(left=140, top=90, right=180, bottom=120),
            ),
        ],
        "blue:spymaster:in_game": [
            OCRDetection(text="fembluca", confidence=0.95, box=box),
        ],
        "red:operative:in_game": [
            OCRDetection(text="Cherry", confidence=0.97, box=box),
        ],
        "red:spymaster:in_game": [
            OCRDetection(text="Zek 67", confidence=0.94, box=box),
        ],
    }

    players = parse_rosters(frame, roi_config, FakeOCRBackend(responses))

    assert len(players) == 5
    assert [player.display_name for player in players] == [
        "Urethra",
        "†",
        "fembluca",
        "Cherry",
        "Zek 67",
    ]
    assert players[1].player_name == "†"
    assert players[1].team_color == TeamColor.BLUE
    assert players[1].role == PlayerRole.OPERATIVE
    assert all(player.avatar_hash is not None for player in players)


def test_parse_rosters_ignores_ui_labels() -> None:
    frame = np.zeros((900, 1400, 3), dtype=np.uint8)
    roi_config = make_roi_config()
    responses = {
        "blue:operative:in_game": [
            OCRDetection(
                text="OPERATIVES",
                confidence=0.99,
                box=ImageBoundingBox(left=20, top=20, right=100, bottom=40),
            ),
            OCRDetection(
                text="Urethra",
                confidence=0.95,
                box=ImageBoundingBox(left=40, top=90, right=120, bottom=120),
            ),
        ],
        "blue:spymaster:in_game": [
            OCRDetection(
                text="JOIN TEAM",
                confidence=0.99,
                box=ImageBoundingBox(left=20, top=20, right=90, bottom=40),
            ),
            OCRDetection(
                text="fembluca",
                confidence=0.92,
                box=ImageBoundingBox(left=40, top=90, right=120, bottom=120),
            ),
        ],
    }

    players = parse_rosters(frame, roi_config, FakeOCRBackend(responses))

    assert [player.display_name for player in players] == ["Urethra", "fembluca"]
