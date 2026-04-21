from __future__ import annotations

import numpy as np
from PIL import Image

from app.parsers.gamelog import ClueEvent, GuessEvent, _ocr_row_fields, log_has_changed, parse_game_log
from app.core.models import BoardCell, BoardState, ImageBoundingBox, OCRDetection, PlayerRole, PlayerRosterEntry, TeamColor
from app.infra.roi_config import ROIConfig


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
            "board_region": {"x": 0.18, "y": 0.10, "width": 0.60, "height": 0.72},
            "game_log_region": {"x": 0.80, "y": 0.50, "width": 0.18, "height": 0.40},
            "left_counter_region": {"x": 0.05, "y": 0.35, "width": 0.05, "height": 0.08},
            "right_counter_region": {"x": 0.90, "y": 0.35, "width": 0.05, "height": 0.08},
            "top_banner_region": {"x": 0.18, "y": 0.0, "width": 0.60, "height": 0.10},
            "end_banner_region": {"x": 0.18, "y": 0.80, "width": 0.64, "height": 0.16},
        }
    )


def make_roster() -> list[PlayerRosterEntry]:
    return [
        PlayerRosterEntry(
            player_name="fembluca",
            display_name="fembluca",
            team_color=TeamColor.BLUE,
            role=PlayerRole.SPYMASTER,
            avatar_hash="a",
        ),
        PlayerRosterEntry(
            player_name="†",
            display_name="†",
            team_color=TeamColor.BLUE,
            role=PlayerRole.OPERATIVE,
            avatar_hash="b",
        ),
        PlayerRosterEntry(
            player_name="Cherry",
            display_name="Cherry",
            team_color=TeamColor.RED,
            role=PlayerRole.OPERATIVE,
            avatar_hash="c",
        ),
        PlayerRosterEntry(
            player_name="Zek 67",
            display_name="Zek 67",
            team_color=TeamColor.RED,
            role=PlayerRole.SPYMASTER,
            avatar_hash="d",
        ),
    ]


def make_board_state() -> BoardState:
    words = ["ANTARCTICA", "NIGHT", "STAFF", "CLOAK", "TAP"]
    cells = [
        BoardCell(
            row=index // 5,
            col=index % 5,
            word=word,
            confidence=0.95,
            box=ImageBoundingBox(left=0, top=0, right=10, bottom=10),
        )
        for index, word in enumerate(words)
    ]
    return BoardState(cells=cells)


def paint_box(frame: np.ndarray, box: ImageBoundingBox, color: tuple[int, int, int]) -> None:
    frame[box.top : box.bottom, box.left : box.right] = color


def test_log_has_changed_detects_visible_pixel_deltas() -> None:
    previous = np.zeros((80, 120, 3), dtype=np.uint8)
    current = previous.copy()
    current[10:20, 10:20] = (255, 255, 255)

    assert log_has_changed(previous, current)
    assert not log_has_changed(previous, previous.copy())


def test_parse_game_log_extracts_clue_and_guess_events() -> None:
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    roi_config = make_roi_config()
    board_state = make_board_state()
    roster = make_roster()

    log_box = ImageBoundingBox(left=800, top=300, right=980, bottom=540)
    clue_row = ImageBoundingBox(left=12, top=12, right=170, bottom=54)
    blue_guess_row = ImageBoundingBox(left=12, top=64, right=170, bottom=96)
    red_clue_row = ImageBoundingBox(left=12, top=116, right=170, bottom=158)
    red_guess_row = ImageBoundingBox(left=12, top=168, right=170, bottom=200)
    paint_box(frame, ImageBoundingBox(left=log_box.left + clue_row.left, top=log_box.top + clue_row.top, right=log_box.left + clue_row.right, bottom=log_box.top + clue_row.bottom), (230, 140, 60))
    paint_box(frame, ImageBoundingBox(left=log_box.left + blue_guess_row.left, top=log_box.top + blue_guess_row.top, right=log_box.left + blue_guess_row.right, bottom=log_box.top + blue_guess_row.bottom), (255, 80, 80))
    paint_box(frame, ImageBoundingBox(left=log_box.left + red_clue_row.left, top=log_box.top + red_clue_row.top, right=log_box.left + red_clue_row.right, bottom=log_box.top + red_clue_row.bottom), (60, 80, 220))
    paint_box(frame, ImageBoundingBox(left=log_box.left + red_guess_row.left, top=log_box.top + red_guess_row.top, right=log_box.left + red_guess_row.right, bottom=log_box.top + red_guess_row.bottom), (60, 80, 220))

    responses = {
        "game_log": [
            OCRDetection(text="fembluca", confidence=0.98, box=ImageBoundingBox(left=8, top=16, right=70, bottom=34)),
            OCRDetection(text="COLD", confidence=0.99, box=ImageBoundingBox(left=78, top=14, right=132, bottom=36)),
            OCRDetection(text="2", confidence=0.97, box=ImageBoundingBox(left=142, top=14, right=156, bottom=36)),
            OCRDetection(text="†", confidence=0.82, box=ImageBoundingBox(left=8, top=68, right=28, bottom=86)),
            OCRDetection(text="ANTARCTlCA", confidence=0.88, box=ImageBoundingBox(left=38, top=66, right=130, bottom=88)),
            OCRDetection(text="Zek 67", confidence=0.95, box=ImageBoundingBox(left=8, top=120, right=62, bottom=138)),
            OCRDetection(text="JAX", confidence=0.97, box=ImageBoundingBox(left=72, top=118, right=118, bottom=140)),
            OCRDetection(text="3", confidence=0.96, box=ImageBoundingBox(left=140, top=118, right=152, bottom=140)),
            OCRDetection(text="Cherry", confidence=0.94, box=ImageBoundingBox(left=8, top=172, right=54, bottom=190)),
            OCRDetection(text="TAP", confidence=0.93, box=ImageBoundingBox(left=64, top=170, right=98, bottom=192)),
        ]
    }

    events = parse_game_log(
        frame,
        roi_config,
        FakeOCRBackend(responses),
        roster,
        board_state,
        timestamp_sec=123.0,
    )

    assert len(events) == 4
    assert isinstance(events[0], ClueEvent)
    assert events[0].clue_text == "COLD"
    assert events[0].team_color is TeamColor.BLUE
    assert isinstance(events[1], GuessEvent)
    assert events[1].player_name == "†"
    assert events[1].word == "ANTARCTICA"
    assert events[1].card_color.name == "BLUE"
    assert isinstance(events[2], ClueEvent)
    assert events[2].team_color is TeamColor.RED
    assert isinstance(events[3], GuessEvent)
    assert events[3].player_name == "Cherry"
    assert events[3].word == "TAP"
    assert events[3].card_color.name == "RED"


def test_parse_game_log_can_use_avatar_match_when_tiny_name_is_missing() -> None:
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    roi_config = make_roi_config()
    board_state = make_board_state()
    roster = make_roster()
    log_box = ImageBoundingBox(left=800, top=300, right=980, bottom=540)
    guess_row = ImageBoundingBox(left=12, top=64, right=170, bottom=96)
    paint_box(frame, ImageBoundingBox(left=log_box.left + guess_row.left, top=log_box.top + guess_row.top, right=log_box.left + guess_row.right, bottom=log_box.top + guess_row.bottom), (255, 80, 80))

    responses = {
        "game_log": [
            OCRDetection(text="ANTARCTICA", confidence=0.92, box=ImageBoundingBox(left=38, top=66, right=130, bottom=88)),
        ]
    }

    events = parse_game_log(
        frame,
        roi_config,
        FakeOCRBackend(responses),
        roster,
        board_state,
        timestamp_sec=123.0,
        avatar_matches_by_row={0: "†"},
    )

    assert len(events) == 1
    assert isinstance(events[0], GuessEvent)
    assert events[0].player_name == "†"


def test_parse_game_log_fuzzy_matches_small_name_ocr_to_roster() -> None:
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    roi_config = make_roi_config()
    board_state = make_board_state()
    roster = make_roster()
    log_box = ImageBoundingBox(left=800, top=300, right=980, bottom=540)
    guess_row = ImageBoundingBox(left=12, top=64, right=170, bottom=96)
    paint_box(
        frame,
        ImageBoundingBox(
            left=log_box.left + guess_row.left,
            top=log_box.top + guess_row.top,
            right=log_box.left + guess_row.right,
            bottom=log_box.top + guess_row.bottom,
        ),
        (255, 80, 80),
    )

    responses = {
        "game_log": [
            OCRDetection(text="Chery", confidence=0.70, box=ImageBoundingBox(left=8, top=68, right=46, bottom=86)),
            OCRDetection(text="TAP", confidence=0.93, box=ImageBoundingBox(left=64, top=66, right=98, bottom=88)),
        ]
    }

    events = parse_game_log(
        frame,
        roi_config,
        FakeOCRBackend(responses),
        roster,
        board_state,
        timestamp_sec=123.0,
    )

    assert len(events) == 1
    assert isinstance(events[0], GuessEvent)
    assert events[0].player_name == "Cherry"


def test_parse_game_log_fuzzy_matches_log_name_to_roster_name() -> None:
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    roi_config = make_roi_config()
    board_state = make_board_state()
    roster = make_roster()
    log_box = ImageBoundingBox(left=800, top=300, right=980, bottom=540)
    guess_row = ImageBoundingBox(left=12, top=168, right=170, bottom=200)
    paint_box(frame, ImageBoundingBox(left=log_box.left + guess_row.left, top=log_box.top + guess_row.top, right=log_box.left + guess_row.right, bottom=log_box.top + guess_row.bottom), (60, 80, 220))

    responses = {
        "game_log": [
            OCRDetection(text="Cherny", confidence=0.81, box=ImageBoundingBox(left=8, top=172, right=54, bottom=190)),
            OCRDetection(text="TAP", confidence=0.93, box=ImageBoundingBox(left=64, top=170, right=98, bottom=192)),
        ]
    }

    events = parse_game_log(
        frame,
        roi_config,
        FakeOCRBackend(responses),
        roster,
        board_state,
        timestamp_sec=123.0,
    )

    assert len(events) == 1
    assert isinstance(events[0], GuessEvent)
    assert events[0].player_name == "Cherry"


def test_parse_game_log_prefers_guess_word_to_the_right_of_player_lane() -> None:
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    roi_config = make_roi_config()
    board_state = BoardState(
        cells=[
            BoardCell(
                row=index // 5,
                col=index % 5,
                word=word,
                confidence=0.95,
                box=ImageBoundingBox(left=0, top=0, right=10, bottom=10),
            )
            for index, word in enumerate(["NIGHT", "TAP"])
        ]
    )
    roster = make_roster()
    log_box = ImageBoundingBox(left=800, top=300, right=980, bottom=540)
    guess_row = ImageBoundingBox(left=12, top=168, right=170, bottom=200)
    paint_box(frame, ImageBoundingBox(left=log_box.left + guess_row.left, top=log_box.top + guess_row.top, right=log_box.left + guess_row.right, bottom=log_box.top + guess_row.bottom), (60, 80, 220))

    responses = {
        "game_log": [
            OCRDetection(text="Cherry", confidence=0.94, box=ImageBoundingBox(left=8, top=172, right=54, bottom=190)),
            OCRDetection(text="NIGHT", confidence=0.93, box=ImageBoundingBox(left=44, top=170, right=82, bottom=192)),
            OCRDetection(text="TAP", confidence=0.97, box=ImageBoundingBox(left=98, top=170, right=128, bottom=192)),
        ]
    }

    events = parse_game_log(
        frame,
        roi_config,
        FakeOCRBackend(responses),
        roster,
        board_state,
        timestamp_sec=123.0,
    )

    assert len(events) == 1
    assert isinstance(events[0], GuessEvent)
    assert events[0].word == "TAP"


def test_parse_game_log_joins_split_clue_fragments_and_prefers_rightmost_count_token() -> None:
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    roi_config = make_roi_config()
    board_state = make_board_state()
    roster = make_roster()
    log_box = ImageBoundingBox(left=800, top=300, right=980, bottom=540)
    clue_row = ImageBoundingBox(left=12, top=116, right=170, bottom=158)
    paint_box(
        frame,
        ImageBoundingBox(
            left=log_box.left + clue_row.left,
            top=log_box.top + clue_row.top,
            right=log_box.left + clue_row.right,
            bottom=log_box.top + clue_row.bottom,
        ),
        (60, 80, 220),
    )

    responses = {
        "game_log": [
            OCRDetection(text="Zek 67", confidence=0.95, box=ImageBoundingBox(left=8, top=120, right=62, bottom=138)),
            OCRDetection(text="LEATH", confidence=0.90, box=ImageBoundingBox(left=72, top=118, right=118, bottom=140)),
            OCRDetection(text="ER", confidence=0.88, box=ImageBoundingBox(left=120, top=118, right=136, bottom=140)),
            OCRDetection(text="64", confidence=0.86, box=ImageBoundingBox(left=140, top=118, right=154, bottom=140)),
        ]
    }

    events = parse_game_log(
        frame,
        roi_config,
        FakeOCRBackend(responses),
        roster,
        board_state,
        timestamp_sec=123.0,
    )

    assert len(events) == 1
    assert isinstance(events[0], ClueEvent)
    assert events[0].spymaster_name == "Zek 67"
    assert events[0].clue_text == "LEATHER"
    assert events[0].clue_count == "4"


def test_parse_game_log_ignores_game_log_header_row_as_clue() -> None:
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    roi_config = make_roi_config()
    board_state = make_board_state()
    roster = make_roster()
    log_box = ImageBoundingBox(left=800, top=300, right=980, bottom=540)
    clue_row = ImageBoundingBox(left=12, top=12, right=170, bottom=54)
    paint_box(
        frame,
        ImageBoundingBox(
            left=log_box.left + clue_row.left,
            top=log_box.top + clue_row.top,
            right=log_box.left + clue_row.right,
            bottom=log_box.top + clue_row.bottom,
        ),
        (60, 80, 220),
    )

    responses = {
        "game_log": [
            OCRDetection(text="Zek 67", confidence=0.95, box=ImageBoundingBox(left=8, top=16, right=62, bottom=34)),
            OCRDetection(text="GAMELOG", confidence=0.91, box=ImageBoundingBox(left=72, top=14, right=136, bottom=36)),
            OCRDetection(text="1", confidence=0.86, box=ImageBoundingBox(left=142, top=14, right=156, bottom=36)),
        ]
    }

    events = parse_game_log(
        frame,
        roi_config,
        FakeOCRBackend(responses),
        roster,
        board_state,
        timestamp_sec=123.0,
    )

    assert events == []


def test_parse_game_log_rejects_board_word_like_clue_rows() -> None:
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    roi_config = make_roi_config()
    board_state = make_board_state()
    roster = make_roster()
    log_box = ImageBoundingBox(left=800, top=300, right=980, bottom=540)
    clue_row = ImageBoundingBox(left=12, top=12, right=170, bottom=54)
    paint_box(
        frame,
        ImageBoundingBox(
            left=log_box.left + clue_row.left,
            top=log_box.top + clue_row.top,
            right=log_box.left + clue_row.right,
            bottom=log_box.top + clue_row.bottom,
        ),
        (60, 80, 220),
    )

    responses = {
        "game_log": [
            OCRDetection(text="Zek 67", confidence=0.95, box=ImageBoundingBox(left=8, top=16, right=62, bottom=34)),
            OCRDetection(text="TAP", confidence=0.90, box=ImageBoundingBox(left=72, top=14, right=118, bottom=36)),
            OCRDetection(text="6", confidence=0.86, box=ImageBoundingBox(left=142, top=14, right=156, bottom=36)),
        ]
    }

    events = parse_game_log(
        frame,
        roi_config,
        FakeOCRBackend(responses),
        roster,
        board_state,
        timestamp_sec=123.0,
    )

    assert len(events) == 1
    assert isinstance(events[0], GuessEvent)
    assert events[0].word == "TAP"


def test_parse_game_log_does_not_snap_guess_to_global_dictionary_only_words() -> None:
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    roi_config = make_roi_config()
    board_state = BoardState(
        cells=[
            BoardCell(
                row=index // 5,
                col=index % 5,
                word=word,
                confidence=0.95,
                box=ImageBoundingBox(left=0, top=0, right=10, bottom=10),
            )
            for index, word in enumerate(["NIGHT", "TAP", "STAFF", "CLOAK", "MAP"])
        ]
    )
    roster = make_roster()
    log_box = ImageBoundingBox(left=800, top=300, right=980, bottom=540)
    guess_row = ImageBoundingBox(left=12, top=168, right=170, bottom=200)
    paint_box(
        frame,
        ImageBoundingBox(
            left=log_box.left + guess_row.left,
            top=log_box.top + guess_row.top,
            right=log_box.left + guess_row.right,
            bottom=log_box.top + guess_row.bottom,
        ),
        (60, 80, 220),
    )

    responses = {
        "game_log": [
            OCRDetection(text="Cherry", confidence=0.94, box=ImageBoundingBox(left=8, top=172, right=54, bottom=190)),
            OCRDetection(text="Antarctlca", confidence=0.93, box=ImageBoundingBox(left=64, top=170, right=130, bottom=192)),
        ]
    }

    events = parse_game_log(
        frame,
        roi_config,
        FakeOCRBackend(responses),
        roster,
        board_state,
        timestamp_sec=123.0,
    )

    assert events == []


def test_blank_game_log_returns_no_rows_not_exception() -> None:
    frame = np.asarray(Image.open("build/debug/game3/smoke/smoke-start-000000.000.png").convert("RGB"))[:, :, ::-1].copy()
    roi_config = make_roi_config()

    events = parse_game_log(
        frame,
        roi_config,
        FakeOCRBackend({}),
        make_roster(),
        make_board_state(),
        timestamp_sec=0.0,
    )

    assert events == []


def test_invalid_row_field_geometry_is_skipped_not_crashed() -> None:
    log_frame = np.zeros((120, 250, 3), dtype=np.uint8)
    row_box = ImageBoundingBox(left=357, top=0, right=400, bottom=100)

    fields = _ocr_row_fields(
        log_frame,
        row_box,
        row_index=0,
        ocr_backend=FakeOCRBackend({}),
        row_detections=[],
    )

    assert fields["team_strip"].box is None
    assert fields["name"].box is None
    assert fields["main"].box is None
    assert fields["count"].box is None
    assert fields["result"].box is None
