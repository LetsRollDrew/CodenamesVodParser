"""Typed smoke-parser result models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.models import GameRecord, PlayerRosterEntry


class SmokeChangeEvent(BaseModel):
    timestamp_sec: float = Field(ge=0)
    new_events: list[dict[str, object]] = Field(default_factory=list)
    left_counter: int | None = Field(default=None, ge=0)
    right_counter: int | None = Field(default=None, ge=0)
    snapshot_path: str | None = None


class SmokeParseResult(BaseModel):
    vod_url: str
    requested_start_sec: float = Field(ge=0)
    stop_sec: float = Field(ge=0)
    stop_reason: Literal["counter_zero", "assassin", "play_next", "winner_banner", "window_exhausted"]
    game_record: GameRecord | None = None
    review_items: list[dict[str, object]] = Field(default_factory=list)
    roster: list[PlayerRosterEntry] = Field(default_factory=list)
    board_words: list[str] = Field(default_factory=list)
    change_events: list[SmokeChangeEvent] = Field(default_factory=list)
    first_frame_snapshot_path: str | None = None
    first_preready_snapshot_path: str | None = None
    roi_probe_overlay_path: str | None = None
    roi_probe_crop_paths: dict[str, str] = Field(default_factory=dict)
    start_snapshot_path: str | None = None
