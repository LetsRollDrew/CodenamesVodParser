"""Typed smoke-parser result models"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from app.core.models import GameRecord, PlayerRosterEntry, TeamColor
from app.parsers.gamelog import ClueEvent, GuessEvent
from app.vision.detection.frame_detectors import CounterObservation


@dataclass(slots=True)
class BannerObservation:
    timestamp_sec: float
    top_banner_text: str = ""
    clue_text: str | None = None
    clue_count: str | None = None

    @property
    def signature(self) -> tuple[str, str | None] | None:
        if not self.clue_text:
            return None
        return (self.clue_text, self.clue_count)


@dataclass(slots=True)
class VisibleLogState:
    previous_log_frame: object | None = None
    previous_visible_events: list[ClueEvent | GuessEvent] = field(default_factory=list)
    quiet_log_frame_count: int = 0
    current_row_signatures: list[tuple[str, str, str, str]] = field(default_factory=list)
    recent_row_registry: dict[tuple[tuple[str, str | None] | None, tuple[str, str, str, str]], float] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class ActiveClueWindow:
    clue_text: str
    clue_count: str | None
    team_color: TeamColor | None
    first_seen_ts: float
    last_seen_ts: float
    confidence: float = 0.0
    sources: set[str] = field(default_factory=set)
    emitted: bool = False

    @property
    def signature(self) -> tuple[str, str | None]:
        return (self.clue_text, self.clue_count)


@dataclass(slots=True)
class PendingClueCandidate:
    event: ClueEvent
    first_seen_ts: float
    last_seen_ts: float
    phase_first_seen: str
    last_phase: str
    consecutive_frames: int = 1
    best_confidence: float = 0.0
    sources: set[str] = field(default_factory=set)
    banner_confirmed: bool = False
    visible_log_confirmed: bool = False
    contamination_flags: set[str] = field(default_factory=set)
    saw_same_frame_guesses: bool = False


@dataclass(slots=True)
class PendingGuessEvidence:
    event: GuessEvent
    source: str
    first_seen_ts: float
    last_seen_ts: float
    confirmations: int = 1


@dataclass(slots=True)
class FrameObservations:
    timestamp_sec: float
    log_changed: bool
    banner: BannerObservation
    visible_events: list[ClueEvent | GuessEvent] = field(default_factory=list)
    board_guess_events: list[GuessEvent] = field(default_factory=list)
    left_counter: int | None = None
    right_counter: int | None = None


@dataclass(slots=True)
class CounterRuntimeState:
    stable_value: int | None = None
    last_observation: CounterObservation | None = None
    zero_streak: int = 0
    confirmed_zero: bool = False


@dataclass(slots=True)
class SmokeRuntimeState:
    history: list[ClueEvent | GuessEvent] = field(default_factory=list)
    visible_log: VisibleLogState = field(default_factory=VisibleLogState)
    active_clue_window: ActiveClueWindow | None = None
    pending_clue: PendingClueCandidate | None = None
    banner_signature: tuple[str, str | None] | None = None
    pending_guess_evidence: list[PendingGuessEvidence] = field(default_factory=list)
    pending_assassin_guess: PendingGuessEvidence | None = None
    last_left_counter: int | None = None
    last_right_counter: int | None = None
    left_counter_state: CounterRuntimeState = field(default_factory=CounterRuntimeState)
    right_counter_state: CounterRuntimeState = field(default_factory=CounterRuntimeState)
    board_locked: bool = False
    board_lock_timestamp: float | None = None


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
