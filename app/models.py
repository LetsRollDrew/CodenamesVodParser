"""Typed parser models and SQLAlchemy ORM schema."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import CheckConstraint, Enum as SAEnum, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def enum_column(enum_cls: type[StrEnum], *, name: str) -> SAEnum:
    """Return a portable SQLAlchemy enum column definition."""

    return SAEnum(enum_cls, name=name, native_enum=False, validate_strings=True)


class SegmentType(StrEnum):
    CODENAMES = "codenames"


class TeamColor(StrEnum):
    BLUE = "blue"
    RED = "red"


class CardColor(StrEnum):
    BLUE = "blue"
    RED = "red"
    NEUTRAL = "neutral"
    BLACK = "black"


class PlayerRole(StrEnum):
    OPERATIVE = "operative"
    SPYMASTER = "spymaster"


class WinReason(StrEnum):
    COUNTER_ZERO = "counter_zero"
    ASSASSIN = "assassin"


class GuessResult(StrEnum):
    CORRECT = "correct"
    ENEMY = "enemy"
    NEUTRAL = "neutral"
    ASSASSIN = "assassin"


class ReviewEntityType(StrEnum):
    VOD = "vod"
    SEGMENT = "segment"
    GAME = "game"
    PLAYER = "player"
    TURN = "turn"
    GUESS = "guess"


class VodMeta(BaseModel):
    vod_id: str
    streamer_login: str
    created_at: datetime
    title: str
    url: str
    duration_seconds: int = Field(ge=0)


class SegmentBounds(BaseModel):
    vod_id: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    segment_type: SegmentType = SegmentType.CODENAMES


class PlayerRosterEntry(BaseModel):
    player_name: str
    display_name: str
    team_color: TeamColor
    role: PlayerRole
    avatar_hash: str | None = None


class ImageBoundingBox(BaseModel):
    left: int = Field(ge=0)
    top: int = Field(ge=0)
    right: int = Field(ge=0)
    bottom: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ImageBoundingBox":
        if self.right <= self.left:
            raise ValueError("right must be greater than left")
        if self.bottom <= self.top:
            raise ValueError("bottom must be greater than top")
        return self

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2


class OCRDetection(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1)
    box: ImageBoundingBox


class BoardCell(BaseModel):
    row: int = Field(ge=0, lt=5)
    col: int = Field(ge=0, lt=5)
    word: str
    confidence: float = Field(ge=0, le=1)
    box: ImageBoundingBox


class BoardState(BaseModel):
    cells: list[BoardCell] = Field(default_factory=list)

    @property
    def words(self) -> list[str]:
        return [cell.word for cell in self.cells]


class GuessRecord(BaseModel):
    player_name: str
    word: str
    card_color: CardColor
    result: GuessResult
    timestamp_sec: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)


class TurnRecord(BaseModel):
    turn_index: int = Field(ge=0)
    team_color: TeamColor
    spymaster_name: str
    clue_text: str
    clue_count: str
    timestamp_sec: float = Field(ge=0)
    guesses: list[GuessRecord] = Field(default_factory=list)


class GameRecord(BaseModel):
    vod_id: str
    game_index: int = Field(ge=0)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    winner_team: TeamColor | None = None
    win_reason: WinReason | None = None
    starting_team: TeamColor | None = None
    parse_confidence: float = Field(ge=0, le=1)
    players: list[PlayerRosterEntry] = Field(default_factory=list)
    turns: list[TurnRecord] = Field(default_factory=list)


class Base(DeclarativeBase):
    """Base ORM model."""


class Vod(Base):
    __tablename__ = "vods"

    vod_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    streamer_login: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False)

    segments: Mapped[list["Segment"]] = relationship(
        back_populates="vod",
        cascade="all, delete-orphan",
        order_by="Segment.start_sec",
    )
    games: Mapped[list["Game"]] = relationship(
        back_populates="vod",
        cascade="all, delete-orphan",
        order_by="Game.game_index",
    )

    __table_args__ = (
        CheckConstraint("duration_seconds >= 0", name="ck_vods_duration_seconds_nonnegative"),
    )


class Segment(Base):
    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    vod_id: Mapped[str] = mapped_column(
        ForeignKey("vods.vod_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    segment_type: Mapped[SegmentType] = mapped_column(
        enum_column(SegmentType, name="segment_type"),
        nullable=False,
        default=SegmentType.CODENAMES,
    )
    start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    end_sec: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    vod: Mapped["Vod"] = relationship(back_populates="segments")
    games: Mapped[list["Game"]] = relationship(
        back_populates="segment",
        cascade="all, delete-orphan",
        order_by="Game.game_index",
    )

    __table_args__ = (
        CheckConstraint("start_sec >= 0", name="ck_segments_start_sec_nonnegative"),
        CheckConstraint("end_sec >= start_sec", name="ck_segments_end_after_start"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_segments_confidence_range"),
        UniqueConstraint("vod_id", "start_sec", "end_sec", name="uq_segments_vod_bounds"),
    )


class Game(Base):
    __tablename__ = "games"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    vod_id: Mapped[str] = mapped_column(
        ForeignKey("vods.vod_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    segment_id: Mapped[int] = mapped_column(
        ForeignKey("segments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    game_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    end_sec: Mapped[float] = mapped_column(Float, nullable=False)
    winner_team: Mapped[TeamColor | None] = mapped_column(
        enum_column(TeamColor, name="winner_team"),
        nullable=True,
    )
    win_reason: Mapped[WinReason | None] = mapped_column(
        enum_column(WinReason, name="win_reason"),
        nullable=True,
    )
    starting_team: Mapped[TeamColor | None] = mapped_column(
        enum_column(TeamColor, name="starting_team"),
        nullable=True,
    )
    parse_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    vod: Mapped["Vod"] = relationship(back_populates="games")
    segment: Mapped["Segment"] = relationship(back_populates="games")
    players: Mapped[list["Player"]] = relationship(
        back_populates="game",
        cascade="all, delete-orphan",
        order_by="Player.id",
    )
    turns: Mapped[list["Turn"]] = relationship(
        back_populates="game",
        cascade="all, delete-orphan",
        order_by="Turn.turn_index",
    )

    __table_args__ = (
        CheckConstraint("game_index >= 0", name="ck_games_index_nonnegative"),
        CheckConstraint("start_sec >= 0", name="ck_games_start_sec_nonnegative"),
        CheckConstraint("end_sec >= start_sec", name="ck_games_end_after_start"),
        CheckConstraint("parse_confidence >= 0 AND parse_confidence <= 1", name="ck_games_confidence_range"),
        UniqueConstraint("segment_id", "game_index", name="uq_games_segment_index"),
    )


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(
        ForeignKey("games.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    player_name: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    team_color: Mapped[TeamColor] = mapped_column(
        enum_column(TeamColor, name="team_color"),
        nullable=False,
    )
    role: Mapped[PlayerRole] = mapped_column(
        enum_column(PlayerRole, name="player_role"),
        nullable=False,
    )
    avatar_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)

    game: Mapped["Game"] = relationship(back_populates="players")

    __table_args__ = (
        UniqueConstraint("game_id", "player_name", "team_color", "role", name="uq_players_identity"),
    )


class Turn(Base):
    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(
        ForeignKey("games.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    team_color: Mapped[TeamColor] = mapped_column(
        enum_column(TeamColor, name="turn_team_color"),
        nullable=False,
    )
    spymaster_name: Mapped[str] = mapped_column(String(255), nullable=False)
    clue_text: Mapped[str] = mapped_column(String(255), nullable=False)
    clue_count: Mapped[str] = mapped_column(String(32), nullable=False)
    timestamp_sec: Mapped[float] = mapped_column(Float, nullable=False)

    game: Mapped["Game"] = relationship(back_populates="turns")
    guesses: Mapped[list["Guess"]] = relationship(
        back_populates="turn",
        cascade="all, delete-orphan",
        order_by="Guess.guess_index",
    )

    __table_args__ = (
        CheckConstraint("turn_index >= 0", name="ck_turns_index_nonnegative"),
        CheckConstraint("timestamp_sec >= 0", name="ck_turns_timestamp_nonnegative"),
        UniqueConstraint("game_id", "turn_index", name="uq_turns_game_index"),
    )


class Guess(Base):
    __tablename__ = "guesses"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    turn_id: Mapped[int] = mapped_column(
        ForeignKey("turns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    guess_index: Mapped[int] = mapped_column(Integer, nullable=False)
    player_name: Mapped[str] = mapped_column(String(255), nullable=False)
    word: Mapped[str] = mapped_column(String(255), nullable=False)
    card_color: Mapped[CardColor] = mapped_column(
        enum_column(CardColor, name="card_color"),
        nullable=False,
    )
    result: Mapped[GuessResult] = mapped_column(
        enum_column(GuessResult, name="guess_result"),
        nullable=False,
    )
    timestamp_sec: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    turn: Mapped["Turn"] = relationship(back_populates="guesses")

    __table_args__ = (
        CheckConstraint("guess_index >= 0", name="ck_guesses_index_nonnegative"),
        CheckConstraint("timestamp_sec >= 0", name="ck_guesses_timestamp_nonnegative"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_guesses_confidence_range"),
        UniqueConstraint("turn_id", "guess_index", name="uq_guesses_turn_index"),
    )


class ReviewQueueItem(Base):
    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    entity_type: Mapped[ReviewEntityType] = mapped_column(
        enum_column(ReviewEntityType, name="review_entity_type"),
        nullable=False,
        index=True,
    )
    entity_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    snapshot_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        CheckConstraint(
            "(timestamp_sec IS NULL) OR (timestamp_sec >= 0)",
            name="ck_review_queue_timestamp_nonnegative",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_review_queue_confidence_range",
        ),
    )
