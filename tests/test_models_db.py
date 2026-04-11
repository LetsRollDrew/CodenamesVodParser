from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from app.db import create_session_factory, create_sqlalchemy_engine, init_db, session_scope
from app.models import (
    Base,
    CardColor,
    Game,
    GameRecord,
    Guess,
    GuessResult,
    Player,
    PlayerRole,
    ReviewEntityType,
    ReviewQueueItem,
    Segment,
    SegmentType,
    TeamColor,
    Turn,
    Vod,
    VodMeta,
    WinReason,
)


def workspace_db_url(filename: str) -> str:
    base_dir = Path(__file__).resolve().parent / ".tmp"
    base_dir.mkdir(exist_ok=True)
    db_path = base_dir / f"{filename}-{uuid4().hex}.db"
    return f"sqlite:///{db_path.as_posix()}"


def test_init_db_creates_expected_tables() -> None:
    db_url = workspace_db_url("parser")

    engine = create_sqlalchemy_engine(url=db_url)
    init_db(engine)

    inspector = inspect(engine)

    assert set(inspector.get_table_names()) == {
        "games",
        "guesses",
        "players",
        "review_queue",
        "segments",
        "turns",
        "vods",
    }
    assert Base.metadata.tables.keys() == {
        "vods",
        "segments",
        "games",
        "players",
        "turns",
        "guesses",
        "review_queue",
    }


def test_typed_models_validate_core_constraints() -> None:
    VodMeta(
        vod_id="vod-001",
        streamer_login="tiewhy",
        created_at=datetime.now(timezone.utc),
        title="Test VOD",
        url="https://twitch.tv/videos/1",
        duration_seconds=3600,
    )

    with pytest.raises(ValidationError):
        GameRecord(
            vod_id="vod-001",
            game_index=0,
            start_sec=0,
            end_sec=10,
            parse_confidence=1.5,
        )


def test_round_trip_relationships_and_cascade_delete() -> None:
    db_url = workspace_db_url("relationships")
    engine = create_sqlalchemy_engine(url=db_url)
    init_db(engine)
    session_factory = create_session_factory(engine)

    with session_scope(session_factory) as session:
        vod = Vod(
            vod_id="vod-001",
            streamer_login="tiewhy",
            created_at=datetime.now(timezone.utc),
            title="Latest tiewhy archive",
            url="https://twitch.tv/videos/1",
            duration_seconds=14400,
        )
        segment = Segment(
            segment_type=SegmentType.CODENAMES,
            start_sec=100.0,
            end_sec=1200.0,
            confidence=0.95,
        )
        game = Game(
            game_index=0,
            start_sec=110.0,
            end_sec=400.0,
            winner_team=TeamColor.BLUE,
            win_reason=WinReason.COUNTER_ZERO,
            starting_team=TeamColor.BLUE,
            parse_confidence=0.91,
        )
        player = Player(
            player_name="fembluca",
            display_name="fembluca",
            team_color=TeamColor.BLUE,
            role=PlayerRole.SPYMASTER,
            avatar_hash="hash-001",
        )
        turn = Turn(
            turn_index=0,
            team_color=TeamColor.BLUE,
            spymaster_name="fembluca",
            clue_text="COLD",
            clue_count="2",
            timestamp_sec=150.0,
        )
        guess = Guess(
            guess_index=0,
            player_name="dagger",
            word="ANTARCTICA",
            card_color=CardColor.BLUE,
            result=GuessResult.CORRECT,
            timestamp_sec=160.0,
            confidence=0.98,
        )
        review_item = ReviewQueueItem(
            entity_type=ReviewEntityType.GUESS,
            entity_id="1",
            reason="Low OCR confidence",
            timestamp_sec=160.0,
            snapshot_path="debug_snapshots/guess-1.png",
            confidence=0.45,
        )

        turn.guesses.append(guess)
        game.players.append(player)
        game.turns.append(turn)
        segment.games.append(game)
        vod.segments.append(segment)
        vod.games.append(game)

        session.add(vod)
        session.add(review_item)

    with session_factory() as session:
        stored_game = session.scalar(select(Game).where(Game.game_index == 0))
        assert stored_game is not None
        assert stored_game.winner_team is TeamColor.BLUE
        assert stored_game.win_reason is WinReason.COUNTER_ZERO
        assert len(stored_game.players) == 1
        assert stored_game.turns[0].guesses[0].result is GuessResult.CORRECT
        assert stored_game.segment.vod.vod_id == "vod-001"

        vod = session.scalar(select(Vod).where(Vod.vod_id == "vod-001"))
        assert vod is not None
        session.delete(vod)
        session.commit()

    with session_factory() as session:
        assert session.scalar(select(Vod)) is None
        assert session.scalar(select(Segment)) is None
        assert session.scalar(select(Game)) is None
        assert session.scalar(select(Player)) is None
        assert session.scalar(select(Turn)) is None
        assert session.scalar(select(Guess)) is None
        assert session.scalar(select(ReviewQueueItem)) is not None


def test_sqlite_foreign_keys_are_enforced() -> None:
    db_url = workspace_db_url("foreign-keys")
    engine = create_sqlalchemy_engine(url=db_url)
    init_db(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        session.add(
            Game(
                vod_id="missing-vod",
                segment_id=999,
                game_index=0,
                start_sec=0.0,
                end_sec=1.0,
                parse_confidence=0.5,
            )
        )

        with pytest.raises(IntegrityError):
            session.commit()
