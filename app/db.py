"""Database engine and session helpers."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.models import Base

SessionFactory = sessionmaker[Session]


def _sqlite_path_from_url(url: str) -> Path | None:
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return None

    raw_path = url.removeprefix(prefix)
    if raw_path == ":memory:":
        return None
    return Path(raw_path)


def _ensure_sqlite_parent_directory(url: str) -> None:
    sqlite_path = _sqlite_path_from_url(url)
    if sqlite_path is None:
        return
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)


def _configure_sqlite_foreign_keys(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: object, connection_record: object) -> None:
        del connection_record
        if not isinstance(dbapi_connection, sqlite3.Connection):
            return

        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def create_sqlalchemy_engine(
    *,
    settings: Settings | None = None,
    url: str | None = None,
    echo: bool = False,
) -> Engine:
    """Create the shared SQLAlchemy engine."""

    resolved_url = url
    if resolved_url is None:
        resolved_settings = settings or get_settings()
        resolved_url = resolved_settings.sqlite_url

    _ensure_sqlite_parent_directory(resolved_url)
    engine = create_engine(resolved_url, echo=echo)
    if resolved_url.startswith("sqlite"):
        _configure_sqlite_foreign_keys(engine)
    return engine


def create_session_factory(engine: Engine) -> SessionFactory:
    """Create a session factory with project defaults."""

    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create all ORM tables."""

    Base.metadata.create_all(engine)


@contextmanager
def session_scope(session_factory: SessionFactory) -> Iterator[Session]:
    """Provide a commit-or-rollback transactional session scope."""

    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
