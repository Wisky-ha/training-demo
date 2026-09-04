"""SQLite/SQLAlchemy database setup.

The module intentionally contains only infrastructure. Domain models and
migrations can be added by later implementation steps without changing the
connection lifecycle used by the API.
"""

from collections.abc import Generator
from pathlib import Path
from typing import Any
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ..core.config import Settings, get_settings


class Base(DeclarativeBase):
    """Base class for future SQLAlchemy models."""


def _ensure_sqlite_parent(database_url: str) -> None:
    """Ensure the parent directory exists for a file-backed SQLite URL."""

    if not database_url.startswith("sqlite:///"):
        return

    database_path = make_url(database_url).database
    if not database_path or database_path == ":memory:":
        return

    # ``make_url`` normalizes relative paths and Windows drive-letter paths,
    # while Path handles both forms used by the configured URL.
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)


def create_engine_from_settings(config: Settings | None = None) -> Engine:
    """Build a SQLAlchemy engine from the unified application settings."""

    config = config or get_settings()
    _ensure_sqlite_parent(config.database_url)
    connect_args: dict[str, Any] = {}
    if config.database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    return create_engine(config.database_url, connect_args=connect_args, pool_pre_ping=True)


engine = create_engine_from_settings()
SessionLocal = sessionmaker(bind=engine, class_=Session, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    """Yield a request-scoped database session for FastAPI dependencies."""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create registered tables for the development/demo database.

    There are no domain models in the foundation step, but calling this at
    startup makes the SQLite setup ready for subsequent steps.
    """

    Base.metadata.create_all(bind=engine)


def check_database_connection() -> bool:
    """Return whether the configured database accepts a simple query."""

    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return True


def dispose_engine() -> None:
    """Close pooled database connections during application shutdown."""

    engine.dispose()
