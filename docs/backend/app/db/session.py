"""SQLAlchemy engine, session, and schema initialization helpers."""

from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ..core.config import Settings, get_settings
from .models import Base


def create_database_engine(settings: Settings | None = None) -> Engine:
    """Create an engine for the configured SQLite database.

    In-memory databases use ``StaticPool`` so all sessions created from the
    engine see the same schema and data.  File databases retain SQLite's safe
    cross-thread option used by the FastAPI application.
    """

    active_settings = settings or get_settings()
    active_settings.ensure_storage_directories()
    url = active_settings.database_url
    engine_kwargs: dict[str, object] = {}
    if url.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
        if url == "sqlite:///:memory:":
            engine_kwargs["poolclass"] = StaticPool
    return create_engine(url, **engine_kwargs)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide application engine."""

    return create_database_engine(get_settings())


def create_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Build a session factory, useful for application and test databases."""

    return sessionmaker(bind=engine or get_engine(), autoflush=False, expire_on_commit=False)


SessionLocal = create_session_factory()


def get_session() -> Generator[Session, None, None]:
    """Yield one SQLAlchemy session and always close it.

    Transaction ownership stays with the caller: a failed request or service
    can explicitly roll back, while a successful service can commit.  The
    dependency rolls back uncommitted work when an exception escapes.
    """

    session = SessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def initialize_database(
    engine: Engine | None = None, settings: Settings | None = None
) -> Engine:
    """Create all ORM tables and return the engine used.

    This intentionally uses SQLAlchemy metadata ``create_all`` rather than
    importing business services or seeding workflow data.  Model-family and
    baseline records belong to later workflow initialization.
    """

    active_engine = engine or create_database_engine(settings)
    Base.metadata.create_all(active_engine)
    return active_engine


# Conventional aliases make the infrastructure easy to discover.
init_db = initialize_database
get_db_session = get_session


__all__ = [
    "SessionLocal",
    "create_database_engine",
    "create_session_factory",
    "get_engine",
    "get_session",
    "get_db_session",
    "initialize_database",
    "init_db",
]
