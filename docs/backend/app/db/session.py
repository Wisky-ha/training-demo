"""SQLAlchemy engine, session, and schema initialization helpers."""

from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ..core.config import Settings, get_settings
from .models import Base


def _configure_sqlite_foreign_keys(engine: Engine) -> None:
    """Enable SQLite foreign-key enforcement for every pooled connection."""

    if engine.dialect.name != "sqlite":
        return
    if not getattr(engine, "_foreign_keys_configured", False):
        @event.listens_for(engine, "connect")
        def _set_foreign_keys(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        setattr(engine, "_foreign_keys_configured", True)
    # An engine supplied by a caller may already have an open pooled
    # connection, so set the pragma on the currently checked-out connection.
    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")


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
    engine = create_engine(url, **engine_kwargs)
    _configure_sqlite_foreign_keys(engine)
    return engine


@lru_cache(maxsize=1)
def _get_default_engine() -> Engine:
    return create_database_engine(get_settings())


def get_engine(settings: Settings | None = None) -> Engine:
    """Return the process-wide engine, or an engine for explicit settings."""

    return _get_default_engine() if settings is None else create_database_engine(settings)


def create_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Build a session factory, useful for application and test databases."""

    active_engine = engine if engine is not None else get_engine()
    return sessionmaker(bind=active_engine, autoflush=False, expire_on_commit=False)


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

    active_engine = engine if engine is not None else create_database_engine(settings)
    _configure_sqlite_foreign_keys(active_engine)
    Base.metadata.create_all(active_engine)
    _upgrade_training_job_columns(active_engine)
    return active_engine


def _upgrade_training_job_columns(engine: Engine) -> None:
    """Add nullable workflow columns when opening a pre-step-8 SQLite DB.

    The project has no migration runner yet.  These additions are deliberately
    nullable/defaulted so existing datasets and production-version records are
    never rewritten or invalidated by the training workflow.
    """
    if engine.dialect.name != "sqlite":
        return
    columns = {item["name"] for item in inspect(engine).get_columns("training_jobs")}
    additions = {
        "preprocessing_task_id": "VARCHAR(36)",
        "started_at": "DATETIME",
        "current_stage": "VARCHAR(100)",
        "config": "JSON NOT NULL DEFAULT '{}'",
        "config_summary": "JSON NOT NULL DEFAULT '{}'",
        "model_version_id": "VARCHAR(36)",
    }
    with engine.begin() as connection:
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(text(f"ALTER TABLE training_jobs ADD COLUMN {name} {definition}"))


# Conventional aliases make the infrastructure easy to discover.
init_db = initialize_database
get_db_session = get_session
get_db = get_session


__all__ = [
    "SessionLocal",
    "create_database_engine",
    "create_session_factory",
    "get_engine",
    "get_session",
    "get_db_session",
    "get_db",
    "initialize_database",
    "init_db",
]
