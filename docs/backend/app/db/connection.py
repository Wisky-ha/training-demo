"""SQLite connection helpers used by the FastAPI application."""

import sqlite3
from collections.abc import Generator
from pathlib import Path

from ..core.config import Settings, get_settings


def connect_database(settings: Settings | None = None) -> sqlite3.Connection:
    """Open a configured SQLite connection and apply safe connection defaults."""

    active_settings = settings or get_settings()
    database_path = active_settings.database_path
    if database_path != ":memory:":
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        database_path = str(path)

    connection = sqlite3.connect(database_path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def get_db() -> Generator[sqlite3.Connection, None, None]:
    """FastAPI dependency yielding one connection per request."""

    connection = connect_database()
    try:
        yield connection
    finally:
        connection.close()
