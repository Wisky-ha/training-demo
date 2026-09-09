"""SQLAlchemy engine, session, and schema initialization helpers."""

from __future__ import annotations

import re
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
    _upgrade_dataset_columns(active_engine)
    _upgrade_audit_event_indexes(active_engine)
    _upgrade_training_job_columns(active_engine)
    _upgrade_rollback_record_columns(active_engine)
    _upgrade_preprocessing_task_columns(active_engine)
    _upgrade_model_version_columns(active_engine)
    _upgrade_model_version_health_constraint(active_engine)
    _upgrade_model_alert_columns(active_engine)
    _upgrade_publish_record_columns(active_engine)
    return active_engine


def _upgrade_dataset_columns(engine: Engine) -> None:
    """Add dataset inspection fields used by the current workflow.

    ``create_all`` does not alter an existing SQLite table.  Databases created
    before the dataset-status/inspection contract therefore need these
    additive columns before an upload can be persisted.  Defaults preserve
    existing rows and do not rewrite the source CSV or its resource IDs.
    """
    if engine.dialect.name != "sqlite":
        return
    inspector = inspect(engine)
    if not inspector.has_table("datasets"):
        return
    columns = {item["name"] for item in inspector.get_columns("datasets")}
    additions = {
        "status": "VARCHAR(10) NOT NULL DEFAULT 'parsed'",
        "numeric_columns": "JSON NOT NULL DEFAULT '[]'",
        "time_parse": "JSON NOT NULL DEFAULT '{}'",
        "time_range": "JSON NOT NULL DEFAULT '{}'",
        "summary": "JSON NOT NULL DEFAULT '{}'",
    }
    with engine.begin() as connection:
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(text(f"ALTER TABLE datasets ADD COLUMN {name} {definition}"))
        connection.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_datasets_status_created_at "
            "ON datasets (status, created_at)"
        ))
        connection.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_datasets_created_at "
            "ON datasets (created_at)"
        ))


def _upgrade_audit_event_indexes(engine: Engine) -> None:
    """Create audit indexes additively for older SQLite schemas.

    ``create_all`` creates the new table on an existing database, but this
    explicit idempotent step also repairs a database that was initialized by a
    partially deployed version.  It only adds indexes and never rewrites or
    removes audit history.
    """
    if engine.dialect.name != "sqlite":
        return
    index_definitions = (
        "CREATE INDEX IF NOT EXISTS ix_audit_events_occurred_at_event_id "
        "ON audit_events (occurred_at, event_id)",
        "CREATE INDEX IF NOT EXISTS ix_audit_events_event_type "
        "ON audit_events (event_type)",
        "CREATE INDEX IF NOT EXISTS ix_audit_events_object_type_object_id "
        "ON audit_events (object_type, object_id)",
        "CREATE INDEX IF NOT EXISTS ix_audit_events_model_type "
        "ON audit_events (model_type)",
        "CREATE INDEX IF NOT EXISTS ix_audit_events_result "
        "ON audit_events (result)",
        "CREATE INDEX IF NOT EXISTS ix_audit_events_model_version_id "
        "ON audit_events (model_version_id)",
        "CREATE INDEX IF NOT EXISTS ix_audit_events_training_job_id "
        "ON audit_events (training_job_id)",
        "CREATE INDEX IF NOT EXISTS ix_audit_events_request_id "
        "ON audit_events (request_id)",
        "CREATE INDEX IF NOT EXISTS ix_audit_events_correlation_id "
        "ON audit_events (correlation_id)",
    )
    with engine.begin() as connection:
        for definition in index_definitions:
            connection.execute(text(definition))


def _upgrade_publish_record_columns(engine: Engine) -> None:
    """Add idempotency support to release records in older SQLite databases."""
    if engine.dialect.name != "sqlite":
        return
    columns = {item["name"] for item in inspect(engine).get_columns("publish_records")}
    with engine.begin() as connection:
        if "idempotency_key" not in columns:
            connection.execute(text("ALTER TABLE publish_records ADD COLUMN idempotency_key VARCHAR(255)"))
        connection.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_publish_records_idempotency_key "
            "ON publish_records (idempotency_key) WHERE idempotency_key IS NOT NULL"
        ))


def _upgrade_rollback_record_columns(engine: Engine) -> None:
    """Add canonical lifecycle audit fields to older SQLite databases."""
    if engine.dialect.name != "sqlite":
        return
    publish_columns = {item["name"] for item in inspect(engine).get_columns("publish_records")}
    rollback_columns = {item["name"] for item in inspect(engine).get_columns("rollback_records")}
    with engine.begin() as connection:
        if "reason" not in publish_columns:
            connection.execute(text("ALTER TABLE publish_records ADD COLUMN reason TEXT"))
        if "idempotency_key" not in rollback_columns:
            connection.execute(text("ALTER TABLE rollback_records ADD COLUMN idempotency_key VARCHAR(255)"))
        connection.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_rollback_records_idempotency_key "
            "ON rollback_records (idempotency_key) WHERE idempotency_key IS NOT NULL"
        ))


def _upgrade_preprocessing_task_columns(engine: Engine) -> None:
    """Add structured failure fields to older preprocessing task tables."""
    if engine.dialect.name != "sqlite":
        return
    columns = {item["name"] for item in inspect(engine).get_columns("preprocessing_tasks")}
    with engine.begin() as connection:
        if "error_code" not in columns:
            connection.execute(text(
                "ALTER TABLE preprocessing_tasks ADD COLUMN error_code VARCHAR(100)"
            ))
        if "error_details" not in columns:
            connection.execute(text(
                "ALTER TABLE preprocessing_tasks ADD COLUMN error_details JSON NOT NULL DEFAULT '{}'"
            ))


def _upgrade_model_version_columns(engine: Engine) -> None:
    """Add model-version fields introduced by the lifecycle workflow."""
    if engine.dialect.name != "sqlite":
        return
    columns = {item["name"] for item in inspect(engine).get_columns("model_versions")}
    additions = {
        # Existing databases only get this column when upgraded. A missing
        # health check must remain UNKNOWN rather than being inferred healthy.
        "health_status": "VARCHAR(10) NOT NULL DEFAULT 'UNKNOWN'",
        "model_artifact_id": "VARCHAR(36)",
        "preprocessor_artifact_id": "VARCHAR(36)",
    }
    with engine.begin() as connection:
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(text(
                    f"ALTER TABLE model_versions ADD COLUMN {name} {definition}"
                ))


def _upgrade_model_version_health_constraint(engine: Engine) -> None:
    """Permit UNKNOWN on SQLite databases created before health was tri-state.

    SQLite cannot alter a CHECK constraint in place. Older deployments have a
    ``healthstatus_enum`` constraint containing only HEALTHY/ABNORMAL, while
    the current contract must persist UNKNOWN for rows without health
    evidence. Rebuild only this table, copying every existing column and
    recreating its explicit indexes; no rows or resource IDs are rewritten.
    """
    if engine.dialect.name != "sqlite":
        return

    with engine.connect() as connection:
        table_sql = connection.scalar(
            text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'model_versions'")
        )
        if not table_sql or not re.search(
            r"health_status\s+IN\s*\(\s*'HEALTHY'\s*,\s*'ABNORMAL'\s*\)",
            table_sql,
            flags=re.IGNORECASE,
        ):
            return
        columns = [
            row[1]
            for row in connection.exec_driver_sql("PRAGMA table_info(model_versions)")
        ]
        if "health_status" not in columns:
            return
        indexes = list(connection.exec_driver_sql(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'index' AND tbl_name = 'model_versions' AND sql IS NOT NULL"
        ))
        # End the implicit read transaction before changing SQLite's foreign
        # key pragma. The migration is serialized by application startup.
        connection.commit()
        connection.exec_driver_sql("PRAGMA foreign_keys = OFF")
        connection.commit()
        try:
            with connection.begin():
                upgraded_sql = re.sub(
                    r"(?i)(CREATE\s+TABLE\s+)([\"]?model_versions[\"]?)",
                    r"\1model_versions_health_upgrade",
                    table_sql,
                    count=1,
                )
                upgraded_sql = re.sub(
                    r"health_status\s+IN\s*\(\s*'HEALTHY'\s*,\s*'ABNORMAL'\s*\)",
                    "health_status IN ('HEALTHY', 'ABNORMAL', 'UNKNOWN')",
                    upgraded_sql,
                    count=1,
                    flags=re.IGNORECASE,
                )
                connection.exec_driver_sql(upgraded_sql)
                quoted_columns = ", ".join(
                    f'"{column.replace(chr(34), chr(34) * 2)}"' for column in columns
                )
                connection.exec_driver_sql(
                    f"INSERT INTO model_versions_health_upgrade ({quoted_columns}) "
                    f"SELECT {quoted_columns} FROM model_versions"
                )
                for name, _ in indexes:
                    safe_name = name.replace(chr(34), chr(34) * 2)
                    connection.exec_driver_sql(f'DROP INDEX IF EXISTS "{safe_name}"')
                connection.exec_driver_sql("DROP TABLE model_versions")
                connection.exec_driver_sql(
                    "ALTER TABLE model_versions_health_upgrade RENAME TO model_versions"
                )
                for _, index_sql in indexes:
                    connection.exec_driver_sql(index_sql)
        finally:
            connection.exec_driver_sql("PRAGMA foreign_keys = ON")
            connection.commit()


def _upgrade_model_alert_columns(engine: Engine) -> None:
    """Upgrade the alert acknowledgement field in pre-lifecycle SQLite DBs."""
    if engine.dialect.name != "sqlite":
        return
    columns = {item["name"] for item in inspect(engine).get_columns("model_alerts")}
    if "acknowledged_at" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE model_alerts ADD COLUMN acknowledged_at DATETIME"))


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
        "stage_started_at": "DATETIME",
        "started_at": "DATETIME",
        "current_stage": "VARCHAR(100)",
        "error_code": "VARCHAR(100)",
        "error_details": "JSON NOT NULL DEFAULT '{}'",
        "config": "JSON NOT NULL DEFAULT '{}'",
        "config_summary": "JSON NOT NULL DEFAULT '{}'",
        "model_version_id": "VARCHAR(36)",
        "log_entries": "JSON NOT NULL DEFAULT '[]'",
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
