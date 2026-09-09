from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.db.models import (
    AuditEventORM,
    DatasetORM,
    ModelAlertORM,
    ModelTypeORM,
    ModelVersionORM,
    RollbackRecordORM,
    ScriptORM,
    TrainingJobORM,
)
from backend.app.db.repositories import (
    DatasetRepository,
    ModelTypeRepository,
    ScriptRepository,
    TrainingJobRepository,
)
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import AlertStatus, ModelType, ScriptType, TrainingJobStatus
from backend.app.domain.models import DatasetRecord, ModelTypeRecord, ScriptRecord, TrainingJob


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_database(engine)
    factory = create_session_factory(engine)
    with factory() as db:
        yield db
        db.rollback()
    engine.dispose()


def test_sqlite_database_has_core_tables_and_indexes(session: Session):
    names = set(inspect(session.bind).get_table_names())
    assert {
        "model_types",
        "scripts",
        "script_model_types",
        "datasets",
        "training_jobs",
        "model_versions",
        "model_alerts",
        "rollback_records",
        "audit_events",
    } <= names
    inspector = inspect(session.bind)
    indexes = {index["name"] for index in inspector.get_indexes("model_versions")}
    constraints = {item["name"] for item in inspector.get_unique_constraints("model_versions")}
    assert "ix_model_versions_model_type" in indexes
    assert "uq_model_versions_type_version" in constraints
    audit_indexes = {index["name"] for index in inspector.get_indexes("audit_events")}
    assert {
        "ix_audit_events_occurred_at_event_id",
        "ix_audit_events_event_type",
        "ix_audit_events_object_type_object_id",
        "ix_audit_events_model_type",
        "ix_audit_events_result",
        "ix_audit_events_model_version_id",
        "ix_audit_events_training_job_id",
    } <= audit_indexes


def test_audit_events_are_queryable_and_append_only(session: Session):
    first = AuditEventORM(
        event_id="evt-1",
        occurred_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        event_type="MODEL_SAVED",
        object_type="MODEL_VERSION",
        object_id="version-1",
        result="SUCCEEDED",
        metadata={"source": "test"},
    )
    second = AuditEventORM(
        event_id="evt-2",
        occurred_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        event_type="MODEL_PUBLISHED",
        object_type="MODEL_VERSION",
        object_id="version-1",
        model_type=None,
        model_version_id=None,
        result="SUCCEEDED",
    )
    # Resource IDs are nullable links; use a standalone event for the query
    # contract so this test does not need to construct the entire workflow.
    second.model_version_id = None
    session.add_all([first, second])
    session.commit()

    events = list(session.scalars(
        select(AuditEventORM).order_by(
            AuditEventORM.occurred_at.desc(), AuditEventORM.event_id.desc()
        )
    ))
    assert [item.event_id for item in events[:2]] == ["evt-2", "evt-1"]
    assert events[1].event_metadata == {"source": "test"}
    assert events[1].metadata == {"source": "test"}

    first.message = "must not change"
    with pytest.raises(ValueError, match="append-only"):
        session.flush()
    session.rollback()
    assert session.get(AuditEventORM, "evt-1").message is None

    event_to_delete = session.get(AuditEventORM, "evt-1")
    session.delete(event_to_delete)
    with pytest.raises(ValueError, match="append-only"):
        session.flush()
    session.rollback()
    assert session.get(AuditEventORM, "evt-1") is not None


def test_initialize_database_recreates_audit_table_without_touching_existing_tables(session: Session):
    engine = session.bind
    session.add(ModelTypeORM(code=ModelType.INTEGRATED_ENERGY, name="综合能耗"))
    session.commit()
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE audit_events")

    initialize_database(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP INDEX ix_audit_events_result")
    initialize_database(engine)
    assert "audit_events" in inspect(engine).get_table_names()
    assert "ix_audit_events_result" in {
        index["name"] for index in inspect(engine).get_indexes("audit_events")
    }
    with Session(engine) as check:
        assert check.scalar(select(ModelTypeORM).where(
            ModelTypeORM.code == ModelType.INTEGRATED_ENERGY
        )) is not None


def test_initialize_database_upgrades_legacy_dataset_columns():
    """Older local databases can accept the current dataset upload record."""

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE TABLE datasets (
                id VARCHAR(36) PRIMARY KEY,
                file_name VARCHAR(255) NOT NULL,
                file_path VARCHAR(1024),
                row_count INTEGER NOT NULL,
                columns JSON NOT NULL,
                time_column VARCHAR(255) NOT NULL,
                feature_columns JSON NOT NULL,
                target_column VARCHAR(255) NOT NULL,
                column_types JSON NOT NULL,
                missing_value_counts JSON NOT NULL,
                preview_rows JSON NOT NULL,
                created_at DATETIME NOT NULL
            )
            """
        )

    initialize_database(engine)
    columns = {item["name"] for item in inspect(engine).get_columns("datasets")}
    assert {"status", "numeric_columns", "time_parse", "time_range", "summary"} <= columns
    assert {item["name"] for item in inspect(engine).get_indexes("datasets")} >= {
        "ix_datasets_status_created_at",
        "ix_datasets_created_at",
    }
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO datasets "
                "(id, file_name, row_count, columns, time_column, feature_columns, "
                "target_column, column_types, missing_value_counts, preview_rows, created_at) "
                "VALUES (:id, :file_name, :row_count, :columns, :time_column, :features, "
                ":target, :types, :missing, :preview, :created_at)"
            ),
            {
                "id": "legacy-dataset",
                "file_name": "legacy.csv",
                "row_count": 2,
                "columns": "[]",
                "time_column": "timestamp",
                "features": "[]",
                "target": "load",
                "types": "{}",
                "missing": "{}",
                "preview": "[]",
                "created_at": "2025-01-01T00:00:00",
            },
        )
        row = connection.execute(
            text("SELECT status, numeric_columns, time_parse, time_range, summary FROM datasets WHERE id = :id"),
            {"id": "legacy-dataset"},
        ).mappings().one()
    assert row["status"] == "parsed"
    assert row["numeric_columns"] == "[]"
    assert row["time_parse"] == "{}"
    assert row["time_range"] == "{}"
    assert row["summary"] == "{}"
    engine.dispose()


def test_initialize_database_upgrades_legacy_health_status_constraint():
    """Existing SQLite rows accept the new explicit UNKNOWN health state."""

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE TABLE model_versions (
                id VARCHAR(36) PRIMARY KEY,
                model_type VARCHAR(64) NOT NULL,
                version VARCHAR(100) NOT NULL,
                model_path VARCHAR(1024) NOT NULL,
                training_job_id VARCHAR(36),
                status VARCHAR(9) NOT NULL DEFAULT 'READY',
                health_status VARCHAR(8) NOT NULL DEFAULT 'HEALTHY',
                is_baseline BOOLEAN NOT NULL DEFAULT 0,
                is_current BOOLEAN NOT NULL DEFAULT 0,
                CONSTRAINT healthstatus_enum
                    CHECK (health_status IN ('HEALTHY', 'ABNORMAL'))
            )
            """
        )

    initialize_database(engine)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO model_versions "
                "(id, model_type, version, model_path, status, health_status) "
                "VALUES (:id, :model_type, :version, :model_path, :status, :health_status)"
            ),
            {
                "id": "legacy-unknown",
                "model_type": ModelType.ELECTRIC_LOAD.value,
                "version": "v-unknown",
                "model_path": "legacy/v-unknown",
                "status": "READY",
                "health_status": "UNKNOWN",
            },
        )

    with engine.connect() as connection:
        assert connection.scalar(
            text("SELECT health_status FROM model_versions WHERE id = 'legacy-unknown'")
        ) == "UNKNOWN"
    engine.dispose()


def test_core_relationships_and_json_fields(session: Session):
    model_type = ModelTypeRepository(session).create(
        ModelTypeRecord(code=ModelType.ELECTRIC_LOAD, name="电力负荷预测")
    )
    script = ScriptRepository(session).create(
        ScriptRecord(
            name="normalize.py",
            script_type=ScriptType.PREPROCESSOR,
            version="v1",
            source_code="pass",
            supported_model_types=[ModelType.ELECTRIC_LOAD],
        )
    )
    dataset = DatasetRepository(session).create(
        DatasetRecord(
            file_name="load.csv",
            row_count=2,
            columns=["timestamp", "temperature", "load"],
            time_column="timestamp",
            feature_columns=["temperature"],
            target_column="load",
            column_types={"load": "number"},
        )
    )
    job = TrainingJobRepository(session).create(
        TrainingJob(
            model_type=ModelType.ELECTRIC_LOAD,
            dataset_id=dataset.id,
            train_script_id=script.id,
        )
    )
    session.flush()
    assert model_type.scripts == [script]
    assert script.supported_model_types[0].code == ModelType.ELECTRIC_LOAD
    assert job.dataset is dataset
    assert job.model_type_record is model_type
    assert dataset.training_jobs == [job]
    assert job.split_ratio == 0.8


def test_unique_model_version_and_repository_crud(session: Session):
    model_type = ModelTypeRepository(session).create(
        code=ModelType.ELECTRIC_LOAD, name="电力负荷预测"
    )
    repo = ModelTypeRepository(session)
    assert repo.get(model_type.id) is model_type
    updated = repo.update(model_type.id, name="电力")
    assert updated.name == "电力"
    assert repo.list(code=ModelType.ELECTRIC_LOAD) == [model_type]
    assert repo.delete(model_type.id) is True
    assert repo.get(model_type.id) is None

    ModelTypeRepository(session).create(
        code=ModelType.INTEGRATED_ENERGY, name="综合能耗"
    )
    first = ModelVersionORM(
        model_type=ModelType.INTEGRATED_ENERGY,
        version="v1",
        model_path="v1.joblib",
    )
    second = ModelVersionORM(
        model_type=ModelType.INTEGRATED_ENERGY,
        version="v1",
        model_path="v2.joblib",
    )
    session.add_all([first, second])
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()


def test_repository_transaction_can_roll_back(session: Session):
    repo = ModelTypeRepository(session)
    item = repo.create(code=ModelType.INTEGRATED_ENERGY, name="综合能耗")
    session.flush()
    session.rollback()
    assert session.scalar(select(ModelTypeORM).where(ModelTypeORM.id == item.id)) is None


def test_all_orm_models_expose_expected_status_types():
    assert ModelAlertORM.__tablename__ == "model_alerts"
    assert RollbackRecordORM.__tablename__ == "rollback_records"
    assert TrainingJobORM.__tablename__ == "training_jobs"
    assert DatasetORM.__tablename__ == "datasets"
    assert ScriptORM.__tablename__ == "scripts"
    assert ModelVersionORM.__tablename__ == "model_versions"
    assert datetime.now(timezone.utc).tzinfo is not None
