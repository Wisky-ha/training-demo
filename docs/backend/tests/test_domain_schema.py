from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.db.models import (
    DatasetORM,
    ModelAlertORM,
    ModelTypeORM,
    ModelVersionORM,
    PublishRecordORM,
    ScriptORM,
    TrainingJobORM,
)
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import (
    AlertStatus,
    DatasetStatus,
    HealthStatus,
    ModelType,
    ModelVersionStatus,
    ScriptStatus,
    ScriptType,
    TrainingJobStatus,
)
from backend.app.domain.models import PublishRecord


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


def test_publish_record_domain_model_has_audit_defaults():
    record = PublishRecord(model_version_id="version-1", published_version="v1")

    assert record.previous_current_version_id is None
    assert record.message is None
    assert record.published_at.tzinfo is not None


def test_initialization_registers_every_domain_table_and_indexes(session: Session):
    inspector = inspect(session.bind)
    assert {
        "model_types",
        "scripts",
        "script_model_types",
        "datasets",
        "training_jobs",
        "model_versions",
        "publish_records",
        "model_alerts",
        "rollback_records",
    } <= set(inspector.get_table_names())

    assert "ix_publish_records_model_version_id" in {
        index["name"] for index in inspector.get_indexes("publish_records")
    }
    assert "uq_model_versions_type_version" in {
        constraint["name"]
        for constraint in inspector.get_unique_constraints("model_versions")
    }


def test_domain_enums_and_relationships_cover_release_and_health_state(session: Session):
    model_type = ModelTypeORM(code=ModelType.ELECTRIC_LOAD, name="电力负荷预测")
    dataset = DatasetORM(
        file_name="load.csv",
        row_count=10,
        columns=["time", "feature", "target"],
        time_column="time",
        feature_columns=["feature"],
        target_column="target",
        status=DatasetStatus.PARSED,
    )
    trainer = ScriptORM(
        name="train.py",
        script_type=ScriptType.TRAINER,
        version="v1",
        source_code="def train(X, y): pass",
        status=ScriptStatus.ENABLED,
    )
    model_type.scripts.append(trainer)
    job = TrainingJobORM(
        model_type=ModelType.ELECTRIC_LOAD,
        dataset=dataset,
        train_script=trainer,
        status=TrainingJobStatus.PENDING,
        stage_started_at=datetime.now(timezone.utc),
    )
    version = ModelVersionORM(
        model_type=ModelType.ELECTRIC_LOAD,
        version="v1",
        model_path="models/v1.joblib",
        training_job=job,
        status=ModelVersionStatus.READY,
        health_status=HealthStatus.HEALTHY,
        is_current=True,
    )
    release = PublishRecordORM(
        model_version=version,
        published_version="v1",
        message="first release",
    )
    model_type.current_version = version
    session.add(model_type)
    session.flush()

    assert version.model_type_record is model_type
    assert version.training_job is job
    assert version.publish_records == [release]
    assert model_type.current_version is version
    assert dataset.training_jobs == [job]
    assert trainer.supported_model_types == [model_type]
    assert version.health_status is HealthStatus.HEALTHY


def test_database_enforces_foreign_keys_and_current_version_uniqueness(session: Session):
    model_type = ModelTypeORM(code=ModelType.INTEGRATED_ENERGY, name="综合能耗")
    session.add(model_type)
    session.flush()

    first = ModelVersionORM(
        model_type=ModelType.INTEGRATED_ENERGY,
        version="v1",
        model_path="models/v1.joblib",
        is_current=True,
    )
    second = ModelVersionORM(
        model_type=ModelType.INTEGRATED_ENERGY,
        version="v2",
        model_path="models/v2.joblib",
        is_current=True,
    )
    session.add_all([first, second])
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    session.add(
        PublishRecordORM(
            model_version_id="missing-version",
            published_version="v1",
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()


def test_database_enforces_training_ratio_and_one_active_alert(session: Session):
    model_type = ModelTypeORM(code=ModelType.HEATING_COOLING_LOAD, name="冷热负荷预测")
    session.add(model_type)
    session.flush()

    invalid_job = TrainingJobORM(
        model_type=ModelType.HEATING_COOLING_LOAD,
        dataset_id="missing-dataset",
        train_script_id="missing-script",
        split_ratio=0.7,
        test_ratio=0.2,
    )
    session.add(invalid_job)
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    first = ModelAlertORM(
        model_type=ModelType.HEATING_COOLING_LOAD,
        reason="drift",
        status=AlertStatus.ACTIVE,
    )
    second = ModelAlertORM(
        model_type=ModelType.HEATING_COOLING_LOAD,
        reason="drift again",
        status=AlertStatus.ACTIVE,
    )
    session.add_all([first, second])
    with pytest.raises(IntegrityError):
        session.flush()