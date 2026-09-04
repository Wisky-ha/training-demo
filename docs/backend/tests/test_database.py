from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.db.models import (
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
    } <= names
    inspector = inspect(session.bind)
    indexes = {index["name"] for index in inspector.get_indexes("model_versions")}
    constraints = {item["name"] for item in inspector.get_unique_constraints("model_versions")}
    assert "ix_model_versions_model_type" in indexes
    assert "uq_model_versions_type_version" in constraints


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

    first = ModelVersionORM(
        model_type=ModelType.ELECTRIC_LOAD,
        version="v1",
        model_path="v1.joblib",
    )
    second = ModelVersionORM(
        model_type=ModelType.ELECTRIC_LOAD,
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
