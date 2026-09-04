from datetime import datetime

import pytest
from pydantic import ValidationError

from backend.app.domain.enums import (
    AlertStatus,
    ModelType,
    ModelVersionStatus,
    RollbackStatus,
    ScriptStatus,
    ScriptType,
    TrainingJobStatus,
)
from backend.app.domain.models import (
    DatasetRecord,
    ModelAlert,
    ModelVersion,
    RollbackRecord,
    ScriptRecord,
    TrainingJob,
)


def test_model_type_values_match_requirements():
    assert [item.value for item in ModelType] == [
        "electric_load",
        "heating_cooling_load",
        "integrated_energy",
    ]


def test_workflow_enum_values_match_requirements():
    assert {item.value for item in ScriptType} == {"preprocessor", "trainer"}
    assert {item.value for item in TrainingJobStatus} == {
        "PENDING",
        "RUNNING",
        "PREPROCESSING",
        "SPLITTING",
        "TRAINING",
        "EVALUATING",
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
    }
    assert {item.value for item in ModelVersionStatus} == {
        "DRAFT",
        "TRAINING",
        "READY",
        "PUBLISHED",
        "RETIRED",
        "ABNORMAL",
        "FAILED",
    }
    assert {item.value for item in AlertStatus} == {"ACTIVE", "RESOLVED"}
    assert {item.value for item in RollbackStatus} == {
        "PENDING",
        "SUCCEEDED",
        "FAILED",
    }


def test_domain_records_can_be_instantiated_with_required_relationships():
    script = ScriptRecord(
        name="normalize.py",
        script_type=ScriptType.PREPROCESSOR,
        version="v1",
        source_code="def transform(data): return data",
        supported_model_types=[ModelType.ELECTRIC_LOAD],
    )
    dataset = DatasetRecord(
        file_name="load.csv",
        row_count=100,
        columns=["timestamp", "temperature", "load"],
        time_column="timestamp",
        feature_columns=["temperature"],
        target_column="load",
    )
    job = TrainingJob(
        model_type=ModelType.ELECTRIC_LOAD,
        dataset_id=dataset.id,
        train_script_id="script-trainer-1",
        preprocess_script_id=script.id,
    )
    version = ModelVersion(
        model_type=job.model_type,
        version="v1",
        model_path="data/models/v1.joblib",
        training_job_id=job.id,
        train_script_source="def train(X_train, y_train): ...",
        input_schema={"timestamp": "datetime", "temperature": "number"},
        time_column="timestamp",
        feature_columns=["temperature"],
        target_column="load",
    )
    alert = ModelAlert(
        model_type=version.model_type,
        model_version_id=version.id,
        reason="prediction drift",
    )
    rollback = RollbackRecord(
        model_type=version.model_type,
        rollback_from=version.id,
        rollback_to="baseline-id",
        alert_id=alert.id,
    )

    assert script.status is ScriptStatus.ENABLED
    assert dataset.column_count == 3
    assert job.split_ratio == 0.8
    assert job.status is TrainingJobStatus.PENDING
    assert version.status is ModelVersionStatus.DRAFT
    assert version.preprocess_script_source is None
    assert alert.status is AlertStatus.ACTIVE
    assert rollback.status is RollbackStatus.PENDING
    assert isinstance(version.created_at, datetime)


def test_nullable_fields_and_defaults_follow_workflow_rules():
    job = TrainingJob(
        model_type=ModelType.INTEGRATED_ENERGY,
        dataset_id="dataset-1",
        train_script_id="script-1",
    )
    version = ModelVersion(
        model_type=ModelType.INTEGRATED_ENERGY,
        version="v0-baseline",
        model_path="baseline.joblib",
    )
    alert = ModelAlert(
        model_type=ModelType.INTEGRATED_ENERGY,
        model_version_id=None,
        reason="baseline unavailable",
    )

    assert job.preprocess_script_id is None
    assert job.finished_at is None
    assert job.error_message is None
    assert version.preprocessor_path is None
    assert version.published_at is None
    assert version.previous_healthy_version_id is None
    assert version.is_baseline is False
    assert version.is_current is False
    assert alert.rollback_from is None
    assert alert.resolved_at is None


def test_invalid_enum_values_are_rejected_by_pydantic():
    with pytest.raises(ValidationError):
        ScriptRecord(
            name="bad.py",
            script_type="unknown",
            version="v1",
            source_code="pass",
            supported_model_types=[ModelType.ELECTRIC_LOAD],
        )

    with pytest.raises(ValidationError):
        TrainingJob(
            model_type="electric_load",
            dataset_id="dataset-1",
            train_script_id="script-1",
            status="NOT_A_STATUS",
        )

    with pytest.raises(ValidationError):
        ModelVersion(
            model_type="electric_load",
            version="v1",
            model_path="model.joblib",
            status="NOT_A_STATUS",
        )

    with pytest.raises(ValidationError):
        ModelAlert(
            model_type="electric_load",
            model_version_id="version-1",
            reason="bad status",
            status="NOT_A_STATUS",
        )
