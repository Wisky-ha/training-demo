"""Pydantic domain records.

These models describe the business data exchanged between future persistence and
service layers.  They intentionally contain no database, API, or workflow
logic.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .enums import (
    AlertStatus,
    DatasetStatus,
    HealthStatus,
    ModelType,
    ModelVersionStatus,
    RollbackStatus,
    ScriptStatus,
    ScriptType,
    SplitStrategy,
    TrainingJobStatus,
)


def _new_id() -> str:
    return str(uuid4())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DomainModel(BaseModel):
    """Common validation settings for domain records."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
    )


class ModelTypeRecord(DomainModel):
    """Registered model family and its current platform state."""

    id: str = Field(default_factory=_new_id)
    code: ModelType
    name: str = Field(min_length=1)
    current_version_id: str | None = None
    alert_status: AlertStatus = AlertStatus.RESOLVED
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


class ScriptRecord(DomainModel):
    """An immutable versioned source entry in the global script library."""

    id: str = Field(default_factory=_new_id)
    name: str = Field(min_length=1)
    script_type: ScriptType
    version: str = Field(min_length=1)
    source_code: str
    supported_model_types: list[ModelType] = Field(min_length=1)
    status: ScriptStatus = ScriptStatus.ENABLED
    created_at: datetime = Field(default_factory=_utc_now)

    @property
    def uploaded_at(self) -> datetime:
        """Compatibility name for the script upload timestamp in the UI spec."""

        return self.created_at


class DatasetRecord(DomainModel):
    """Parsed dataset metadata; the CSV contents are stored outside this record."""

    id: str = Field(default_factory=_new_id)
    file_name: str = Field(min_length=1)
    file_path: str | None = None
    status: DatasetStatus = DatasetStatus.PARSED
    row_count: int = Field(ge=0)
    columns: list[str] = Field(min_length=3)
    time_column: str = Field(min_length=1)
    feature_columns: list[str] = Field(min_length=1)
    target_column: str = Field(min_length=1)
    column_types: dict[str, str] = Field(default_factory=dict)
    missing_value_counts: dict[str, int] = Field(default_factory=dict)
    preview_rows: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utc_now)

    @property
    def column_count(self) -> int:
        """Number of fields in the parsed source data."""

        return len(self.columns)


class TrainingJob(DomainModel):
    """Configuration and lifecycle metadata for one training run."""

    id: str = Field(default_factory=_new_id)
    model_type: ModelType
    dataset_id: str = Field(min_length=1)
    preprocess_script_id: str | None = None
    train_script_id: str = Field(min_length=1)
    split_strategy: SplitStrategy = SplitStrategy.TIME_ORDERED
    # ``split_ratio`` is the required 80% training portion from the spec.
    split_ratio: float = Field(default=0.8, gt=0, lt=1)
    test_ratio: float = Field(default=0.2, gt=0, lt=1)
    status: TrainingJobStatus = TrainingJobStatus.PENDING
    progress_stage: str | None = None
    stage_started_at: datetime | None = None
    logs: list[str] = Field(default_factory=list)
    error_message: str | None = None
    train_row_count: int | None = Field(default=None, ge=0)
    test_row_count: int | None = Field(default=None, ge=0)
    train_time_start: datetime | None = None
    train_time_end: datetime | None = None
    test_time_start: datetime | None = None
    test_time_end: datetime | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    finished_at: datetime | None = None


class ModelVersion(DomainModel):
    """Traceable model artifact metadata and its release state."""

    id: str = Field(default_factory=_new_id)
    model_type: ModelType
    version: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    preprocessor_path: str | None = None
    training_job_id: str | None = None

    # Script snapshots make a version independent of later library changes.
    train_script_id: str | None = None
    train_script_version: str | None = None
    train_script_source: str | None = None
    preprocess_script_id: str | None = None
    preprocess_script_version: str | None = None
    preprocess_script_source: str | None = None
    preprocess_used: bool = False
    preprocessor_state: dict[str, Any] | None = None

    input_schema: dict[str, Any] = Field(default_factory=dict)
    time_column: str | None = None
    feature_columns: list[str] = Field(default_factory=list)
    target_column: str | None = None
    split_strategy: SplitStrategy = SplitStrategy.TIME_ORDERED
    split_ratio: float = Field(default=0.8, gt=0, lt=1)
    test_ratio: float = Field(default=0.2, gt=0, lt=1)
    train_data_summary: dict[str, Any] = Field(default_factory=dict)
    test_data_summary: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)

    status: ModelVersionStatus = ModelVersionStatus.DRAFT
    health_status: HealthStatus = HealthStatus.HEALTHY
    is_baseline: bool = False
    is_current: bool = False
    previous_healthy_version_id: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    published_at: datetime | None = None


class PublishRecord(DomainModel):
    """Immutable audit entry for making a model version current."""

    id: str = Field(default_factory=_new_id)
    model_version_id: str = Field(min_length=1)
    published_version: str = Field(min_length=1)
    previous_current_version_id: str | None = None
    published_at: datetime = Field(default_factory=_utc_now)
    message: str | None = None


class ModelAlert(DomainModel):
    """An anomaly alert kept open until a successful replacement is published."""

    id: str = Field(default_factory=_new_id)
    model_type: ModelType
    model_version_id: str | None = None
    reason: str = Field(min_length=1)
    rollback_from: str | None = None
    rollback_to: str | None = None
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = Field(default_factory=_utc_now)
    resolved_at: datetime | None = None


class RollbackRecord(DomainModel):
    """Audit record for a manual or automatic version switch."""

    id: str = Field(default_factory=_new_id)
    model_type: ModelType
    rollback_from: str | None = None
    rollback_to: str | None = None
    alert_id: str | None = None
    reason: str | None = None
    status: RollbackStatus = RollbackStatus.PENDING
    created_at: datetime = Field(default_factory=_utc_now)
    finished_at: datetime | None = None


# Short names are useful to service code while the *Record names match the
# storage terminology in the requirements.  They refer to the same Pydantic
# classes, not subclasses with different validation behaviour.
Script = ScriptRecord
Dataset = DatasetRecord
TrainingJobRecord = TrainingJob
ModelVersionRecord = ModelVersion
Alert = ModelAlert
AlertRecord = ModelAlert
ModelRelease = PublishRecord
ReleaseRecord = PublishRecord
Rollback = RollbackRecord
