"""HTTP contracts for model versions and their lifecycle operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from ..domain.enums import (
    AlertStatus,
    HealthStatus,
    ModelType,
    ModelVersionStatus,
    RollbackStatus,
    SplitStrategy,
)


class ModelSaveRequest(BaseModel):
    """Metadata for an independently saved model artifact.

    Training creates a DRAFT automatically.  This contract also permits an
    integration to save a ready artifact without going through a training job.
    ``model_content_base64`` is optional for callers that already manage the
    artifact and only need to register its immutable metadata.
    """

    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(default=None, min_length=1, max_length=36)
    model_type: ModelType
    version: str | None = Field(default=None, min_length=1, max_length=100)
    model_path: str | None = Field(default=None, min_length=1, max_length=1024)
    model_content_base64: str | None = None
    preprocessor_path: str | None = Field(default=None, max_length=1024)
    training_job_id: str | None = Field(default=None, max_length=36)
    train_script_id: str | None = Field(default=None, max_length=36)
    train_script_version: str | None = Field(default=None, max_length=100)
    train_script_source: str | None = None
    preprocess_script_id: str | None = Field(default=None, max_length=36)
    preprocess_script_version: str | None = Field(default=None, max_length=100)
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
    status: ModelVersionStatus = ModelVersionStatus.READY

    @model_validator(mode="after")
    def only_saveable_statuses(self):
        if self.status not in {ModelVersionStatus.DRAFT, ModelVersionStatus.READY}:
            raise ValueError("保存模型的状态只能是 DRAFT 或 READY")
        return self


class ModelVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    model_type: ModelType
    version: str
    model_path: str
    model_artifact_id: str | None
    preprocessor_path: str | None
    preprocessor_artifact_id: str | None
    training_job_id: str | None
    train_script_id: str | None
    train_script_version: str | None
    train_script_source: str | None
    preprocess_script_id: str | None
    preprocess_script_version: str | None
    preprocess_script_source: str | None
    preprocess_used: bool
    preprocessor_state: dict[str, Any] | None
    input_schema: dict[str, Any]
    time_column: str | None
    feature_columns: list[str]
    target_column: str | None
    split_strategy: SplitStrategy
    split_ratio: float
    test_ratio: float
    train_data_summary: dict[str, Any]
    test_data_summary: dict[str, Any]
    metrics: dict[str, Any]
    status: ModelVersionStatus
    health_status: HealthStatus
    is_baseline: bool
    is_current: bool
    previous_healthy_version_id: str | None
    created_at: datetime
    published_at: datetime | None
    model_file_metadata: dict[str, Any] | None = None
    preprocessor_file_metadata: dict[str, Any] | None = None


class PublishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # False is the safe default.  ``confirmed`` is accepted as a compatibility
    # spelling used by clients that present a second confirmation dialog.
    confirm: bool = False
    confirmed: bool | None = None
    confirmation: bool | None = None
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=255)
    message: str | None = Field(default=None, max_length=2000)
    reason: str | None = Field(default=None, max_length=2000)

    @property
    def is_confirmed(self) -> bool:
        return self.confirm or self.confirmed is True or self.confirmation is True


class RollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_version_id: str | None = Field(default=None, min_length=1, max_length=36)
    target_version: str | None = Field(default=None, min_length=1, max_length=100)
    # Compatibility spellings for clients whose rollback dialog calls the
    # destination simply ``version_id``/``version``.
    version_id: str | None = Field(default=None, min_length=1, max_length=36)
    version: str | None = Field(default=None, min_length=1, max_length=100)
    reason: str = Field(default="手动回滚", min_length=1, max_length=2000)


class AbnormalRequest(BaseModel):
    """Compatibility payload for the version-id lifecycle endpoint."""

    model_config = ConfigDict(extra="forbid")

    abnormal: bool = True
    reason: str = Field(default="健康检查异常", min_length=1, max_length=2000)


class ModelAbnormalRequest(BaseModel):
    """Type/version based anomaly command used by API and MCP adapters."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    model_type: ModelType
    model_version: str = Field(
        min_length=1,
        max_length=100,
        validation_alias=AliasChoices("model_version", "version", "model_version_id"),
    )
    abnormal: bool = True
    reason: str = Field(default="健康检查异常", min_length=1, max_length=2000)


class ModelAlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    model_type: ModelType
    model_version_id: str | None
    reason: str
    rollback_from: str | None
    rollback_to: str | None
    status: AlertStatus
    created_at: datetime
    acknowledged_at: datetime | None
    resolved_at: datetime | None


class RollbackResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    model_type: ModelType
    rollback_from: str | None
    rollback_to: str | None
    alert_id: str | None
    reason: str | None
    status: RollbackStatus
    created_at: datetime
    finished_at: datetime | None


class LifecycleOperationResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    operation: str
    model: ModelVersionResponse
    rollback: RollbackResponse | None = None
    alert: ModelAlertResponse | None = None


__all__ = [
    "AbnormalRequest", "LifecycleOperationResponse", "ModelAbnormalRequest", "ModelAlertResponse",
    "ModelSaveRequest", "ModelVersionResponse", "PublishRequest",
    "RollbackRequest", "RollbackResponse",
]
