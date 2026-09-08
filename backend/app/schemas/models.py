"""HTTP contracts for model versions and their lifecycle operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from ..domain.models import ResourceId
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

    id: ResourceId | None = None
    model_type: ModelType
    version: str | None = Field(default=None, min_length=1, max_length=100)
    model_path: str | None = Field(default=None, min_length=1, max_length=1024)
    model_content_base64: str | None = None
    preprocessor_path: str | None = Field(default=None, max_length=1024)
    training_job_id: ResourceId | None = None
    train_script_id: ResourceId | None = None
    train_script_version: str | None = Field(default=None, max_length=100)
    train_script_source: str | None = None
    preprocess_script_id: ResourceId | None = None
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
    # None means no health evidence; persistence maps it to UNKNOWN, never
    # to HEALTHY. The optional form preserves existing training health on a
    # compatibility save of an already-created draft.
    health_status: HealthStatus | None = None
    status: ModelVersionStatus = ModelVersionStatus.READY

    @model_validator(mode="after")
    def only_saveable_statuses(self):
        if self.status not in {ModelVersionStatus.DRAFT, ModelVersionStatus.READY}:
            raise ValueError("保存模型的状态只能是 DRAFT 或 READY")
        return self


class ModelVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: ResourceId
    model_type: ModelType
    version: str
    model_path: str
    model_artifact_id: ResourceId | None
    preprocessor_path: str | None
    preprocessor_artifact_id: ResourceId | None
    training_job_id: ResourceId | None
    train_script_id: ResourceId | None
    train_script_version: str | None
    train_script_source: str | None
    preprocess_script_id: ResourceId | None
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
    previous_healthy_version_id: ResourceId | None
    created_at: datetime
    published_at: datetime | None
    model_file_metadata: dict[str, Any] | None = None
    preprocessor_file_metadata: dict[str, Any] | None = None


class PublishRequest(BaseModel):
    """Canonical publication command with a parser for old client names."""

    model_config = ConfigDict(extra="forbid")

    confirmed: bool = False
    reason: str | None = Field(default=None, max_length=2000)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=255)

    @model_validator(mode="before")
    @classmethod
    def parse_compatibility_names(cls, value: Any):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "confirmed" not in data:
            data["confirmed"] = data.get("confirm", data.get("confirmation", False))
        if "reason" not in data and "message" in data:
            data["reason"] = data["message"]
        # These names are deliberately removed so they do not appear as
        # normative OpenAPI request fields, while old integrations still work.
        data.pop("confirm", None)
        data.pop("confirmation", None)
        data.pop("message", None)
        return data


class RollbackRequest(BaseModel):
    """Canonical explicit-target rollback command.

    ``version_id``/``version`` are parsed only as compatibility aliases.  A
    path model id is never used as an implicit rollback destination.
    """

    model_config = ConfigDict(extra="forbid")

    # Canonical callers provide all three fields. Defaults preserve parsing of
    # pre-step-6 clients; the router still rejects an omitted target rather
    # than silently using the path model.
    target_version_id: ResourceId | None = None
    reason: str = Field(default="手动回滚", min_length=1, max_length=2000)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=255)

    @model_validator(mode="before")
    @classmethod
    def parse_compatibility_target(cls, value: Any):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "target_version_id" not in data:
            if data.get("version_id") is not None:
                data["target_version_id"] = data["version_id"]
            # A version label was historically accepted by the service. It is
            # retained under a private compatibility key for the router.
            elif data.get("version") is not None:
                data["target_version_id"] = data["version"]
        data.pop("version_id", None)
        data.pop("version", None)
        return data


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


class AlertStatistics(BaseModel):
    total: int
    active: int
    acknowledged: int
    resolved: int


class ModelAlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: ResourceId
    model_type: ModelType
    model_version_id: ResourceId | None
    reason: str
    rollback_from: ResourceId | None
    rollback_to: ResourceId | None
    status: AlertStatus
    created_at: datetime
    acknowledged_at: datetime | None
    resolved_at: datetime | None


class AlertAcknowledgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmed: bool = True


class AlertListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ModelAlertResponse]
    total: int
    page: int | None = None
    page_size: int | None = None
    limit: int | None = None
    next_cursor: str | None = None
    statistics: AlertStatistics


class AlertAcknowledgeResponse(ModelAlertResponse):
    statistics: AlertStatistics


class RollbackResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: ResourceId
    model_type: ModelType
    rollback_from: ResourceId | None
    rollback_to: ResourceId | None
    alert_id: ResourceId | None
    reason: str | None
    idempotency_key: str | None = None
    status: RollbackStatus
    created_at: datetime
    finished_at: datetime | None


class PublishRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: ResourceId
    model_version_id: ResourceId
    published_version: str
    previous_current_version_id: ResourceId | None
    published_at: datetime
    reason: str | None = None
    # Compatibility response projection for old record consumers.
    message: str | None = None
    idempotency_key: str | None = None


class LifecycleOperationResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    operation: str
    model: ModelVersionResponse
    record: PublishRecordResponse | None = None
    rollback: RollbackResponse | None = None
    alert: ModelAlertResponse | None = None


__all__ = [
    "AbnormalRequest", "AlertAcknowledgeRequest", "AlertAcknowledgeResponse", "AlertListResponse",
    "AlertStatistics", "LifecycleOperationResponse", "ModelAbnormalRequest", "ModelAlertResponse",
    "ModelSaveRequest", "ModelVersionResponse", "PublishRecordResponse", "PublishRequest",
    "RollbackRequest", "RollbackResponse",
]
