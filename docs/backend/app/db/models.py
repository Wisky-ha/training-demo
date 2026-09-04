"""SQLAlchemy persistence models for the model-training domain.

The Pydantic records in :mod:`backend.app.domain.models` remain the public
validation models.  These classes are deliberately persistence-focused and
use JSON columns for the variable-shaped dataset/model metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Table,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from ..domain.enums import (
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


def _enum_column(enum_class: type) -> SAEnum:
    """Store enum values (rather than Python member names) in SQLite."""

    return SAEnum(
        enum_class,
        name=f"{enum_class.__name__.lower()}_enum",
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
        values_callable=lambda values: [member.value for member in values],
    )


def _new_id() -> str:
    return str(uuid4())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base for all application ORM models."""


script_model_types = Table(
    "script_model_types",
    Base.metadata,
    Column("script_id", String(36), ForeignKey("scripts.id", ondelete="CASCADE"), primary_key=True),
    Column(
        "model_type_code",
        String(64),
        ForeignKey("model_types.code", ondelete="CASCADE"),
        primary_key=True,
    ),
    Index("ix_script_model_types_model_type_code", "model_type_code"),
)


class ModelTypeORM(Base):
    __tablename__ = "model_types"
    __table_args__ = (
        UniqueConstraint("code", name="uq_model_types_code"),
        Index("ix_model_types_alert_status", "alert_status"),
        Index("ix_model_types_current_version_id", "current_version_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    code: Mapped[ModelType] = mapped_column(_enum_column(ModelType), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    current_version_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey(
            "model_versions.id",
            ondelete="SET NULL",
            use_alter=True,
            name="fk_model_types_current_version",
        ),
        nullable=True,
    )
    alert_status: Mapped[AlertStatus] = mapped_column(
        _enum_column(AlertStatus), nullable=False, default=AlertStatus.RESOLVED
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, onupdate=_utc_now, nullable=False
    )

    current_version: Mapped[ModelVersionORM | None] = relationship(
        "ModelVersionORM", foreign_keys=[current_version_id], post_update=True
    )
    model_versions: Mapped[list[ModelVersionORM]] = relationship(
        "ModelVersionORM",
        foreign_keys="ModelVersionORM.model_type",
        back_populates="model_type_record",
    )
    scripts: Mapped[list[ScriptORM]] = relationship(
        "ScriptORM", secondary=script_model_types, back_populates="supported_model_types"
    )
    training_jobs: Mapped[list[TrainingJobORM]] = relationship(
        "TrainingJobORM", foreign_keys="TrainingJobORM.model_type", back_populates="model_type_record"
    )
    alerts: Mapped[list[ModelAlertORM]] = relationship(
        "ModelAlertORM", foreign_keys="ModelAlertORM.model_type", back_populates="model_type_record"
    )
    rollback_records: Mapped[list[RollbackRecordORM]] = relationship(
        "RollbackRecordORM", foreign_keys="RollbackRecordORM.model_type", back_populates="model_type_record"
    )


class ScriptORM(Base):
    __tablename__ = "scripts"
    __table_args__ = (
        UniqueConstraint("name", "script_type", "version", name="uq_scripts_name_type_version"),
        Index("ix_scripts_type_status", "script_type", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    script_type: Mapped[ScriptType] = mapped_column(_enum_column(ScriptType), nullable=False)
    version: Mapped[str] = mapped_column(String(100), nullable=False)
    source_code: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[ScriptStatus] = mapped_column(
        _enum_column(ScriptStatus), nullable=False, default=ScriptStatus.ENABLED
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)

    supported_model_types: Mapped[list[ModelTypeORM]] = relationship(
        "ModelTypeORM", secondary=script_model_types, back_populates="scripts"
    )
    training_jobs: Mapped[list[TrainingJobORM]] = relationship(
        "TrainingJobORM", foreign_keys="TrainingJobORM.train_script_id", back_populates="train_script"
    )
    preprocess_training_jobs: Mapped[list[TrainingJobORM]] = relationship(
        "TrainingJobORM", foreign_keys="TrainingJobORM.preprocess_script_id", back_populates="preprocess_script"
    )
    model_versions_as_train_script: Mapped[list[ModelVersionORM]] = relationship(
        "ModelVersionORM", foreign_keys="ModelVersionORM.train_script_id", back_populates="train_script"
    )
    model_versions_as_preprocess_script: Mapped[list[ModelVersionORM]] = relationship(
        "ModelVersionORM", foreign_keys="ModelVersionORM.preprocess_script_id", back_populates="preprocess_script"
    )


class DatasetORM(Base):
    __tablename__ = "datasets"
    __table_args__ = (
        CheckConstraint("row_count >= 0", name="ck_datasets_row_count_nonnegative"),
        Index("ix_datasets_status_created_at", "status", "created_at"),
        Index("ix_datasets_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    status: Mapped[DatasetStatus] = mapped_column(
        _enum_column(DatasetStatus), nullable=False, default=DatasetStatus.PARSED
    )
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    columns: Mapped[list[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=False)
    time_column: Mapped[str] = mapped_column(String(255), nullable=False)
    feature_columns: Mapped[list[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=False)
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    column_types: Mapped[dict[str, str]] = mapped_column(
        MutableDict.as_mutable(JSON), nullable=False, default=dict
    )
    missing_value_counts: Mapped[dict[str, int]] = mapped_column(
        MutableDict.as_mutable(JSON), nullable=False, default=dict
    )
    preview_rows: Mapped[list[dict[str, Any]]] = mapped_column(
        MutableList.as_mutable(JSON), nullable=False, default=list
    )
    # Inspection metadata is persisted with the dataset record so a later
    # request does not need to re-read and re-parse a potentially large CSV.
    numeric_columns: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(JSON), nullable=False, default=list
    )
    time_parse: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSON), nullable=False, default=dict
    )
    time_range: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSON), nullable=False, default=dict
    )
    summary: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSON), nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)

    training_jobs: Mapped[list[TrainingJobORM]] = relationship(
        "TrainingJobORM", back_populates="dataset"
    )


class FileArtifactORM(Base):
    """Metadata for a file kept by the local artifact storage service.

    The path is always relative to the configured storage root.  Keeping one
    table for all artifact kinds makes integrity checks and future storage
    backends independent from the workflow tables that reference the files.
    """

    __tablename__ = "file_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "artifact_type",
            "artifact_id",
            name="uq_file_artifacts_type_id",
        ),
        Index("ix_file_artifacts_type_id", "artifact_type", "artifact_id"),
        Index("ix_file_artifacts_checksum", "checksum_sha256"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    artifact_type: Mapped[str] = mapped_column(String(32), nullable=False)
    artifact_id: Mapped[str] = mapped_column(String(255), nullable=False)
    relative_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, onupdate=_utc_now, nullable=False
    )

    # Readable compatibility aliases for callers using generic file metadata
    # terminology.  The canonical persisted names remain explicit above.
    @property
    def file_size(self) -> int:
        return self.size_bytes

    @property
    def checksum(self) -> str:
        return self.checksum_sha256

    @property
    def size(self) -> int:
        return self.size_bytes

    @property
    def sha256(self) -> str:
        return self.checksum_sha256

    @property
    def identifier(self) -> str:
        return self.artifact_id


class TrainingJobORM(Base):
    __tablename__ = "training_jobs"
    __table_args__ = (
        CheckConstraint(
            "split_ratio > 0 AND split_ratio < 1 AND test_ratio > 0 AND test_ratio < 1 "
            "AND abs((split_ratio + test_ratio) - 1.0) < 0.000001",
            name="ck_training_jobs_split_ratios_valid",
        ),
        CheckConstraint(
            "(train_row_count IS NULL OR train_row_count >= 0) AND "
            "(test_row_count IS NULL OR test_row_count >= 0)",
            name="ck_training_jobs_row_counts_nonnegative",
        ),
        Index("ix_training_jobs_model_type_status", "model_type", "status"),
        Index("ix_training_jobs_dataset_id", "dataset_id"),
        Index("ix_training_jobs_train_script_id", "train_script_id"),
        Index("ix_training_jobs_preprocess_script_id", "preprocess_script_id"),
        Index("ix_training_jobs_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    model_type: Mapped[ModelType] = mapped_column(
        _enum_column(ModelType), ForeignKey("model_types.code"), nullable=False
    )
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id"), nullable=False
    )
    preprocess_script_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("scripts.id"), nullable=True
    )
    train_script_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("scripts.id"), nullable=False
    )
    split_strategy: Mapped[SplitStrategy] = mapped_column(
        _enum_column(SplitStrategy), nullable=False, default=SplitStrategy.TIME_ORDERED
    )
    split_ratio: Mapped[float] = mapped_column(nullable=False, default=0.8)
    test_ratio: Mapped[float] = mapped_column(nullable=False, default=0.2)
    status: Mapped[TrainingJobStatus] = mapped_column(
        _enum_column(TrainingJobStatus), nullable=False, default=TrainingJobStatus.PENDING
    )
    progress_stage: Mapped[str | None] = mapped_column(String(100), nullable=True)
    stage_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    logs: Mapped[list[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=False, default=list)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    train_row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    test_row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    train_time_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    train_time_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    test_time_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    test_time_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    model_type_record: Mapped[ModelTypeORM] = relationship(
        "ModelTypeORM", foreign_keys=[model_type], back_populates="training_jobs"
    )
    dataset: Mapped[DatasetORM] = relationship("DatasetORM", back_populates="training_jobs")
    preprocess_script: Mapped[ScriptORM | None] = relationship(
        "ScriptORM", foreign_keys=[preprocess_script_id], back_populates="preprocess_training_jobs"
    )
    train_script: Mapped[ScriptORM] = relationship(
        "ScriptORM", foreign_keys=[train_script_id], back_populates="training_jobs"
    )
    model_versions: Mapped[list[ModelVersionORM]] = relationship(
        "ModelVersionORM", back_populates="training_job"
    )


class ModelVersionORM(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("model_type", "version", name="uq_model_versions_type_version"),
        Index("ix_model_versions_model_type", "model_type"),
        Index("ix_model_versions_training_job_id", "training_job_id"),
        Index("ix_model_versions_status", "status"),
        Index(
            "uq_model_versions_one_current",
            "model_type",
            unique=True,
            sqlite_where=text("is_current = 1"),
        ),
        Index("ix_model_versions_health_status", "health_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    model_type: Mapped[ModelType] = mapped_column(
        _enum_column(ModelType), ForeignKey("model_types.code"), nullable=False
    )
    version: Mapped[str] = mapped_column(String(100), nullable=False)
    model_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    preprocessor_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    training_job_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("training_jobs.id"), nullable=True
    )
    train_script_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("scripts.id"), nullable=True)
    train_script_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    train_script_source: Mapped[str | None] = mapped_column(Text, nullable=True)
    preprocess_script_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("scripts.id"), nullable=True)
    preprocess_script_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    preprocess_script_source: Mapped[str | None] = mapped_column(Text, nullable=True)
    preprocess_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    preprocessor_state: Mapped[dict[str, Any] | None] = mapped_column(MutableDict.as_mutable(JSON), nullable=True)
    input_schema: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSON), nullable=False, default=dict
    )
    time_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    feature_columns: Mapped[list[str]] = mapped_column(MutableList.as_mutable(JSON), nullable=False, default=list)
    target_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    split_strategy: Mapped[SplitStrategy] = mapped_column(
        _enum_column(SplitStrategy), nullable=False, default=SplitStrategy.TIME_ORDERED
    )
    split_ratio: Mapped[float] = mapped_column(nullable=False, default=0.8)
    test_ratio: Mapped[float] = mapped_column(nullable=False, default=0.2)
    train_data_summary: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSON), nullable=False, default=dict
    )
    test_data_summary: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSON), nullable=False, default=dict
    )
    metrics: Mapped[dict[str, Any]] = mapped_column(MutableDict.as_mutable(JSON), nullable=False, default=dict)
    status: Mapped[ModelVersionStatus] = mapped_column(
        _enum_column(ModelVersionStatus), nullable=False, default=ModelVersionStatus.DRAFT
    )
    health_status: Mapped[HealthStatus] = mapped_column(
        _enum_column(HealthStatus), nullable=False, default=HealthStatus.HEALTHY
    )
    is_baseline: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    previous_healthy_version_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("model_versions.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    model_type_record: Mapped[ModelTypeORM] = relationship(
        "ModelTypeORM", foreign_keys=[model_type], back_populates="model_versions"
    )
    training_job: Mapped[TrainingJobORM | None] = relationship("TrainingJobORM", back_populates="model_versions")
    train_script: Mapped[ScriptORM | None] = relationship(
        "ScriptORM", foreign_keys=[train_script_id], back_populates="model_versions_as_train_script"
    )
    preprocess_script: Mapped[ScriptORM | None] = relationship(
        "ScriptORM", foreign_keys=[preprocess_script_id], back_populates="model_versions_as_preprocess_script"
    )
    previous_healthy_version: Mapped[ModelVersionORM | None] = relationship(
        "ModelVersionORM", remote_side=[id], back_populates="superseding_versions"
    )
    superseding_versions: Mapped[list[ModelVersionORM]] = relationship(
        "ModelVersionORM", back_populates="previous_healthy_version"
    )
    publish_records: Mapped[list[PublishRecordORM]] = relationship(
        "PublishRecordORM",
        foreign_keys="PublishRecordORM.model_version_id",
        back_populates="model_version",
    )
    previous_publish_records: Mapped[list[PublishRecordORM]] = relationship(
        "PublishRecordORM",
        foreign_keys="PublishRecordORM.previous_current_version_id",
        back_populates="previous_current_version",
    )
    alerts: Mapped[list[ModelAlertORM]] = relationship(
        "ModelAlertORM", foreign_keys="ModelAlertORM.model_version_id", back_populates="model_version"
    )
    alerts_rollback_from: Mapped[list[ModelAlertORM]] = relationship(
        "ModelAlertORM", foreign_keys="ModelAlertORM.rollback_from", back_populates="rollback_from_version"
    )
    alerts_rollback_to: Mapped[list[ModelAlertORM]] = relationship(
        "ModelAlertORM", foreign_keys="ModelAlertORM.rollback_to", back_populates="rollback_to_version"
    )
    rollback_from_records: Mapped[list[RollbackRecordORM]] = relationship(
        "RollbackRecordORM", foreign_keys="RollbackRecordORM.rollback_from", back_populates="rollback_from_version"
    )
    rollback_to_records: Mapped[list[RollbackRecordORM]] = relationship(
        "RollbackRecordORM", foreign_keys="RollbackRecordORM.rollback_to", back_populates="rollback_to_version"
    )


class PublishRecordORM(Base):
    """Immutable history of a successful model publication."""

    __tablename__ = "publish_records"
    __table_args__ = (
        Index("ix_publish_records_model_version_id", "model_version_id"),
        Index("ix_publish_records_published_at", "published_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    model_version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("model_versions.id"), nullable=False
    )
    # Keep the displayed version as an audit snapshot, even if presentation
    # rules for version strings change in a future release.
    published_version: Mapped[str] = mapped_column(String(100), nullable=False)
    previous_current_version_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("model_versions.id"), nullable=True
    )
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, nullable=False
    )
    message: Mapped[str | None] = mapped_column(Text, nullable=True)

    model_version: Mapped[ModelVersionORM] = relationship(
        "ModelVersionORM",
        foreign_keys=[model_version_id],
        back_populates="publish_records",
    )
    previous_current_version: Mapped[ModelVersionORM | None] = relationship(
        "ModelVersionORM",
        foreign_keys=[previous_current_version_id],
        back_populates="previous_publish_records",
    )


class ModelAlertORM(Base):
    __tablename__ = "model_alerts"
    __table_args__ = (
        Index("ix_model_alerts_model_type_status", "model_type", "status"),
        Index("ix_model_alerts_model_version_id", "model_version_id"),
        Index("ix_model_alerts_created_at", "created_at"),
        Index(
            "uq_model_alerts_one_active_per_type",
            "model_type",
            unique=True,
            sqlite_where=text("status = 'ACTIVE'"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    model_type: Mapped[ModelType] = mapped_column(
        _enum_column(ModelType), ForeignKey("model_types.code"), nullable=False
    )
    model_version_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("model_versions.id"), nullable=True
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    rollback_from: Mapped[str | None] = mapped_column(String(36), ForeignKey("model_versions.id"), nullable=True)
    rollback_to: Mapped[str | None] = mapped_column(String(36), ForeignKey("model_versions.id"), nullable=True)
    status: Mapped[AlertStatus] = mapped_column(
        _enum_column(AlertStatus), nullable=False, default=AlertStatus.ACTIVE
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    model_type_record: Mapped[ModelTypeORM] = relationship(
        "ModelTypeORM", foreign_keys=[model_type], back_populates="alerts"
    )
    model_version: Mapped[ModelVersionORM | None] = relationship(
        "ModelVersionORM", foreign_keys=[model_version_id], back_populates="alerts"
    )
    rollback_from_version: Mapped[ModelVersionORM | None] = relationship(
        "ModelVersionORM", foreign_keys=[rollback_from], back_populates="alerts_rollback_from"
    )
    rollback_to_version: Mapped[ModelVersionORM | None] = relationship(
        "ModelVersionORM", foreign_keys=[rollback_to], back_populates="alerts_rollback_to"
    )
    rollback_records: Mapped[list[RollbackRecordORM]] = relationship(
        "RollbackRecordORM", back_populates="alert"
    )


class RollbackRecordORM(Base):
    __tablename__ = "rollback_records"
    __table_args__ = (
        Index("ix_rollback_records_model_type_created_at", "model_type", "created_at"),
        Index("ix_rollback_records_status", "status"),
        Index("ix_rollback_records_alert_id", "alert_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    model_type: Mapped[ModelType] = mapped_column(
        _enum_column(ModelType), ForeignKey("model_types.code"), nullable=False
    )
    rollback_from: Mapped[str | None] = mapped_column(String(36), ForeignKey("model_versions.id"), nullable=True)
    rollback_to: Mapped[str | None] = mapped_column(String(36), ForeignKey("model_versions.id"), nullable=True)
    alert_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("model_alerts.id"), nullable=True
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[RollbackStatus] = mapped_column(
        _enum_column(RollbackStatus), nullable=False, default=RollbackStatus.PENDING
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    model_type_record: Mapped[ModelTypeORM] = relationship(
        "ModelTypeORM", foreign_keys=[model_type], back_populates="rollback_records"
    )
    rollback_from_version: Mapped[ModelVersionORM | None] = relationship(
        "ModelVersionORM", foreign_keys=[rollback_from], back_populates="rollback_from_records"
    )
    rollback_to_version: Mapped[ModelVersionORM | None] = relationship(
        "ModelVersionORM", foreign_keys=[rollback_to], back_populates="rollback_to_records"
    )
    alert: Mapped[ModelAlertORM | None] = relationship("ModelAlertORM", back_populates="rollback_records")


# Convenient names for callers that do not need to distinguish ORM from domain
# records.  The explicit ORM suffixes remain available for unambiguous imports.
ModelTypeModel = ModelTypeORM
ScriptModel = ScriptORM
DatasetModel = DatasetORM
FileArtifactModel = FileArtifactORM
TrainingJobModel = TrainingJobORM
ModelVersionModel = ModelVersionORM
PublishRecordModel = PublishRecordORM
ModelReleaseORM = PublishRecordORM
ReleaseRecordORM = PublishRecordORM
ModelAlertModel = ModelAlertORM
RollbackModel = RollbackRecordORM

__all__ = [
    "Base",
    "script_model_types",
    "ModelTypeORM",
    "ScriptORM",
    "DatasetORM",
    "FileArtifactORM",
    "TrainingJobORM",
    "ModelVersionORM",
    "PublishRecordORM",
    "ModelReleaseORM",
    "ReleaseRecordORM",
    "ModelAlertORM",
    "RollbackRecordORM",
    "ModelTypeModel",
    "ScriptModel",
    "DatasetModel",
    "FileArtifactModel",
    "TrainingJobModel",
    "ModelVersionModel",
    "PublishRecordModel",
    "ModelAlertModel",
    "RollbackModel",
]
