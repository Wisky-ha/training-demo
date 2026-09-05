"""HTTP contracts for training jobs and their persisted evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..domain.enums import ModelType, SplitStrategy, TrainingJobStatus


class TrainingJobCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: ModelType
    dataset_id: str = Field(min_length=1, max_length=255)
    preprocess_script_id: str | None = Field(default=None, max_length=255)
    # Usually this is inferred from the dataset's immutable split.  It is
    # accepted explicitly so a client cannot accidentally pair a script with
    # a different completed preprocessing task.
    preprocessing_task_id: str | None = Field(default=None, max_length=255)
    train_script_id: str = Field(min_length=1, max_length=255)
    config: dict[str, Any] = Field(default_factory=dict)


class TrainingJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    model_type: ModelType
    dataset_id: str
    preprocess_script_id: str | None
    preprocessing_task_id: str | None
    train_script_id: str
    split_strategy: SplitStrategy
    split_ratio: float
    test_ratio: float
    status: TrainingJobStatus
    progress_stage: str | None
    current_stage: str | None
    # Alias retained for clients that call the current phase simply stage.
    stage: str | None = None
    stage_started_at: datetime | None
    logs: list[str]
    error_message: str | None
    error_code: str | None = None
    error_details: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any]
    config_summary: dict[str, Any]
    model_version_id: str | None
    train_row_count: int | None
    test_row_count: int | None
    train_time_start: datetime | None
    train_time_end: datetime | None
    test_time_start: datetime | None
    test_time_end: datetime | None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None


class TrainingJobLogsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    items: list[str]


class MetricSet(BaseModel):
    model_config = ConfigDict(extra="allow")

    mae: float | None
    rmse: float | None
    mape: float | None
    r2: float | None
    sample_count: int
    mape_valid_count: int
    mape_excluded_count: int
    mape_note: str


class EvaluationResponse(BaseModel):
    """Metrics use the complete test set; chart rows may be sampled."""

    model_config = ConfigDict(extra="allow")

    job_id: str
    model_version_id: str
    metrics: MetricSet
    chart_data: list[dict[str, Any]]
    # These persisted series are never chart-sampled and cover every test row.
    test_time_series: list[Any] = Field(default_factory=list)
    timestamps: list[Any] = Field(default_factory=list)
    actual_values: list[float] = Field(default_factory=list)
    candidate_predictions: list[float] = Field(default_factory=list)
    baseline_predictions: list[float] = Field(default_factory=list)
    candidate_errors: list[float] = Field(default_factory=list)
    baseline_errors: list[float] = Field(default_factory=list)
    error_series: list[float] = Field(default_factory=list)
    error_data: list[dict[str, Any]]
    model_metrics: dict[str, Any] = Field(default_factory=dict)
    metric_differences: dict[str, float | None] = Field(default_factory=dict)
    metrics_difference: dict[str, float | None] = Field(default_factory=dict)
    chart_sampled: bool
    chart_total_count: int
    chart_sample_count: int
    model_comparison: dict[str, Any]


__all__ = [
    "EvaluationResponse",
    "MetricSet",
    "TrainingJobCreate",
    "TrainingJobLogsResponse",
    "TrainingJobResponse",
]
