"""API contracts for the optional preprocessing workflow stage."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..domain.enums import ModelType, PreprocessingStage, PreprocessingTaskStatus


class PreprocessingTaskCreate(BaseModel):
    """Selection made on the preprocessing page.

    ``mode=skip`` is deliberately explicit.  For compatibility with the
    training-job contract, omitting a script ID also means skip.
    """

    model_config = ConfigDict(extra="forbid")

    model_type: ModelType
    dataset_id: str = Field(min_length=1, max_length=255)
    preprocess_script_id: str | None = Field(default=None, max_length=255)
    mode: Literal["use", "skip"] | None = None
    # ``skip`` is an ergonomic alias for clients that model the option as a
    # checkbox.  When it is true the service ignores any script ID and never
    # resolves or executes that script.
    skip: bool = False
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_selection(self) -> "PreprocessingTaskCreate":
        if self.mode == "use" and (self.skip or not self.preprocess_script_id):
            raise ValueError("preprocess_script_id is required when mode is use")
        return self

    @property
    def should_skip(self) -> bool:
        return self.skip or self.mode == "skip" or self.preprocess_script_id is None


class PreprocessingTaskResponse(BaseModel):
    """Stage, audit, and before/after data summary returned to the UI."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    model_type: ModelType
    dataset_id: str
    preprocess_script_id: str | None
    preprocess_used: bool
    preprocess_status: Literal["used", "unused"]
    preprocess_message: str
    status: PreprocessingTaskStatus
    stage: PreprocessingStage
    progress_stage: PreprocessingStage
    logs: list[str]
    error_message: str | None
    config: dict[str, Any]
    input_row_count: int | None
    output_row_count: int | None
    input_columns: list[str]
    output_columns: list[str]
    input_summary: dict[str, Any]
    output_summary: dict[str, Any]
    preprocessor_path: str | None
    preprocessor_state: dict[str, Any] | None
    started_at: datetime | None
    finished_at: datetime | None
    stage_started_at: datetime | None
    created_at: datetime
    next_step: Literal["dataset_split"] | None
    data_source: Literal["raw", "preprocessed"]


class PreprocessingTransformRequest(BaseModel):
    """Request used by future test/prediction adapters to reuse fitted state."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1, max_length=255)
    config: dict[str, Any] | None = None


class PreprocessingTransformResponse(BaseModel):
    """Small, JSON-safe result for a state-reuse operation."""

    task_id: str
    preprocess_used: bool
    data_source: Literal["raw", "preprocessed"]
    row_count: int
    columns: list[str]
    summary: dict[str, Any]
