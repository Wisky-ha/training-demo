"""API contracts for the fixed, time-ordered dataset split."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..domain.enums import SplitStrategy


class DatasetSplitRequest(BaseModel):
    """Optional selection of an already completed preprocessing result.

    The ratios and strategy intentionally do not exist in this request model:
    clients cannot change the platform's fixed 80/20 time split.
    """

    model_config = ConfigDict(extra="forbid")

    preprocessing_task_id: str | None = Field(default=None, min_length=1, max_length=255)


class TimeRange(BaseModel):
    start: str
    end: str


class DatasetSplitResponse(BaseModel):
    """Persisted split boundary and metadata, without exposing dataset rows."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    preprocessing_task_id: str | None
    data_source: Literal["raw", "preprocessed"]
    split_strategy: SplitStrategy
    split_ratio: float
    test_ratio: float
    total_row_count: int
    train_row_count: int
    test_row_count: int
    train_time_range: TimeRange
    test_time_range: TimeRange
    # Flat aliases make the range convenient to consume in a table while the
    # nested fields above are the canonical API representation.
    train_time_start: str
    train_time_end: str
    test_time_start: str
    test_time_end: str
    sort_order: Literal["ascending"]
    rounding_rule: Literal["floor(total_row_count * 0.8)"]
    sorted_before_split: bool
    created_at: datetime

    @classmethod
    def from_orm_split(cls, split: Any) -> "DatasetSplitResponse":
        train_range = dict(split.train_time_range or {})
        test_range = dict(split.test_time_range or {})
        return cls(
            id=split.id,
            dataset_id=split.dataset_id,
            preprocessing_task_id=split.preprocessing_task_id,
            data_source=split.data_source,
            split_strategy=split.split_strategy,
            split_ratio=split.split_ratio,
            test_ratio=split.test_ratio,
            total_row_count=split.total_row_count,
            train_row_count=split.train_row_count,
            test_row_count=split.test_row_count,
            train_time_range=train_range,
            test_time_range=test_range,
            train_time_start=train_range["start"],
            train_time_end=train_range["end"],
            test_time_start=test_range["start"],
            test_time_end=test_range["end"],
            sort_order=split.sort_order,
            rounding_rule=split.rounding_rule,
            sorted_before_split=split.sorted_before_split,
            created_at=split.created_at,
        )
