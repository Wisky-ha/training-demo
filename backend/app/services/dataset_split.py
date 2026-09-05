"""Fixed time-ordered 80/20 dataset splitting."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pandas as pd
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..core.config import Settings, get_settings
from ..db.models import DatasetORM, DatasetSplitORM, PreprocessingTaskORM
from ..db.repositories import DatasetSplitRepository
from ..domain.enums import PreprocessingStage, PreprocessingTaskStatus, SplitStrategy
from ..schemas.dataset_split import DatasetSplitResponse
from .preprocessing import PreprocessingError, PreprocessingService
from ..datasets.service import DatasetService


class DatasetSplitError(ValueError):
    """User-facing split failure with a stable machine-readable code."""

    def __init__(self, message: str, code: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


class DatasetSplitNotFoundError(DatasetSplitError):
    def __init__(self, message: str, code: str = "DATASET_SPLIT_NOT_FOUND", **details: Any) -> None:
        super().__init__(message, code, **details)


class DatasetSplitService:
    """Calculate and persist one immutable split metadata record per dataset."""

    TRAIN_RATIO = 0.8
    TEST_RATIO = 0.2
    ROUNDING_RULE = "floor(total_row_count * 0.8)"
    MINIMUM_ROWS = 2

    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.repository = DatasetSplitRepository(session)
        self.preprocessing = PreprocessingService(session, settings=self.settings)

    @staticmethod
    def _detail(exc: DatasetSplitError) -> dict[str, Any]:
        return {"code": exc.code, "message": str(exc), **exc.details}

    def _frame(
        self, dataset: DatasetORM, preprocessing_task_id: str | None
    ) -> tuple[pd.DataFrame, str, str | None]:
        """Read raw data or reuse the selected completed preprocessing state."""

        task = None
        if preprocessing_task_id is not None:
            task = self.session.get(PreprocessingTaskORM, preprocessing_task_id)
            if task is None:
                raise DatasetSplitError("预处理任务不存在", "PREPROCESSING_TASK_NOT_FOUND")
            if task.dataset_id != dataset.id:
                raise DatasetSplitError(
                    "预处理任务与数据集不匹配",
                    "PREPROCESSING_TASK_DATASET_MISMATCH",
                    dataset_id=dataset.id,
                    preprocessing_task_id=preprocessing_task_id,
                )
            if task.stage is not PreprocessingStage.COMPLETED or task.status not in {
                PreprocessingTaskStatus.SUCCEEDED,
                PreprocessingTaskStatus.SKIPPED,
            }:
                raise DatasetSplitError(
                    "预处理任务尚未成功完成，不能用于数据集划分",
                    "PREPROCESSING_STATE_UNAVAILABLE",
                    preprocessing_task_id=preprocessing_task_id,
                )

        try:
            frame = self.preprocessing._dataset_frame(dataset)
        except PreprocessingError as exc:
            raise DatasetSplitError(str(exc), exc.code) from exc

        if task is None:
            return frame, "raw", None

        try:
            output = self.preprocessing.transform_with_saved_state(task, frame)
        except PreprocessingError as exc:
            raise DatasetSplitError(str(exc), exc.code) from exc
        return output, "preprocessed" if task.preprocess_used else "raw", task.id

    @staticmethod
    def _time_ranges(frame: pd.DataFrame, time_column: str) -> tuple[dict[str, str], dict[str, str], int, int]:
        if time_column not in frame.columns:
            raise DatasetSplitError(
                f"时间列“{time_column}”不存在",
                "SPLIT_TIME_COLUMN_MISSING",
                column=time_column,
            )

        values = frame[time_column]
        missing = DatasetService._missing_mask(values)
        if bool(missing.any()):
            raise DatasetSplitError(
                f"时间列“{time_column}”包含缺失值，无法划分",
                "SPLIT_TIME_VALUE_MISSING",
                column=time_column,
                count=int(missing.sum()),
            )
        parsed = DatasetService._parse_datetimes(values)
        invalid = parsed.isna()
        if bool(invalid.any()):
            raise DatasetSplitError(
                f"时间列“{time_column}”存在无法解析的值，无法划分",
                "SPLIT_TIME_PARSE_FAILED",
                column=time_column,
                count=int(invalid.sum()),
            )
        duplicate_count = int(parsed.duplicated().sum())
        if duplicate_count:
            raise DatasetSplitError(
                f"时间列“{time_column}”包含重复时间值，无法确定划分边界",
                "SPLIT_TIME_DUPLICATE",
                column=time_column,
                count=duplicate_count,
            )

        # ``mergesort`` is stable.  It is deliberately applied to a copy so
        # sorting can never rewrite the uploaded source or preprocessing output.
        ordered = pd.DataFrame({"_parsed_time": parsed}).sort_values(
            "_parsed_time", kind="mergesort"
        )["_parsed_time"].reset_index(drop=True)
        row_count = len(ordered)
        if row_count < DatasetSplitService.MINIMUM_ROWS:
            raise DatasetSplitError(
                "数据量不足以完成 80%/20% 划分，训练集和测试集都必须非空",
                "SPLIT_DATASET_TOO_SMALL",
                count=row_count,
                minimum=DatasetSplitService.MINIMUM_ROWS,
            )
        # Exact integer arithmetic is floor(n * 4 / 5), avoiding floating
        # point boundary surprises.  The remainder always belongs to test.
        train_count = row_count * 4 // 5
        test_count = row_count - train_count
        if train_count < 1 or test_count < 1:
            raise DatasetSplitError(
                "数据量不足以保证训练集和测试集非空",
                "SPLIT_DATASET_TOO_SMALL",
                count=row_count,
                minimum=DatasetSplitService.MINIMUM_ROWS,
            )

        def timestamp(value: Any) -> str:
            result = DatasetService._json_value(value)
            return str(result)

        train_values = ordered.iloc[:train_count]
        test_values = ordered.iloc[train_count:]
        return (
            {"start": timestamp(train_values.iloc[0]), "end": timestamp(train_values.iloc[-1])},
            {"start": timestamp(test_values.iloc[0]), "end": timestamp(test_values.iloc[-1])},
            train_count,
            test_count,
        )

    def get(self, dataset_id: str) -> DatasetSplitORM | None:
        dataset = self.session.get(DatasetORM, dataset_id)
        if dataset is None:
            raise DatasetSplitNotFoundError("数据集不存在", "DATASET_NOT_FOUND", dataset_id=dataset_id)
        return self.repository.get_for_dataset(dataset_id)

    def split(self, dataset_id: str, preprocessing_task_id: str | None = None) -> DatasetSplitORM:
        dataset = self.session.get(DatasetORM, dataset_id)
        if dataset is None:
            raise DatasetSplitNotFoundError("数据集不存在", "DATASET_NOT_FOUND", dataset_id=dataset_id)
        existing = self.repository.get_for_dataset(dataset_id)
        if existing is not None:
            raise DatasetSplitError(
                "该数据集已经完成划分，不能重复划分",
                "DATASET_SPLIT_ALREADY_EXISTS",
                dataset_id=dataset_id,
                split_id=existing.id,
            )

        frame, data_source, selected_task_id = self._frame(dataset, preprocessing_task_id)
        train_range, test_range, train_count, test_count = self._time_ranges(
            frame, dataset.time_column
        )
        split = DatasetSplitORM(
            id=str(uuid4()),
            dataset_id=dataset_id,
            preprocessing_task_id=selected_task_id,
            data_source=data_source,
            split_strategy=SplitStrategy.TIME_ORDERED,
            split_ratio=self.TRAIN_RATIO,
            test_ratio=self.TEST_RATIO,
            total_row_count=train_count + test_count,
            train_row_count=train_count,
            test_row_count=test_count,
            train_time_range=train_range,
            test_time_range=test_range,
            sort_order="ascending",
            rounding_rule=self.ROUNDING_RULE,
            sorted_before_split=True,
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(split)
        try:
            self.session.flush()
        except IntegrityError as exc:
            self.session.rollback()
            existing = self.repository.get_for_dataset(dataset_id)
            if existing is not None:
                raise DatasetSplitError(
                    "该数据集已经完成划分，不能重复划分",
                    "DATASET_SPLIT_ALREADY_EXISTS",
                    dataset_id=dataset_id,
                    split_id=existing.id,
                ) from exc
            raise DatasetSplitError(
                "数据集划分结果无法持久化",
                "DATASET_SPLIT_PERSIST_FAILED",
            ) from exc
        self.session.commit()
        return split

    @staticmethod
    def to_response(split: DatasetSplitORM) -> DatasetSplitResponse:
        return DatasetSplitResponse.from_orm_split(split)


__all__ = ["DatasetSplitError", "DatasetSplitNotFoundError", "DatasetSplitService"]
