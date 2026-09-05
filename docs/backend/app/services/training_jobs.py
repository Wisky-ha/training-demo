"""Training-task orchestration, model-artifact persistence, and evaluation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
import copy
import logging
import threading
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import Settings, get_settings
from ..datasets.service import DatasetService
from ..db.models import (
    DatasetORM,
    DatasetSplitORM,
    ModelTypeORM,
    ModelVersionORM,
    PreprocessingTaskORM,
    ScriptORM,
    TrainingJobORM,
)
from ..db.repositories import ModelVersionRepository, TrainingJobRepository
from ..domain.enums import (
    DatasetStatus,
    ModelType,
    ModelVersionStatus,
    PreprocessingStage,
    PreprocessingTaskStatus,
    ScriptStatus,
    ScriptType,
    SplitStrategy,
    TrainingJobStatus,
)
from ..schemas.training_jobs import TrainingJobCreate
from ..storage import ArtifactNotFoundError, ArtifactType, FileStorageService
from .model_baseline import ModelBaselineService
from .model_evaluation import ModelEvaluationError, ModelEvaluationService
from .preprocessing import PreprocessingError, PreprocessingService
from .training_executor import TrainingScriptExecutor


logger = logging.getLogger(__name__)


class TrainingJobError(ValueError):
    """A safe API error with a stable machine-readable code and details."""

    def __init__(self, message: str, code: str = "TRAINING_JOB_FAILED", **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details
        self.job_id: str | None = None


class TrainingJobNotFoundError(TrainingJobError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "TRAINING_JOB_NOT_FOUND")


class _CancellationRequested(Exception):
    pass


class TrainingJobService:
    """Create and run jobs without ever changing the production pointer."""

    MAX_CHART_POINTS = 1000
    _events: dict[str, threading.Event] = {}
    _events_lock = threading.RLock()
    _version_locks: dict[ModelType, threading.Lock] = {}
    _version_locks_guard = threading.RLock()

    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.repository = TrainingJobRepository(session)
        self.version_repository = ModelVersionRepository(session)
        self.baseline_service = ModelBaselineService(session)
        self.storage = FileStorageService(self.settings.file_storage_root, session=session)

    @classmethod
    def _event(cls, job_id: str) -> threading.Event:
        with cls._events_lock:
            return cls._events.setdefault(job_id, threading.Event())

    @classmethod
    def _remove_event(cls, job_id: str) -> None:
        with cls._events_lock:
            cls._events.pop(job_id, None)

    @classmethod
    def _cancelled(cls, job_id: str) -> bool:
        with cls._events_lock:
            event = cls._events.get(job_id)
            return event.is_set() if event is not None else False

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def _script(
        self, script_id: str, model_type: ModelType, script_type: ScriptType, label: str
    ) -> ScriptORM:
        script = self.session.get(ScriptORM, script_id)
        if script is None:
            raise TrainingJobError(f"{label}脚本不存在", "SCRIPT_NOT_FOUND")
        if script.script_type is not script_type:
            raise TrainingJobError(f"所选脚本不是{label}脚本", "SCRIPT_TYPE_INVALID")
        if script.status is not ScriptStatus.ENABLED:
            raise TrainingJobError(f"所选{label}脚本已停用", "SCRIPT_DISABLED")
        if model_type not in {item.code for item in script.supported_model_types}:
            raise TrainingJobError(f"{label}脚本不适用于所选模型类型", "MODEL_TYPE_INCOMPATIBLE")
        return script

    def _validate_preprocessing(
        self,
        request: TrainingJobCreate,
        dataset: DatasetORM,
        split: DatasetSplitORM,
    ) -> PreprocessingTaskORM | None:
        selected_id = request.preprocessing_task_id or split.preprocessing_task_id
        task = self.session.get(PreprocessingTaskORM, selected_id) if selected_id else None
        if request.preprocessing_task_id and task is None:
            raise TrainingJobError("预处理任务不存在", "PREPROCESSING_TASK_NOT_FOUND")
        if task is not None:
            if task.dataset_id != dataset.id:
                raise TrainingJobError("预处理任务与数据集不匹配", "PREPROCESSING_TASK_DATASET_MISMATCH")
            if task.model_type is not request.model_type:
                raise TrainingJobError("预处理任务与模型类型不匹配", "MODEL_TYPE_INCOMPATIBLE")
            if task.stage is not PreprocessingStage.COMPLETED or task.status not in {
                PreprocessingTaskStatus.SUCCEEDED,
                PreprocessingTaskStatus.SKIPPED,
            }:
                raise TrainingJobError("预处理任务尚未成功完成", "PREPROCESSING_STATE_UNAVAILABLE")
        if request.preprocess_script_id:
            script = self._script(
                request.preprocess_script_id, request.model_type, ScriptType.PREPROCESSOR, "预处理"
            )
            if task is None or task.preprocess_script_id != script.id or not task.preprocess_used:
                raise TrainingJobError(
                    "训练任务选择的预处理脚本没有对应的已完成预处理任务",
                    "PREPROCESSING_STATE_UNAVAILABLE",
                )
        elif task is not None and task.preprocess_used:
            raise TrainingJobError(
                "数据集划分使用了预处理结果，必须选择相同的预处理脚本",
                "PREPROCESSING_SCRIPT_REQUIRED",
            )
        if split.preprocessing_task_id != (task.id if task else None):
            raise TrainingJobError(
                "训练任务的预处理任务必须与已完成的数据集划分一致",
                "PREPROCESSING_SPLIT_MISMATCH",
            )
        return task

    def create(self, request: TrainingJobCreate) -> TrainingJobORM:
        """Validate all immutable inputs and commit a waiting job."""

        if self.session.scalar(select(ModelTypeORM.id).where(ModelTypeORM.code == request.model_type)) is None:
            raise TrainingJobError("模型类型不存在", "MODEL_TYPE_NOT_FOUND")
        dataset = self.session.get(DatasetORM, request.dataset_id)
        if dataset is None:
            raise TrainingJobError("数据集不存在", "DATASET_NOT_FOUND")
        if dataset.status is not DatasetStatus.PARSED:
            raise TrainingJobError("数据集尚未完成解析", "DATASET_NOT_READY")
        split = self.session.scalar(
            select(DatasetSplitORM).where(DatasetSplitORM.dataset_id == dataset.id)
        )
        if split is None:
            raise TrainingJobError("数据集尚未完成划分", "DATASET_SPLIT_NOT_FOUND")
        if (
            split.split_strategy is not SplitStrategy.TIME_ORDERED
            or split.split_ratio != 0.8
            or split.test_ratio != 0.2
            or split.total_row_count != dataset.row_count
        ):
            raise TrainingJobError("数据集划分配置无效或已过期", "DATASET_SPLIT_INVALID")
        train_script = self._script(
            request.train_script_id, request.model_type, ScriptType.TRAINER, "训练"
        )
        task = self._validate_preprocessing(request, dataset, split)
        baseline = self.baseline_service.get_retraining_baseline(request.model_type)
        summary = {
            "model_type": {"code": request.model_type.value},
            "dataset": {
                "id": dataset.id,
                "file_name": dataset.file_name,
                "row_count": dataset.row_count,
                "time_column": dataset.time_column,
                "feature_columns": list(dataset.feature_columns),
                "target_column": dataset.target_column,
            },
            "split": {
                "id": split.id,
                "strategy": split.split_strategy.value,
                "split_ratio": split.split_ratio,
                "test_ratio": split.test_ratio,
                "train_row_count": split.train_row_count,
                "test_row_count": split.test_row_count,
                "train_time_range": dict(split.train_time_range),
                "test_time_range": dict(split.test_time_range),
            },
            "preprocessing": {
                "task_id": task.id if task else None,
                "script_id": task.preprocess_script_id if task and task.preprocess_used else None,
                "used": bool(task and task.preprocess_used),
                "data_source": split.data_source,
            },
            "train_script": {
                "id": train_script.id,
                "name": train_script.name,
                "version": train_script.version,
                "status": train_script.status.value,
            },
            "baseline": {
                "model_version_id": baseline.id,
                "version": baseline.version,
                "is_baseline": baseline.is_baseline,
                "source": "system_baseline" if baseline.is_baseline else "current_production",
            },
        }
        job = self.repository.create(
            id=str(uuid4()),
            model_type=request.model_type,
            dataset_id=dataset.id,
            preprocess_script_id=(request.preprocess_script_id or None),
            preprocessing_task_id=task.id if task else None,
            train_script_id=train_script.id,
            split_strategy=split.split_strategy,
            split_ratio=split.split_ratio,
            test_ratio=split.test_ratio,
            status=TrainingJobStatus.PENDING,
            progress_stage="等待执行",
            current_stage="等待执行",
            config=dict(request.config),
            config_summary=summary,
            logs=["PENDING：任务已创建，等待后台执行"],
        )
        self.session.commit()
        self._event(job.id)
        return job

    def get(self, job_id: str) -> TrainingJobORM | None:
        return self.repository.get(job_id)

    def retry(self, job_id: str) -> TrainingJobORM:
        job = self.get(job_id)
        if job is None:
            raise TrainingJobNotFoundError(f"训练任务不存在：{job_id}")
        if job.status is not TrainingJobStatus.FAILED:
            raise TrainingJobError("只有失败任务可以重试", "TRAINING_RETRY_NOT_ALLOWED")
        # Retry is a new execution attempt, not a bypass around the selection
        # guards.  In particular, a script disabled after the failure cannot
        # be executed by retry.
        dataset = self.session.get(DatasetORM, job.dataset_id)
        split = self.session.scalar(select(DatasetSplitORM).where(DatasetSplitORM.dataset_id == job.dataset_id))
        if dataset is None or split is None:
            raise TrainingJobError("数据集或数据集划分不存在", "DATASET_SPLIT_NOT_FOUND")
        if dataset.status is not DatasetStatus.PARSED:
            raise TrainingJobError("数据集尚未完成解析", "DATASET_NOT_READY")
        if split.total_row_count != dataset.row_count or split.split_ratio != 0.8 or split.test_ratio != 0.2:
            raise TrainingJobError("数据集划分配置无效或已过期", "DATASET_SPLIT_INVALID")
        self._script(job.train_script_id, job.model_type, ScriptType.TRAINER, "训练")
        request = TrainingJobCreate(
            model_type=job.model_type,
            dataset_id=job.dataset_id,
            preprocess_script_id=job.preprocess_script_id,
            preprocessing_task_id=job.preprocessing_task_id,
            train_script_id=job.train_script_id,
            config=dict(job.config or {}),
        )
        self._validate_preprocessing(request, dataset, split)
        job.status = TrainingJobStatus.PENDING
        job.progress_stage = "等待执行"
        job.current_stage = "等待执行"
        job.stage_started_at = None
        job.started_at = None
        job.finished_at = None
        job.error_message = None
        job.error_code = None
        job.error_details = {}
        job.model_version_id = None
        job.logs = [*(job.logs or []), "PENDING：已提交失败重试"]
        self.session.commit()
        self._event(job.id).clear()
        return job

    def cancel(self, job_id: str) -> TrainingJobORM:
        job = self.get(job_id)
        if job is None:
            raise TrainingJobNotFoundError(f"训练任务不存在：{job_id}")
        if job.status in {TrainingJobStatus.SUCCEEDED, TrainingJobStatus.FAILED, TrainingJobStatus.CANCELLED}:
            return job
        self._event(job_id).set()
        if job.status is TrainingJobStatus.PENDING:
            now = self._now()
            job.status = TrainingJobStatus.CANCELLED
            job.progress_stage = "CANCELLED"
            job.current_stage = "已取消"
            job.stage_started_at = now
            job.finished_at = now
            job.logs = [*(job.logs or []), "CANCELLED：任务在后台执行前已取消"]
            self.session.commit()
        else:
            job.logs = [*(job.logs or []), "已收到取消请求，任务将在当前操作结束后停止"]
            self.session.commit()
        return job

    def _stage(self, job: TrainingJobORM, stage: str, message: str) -> None:
        if self._cancelled(job.id) or job.status is TrainingJobStatus.CANCELLED:
            raise _CancellationRequested
        now = self._now()
        if job.started_at is None:
            job.started_at = now
        job.status = TrainingJobStatus.RUNNING
        job.progress_stage = stage
        job.current_stage = stage
        job.stage_started_at = now
        job.logs = [*(job.logs or []), message]
        self.session.commit()

    def _check_cancel(self, job: TrainingJobORM) -> None:
        if self._cancelled(job.id) or job.status is TrainingJobStatus.CANCELLED:
            raise _CancellationRequested

    def _fail(
        self,
        job_id: str,
        message: str,
        error_code: str = "TRAINING_JOB_FAILED",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.session.rollback()
        job = self.repository.get(job_id)
        if job is None or job.status is TrainingJobStatus.CANCELLED:
            return
        now = self._now()
        safe_details = dict(details or {})
        safe_details.pop("traceback", None)
        safe_details.pop("stack_trace", None)
        job.status = TrainingJobStatus.FAILED
        job.progress_stage = "FAILED"
        job.current_stage = "失败"
        job.stage_started_at = now
        job.error_message = message
        job.error_code = error_code
        job.error_details = safe_details
        job.finished_at = now
        entries = [*(job.logs or []), f"FAILED[{error_code}]：{message}"]
        if safe_details:
            entries.append(f"异常详情：{safe_details}")
        job.logs = entries
        self.session.commit()

    def _discard_candidate(self, version_id: str) -> None:
        """Compensate a draft row and file if a later stage aborts."""
        self.session.rollback()
        version = self.session.get(ModelVersionORM, version_id)
        if version is not None:
            self.session.delete(version)
            self.session.flush()
        try:
            self.storage.remove(ArtifactType.MODEL, version_id)
            self.session.commit()
        except Exception:
            self.session.rollback()

    def _cancel_finish(self, job_id: str) -> None:
        self.session.rollback()
        job = self.repository.get(job_id)
        if job is None or job.status is TrainingJobStatus.CANCELLED:
            return
        now = self._now()
        job.status = TrainingJobStatus.CANCELLED
        job.progress_stage = "CANCELLED"
        job.current_stage = "已取消"
        job.stage_started_at = now
        job.finished_at = now
        job.logs = [*(job.logs or []), "CANCELLED：任务已取消，未生成模型版本"]
        self.session.commit()

    @staticmethod
    def _ordered_training_data(
        frame: pd.DataFrame, dataset: DatasetORM
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, int, int, pd.Series]:
        if dataset.time_column not in frame.columns or dataset.target_column not in frame.columns:
            raise TrainingJobError("训练数据缺少时间列或目标列", "TRAINING_DATA_SCHEMA_INVALID")
        parsed = DatasetService._parse_datetimes(frame[dataset.time_column])
        if parsed.isna().any():
            raise TrainingJobError("训练数据时间列存在无法解析的值", "SPLIT_TIME_PARSE_FAILED")
        if parsed.duplicated().any():
            raise TrainingJobError("训练数据时间列包含重复值", "SPLIT_TIME_DUPLICATE")
        ordered = frame.copy(deep=True)
        ordered["__training_time"] = parsed
        ordered = ordered.sort_values("__training_time", kind="mergesort").reset_index(drop=True)
        ordered_times = ordered.pop("__training_time")
        train_count = len(ordered) * 4 // 5
        test_count = len(ordered) - train_count
        if train_count < 1 or test_count < 1:
            raise TrainingJobError("训练集和测试集都必须非空", "SPLIT_DATASET_TOO_SMALL")
        features = [str(column) for column in ordered.columns if column not in {dataset.time_column, dataset.target_column}]
        if not features:
            raise TrainingJobError("训练数据没有特征字段", "TRAINING_FEATURES_EMPTY")
        target = pd.to_numeric(ordered[dataset.target_column], errors="coerce")
        if target.isna().any() or not np.isfinite(target.to_numpy(dtype=float)).all():
            raise TrainingJobError("训练数据目标列包含无效值", "TRAINING_TARGET_INVALID")
        train = ordered.iloc[:train_count]
        test = ordered.iloc[train_count:]
        return train[features].copy(), target.iloc[:train_count].copy(), test[features].copy(), target.iloc[train_count:].copy(), train_count, test_count, ordered_times

    @staticmethod
    def _prediction(model: Any, X_test: pd.DataFrame, expected: int) -> np.ndarray:
        try:
            values = model.predict(X_test)
            array = np.asarray(values, dtype=float)
        except Exception as exc:
            raise TrainingJobError(f"模型预测失败：{exc}", "PREDICTION_FAILED") from exc
        if array.ndim != 1 or len(array) != expected:
            raise TrainingJobError(
                f"预测输出长度错误：得到 {len(array)}，预期 {expected}",
                "PREDICTION_LENGTH_INVALID",
            )
        if not np.isfinite(array).all():
            raise TrainingJobError("预测输出包含无效数值", "PREDICTION_VALUES_INVALID")
        return array

    @classmethod
    def _metrics(cls, actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
        """Compatibility wrapper for callers of the former inline evaluator."""
        try:
            return ModelEvaluationService.metrics(actual, predicted)
        except ModelEvaluationError as exc:
            raise TrainingJobError(str(exc), exc.code) from exc

    @staticmethod
    def _json(value: Any) -> Any:
        return DatasetService._json_value(value)

    @classmethod
    def _chart(cls, times: pd.Series, actual: np.ndarray, predicted: np.ndarray) -> tuple[list[dict[str, Any]], bool, int]:
        total = len(actual)
        if total <= cls.MAX_CHART_POINTS:
            indexes = np.arange(total, dtype=int)
        else:
            indexes = np.unique(np.linspace(0, total - 1, cls.MAX_CHART_POINTS, dtype=int))
        rows = []
        for index in indexes:
            signed_error = float(actual[index] - predicted[index])
            percentage = None if actual[index] == 0 else float(abs(signed_error / actual[index]) * 100.0)
            rows.append({
                "time": cls._json(times.iloc[index]),
                "timestamp": cls._json(times.iloc[index]),
                "actual": float(actual[index]),
                "predicted": float(predicted[index]),
                "error": signed_error,
                "signed_error": signed_error,
                "absolute_error": abs(signed_error),
                "percentage_error": percentage,
            })
        return rows, len(indexes) != total, total

    def _baseline_prediction(
        self,
        job: TrainingJobORM,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
    ) -> tuple[ModelVersionORM, np.ndarray, str]:
        """Load the selected baseline and predict on the candidate's X_test.

        The system v0-baseline is metadata-only in this release.  Its stable
        fallback is the training-set mean, while user production versions use
        their persisted artifact.  Both paths receive the exact same X_test.
        """
        selected_id = (job.config_summary or {}).get("baseline", {}).get("model_version_id")
        baseline = self.session.get(ModelVersionORM, selected_id) if selected_id else None
        if baseline is None:
            baseline = self.baseline_service.get_retraining_baseline(job.model_type)
        if baseline.model_type is not job.model_type:
            raise TrainingJobError("基线模型与任务模型类型不匹配", "BASELINE_MODEL_TYPE_MISMATCH")
        try:
            raw = self.storage.read_model(baseline.id)
            model = joblib.load(BytesIO(raw))
            return baseline, self._prediction(model, X_test, len(X_test)), "artifact"
        except ArtifactNotFoundError as exc:
            if not baseline.is_baseline:
                raise TrainingJobError(f"基线模型制品不存在：{exc}", "BASELINE_ARTIFACT_NOT_FOUND") from exc
            # v0-baseline has no model file yet.  Do not fabricate a comparison
            # from stale persisted metrics: calculate all baseline metrics on
            # this same test set using the deterministic mean predictor.
            values = y_train.to_numpy(dtype=float)
            if values.size == 0 or not np.isfinite(values).all():
                raise TrainingJobError("基线模型训练数据无效", "BASELINE_DATA_INVALID")
            return baseline, np.full(len(X_test), float(np.mean(values))), "training_mean_fallback"
        except Exception as exc:
            raise TrainingJobError(f"基线模型预测失败：{exc}", "BASELINE_PREDICTION_FAILED") from exc

    @classmethod
    def _version_lock(cls, model_type: ModelType) -> threading.Lock:
        with cls._version_locks_guard:
            return cls._version_locks.setdefault(model_type, threading.Lock())

    @staticmethod
    def _next_version(session: Session, model_type: ModelType) -> str:
        return ModelVersionRepository(session).next_version(model_type)

    @staticmethod
    def _frame_data_summary(
        times: pd.Series, features: pd.DataFrame, target: pd.Series,
    ) -> dict[str, Any]:
        """Persist a JSON-safe summary of exactly one split."""
        values = target.to_numpy(dtype=float)
        return {
            "row_count": int(len(target)),
            "columns": [str(column) for column in features.columns],
            "time_range": {
                "start": DatasetService._json_value(times.iloc[0]) if len(times) else None,
                "end": DatasetService._json_value(times.iloc[-1]) if len(times) else None,
            },
            "target": {
                "min": float(np.min(values)) if len(values) else None,
                "max": float(np.max(values)) if len(values) else None,
                "mean": float(np.mean(values)) if len(values) else None,
            },
        }

    @staticmethod
    def _input_schema(frame: pd.DataFrame, dataset: DatasetORM, feature_columns: list[str]) -> dict[str, Any]:
        columns = [str(column) for column in frame.columns]
        types = {
            column: dataset.column_types.get(column, "number" if pd.api.types.is_numeric_dtype(frame[column]) else "string")
            for column in columns
        }
        types[dataset.time_column] = "datetime"
        return {
            "columns": columns,
            "required_columns": [dataset.time_column, *feature_columns],
            "column_types": types,
            "time_column": dataset.time_column,
            "time_format": (dataset.time_parse or {}).get("format"),
            "target_column": dataset.target_column,
            "extra_columns": "reject",
        }

    def _save_version(
        self, job: TrainingJobORM, dataset: DatasetORM, train_script: ScriptORM,
        task: PreprocessingTaskORM | None, frame: pd.DataFrame, X_train: pd.DataFrame,
        X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
        ordered_times: pd.Series, result: Any, predictions: np.ndarray,
    ) -> tuple[ModelVersionORM, dict[str, Any], str]:
        # Version labels are generated while holding a per-model-family lock;
        # the database unique constraint remains the cross-process safeguard.
        with self._version_lock(job.model_type):
            return self._save_version_locked(
                job, dataset, train_script, task, frame, X_train, X_test,
                y_train, y_test, ordered_times, result, predictions,
            )

    def _save_version_locked(
        self, job: TrainingJobORM, dataset: DatasetORM, train_script: ScriptORM,
        task: PreprocessingTaskORM | None, frame: pd.DataFrame, X_train: pd.DataFrame,
        X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
        ordered_times: pd.Series, result: Any, predictions: np.ndarray,
    ) -> tuple[ModelVersionORM, dict[str, Any], str]:
        actual = y_test.to_numpy(dtype=float)
        train_times = ordered_times.iloc[:len(y_train)].reset_index(drop=True)
        test_times = ordered_times.iloc[len(y_train):].reset_index(drop=True)
        feature_columns = [str(item) for item in X_train.columns]
        version_label = self._next_version(self.session, job.model_type)
        baseline, baseline_predictions, baseline_source = self._baseline_prediction(
            job, X_train, y_train, X_test
        )
        try:
            evaluation = ModelEvaluationService.evaluate(
                times=[self._json(item) for item in test_times],
                actual=actual,
                candidate_predictions=predictions,
                baseline_predictions=baseline_predictions,
                candidate_id=None,
                baseline_id=baseline.id,
                baseline_version=baseline.version,
                candidate_version=version_label,
            )
        except ModelEvaluationError as exc:
            raise TrainingJobError(str(exc), exc.code) from exc
        chart_data, sampled, total = self._chart(test_times, actual, predictions)
        evaluation["chart_data"] = chart_data
        evaluation["chart_sampled"] = sampled
        evaluation["chart_total_count"] = total
        evaluation["chart_sample_count"] = len(chart_data)
        comparison = evaluation["model_comparison"]
        comparison["baseline"]["source"] = baseline_source
        comparison["production"] = comparison["baseline"]
        comparison["current_model"] = comparison["baseline"]
        model_bytes = None
        serialization_error: Exception | None = None
        # Prefer cloudpickle so source-defined model classes remain loadable
        # after the executor removes its temporary module from sys.modules.
        try:
            import cloudpickle
            model_bytes = cloudpickle.dumps(result.model)
        except Exception as exc:
            serialization_error = exc
        if model_bytes is None:
            try:
                buffer = BytesIO()
                joblib.dump(result.model, buffer)
                model_bytes = buffer.getvalue()
            except Exception as exc:
                cause = serialization_error or exc
                raise TrainingJobError(f"模型制品保存失败：{cause}", "MODEL_SERIALIZATION_FAILED") from exc
        version_id = str(uuid4())
        # Allocate the candidate ID before inserting JSON metrics so the
        # comparison payload is complete in the initial row.
        comparison["candidate"]["model_version_id"] = version_id
        artifact = self.storage.save_model(version_id, model_bytes)
        preprocessor_artifact = None
        try:
            preprocessor_artifact = (
                self.storage.metadata(ArtifactType.PREPROCESSOR, task.id)
                if task and task.preprocess_used else None
            )
            version = self.version_repository.create(
                id=version_id,
                model_type=job.model_type,
                version=version_label,
                model_path=artifact.relative_path,
                model_artifact_id=artifact.id,
                training_job_id=job.id,
                train_script_id=train_script.id,
                train_script_version=train_script.version,
                train_script_source=str(train_script.source_code),
                preprocess_script_id=task.preprocess_script_id if task and task.preprocess_used else None,
                preprocess_script_version=(task.preprocess_script.version if task and task.preprocess_used and task.preprocess_script else None),
                preprocess_script_source=(str(task.preprocess_script.source_code) if task and task.preprocess_used and task.preprocess_script else None),
                preprocess_used=bool(task and task.preprocess_used),
                preprocessor_path=task.preprocessor_path if task and task.preprocess_used else None,
                preprocessor_artifact_id=preprocessor_artifact.id if preprocessor_artifact else None,
                preprocessor_state=copy.deepcopy(task.preprocessor_state) if task and task.preprocess_used else None,
                input_schema=self._input_schema(frame, dataset, feature_columns),
                time_column=dataset.time_column,
                feature_columns=feature_columns,
                target_column=dataset.target_column,
                split_strategy=job.split_strategy,
                split_ratio=job.split_ratio,
                test_ratio=job.test_ratio,
                train_data_summary=self._frame_data_summary(train_times, X_train, y_train),
                test_data_summary=self._frame_data_summary(test_times, X_test, y_test),
                metrics=copy.deepcopy(evaluation),
                status=ModelVersionStatus.DRAFT,
            )
        except Exception:
            # File storage and the SQL transaction cannot be atomic together.
            # Compensate the artifact before allowing the worker to mark the
            # job failed.
            self.session.rollback()
            try:
                self.storage.remove(ArtifactType.MODEL, version_id)
                self.session.commit()
            except Exception:
                self.session.rollback()
            raise
        # Keep a detached copy on the ORM value; nested JSON mutations are
        # not tracked by SQLAlchemy's MutableDict.
        version.metrics = copy.deepcopy(evaluation)
        self.session.flush()
        return version, evaluation, artifact.relative_path

    def run(self, job_id: str, config: dict[str, Any] | None = None) -> None:
        """Execute in a worker-owned session and persist a READY candidate."""

        job = self.repository.get(job_id)
        if job is None or job.status is TrainingJobStatus.CANCELLED:
            return
        artifact_id: str | None = None
        try:
            self._stage(job, "准备数据", "后台任务开始：RUNNING：准备数据（预处理状态和数据集划分已校验）")
            dataset = self.session.get(DatasetORM, job.dataset_id)
            split = self.session.scalar(select(DatasetSplitORM).where(DatasetSplitORM.dataset_id == job.dataset_id))
            if dataset is None or split is None:
                raise TrainingJobError("训练数据集或划分不存在", "DATASET_SPLIT_NOT_FOUND")
            task = self.session.get(PreprocessingTaskORM, job.preprocessing_task_id) if job.preprocessing_task_id else None
            preprocessing = PreprocessingService(self.session, settings=self.settings)
            frame = preprocessing._dataset_frame(dataset)
            if task and task.preprocess_used:
                frame = preprocessing.transform_with_saved_state(task, frame, config or job.config)
            X_train, y_train, X_test, y_test, train_count, test_count, ordered_times = self._ordered_training_data(frame, dataset)
            job.train_row_count = train_count
            job.test_row_count = test_count
            job.train_time_start = ordered_times.iloc[0].to_pydatetime()
            job.train_time_end = ordered_times.iloc[train_count - 1].to_pydatetime()
            job.test_time_start = ordered_times.iloc[train_count].to_pydatetime()
            job.test_time_end = ordered_times.iloc[-1].to_pydatetime()
            self.session.commit()
            self._check_cancel(job)

            self._stage(job, "加载训练脚本", "RUNNING：加载训练脚本")
            train_script = self.session.get(ScriptORM, job.train_script_id)
            if train_script is None:
                raise TrainingJobError("训练脚本不存在", "SCRIPT_NOT_FOUND")
            self._stage(job, "执行训练", "RUNNING：执行训练")
            executor = TrainingScriptExecutor()
            try:
                result = executor.execute(
                    script=train_script, X_train=X_train, y_train=y_train, X_test=X_test,
                    y_test=y_test, config=config or job.config, persist_model=True,
                )
            except TypeError as exc:
                # Keep service tests and older internal adapters compatible
                # while the persistence flag is being rolled out.
                if "persist_model" not in str(exc):
                    raise
                result = executor.execute(
                    script=train_script, X_train=X_train, y_train=y_train, X_test=X_test,
                    y_test=y_test, config=config or job.config,
                )
            job.logs = [*(job.logs or []), *result.logs]
            self.session.commit()
            self._check_cancel(job)
            if not result.success or result.model is None:
                raise TrainingJobError(result.error or "训练脚本执行失败", result.error_code or "TRAIN_EXECUTION_FAILED")
            predictions = self._prediction(result.model, X_test, len(y_test))
            self._stage(job, "保存模型", "RUNNING：保存模型草稿")
            version, evaluation, model_path = self._save_version(
                job, dataset, train_script, task, frame, X_train, X_test, y_train,
                y_test, ordered_times, result, predictions,
            )
            artifact_id = version.id
            job.model_version_id = version.id
            self._stage(job, "进入评估", "RUNNING：进入评估")
            self._check_cancel(job)
            # A candidate is transiently DRAFT while its artifact and
            # evaluation are being assembled.  Only a fully successful
            # training run exposes it to the lifecycle publisher as READY.
            version.status = ModelVersionStatus.READY
            # Evaluation was calculated from every test row before this final
            # state transition.  No epoch/percentage progress is fabricated.
            job.status = TrainingJobStatus.SUCCEEDED
            job.progress_stage = "SUCCEEDED"
            job.current_stage = "进入评估"
            job.stage_started_at = self._now()
            job.finished_at = self._now()
            job.logs = [*(job.logs or []), f"SUCCEEDED：评估完成，模型已保存为 READY（{model_path}）"]
            self.session.commit()
        except _CancellationRequested:
            if artifact_id:
                self._discard_candidate(artifact_id)
            self._cancel_finish(job_id)
        except (TrainingJobError, PreprocessingError) as exc:
            if artifact_id:
                self._discard_candidate(artifact_id)
            details = dict(getattr(exc, "details", {}) or {})
            if getattr(exc, "task_id", None):
                details["task_id"] = exc.task_id
            details.setdefault("exception_type", type(exc).__name__)
            self._fail(job_id, str(exc), getattr(exc, "code", "TRAINING_JOB_FAILED"), details)
        except Exception as exc:
            if artifact_id:
                self._discard_candidate(artifact_id)
            logger.exception("training job failed job_id=%s", job_id)
            self._fail(
                job_id,
                "训练任务执行失败",
                "TRAINING_JOB_FAILED",
                {"exception_type": type(exc).__name__},
            )
        finally:
            self._remove_event(job_id)

    @staticmethod
    def submit(executor: ThreadPoolExecutor, session_factory: Any, job_id: str, config: dict[str, Any] | None, settings: Settings) -> None:
        def worker() -> None:
            with session_factory() as session:
                TrainingJobService(session, settings=settings).run(job_id, config=config)
        executor.submit(worker)

    @classmethod
    def evaluation_response(cls, job: TrainingJobORM) -> dict[str, Any]:
        if job.status is not TrainingJobStatus.SUCCEEDED or not job.model_version_id:
            raise TrainingJobError("训练任务尚未完成评估", "EVALUATION_NOT_READY")
        version = job.model_version
        if version is None:
            version = next((item for item in job.model_versions if item.id == job.model_version_id), None)
        if version is None or not version.metrics:
            raise TrainingJobError("评估结果不存在", "EVALUATION_NOT_FOUND")
        metrics = {key: version.metrics.get(key) for key in ("mae", "rmse", "mape", "r2", "sample_count", "mape_valid_count", "mape_excluded_count", "mape_note")}
        chart_data = version.metrics.get("chart_data", [])
        comparison = version.metrics.get("model_comparison", {})
        return {
            "job_id": job.id,
            "model_version_id": version.id,
            "metrics": metrics,
            "mae": metrics["mae"], "rmse": metrics["rmse"], "mape": metrics["mape"], "r2": metrics["r2"],
            "chart_data": chart_data,
            "chart": {
                "timestamps": [row.get("timestamp", row.get("time")) for row in chart_data],
                "actual_values": [row.get("actual") for row in chart_data],
                "predicted_values": [row.get("predicted") for row in chart_data],
            },
            "error_data": version.metrics.get("error_data", []),
            "test_time_series": version.metrics.get("test_time_series", []),
            "timestamps": version.metrics.get("timestamps", version.metrics.get("test_time_series", [])),
            "actual_values": version.metrics.get("actual_values", []),
            "candidate_predictions": version.metrics.get("candidate_predictions", []),
            "baseline_predictions": version.metrics.get("baseline_predictions", []),
            "candidate_errors": version.metrics.get("candidate_errors", []),
            "baseline_errors": version.metrics.get("baseline_errors", []),
            "error_series": version.metrics.get("error_series", version.metrics.get("candidate_errors", [])),
            "model_metrics": version.metrics.get("metrics", {}),
            "metric_differences": version.metrics.get("metric_differences", {}),
            "metrics_difference": version.metrics.get("metrics_difference", version.metrics.get("metric_differences", {})),
            "chart_sampled": version.metrics.get("chart_sampled", False),
            "chart_total_count": version.metrics.get("chart_total_count", metrics["sample_count"]),
            "chart_sample_count": version.metrics.get("chart_sample_count", len(version.metrics.get("chart_data", []))),
            "model_comparison": comparison,
            "comparison": comparison,
        }

    @staticmethod
    def to_response(job: TrainingJobORM) -> dict[str, Any]:
        return {
            "id": job.id, "model_type": job.model_type.value, "dataset_id": job.dataset_id,
            "preprocess_script_id": job.preprocess_script_id,
            "preprocessing_task_id": job.preprocessing_task_id,
            "train_script_id": job.train_script_id,
            "split_strategy": job.split_strategy.value, "split_ratio": job.split_ratio, "test_ratio": job.test_ratio,
            "status": job.status.value, "progress_stage": job.progress_stage, "current_stage": job.current_stage,
            "stage": job.current_stage,
            "stage_started_at": job.stage_started_at, "started_at": job.started_at,
            "logs": list(job.logs or []), "error_message": job.error_message,
            "error_code": job.error_code,
            "error_details": dict(job.error_details or {}),
            "config": dict(job.config or {}), "config_summary": dict(job.config_summary or {}),
            "model_version_id": job.model_version_id,
            "train_row_count": job.train_row_count, "test_row_count": job.test_row_count,
            "train_time_start": job.train_time_start, "train_time_end": job.train_time_end,
            "test_time_start": job.test_time_start, "test_time_end": job.test_time_end,
            "created_at": job.created_at, "finished_at": job.finished_at,
        }


__all__ = ["TrainingJobError", "TrainingJobNotFoundError", "TrainingJobService"]
