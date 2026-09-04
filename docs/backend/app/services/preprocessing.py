"""Execution service for the optional, script-backed preprocessing stage.

The service executes source selected by a database ID only.  It never accepts a
Python module or filesystem path from a request, and it keeps the fitted object
as a local artifact owned by the task.  This is an execution guard for the
internal demo (not a replacement for a process/container sandbox).
"""

from __future__ import annotations

import ast
import builtins
from contextlib import contextmanager
from datetime import datetime, timezone
from io import BytesIO, StringIO
import inspect
import sys
import types
from typing import Any, Iterator
from uuid import uuid4

import joblib
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import Settings, get_settings
from ..db.models import DatasetORM, ModelTypeORM, PreprocessingTaskORM, ScriptORM
from ..db.repositories import PreprocessingTaskRepository
from ..domain.enums import (
    ModelType,
    PreprocessingStage,
    PreprocessingTaskStatus,
    ScriptStatus,
    ScriptType,
)
from ..schemas.preprocessing import (
    PreprocessingTaskCreate,
    PreprocessingTaskResponse,
    PreprocessingTransformResponse,
)
from ..storage import ArtifactNotFoundError, ArtifactType, FileStorageService
from ..datasets.service import DatasetService


class PreprocessingError(ValueError):
    """A safe, user-facing preprocessing error with a stable error code."""

    def __init__(self, message: str, code: str = "PREPROCESS_FAILED") -> None:
        super().__init__(message)
        self.code = code
        self.task_id: str | None = None


class PreprocessingNotFoundError(PreprocessingError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "PREPROCESSING_TASK_NOT_FOUND")


_ALLOWED_IMPORTS = {
    "pandas",
    "numpy",
    "sklearn",
    "joblib",
    "math",
    "statistics",
    "datetime",
    "typing",
    "collections",
    "itertools",
}
_FORBIDDEN_CALLS = {
    "__import__",
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "breakpoint",
}
_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "__build_class__", "object", "type", "super", "staticmethod", "classmethod",
        "property", "Exception", "ValueError", "TypeError", "RuntimeError",
        "AttributeError", "KeyError", "IndexError", "isinstance", "issubclass",
        "getattr", "setattr", "hasattr", "len", "min", "max", "sum", "abs",
        "round", "all", "any", "enumerate", "range", "zip", "map", "filter",
        "list", "dict", "set", "tuple", "float", "int", "str", "bool",
        "print", "repr", "sorted", "reversed", "slice",
    )
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _module_name(script_id: str) -> str:
    # IDs are database values, not import paths.  Hex encoding prevents a
    # malformed value from becoming a dotted module name.
    return "platform_preprocessor_" + script_id.encode("utf-8").hex()


def _validate_source(source: str) -> ast.Module:
    try:
        tree = ast.parse(source, filename="preprocessor.py")
    except (SyntaxError, ValueError, TypeError) as exc:
        raise PreprocessingError("预处理脚本语法无效", "INVALID_SCRIPT") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.level:
                raise PreprocessingError("预处理脚本不允许相对导入", "INVALID_SCRIPT")
            names = node.names
            for item in names:
                top_level = item.name.split(".", 1)[0]
                if top_level not in _ALLOWED_IMPORTS:
                    raise PreprocessingError(
                        f"预处理脚本导入了不允许的模块：{top_level}",
                        "UNSAFE_SCRIPT",
                    )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALLS:
                raise PreprocessingError(
                    f"预处理脚本调用了不允许的函数：{node.func.id}",
                    "UNSAFE_SCRIPT",
                )
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Preprocessor"]
    if len(classes) != 1:
        raise PreprocessingError(
            "预处理脚本必须定义且只能定义一个 Preprocessor 类", "INVALID_PREPROCESSOR"
        )
    return tree


@contextmanager
def _loaded_module(script_id: str, source: str) -> Iterator[types.ModuleType]:
    """Compile a validated source in a generated, non-user-selectable module."""

    tree = _validate_source(source)
    name = _module_name(script_id)
    module = types.ModuleType(name)
    module.__file__ = f"<preprocessor:{script_id}>"

    def safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split(".", 1)[0] not in _ALLOWED_IMPORTS:
            raise ImportError(f"module is not allowed: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    module.__dict__.update(
        {
            "__name__": name,
            "__package__": "",
            "__builtins__": {**_SAFE_BUILTINS, "__import__": safe_import},
        }
    )
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(tree, module.__file__, "exec"), module.__dict__, module.__dict__)
        yield module
    except PreprocessingError:
        raise
    except Exception as exc:
        raise PreprocessingError(f"预处理脚本导入失败：{exc}", "SCRIPT_IMPORT_FAILED") from exc
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


@contextmanager
def _pickle_module(instance: Any, script_id: str) -> Iterator[None]:
    """Temporarily register the exact module that owns a fitted class.

    Dynamic modules are removed after execution.  Re-executing source before
    pickling would create a different class object and makes pickle reject the
    instance, so the original function globals are registered instead.
    """

    name = _module_name(script_id)
    globals_dict = getattr(getattr(instance, "fit", None), "__func__", getattr(instance, "fit", None)).__globals__
    module = types.ModuleType(name)
    module.__dict__.update(globals_dict)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def _check_method_signature(instance: Any, method_name: str) -> None:
    method = getattr(instance, method_name, None)
    if not callable(method):
        raise PreprocessingError(
            f"Preprocessor.{method_name}(df, config) 未实现", "INVALID_PREPROCESSOR"
        )
    try:
        signature = inspect.signature(method)
        parameters = list(signature.parameters.values())
        signature.bind(pd.DataFrame(), {})
        if len(parameters) != 2:
            raise TypeError("expected exactly two parameters")
    except (TypeError, ValueError) as exc:
        raise PreprocessingError(
            f"Preprocessor.{method_name} 的签名必须为 (df, config)",
            "INVALID_PREPROCESSOR",
        ) from exc
    if any(parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
           for parameter in parameters):
        raise PreprocessingError(
            f"Preprocessor.{method_name} 不允许使用可变参数", "INVALID_PREPROCESSOR"
        )


def _frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Build a compact JSON-safe summary without inventing progress values."""

    def safe(value: Any) -> Any:
        return DatasetService._json_value(value)

    columns: dict[str, Any] = {}
    for column in frame.columns:
        series = frame[column]
        missing = int(series.isna().sum())
        item: dict[str, Any] = {
            "type": str(series.dtype),
            "missing_count": missing,
            "non_missing_count": int(series.notna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
        }
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any() and numeric.notna().sum() == series.notna().sum():
            item.update({
                "min": safe(numeric.min()),
                "max": safe(numeric.max()),
                "mean": safe(numeric.mean()),
            })
        columns[str(column)] = item
    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": columns,
        "column_names": [str(column) for column in frame.columns],
        "missing_value_counts": {str(column): int(frame[column].isna().sum()) for column in frame.columns},
    }


class PreprocessingService:
    """Create, execute, inspect, and reuse preprocessing task state."""

    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.repository = PreprocessingTaskRepository(session)
        self.storage = FileStorageService(self.settings.file_storage_root, session=session)

    def _dataset_frame(self, dataset: DatasetORM) -> pd.DataFrame:
        try:
            content = self.storage.read_dataset(dataset.id)
        except (ArtifactNotFoundError, OSError) as exc:
            raise PreprocessingError("数据集文件不存在或无法读取", "DATASET_READ_FAILED") from exc
        try:
            text = DatasetService._decode(content)
            frame = pd.read_csv(StringIO(text), dtype=object, index_col=False)
        except Exception as exc:
            raise PreprocessingError(f"数据集读取失败：{exc}", "DATASET_READ_FAILED") from exc
        frame.columns = [str(column).strip() for column in frame.columns]
        if list(frame.columns) != list(dataset.columns) or len(frame) != dataset.row_count:
            raise PreprocessingError("数据集文件与已保存的数据摘要不一致", "DATASET_SCHEMA_CHANGED")
        return frame

    def _script(self, script_id: str, model_type: ModelType) -> ScriptORM:
        script = self.session.get(ScriptORM, script_id)
        if script is None:
            raise PreprocessingError("预处理脚本不存在", "SCRIPT_NOT_FOUND")
        if script.script_type is not ScriptType.PREPROCESSOR:
            raise PreprocessingError("所选脚本不是预处理脚本", "SCRIPT_TYPE_INVALID")
        if script.status is not ScriptStatus.ENABLED:
            raise PreprocessingError("所选预处理脚本已停用", "SCRIPT_DISABLED")
        if model_type not in {item.code for item in script.supported_model_types}:
            raise PreprocessingError("预处理脚本不适用于所选模型类型", "MODEL_TYPE_INCOMPATIBLE")
        # The DB source is the immutable library snapshot.  Storage is only a
        # local artifact; no request can replace it with a path/module.
        return script

    @staticmethod
    def _instantiate(script: ScriptORM) -> Any:
        with _loaded_module(script.id, script.source_code) as module:
            preprocessor = module.__dict__.get("Preprocessor")
            if not isinstance(preprocessor, type):
                raise PreprocessingError("Preprocessor 必须是一个类", "INVALID_PREPROCESSOR")
            try:
                instance = preprocessor()
            except Exception as exc:
                raise PreprocessingError(f"Preprocessor 初始化失败：{exc}", "INVALID_PREPROCESSOR") from exc
            _check_method_signature(instance, "fit")
            _check_method_signature(instance, "transform")
            return instance

    @staticmethod
    def _execute(instance: Any, frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
        try:
            fitted = instance.fit(frame.copy(deep=True), config)
        except Exception as exc:
            raise PreprocessingError(f"预处理 fit 失败：{exc}", "PREPROCESS_FIT_FAILED") from exc
        if fitted is not instance:
            raise PreprocessingError("Preprocessor.fit 必须返回 self", "PREPROCESS_FIT_RETURN_INVALID")
        try:
            output = instance.transform(frame.copy(deep=True), config)
        except Exception as exc:
            raise PreprocessingError(f"预处理 transform 失败：{exc}", "PREPROCESS_TRANSFORM_FAILED") from exc
        if not isinstance(output, pd.DataFrame):
            raise PreprocessingError("Preprocessor.transform 必须返回 pandas.DataFrame", "PREPROCESS_RESULT_NOT_DATAFRAME")
        return output.copy(deep=True)

    @staticmethod
    def _validate_output(output: pd.DataFrame, input_frame: pd.DataFrame, dataset: DatasetORM) -> None:
        if len(output) != len(input_frame):
            raise PreprocessingError(
                f"预处理结果行数错误：得到 {len(output)}，预期 {len(input_frame)}",
                "PREPROCESS_ROW_COUNT_INVALID",
            )
        columns = [str(column) for column in output.columns]
        if any(not column.strip() for column in columns) or len(set(columns)) != len(columns):
            raise PreprocessingError("预处理结果字段名必须非空且不能重复", "PREPROCESS_FIELDS_INVALID")
        for required in (dataset.time_column, dataset.target_column):
            if required not in columns:
                raise PreprocessingError(
                    f"预处理结果缺少必需字段：{required}", "PREPROCESS_FIELDS_INVALID"
                )
        feature_columns = [column for column in columns if column not in {dataset.time_column, dataset.target_column}]
        if not feature_columns:
            raise PreprocessingError("预处理结果不能没有特征字段", "PREPROCESS_FEATURES_EMPTY")
        # Infinite values are never valid training input.  Ordinary NaN values
        # remain possible for a later, explicitly configured imputer.
        for column in columns:
            numeric = pd.to_numeric(output[column], errors="coerce")
            if numeric.notna().any() and not pd.Series(numeric.dropna()).map(lambda value: pd.notna(value) and value != float("inf") and value != float("-inf")).all():
                raise PreprocessingError(
                    f"预处理结果字段包含无法使用的数值：{column}", "PREPROCESS_VALUES_INVALID"
                )

    def _set_stage(self, task: PreprocessingTaskORM, stage: PreprocessingStage, message: str) -> None:
        task.stage = stage
        task.stage_started_at = _now()
        task.logs = [*task.logs, message]
        self.session.commit()

    def create_and_execute(self, request: PreprocessingTaskCreate) -> PreprocessingTaskORM:
        dataset = self.session.get(DatasetORM, request.dataset_id)
        if dataset is None:
            raise PreprocessingError("数据集不存在", "DATASET_NOT_FOUND")
        if self.session.scalar(select(ModelTypeORM.id).where(ModelTypeORM.code == request.model_type)) is None:
            raise PreprocessingError("模型类型不存在", "MODEL_TYPE_NOT_FOUND")

        script = None if request.should_skip else self._script(request.preprocess_script_id or "", request.model_type)
        task = self.repository.create(
            id=str(uuid4()),
            model_type=request.model_type,
            dataset_id=dataset.id,
            preprocess_script_id=script.id if script else None,
            preprocess_used=script is not None,
            status=PreprocessingTaskStatus.WAITING,
            stage=PreprocessingStage.WAITING,
            config=request.config,
            logs=["等待执行"],
        )
        self.session.commit()
        return self.execute(task.id)

    def execute(self, task_id: str) -> PreprocessingTaskORM:
        task = self.repository.get(task_id)
        if task is None:
            raise PreprocessingNotFoundError(f"预处理任务不存在：{task_id}")
        try:
            task.status = PreprocessingTaskStatus.RUNNING
            task.started_at = _now()
            self.session.commit()
            self._set_stage(task, PreprocessingStage.DATA_READING, "数据读取")
            dataset = self.session.get(DatasetORM, task.dataset_id)
            if dataset is None:
                raise PreprocessingError("数据集不存在", "DATASET_NOT_FOUND")
            frame = self._dataset_frame(dataset)
            task.input_row_count = len(frame)
            task.input_columns = [str(column) for column in frame.columns]
            task.input_summary = _frame_summary(frame)
            self.session.commit()

            self._set_stage(task, PreprocessingStage.PREPROCESSING, "执行预处理")
            if not task.preprocess_used:
                output = frame.copy(deep=True)
                task.logs = [*task.logs, "未使用预处理，后续使用原始特征"]
                task.status = PreprocessingTaskStatus.SKIPPED
            else:
                script = self.session.get(ScriptORM, task.preprocess_script_id)
                if script is None:
                    raise PreprocessingError("预处理脚本不存在", "SCRIPT_NOT_FOUND")
                instance = self._instantiate(script)
                output = self._execute(instance, frame, dict(task.config or {}))
                task.logs = [*task.logs, f"已执行脚本 {script.name} {script.version}"]
                self._validate_output(output, frame, dataset)
                # Keep the fitted object (fit is intentionally called exactly
                # once).  A generated module is present while pickling so the
                # class reference is importable when a later request reloads it.
                with _pickle_module(instance, script.id):
                    buffer = BytesIO()
                    joblib.dump(instance, buffer)
                artifact = self.storage.save_preprocessor_state(task.id, buffer.getvalue())
                task.preprocessor_path = artifact.relative_path
                task.preprocessor_state = {
                    "fitted": True,
                    "artifact_type": ArtifactType.PREPROCESSOR.value,
                    "relative_path": artifact.relative_path,
                    "size_bytes": artifact.size_bytes,
                    "checksum_sha256": artifact.checksum_sha256,
                }

            self._set_stage(task, PreprocessingStage.VALIDATING, "结果校验")
            self._validate_output(output, frame, dataset)
            task.output_row_count = len(output)
            task.output_columns = [str(column) for column in output.columns]
            task.output_summary = _frame_summary(output)
            task.stage = PreprocessingStage.COMPLETED
            task.stage_started_at = _now()
            task.finished_at = _now()
            task.logs = [*task.logs, "完成：可进入数据集划分"]
            task.status = task.status if task.status is PreprocessingTaskStatus.SKIPPED else PreprocessingTaskStatus.SUCCEEDED
            self.session.commit()
            return task
        except Exception as exc:
            message = str(exc)
            # Roll back first so compensation of the file also commits the
            # deletion of its FileArtifact metadata, rather than having that
            # deletion undone by the rollback.
            self.session.rollback()
            if task.preprocess_used:
                try:
                    # Remove even when save_preprocessor_state raised after
                    # writing bytes but before returning metadata.
                    self.storage.remove(ArtifactType.PREPROCESSOR, task.id)
                    self.session.commit()
                except Exception:
                    self.session.rollback()
            failed = self.repository.get(task_id)
            if failed is not None:
                failed.status = PreprocessingTaskStatus.FAILED
                failed.stage = PreprocessingStage.FAILED
                failed.stage_started_at = _now()
                failed.error_message = message
                failed.finished_at = _now()
                failed.logs = [*failed.logs, f"失败：{message}"]
                failed.preprocessor_path = None
                failed.preprocessor_state = None
                self.session.commit()
            if isinstance(exc, PreprocessingError):
                exc.task_id = task_id
                raise
            wrapped = PreprocessingError(f"预处理执行失败：{message}")
            wrapped.task_id = task_id
            raise wrapped from exc

    def get(self, task_id: str) -> PreprocessingTaskORM | None:
        return self.repository.get(task_id)

    def to_response(self, task: PreprocessingTaskORM) -> PreprocessingTaskResponse:
        return PreprocessingTaskResponse(
            id=task.id,
            model_type=task.model_type,
            dataset_id=task.dataset_id,
            preprocess_script_id=task.preprocess_script_id,
            preprocess_used=task.preprocess_used,
            preprocess_status="used" if task.preprocess_used else "unused",
            preprocess_message="已使用预处理" if task.preprocess_used else "未使用预处理",
            status=task.status,
            stage=task.stage,
            progress_stage=task.stage,
            logs=list(task.logs or []),
            error_message=task.error_message,
            config=dict(task.config or {}),
            input_row_count=task.input_row_count,
            output_row_count=task.output_row_count,
            input_columns=list(task.input_columns or []),
            output_columns=list(task.output_columns or []),
            input_summary=dict(task.input_summary or {}),
            output_summary=dict(task.output_summary or {}),
            preprocessor_path=task.preprocessor_path,
            preprocessor_state=task.preprocessor_state,
            started_at=task.started_at,
            finished_at=task.finished_at,
            stage_started_at=task.stage_started_at,
            created_at=task.created_at,
            next_step="dataset_split" if task.stage is PreprocessingStage.COMPLETED else None,
            data_source="preprocessed" if task.preprocess_used else "raw",
        )

    def transform_with_saved_state(
        self, task: PreprocessingTaskORM, frame: pd.DataFrame, config: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Transform test/prediction data without calling fit again."""

        if not task.preprocess_used:
            return frame.copy(deep=True)
        if task.status not in {PreprocessingTaskStatus.SUCCEEDED} or not task.preprocess_script_id:
            raise PreprocessingError("预处理任务没有可复用的拟合状态", "PREPROCESS_STATE_UNAVAILABLE")
        script = self.session.get(ScriptORM, task.preprocess_script_id)
        if script is None or not task.preprocessor_path:
            raise PreprocessingError("预处理器状态不存在", "PREPROCESS_STATE_UNAVAILABLE")
        try:
            raw_state = self.storage.read_preprocessor_state(task.id)
            with _loaded_module(script.id, script.source_code):
                instance = joblib.load(BytesIO(raw_state))
            _check_method_signature(instance, "transform")
            output = instance.transform(frame.copy(deep=True), dict(config if config is not None else task.config or {}))
        except PreprocessingError:
            raise
        except Exception as exc:
            raise PreprocessingError(f"复用预处理状态失败：{exc}", "PREPROCESS_STATE_LOAD_FAILED") from exc
        if not isinstance(output, pd.DataFrame):
            raise PreprocessingError("预处理 transform 必须返回 pandas.DataFrame", "PREPROCESS_RESULT_NOT_DATAFRAME")
        dataset = self.session.get(DatasetORM, task.dataset_id)
        if dataset is not None:
            self._validate_output(output, frame, dataset)
        return output.copy(deep=True)

    def transform_dataset(
        self, task_id: str, dataset_id: str, config: dict[str, Any] | None = None
    ) -> PreprocessingTransformResponse:
        task = self.get(task_id)
        if task is None:
            raise PreprocessingNotFoundError(f"预处理任务不存在：{task_id}")
        dataset = self.session.get(DatasetORM, dataset_id)
        if dataset is None:
            raise PreprocessingError("数据集不存在", "DATASET_NOT_FOUND")
        frame = self._dataset_frame(dataset)
        output = self.transform_with_saved_state(task, frame, config=config)
        return PreprocessingTransformResponse(
            task_id=task.id,
            preprocess_used=task.preprocess_used,
            data_source="preprocessed" if task.preprocess_used else "raw",
            row_count=len(output),
            columns=[str(column) for column in output.columns],
            summary=_frame_summary(output),
        )


__all__ = [
    "PreprocessingError",
    "PreprocessingNotFoundError",
    "PreprocessingService",
]
