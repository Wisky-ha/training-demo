"""Execution adapter for the platform's Python training-script contract.

This module deliberately has no database session, model-version repository, or
workflow orchestration dependency.  It executes one immutable script snapshot
with the five values in the ``train`` contract and returns an audit-friendly
result.  A later training-task service can decide how to persist a valid model.
"""

from __future__ import annotations

import ast
import builtins
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import BytesIO
import inspect
import logging
import joblib
import sys
import threading
import time
import traceback
import types
from typing import Any, Iterator, Mapping


_ALLOWED_IMPORTS = {
    "collections",
    "datetime",
    "itertools",
    "joblib",
    "logging",
    "math",
    "numpy",
    "pandas",
    "scikit_learn",
    "sklearn",
    "statistics",
    "typing",
    "warnings",
}
_FORBIDDEN_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
}
_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "__build_class__",
        "AttributeError",
        "Exception",
        "IndexError",
        "KeyError",
        "RuntimeError",
        "TypeError",
        "ValueError",
        "all",
        "any",
        "abs",
        "bool",
        "callable",
        "dict",
        "enumerate",
        "filter",
        "float",
        "getattr",
        "hasattr",
        "int",
        "isinstance",
        "issubclass",
        "len",
        "list",
        "map",
        "max",
        "min",
        "object",
        "property",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "setattr",
        "slice",
        "sorted",
        "staticmethod",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "zip",
    )
}

# Redirecting stdout/stderr and temporarily changing the root logger are
# process-global operations.  Serializing them avoids interleaving two
# in-process script executions into one another's audit records.
_EXECUTION_LOCK = threading.RLock()


class TrainingScriptExecutionError(ValueError):
    """Internal, stable error used to turn validation failures into results."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class _LogCollector:
    def __init__(self) -> None:
        self.entries: list[str] = []
        self._partial: dict[str, str] = {"stdout": "", "stderr": ""}

    def add(self, message: str) -> None:
        for line in str(message).splitlines():
            if line:
                self.entries.append(line)

    def write(self, stream: str, text: str) -> int:
        if not text:
            return 0
        combined = self._partial[stream] + text
        lines = combined.split("\n")
        self._partial[stream] = lines.pop()
        for line in lines:
            if line:
                self.entries.append(line.rstrip("\r"))
        return len(text)

    def flush(self, stream: str) -> None:
        partial = self._partial[stream]
        if partial:
            self.entries.append(partial.rstrip("\r"))
            self._partial[stream] = ""


class _CollectedStream:
    def __init__(self, collector: _LogCollector, stream: str) -> None:
        self.collector = collector
        self.stream = stream

    def write(self, text: str) -> int:
        return self.collector.write(self.stream, text)

    def flush(self) -> None:
        self.collector.flush(self.stream)


class _CollectedLogHandler(logging.Handler):
    def __init__(self, collector: _LogCollector) -> None:
        super().__init__(level=logging.NOTSET)
        self.collector = collector

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.collector.add(self.format(record))
        except Exception:
            # Logging must never hide the training result.
            pass


@dataclass(slots=True)
class TrainingExecutionResult(Mapping[str, Any]):
    """Uniform result returned for both successful and failed executions."""

    success: bool
    model: Any | None
    logs: list[str]
    error: str | None = None
    error_code: str | None = None
    exception_type: str | None = None
    traceback: str | None = None
    duration_ms: float = 0.0
    # Internal-only bytes for the training workflow.  They are intentionally
    # not exposed by ``to_dict`` or an HTTP response.
    model_bytes: bytes | None = None

    @property
    def status(self) -> str:
        return "SUCCEEDED" if self.success else "FAILED"

    @property
    def outcome(self) -> str:
        return "succeeded" if self.success else "failed"

    @property
    def result(self) -> Any | None:
        """Alias used by adapters that call the returned model the result."""

        return self.model

    @property
    def error_message(self) -> str | None:
        return self.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "model": self.model,
            "result": self.model,
            "logs": list(self.logs),
            "error": self.error,
            "error_message": self.error,
            "error_code": self.error_code,
            "exception_type": self.exception_type,
            "traceback": self.traceback,
            "duration_ms": self.duration_ms,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())



def _module_name(script_id: str | None) -> str:
    value = script_id or "anonymous"
    return "platform_trainer_" + value.encode("utf-8").hex()


def _validate_source(source: str) -> ast.Module:
    if not isinstance(source, str) or not source.strip():
        raise TrainingScriptExecutionError("训练脚本源码不能为空", "INVALID_SCRIPT")
    try:
        tree = ast.parse(source, filename="train.py")
    except (SyntaxError, ValueError, TypeError) as exc:
        raise TrainingScriptExecutionError("训练脚本语法无效", "INVALID_SCRIPT") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.level:
                raise TrainingScriptExecutionError(
                    "训练脚本不允许相对导入", "UNSAFE_SCRIPT"
                )
            for imported in node.names:
                top_level = imported.name.split(".", 1)[0]
                if top_level not in _ALLOWED_IMPORTS:
                    raise TrainingScriptExecutionError(
                        f"训练脚本导入了不允许的模块：{top_level}", "UNSAFE_SCRIPT"
                    )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALLS:
                raise TrainingScriptExecutionError(
                    f"训练脚本调用了不允许的函数：{node.func.id}", "UNSAFE_SCRIPT"
                )
    return tree


@contextmanager
def _loaded_module(source: str, script_id: str | None) -> Iterator[types.ModuleType]:
    tree = _validate_source(source)
    name = _module_name(script_id)
    module = types.ModuleType(name)
    module.__file__ = f"<trainer:{script_id or 'anonymous'}>"

    def safe_import(module_name: str, globals=None, locals=None, fromlist=(), level=0):
        if level or module_name.split(".", 1)[0] not in _ALLOWED_IMPORTS:
            raise ImportError(f"module is not allowed: {module_name}")
        return builtins.__import__(module_name, globals, locals, fromlist, level)

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
        try:
            exec(compile(tree, module.__file__, "exec"), module.__dict__, module.__dict__)
        except Exception as exc:
            raise TrainingScriptExecutionError(
                f"训练脚本加载失败：{exc}", "SCRIPT_IMPORT_FAILED"
            ) from exc
        yield module
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class TrainingScriptExecutor:
    """Load and execute one trainer script without touching platform state."""

    def execute(
        self,
        script: Any | None = None,
        X_train: Any = None,
        y_train: Any = None,
        X_test: Any = None,
        y_test: Any = None,
        config: dict[str, Any] | None = None,
        *,
        source_code: str | None = None,
        persist_model: bool = False,
        script_id: str | None = None,
    ) -> TrainingExecutionResult:
        """Execute ``train(X_train, y_train, X_test, y_test, config)``.

        ``script`` may be source text or a script-library ORM-like object with
        ``source_code`` and optional ``id``, ``script_type``, and ``status``
        attributes.  ``source_code`` is a keyword alias for direct callers.
        Script failures are represented in the result rather than escaping,
        which gives all callers the same log and exception contract.
        """

        started = time.perf_counter()
        runtime_config = {} if config is None else config
        collector = _LogCollector()
        collector.add("训练脚本开始执行")
        model: Any | None = None
        model_bytes: bytes | None = None
        error: str | None = None
        error_code: str | None = None
        exception_type: str | None = None
        exception_trace: str | None = None

        with _EXECUTION_LOCK:
            stdout = _CollectedStream(collector, "stdout")
            stderr = _CollectedStream(collector, "stderr")
            handler = _CollectedLogHandler(collector)
            root = logging.getLogger()
            old_level = root.level
            root.addHandler(handler)
            # A script logger may use INFO while the application root defaults
            # to WARNING.  Lower the root threshold only for this execution.
            root.setLevel(min(old_level, logging.DEBUG))
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    try:
                        source, selected_id = self._script_source(
                            script, source_code, script_id
                        )
                        with _loaded_module(source, selected_id) as module:
                            train = module.__dict__.get("train")
                            if not callable(train):
                                raise TrainingScriptExecutionError(
                                    "训练脚本必须定义可调用的 train 函数",
                                    "TRAIN_FUNCTION_INVALID",
                                )
                            try:
                                inspect.signature(train).bind(
                                    X_train, y_train, X_test, y_test, runtime_config
                                )
                            except (TypeError, ValueError) as exc:
                                raise TrainingScriptExecutionError(
                                    "train 函数签名必须为 (X_train, y_train, X_test, y_test, config)",
                                    "TRAIN_SIGNATURE_INVALID",
                                ) from exc
                            collector.add("执行 train")
                            model = train(
                                X_train, y_train, X_test, y_test, runtime_config
                            )
                            try:
                                predict = getattr(model, "predict", None)
                            except Exception as exc:
                                raise TrainingScriptExecutionError(
                                    f"训练脚本返回对象的 predict 属性读取失败：{exc}",
                                    "MODEL_PREDICT_INVALID",
                                ) from exc
                            if not callable(predict):
                                raise TrainingScriptExecutionError(
                                    "训练脚本返回对象的 predict must be callable (predict(X))",
                                    "MODEL_PREDICT_INVALID",
                                )
                            collector.add("训练脚本执行完成")
                            if persist_model:
                                try:
                                    buffer = BytesIO()
                                    # The generated module remains registered
                                    # until this context exits, which makes
                                    # source-defined model classes pickleable.
                                    joblib.dump(model, buffer)
                                    model_bytes = buffer.getvalue()
                                except Exception as exc:
                                    try:
                                        import cloudpickle
                                        model_bytes = cloudpickle.dumps(model)
                                    except Exception:
                                        raise TrainingScriptExecutionError(
                                            f"模型制品序列化失败：{exc}",
                                            "MODEL_SERIALIZATION_FAILED",
                                        ) from exc
                    except Exception as exc:
                        error = self._error_message(exc)
                        error_code = getattr(exc, "code", "TRAIN_EXECUTION_FAILED")
                        exception_type = type(exc).__name__
                        exception_trace = traceback.format_exc()
                        collector.add(f"训练脚本执行失败：{error}")
                        model = None
            finally:
                stdout.flush()
                stderr.flush()
                root.removeHandler(handler)
                root.setLevel(old_level)
                handler.close()

        return TrainingExecutionResult(
            success=error is None,
            model=model,
            logs=collector.entries,
            error=error,
            error_code=error_code,
            exception_type=exception_type,
            traceback=exception_trace,
            duration_ms=(time.perf_counter() - started) * 1000,
            model_bytes=model_bytes,
        )

    # Names kept as small adapters for callers that use "run" or explicitly
    # describe the operation as script execution.
    run = execute
    execute_script = execute

    @staticmethod
    def _script_source(
        script: Any | None, source_code: str | None, script_id: str | None
    ) -> tuple[str, str | None]:
        if script is not None and source_code is not None:
            raise TrainingScriptExecutionError(
                "script 与 source_code 只能提供一个", "INVALID_SCRIPT"
            )
        selected = script if script is not None else source_code
        if selected is None:
            raise TrainingScriptExecutionError("训练脚本源码不能为空", "INVALID_SCRIPT")
        if isinstance(selected, str):
            return selected, script_id

        selected_type = getattr(selected, "script_type", None)
        selected_type = getattr(selected_type, "value", selected_type)
        if selected_type is not None and selected_type not in {"trainer", "TRAINER"}:
            raise TrainingScriptExecutionError(
                "所选脚本不是训练脚本", "SCRIPT_TYPE_INVALID"
            )
        selected_status = getattr(selected, "status", None)
        selected_status = getattr(selected_status, "value", selected_status)
        if selected_status is not None and selected_status not in {"enabled", "ENABLED"}:
            raise TrainingScriptExecutionError(
                "所选训练脚本已停用", "SCRIPT_DISABLED"
            )
        selected_source = getattr(selected, "source_code", None)
        if not isinstance(selected_source, str):
            raise TrainingScriptExecutionError(
                "训练脚本对象缺少 source_code", "INVALID_SCRIPT"
            )
        return selected_source, script_id or getattr(selected, "id", None)

    @staticmethod
    def _error_message(exc: Exception) -> str:
        message = str(exc)
        return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


__all__ = [
    "TrainingExecutionResult",
    "TrainingScriptExecutionError",
    "TrainingScriptExecutor",
]
