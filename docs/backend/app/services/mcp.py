"""Application services backing the MCP model tools.

The project does not depend on an MCP SDK.  This module is the protocol-neutral
boundary: an HTTP adapter (or a future MCP transport) supplies a validated
request and receives a JSON-safe result or a structured ``MCPServiceError``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from io import BytesIO
from typing import Any

import cloudpickle
import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..core.config import Settings, get_settings
from ..db.models import FileArtifactORM, ModelVersionORM
from ..domain.enums import ModelType
from ..services.model_lifecycle import ModelLifecycleError, ModelLifecycleService
from ..services.preprocessing import (
    PreprocessingError,
    _check_method_signature,
    _loaded_module,
)
from ..services.prediction_validation import (
    PredictionInputValidationError,
    PredictionInputValidationService,
)
from ..storage import ArtifactNotFoundError, FileStorageService


class MCPServiceError(ValueError):
    """A safe error that transports one MCP tool failure."""

    def __init__(self, message: str, code: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


class MCPModelService:
    """Resolve, prepare, and execute versioned model operations."""

    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.storage = FileStorageService(self.settings.file_storage_root, session=session)
        self.validation = PredictionInputValidationService(session)
        self.lifecycle = ModelLifecycleService(session, settings=self.settings)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Recursively convert numpy/pandas values for a strict JSON response."""

        if isinstance(value, Mapping):
            return {str(key): MCPModelService._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [MCPModelService._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return MCPModelService._json_safe(value.tolist())
        if isinstance(value, (pd.Series, pd.Index)):
            return MCPModelService._json_safe(value.tolist())
        if isinstance(value, pd.DataFrame):
            return MCPModelService._json_safe(value.to_numpy().tolist())
        if hasattr(value, "item"):
            try:
                return MCPModelService._json_safe(value.item())
            except (ValueError, TypeError):
                pass
        if isinstance(value, (pd.Timestamp, datetime, date)):
            return value.isoformat()
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value

    @staticmethod
    def _preprocess_error(exc: Exception, **details: Any) -> MCPServiceError:
        cause_code = getattr(exc, "code", None)
        if cause_code:
            details = {"cause_code": cause_code, **details}
        details.setdefault("exception_type", type(exc).__name__)
        # The underlying exception is logged by the application boundary; the
        # tool response only carries a stable, user-readable message.
        return MCPServiceError("预处理失败", "PREPROCESS_FAILED", **details)

    def _load_model(self, version: ModelVersionORM) -> Any:
        """Verify and restore the managed model artifact for one version."""

        try:
            self.lifecycle._validate_model_artifact(version)
        except ModelLifecycleError as exc:
            if exc.code == "MODEL_ARTIFACT_NOT_FOUND":
                raise MCPServiceError(str(exc), exc.code, **exc.details) from exc
            raise MCPServiceError("模型加载失败", "MODEL_LOAD_FAILED", cause_code=exc.code) from exc

        try:
            raw = self.storage.read_model(version.id)
            try:
                model = cloudpickle.load(BytesIO(raw))
            except Exception:
                model = joblib.load(BytesIO(raw))
            predict = getattr(model, "predict", None)
            if not callable(predict):
                raise TypeError("model.predict 不可调用")
            return model
        except (ArtifactNotFoundError, OSError) as exc:
            raise MCPServiceError("模型文件不存在", "MODEL_ARTIFACT_NOT_FOUND") from exc
        except Exception as exc:
            raise MCPServiceError("模型加载失败", "MODEL_LOAD_FAILED", exception_type=type(exc).__name__) from exc

    def _transform_with_version_state(
        self, version: ModelVersionORM, frame: pd.DataFrame
    ) -> pd.DataFrame:
        """Reuse the fitted preprocessor snapshot without calling ``fit``."""

        if not version.preprocess_used:
            return frame.copy(deep=True)
        try:
            self.lifecycle._validate_preprocessor(version)
            artifact = self.session.get(FileArtifactORM, version.preprocessor_artifact_id)
            source = version.preprocess_script_source
            if artifact is None or not source or not version.preprocess_script_id:
                raise PreprocessingError("模型预处理器状态不存在", "PREPROCESS_STATE_UNAVAILABLE")
            raw_state = self.storage.read_preprocessor_state(artifact.artifact_id)
            config: dict[str, Any] = {}
            state = version.preprocessor_state or {}
            if isinstance(state.get("config"), dict):
                config = dict(state["config"])
            elif version.training_job is not None and isinstance(version.training_job.config, dict):
                config = dict(version.training_job.config)
            with _loaded_module(version.preprocess_script_id, source):
                instance = joblib.load(BytesIO(raw_state))
            _check_method_signature(instance, "transform")
            output = instance.transform(frame.copy(deep=True), config)
            if not isinstance(output, pd.DataFrame):
                raise PreprocessingError(
                    "Preprocessor.transform 必须返回 pandas.DataFrame",
                    "PREPROCESS_RESULT_NOT_DATAFRAME",
                )
            if len(output) != len(frame):
                raise PreprocessingError("预处理结果行数错误", "PREPROCESS_ROW_COUNT_INVALID")
            return output.copy(deep=True)
        except MCPServiceError:
            raise
        except Exception as exc:
            raise self._preprocess_error(exc)

    def _prediction_values(self, raw: Any, expected_count: int) -> list[Any]:
        if isinstance(raw, pd.DataFrame):
            values: Any = raw.to_numpy().tolist()
        elif isinstance(raw, (pd.Series, pd.Index, np.ndarray, tuple, list)):
            values = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        else:
            values = [raw]
        if not isinstance(values, list):
            values = [values]
        if len(values) != expected_count:
            raise MCPServiceError(
                "模型返回的预测数量与输入行数不一致",
                "PREDICTION_FAILED",
                expected_count=expected_count,
                actual_count=len(values),
            )
        return self._json_safe(values)

    def predict(
        self, model_type: ModelType | str, data: Any, model_version: str | None = None
    ) -> dict[str, Any]:
        """Execute the versioned prediction contract used by the MCP tool."""

        try:
            version = self.validation.resolve_model_version(model_type, model_version)
            validated = self.validation.validate_input(version, data)
        except PredictionInputValidationError as exc:
            raise MCPServiceError(str(exc), exc.code, **exc.details) from exc

        try:
            frame = self._transform_with_version_state(version, validated.frame)
        except MCPServiceError:
            raise
        except Exception as exc:
            # Keep even an adapter/implementation failure inside the safe
            # preprocessing boundary; transports must never expose a raw 500.
            raise self._preprocess_error(exc) from exc
        missing = [column for column in validated.feature_columns if column not in frame.columns]
        if missing:
            raise MCPServiceError(
                "预处理结果缺少模型特征字段",
                "PREPROCESS_FAILED",
                missing_fields=missing,
            )
        features = frame.loc[:, validated.feature_columns]
        model = self._load_model(version)
        try:
            raw_predictions = model.predict(features)
            predictions = self._prediction_values(raw_predictions, len(features))
        except MCPServiceError:
            raise
        except Exception as exc:
            raise MCPServiceError(
                "模型预测失败", "PREDICTION_FAILED", exception_type=type(exc).__name__
            ) from exc
        return {
            "success": True,
            "model_type": version.model_type.value,
            "model_version": version.version,
            "preprocess_used": bool(version.preprocess_used),
            "predictions": predictions,
        }

    def mark_model_abnormal(
        self,
        model_type: ModelType | str,
        model_version: str,
        abnormal: bool = True,
        reason: str = "健康检查异常",
    ) -> dict[str, Any]:
        """Delegate anomaly state changes and automatic failover to step 16."""

        try:
            alert, rollback, target = self.lifecycle.mark_model_abnormal(
                model_type, model_version, abnormal=abnormal, reason=reason
            )
            requested = self.lifecycle._resolve_model_version(
                model_type if isinstance(model_type, ModelType) else ModelType(model_type),
                model_version,
            )
            current = target or self.lifecycle._current(
                self.lifecycle._model_type(requested.model_type), include_unhealthy=True
            ) or requested
            return {
                "success": True,
                "model_type": requested.model_type.value,
                "model_version": requested.version,
                "abnormal": abnormal,
                "current_model_version": current.version,
                "alert": self.lifecycle.to_alert_response(alert) if alert else None,
                "rollback": self.lifecycle.to_rollback_response(rollback) if rollback else None,
                "rollback_triggered": rollback is not None,
                "alert_cleared": False,
            }
        except ModelLifecycleError as exc:
            raise MCPServiceError(str(exc), exc.code, **exc.details) from exc
        except (TypeError, ValueError) as exc:
            raise MCPServiceError("模型类型或版本无效", "MODEL_VERSION_NOT_FOUND") from exc


# Names useful to transports and integrations that use either MCP or tool
# vocabulary.  No external MCP dependency is implied by these aliases.
MCPPredictService = MCPModelService
MCPPredictionService = MCPModelService
MCPService = MCPModelService

__all__ = [
    "MCPModelService",
    "MCPPredictService",
    "MCPPredictionService",
    "MCPService",
    "MCPServiceError",
]
