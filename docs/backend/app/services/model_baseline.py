"""System model baselines and retraining-baseline selection.

Baselines are metadata for the built-in model family defaults.  They are
separate model-version rows, never current versions, and are not changed when
users save or publish a model.
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from threading import RLock
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import joblib

from ..core.config import Settings, get_settings
from ..db.models import ModelTypeORM, ModelVersionORM
from ..db.repositories import ModelTypeRepository, ModelVersionRepository
from ..domain.enums import HealthStatus, ModelType, ModelVersionStatus
from ..storage import ArtifactType, FileStorageService


BASELINE_VERSION = "v0-baseline"

_MODEL_TYPE_NAMES: dict[ModelType, str] = {
    ModelType.ELECTRIC_LOAD: "电力负荷预测",
    ModelType.HEATING_COOLING_LOAD: "冷热负荷预测",
    ModelType.INTEGRATED_ENERGY: "综合能耗",
}


class ModelBaselineError(ValueError):
    """A safe error raised by baseline lookup and initialization."""

    def __init__(self, message: str, code: str = "MODEL_BASELINE_FAILED") -> None:
        super().__init__(message)
        self.code = code


class _ZeroBaselineModel:
    """Deterministic, artifact-backed fallback for a model family."""

    def predict(self, values):
        return [0.0] * len(values)


class ModelBaselineService:
    """Provision one immutable, predictable baseline for every model family."""

    _initialization_lock = RLock()

    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.model_types = ModelTypeRepository(session)
        self.versions = ModelVersionRepository(session)
        self.storage = FileStorageService(self.settings.file_storage_root, session=session)

    @staticmethod
    def _input_schema() -> dict[str, object]:
        # A baseline has no learned feature contract.  It accepts the common
        # timestamp and ignores caller feature columns while returning its
        # deterministic fallback value.
        return {
            "columns": ["time", "target"],
            "required_columns": ["time"],
            "column_types": {"time": "datetime", "target": "number"},
            "time_column": "time",
            "time_format": None,
            "target_column": "target",
            "extra_columns": "ignore",
        }

    def _ensure_artifact(self, baseline: ModelVersionORM) -> None:
        if baseline.model_artifact_id:
            return
        buffer = BytesIO()
        joblib.dump(_ZeroBaselineModel(), buffer)
        artifact = self.storage.save(ArtifactType.BASELINE, baseline.id, buffer.getvalue())
        baseline.model_path = artifact.relative_path
        baseline.model_artifact_id = artifact.id
        baseline.input_schema = self._input_schema()
        baseline.time_column = "time"
        baseline.feature_columns = []
        baseline.target_column = "target"

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _model_type(value: ModelType | str) -> ModelType:
        try:
            return value if isinstance(value, ModelType) else ModelType(value)
        except (TypeError, ValueError) as exc:
            raise ModelBaselineError("模型类型无效", "MODEL_TYPE_INVALID") from exc

    def initialize_baselines(self) -> list[ModelVersionORM]:
        """Create missing model families and their baselines idempotently.

        This method owns its small initialization transaction.  Existing model
        type names and version rows are left untouched, including a conflicting
        user-created ``v0-baseline`` row; such a conflict is reported instead
        of silently converting a historical user model into a system model.
        """

        with self._initialization_lock:
            try:
                for code, name in _MODEL_TYPE_NAMES.items():
                    model_type = self.model_types.get_by(code=code)
                    if model_type is None:
                        model_type = self.model_types.create(code=code, name=name)

                # Flush all model-family rows before resolving their foreign
                # keys.  Repository writes remain reusable and transaction
                # boundaries stay explicit here at the initialization edge.
                self.session.flush()
                result: list[ModelVersionORM] = []
                for code in ModelType:
                    existing = self.session.scalar(
                        select(ModelVersionORM).where(
                            ModelVersionORM.model_type == code,
                            ModelVersionORM.version == BASELINE_VERSION,
                        )
                    )
                    if existing is not None:
                        if not existing.is_baseline:
                            raise ModelBaselineError(
                                f"模型类型 {code.value} 的 v0-baseline 已被用户版本占用",
                                "BASELINE_VERSION_CONFLICT",
                            )
                        self._ensure_artifact(existing)
                        result.append(existing)
                        continue
                    baseline = self.versions.create(
                        model_type=code,
                        version=BASELINE_VERSION,
                        model_path=f"system/baselines/{code.value}/{BASELINE_VERSION}.joblib",
                        status=ModelVersionStatus.READY,
                        health_status=HealthStatus.HEALTHY,
                        is_baseline=True,
                        is_current=False,
                    )
                    self._ensure_artifact(baseline)
                    result.append(baseline)
                self.session.commit()
                return result
            except ModelBaselineError:
                self.session.rollback()
                raise
            except IntegrityError as exc:
                self.session.rollback()
                raise ModelBaselineError(
                    "系统基线初始化发生并发冲突，请重试", "BASELINE_INITIALIZATION_CONFLICT"
                ) from exc

    # A concise alias is useful for application startup and integrations.
    initialize = initialize_baselines

    def get_baseline(self, model_type: ModelType | str) -> ModelVersionORM:
        """Return the system baseline for one model family."""

        code = self._model_type(model_type)
        if self.model_types.get_by(code=code) is None:
            raise ModelBaselineError("模型类型不存在", "MODEL_TYPE_NOT_FOUND")
        baseline = self.versions.get_baseline(code)
        if baseline is None:
            raise ModelBaselineError("系统基线不存在", "BASELINE_NOT_FOUND")
        return baseline

    def get_current_production(self, model_type: ModelType | str) -> ModelVersionORM | None:
        """Return a healthy, published user version selected as production."""

        code = self._model_type(model_type)
        if self.model_types.get_by(code=code) is None:
            raise ModelBaselineError("模型类型不存在", "MODEL_TYPE_NOT_FOUND")
        return self.versions.get_current_production(code)

    def get_retraining_baseline(self, model_type: ModelType | str) -> ModelVersionORM:
        """Prefer current production and otherwise return the system baseline."""

        code = self._model_type(model_type)
        current = self.get_current_production(code)
        return current if current is not None else self.get_baseline(code)

    # Compatibility vocabulary for callers that use ``select``/``for``.
    baseline_for = get_baseline
    select_retraining_baseline = get_retraining_baseline
    retraining_baseline = get_retraining_baseline


__all__ = [
    "BASELINE_VERSION",
    "ModelBaselineError",
    "ModelBaselineService",
]
