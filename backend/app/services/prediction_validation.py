"""Validation of prediction records against an immutable model-version contract.

This module intentionally stops before model loading or prediction.  The later
MCP adapter can use the returned, ordered feature frame, while this boundary
keeps model selection and request validation independent and read-only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
import math
from numbers import Number
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import ModelTypeORM, ModelVersionORM
from ..db.repositories import ModelVersionRepository
from ..domain.enums import HealthStatus, ModelType, ModelVersionStatus
from .model_baseline import ModelBaselineService
from .model_lifecycle import ModelLifecycleError, ModelLifecycleService


class PredictionInputValidationError(ValueError):
    """A stable, structured validation failure for prediction adapters."""

    def __init__(self, message: str, code: str = "PREDICTION_INPUT_INVALID", **details: Any):
        super().__init__(message)
        self.code = code
        self.details = details


@dataclass(slots=True)
class PredictionInputValidationResult:
    """Validated input in the exact order required by the saved version."""

    model_version: ModelVersionORM
    frame: pd.DataFrame
    timestamps: list[datetime]
    time_column: str
    feature_columns: list[str]
    target_column: str
    records: list[dict[str, Any]]

    @property
    def model_type(self) -> ModelType:
        return self.model_version.model_type

    @property
    def version(self) -> str:
        return self.model_version.version

    @property
    def features(self) -> pd.DataFrame:
        """Return only the training features, excluding time and target."""

        return self.frame.loc[:, self.feature_columns]


class PredictionInputValidationService:
    """Resolve an available model version and validate prediction records.

    The service never inspects a dataset and never infers a field contract from
    the request.  All names, types, and time-format rules come from the
    immutable ``ModelVersionORM`` snapshot created by the training workflow.
    """

    _KNOWN_FIELD_TYPES = {"datetime", "number", "string", "boolean"}
    _IGNORE_EXTRA_POLICIES = {"ignore", "allow", "allowed", "drop"}
    _REJECT_EXTRA_POLICIES = {"reject", "forbid", "forbidden", "error"}

    def __init__(self, session: Session):
        self.session = session
        self.versions = ModelVersionRepository(session)

    @staticmethod
    def _model_type(value: ModelType | str) -> ModelType:
        try:
            return value if isinstance(value, ModelType) else ModelType(value)
        except (TypeError, ValueError) as exc:
            # A caller-facing lookup must not leak the enum implementation's
            # ValueError.  Unknown values use the same code as absent rows.
            raise PredictionInputValidationError(
                "模型类型不存在", "MODEL_TYPE_NOT_FOUND", model_type=value
            ) from exc

    def _model_type_exists(self, value: ModelType | str) -> ModelType:
        model_type = self._model_type(value)
        if self.session.scalar(
            select(ModelTypeORM.code).where(ModelTypeORM.code == model_type)
        ) is None:
            raise PredictionInputValidationError(
                "模型类型不存在", "MODEL_TYPE_NOT_FOUND", model_type=model_type.value
            )
        return model_type

    @staticmethod
    def _unavailable(version: ModelVersionORM, *, requested: str | None = None):
        raise PredictionInputValidationError(
            "模型版本当前不可用于预测",
            "MODEL_VERSION_UNAVAILABLE",
            model_type=version.model_type.value,
            model_version=version.version,
            model_version_id=version.id,
            status=version.status.value,
            health_status=version.health_status.value,
            requested_version=requested,
        )

    def resolve_model_version(
        self,
        model_type: ModelType | str,
        model_version: str | None = None,
    ) -> ModelVersionORM:
        """Resolve a type-scoped version label/id, or the current production one."""

        code = self._model_type_exists(model_type)
        version: ModelVersionORM | None
        if model_version is None:
            # The pointer is the lifecycle source of truth.  The repository's
            # compatibility fallback also supports databases created before
            # current_version_id was introduced.
            version = self.versions.get_current_production(code)
            if version is None:
                try:
                    version = ModelBaselineService(self.session).get_baseline(code)
                except Exception:
                    raise PredictionInputValidationError(
                        "模型类型没有可用的当前生产版本",
                        "MODEL_VERSION_UNAVAILABLE",
                        model_type=code.value,
                    )
        else:
            version = self.session.scalar(select(ModelVersionORM).where(
                ModelVersionORM.model_type == code,
                (ModelVersionORM.id == model_version)
                | (ModelVersionORM.version == model_version),
            ))
            if version is None:
                raise PredictionInputValidationError(
                    "模型版本不存在",
                    "MODEL_VERSION_NOT_FOUND",
                    model_type=code.value,
                    model_version=model_version,
                )

        if version.is_baseline:
            if version.status is not ModelVersionStatus.READY or version.health_status is not HealthStatus.HEALTHY:
                self._unavailable(version, requested=model_version)
        elif (
            version.status is not ModelVersionStatus.PUBLISHED
            or version.health_status is not HealthStatus.HEALTHY
        ):
            self._unavailable(version, requested=model_version)
        return version

    @staticmethod
    def _validate_saved_schema(version: ModelVersionORM) -> dict[str, Any]:
        """Check the persisted contract before applying it to a request."""

        try:
            # Keep this in lockstep with the publication boundary: a version
            # that was not saved with a complete, internally consistent
            # contract must not receive a best-effort inferred one here.
            ModelLifecycleService._validate_input_schema(version)
        except ModelLifecycleError as exc:
            raise PredictionInputValidationError(
                str(exc), exc.code, **exc.details
            ) from exc

        schema = version.input_schema
        columns = schema["columns"]
        required = schema["required_columns"]
        types = schema["column_types"]
        if any(
            not isinstance(types[column], str)
            or types[column] not in PredictionInputValidationService._KNOWN_FIELD_TYPES
            for column in columns
        ):
            raise PredictionInputValidationError(
                "输入字段规范包含不支持的字段类型", "MODEL_INPUT_SCHEMA_INVALID"
            )
        if schema["extra_columns"] is None or not isinstance(schema["extra_columns"], str):
            raise PredictionInputValidationError(
                "输入字段规范的额外字段策略无效", "MODEL_INPUT_SCHEMA_INVALID"
            )
        time_format = schema.get("time_format")
        if time_format is not None and not isinstance(time_format, str):
            raise PredictionInputValidationError(
                "输入字段规范的时间格式无效", "MODEL_INPUT_SCHEMA_INVALID"
            )
        policy = schema["extra_columns"].strip().lower()
        if policy not in (
            PredictionInputValidationService._IGNORE_EXTRA_POLICIES
            | PredictionInputValidationService._REJECT_EXTRA_POLICIES
        ):
            raise PredictionInputValidationError(
                "输入字段规范的额外字段策略无效", "MODEL_INPUT_SCHEMA_INVALID"
            )
        if any(column not in types for column in required):
            raise PredictionInputValidationError(
                "输入字段规范缺少必填字段类型", "MODEL_INPUT_SCHEMA_INVALID"
            )
        return schema

    @staticmethod
    def _records(data: Any) -> list[dict[str, Any]]:
        if isinstance(data, pd.DataFrame):
            if len(set(data.columns)) != len(data.columns):
                raise PredictionInputValidationError(
                    "预测输入包含重复字段", "INVALID_INPUT_DATA"
                )
            rows = data.to_dict(orient="records")
        elif isinstance(data, Mapping):
            rows = [dict(data)]
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            rows = list(data)
        else:
            raise PredictionInputValidationError(
                "预测输入必须是记录数组", "INVALID_INPUT_DATA"
            )
        if not rows:
            raise PredictionInputValidationError(
                "预测输入不能为空", "PREDICTION_INPUT_EMPTY"
            )
        if not all(isinstance(row, Mapping) for row in rows):
            raise PredictionInputValidationError(
                "预测输入的每一行必须是对象", "INVALID_INPUT_DATA"
            )
        return [dict(row) for row in rows]

    @staticmethod
    def _is_null(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (str, bytes, bytearray)):
            return False
        try:
            result = pd.isna(value)
            if isinstance(result, bool):
                return result
            item = getattr(result, "item", None)
            return bool(item()) if callable(item) else False
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _parse_time(value: Any, time_format: str | None) -> datetime:
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        if not isinstance(value, str):
            raise PredictionInputValidationError(
                "时间字段类型无效", "INVALID_FIELD_TYPE", expected="datetime"
            )
        try:
            if time_format:
                return datetime.strptime(value, time_format)
            parsed = pd.to_datetime(value, errors="raise", format="mixed")
            if isinstance(parsed, pd.Timestamp):
                return parsed.to_pydatetime()
            return pd.Timestamp(parsed).to_pydatetime()
        except (TypeError, ValueError, OverflowError) as exc:
            raise PredictionInputValidationError(
                "时间字段格式无效", "INVALID_TIME_FORMAT", value=value
            ) from exc

    @classmethod
    def _validate_value(cls, field: str, value: Any, expected: str,
                        time_format: str | None) -> Any:
        if cls._is_null(value):
            raise PredictionInputValidationError(
                "字段不能为空", "NULL_VALUE_NOT_ALLOWED", field=field
            )
        if expected == "datetime":
            return cls._parse_time(value, time_format)
        valid = (
            expected == "number" and isinstance(value, Number)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ) or (
            expected == "string" and isinstance(value, str)
        ) or (
            expected == "boolean" and isinstance(value, bool)
        )
        if not valid:
            raise PredictionInputValidationError(
                "字段值类型无效",
                "INVALID_FIELD_TYPE",
                field=field,
                expected=expected,
                actual=type(value).__name__,
            )
        return value

    def validate_input(
        self, version: ModelVersionORM, data: Any
    ) -> PredictionInputValidationResult:
        """Validate rows for an already resolved model version."""

        schema = self._validate_saved_schema(version)
        rows = self._records(data)
        required = list(schema["required_columns"])
        columns = list(schema["columns"])
        types = dict(schema["column_types"])
        time_column = version.time_column
        feature_columns = list(version.feature_columns)
        target_column = version.target_column
        time_format = schema.get("time_format")
        policy = schema["extra_columns"].strip().lower()
        allowed = set(columns)
        input_rows: list[dict[str, Any]] = []
        timestamps: list[datetime] = []

        for row_index, row in enumerate(rows):
            if time_column not in row:
                raise PredictionInputValidationError(
                    "预测输入缺少时间字段",
                    "MISSING_TIME_FIELD",
                    field=time_column,
                    row_index=row_index,
                    required_fields=required,
                )
            missing_features = [field for field in feature_columns if field not in row]
            if missing_features:
                raise PredictionInputValidationError(
                    "预测输入缺少训练特征字段",
                    "MISSING_FEATURE",
                    fields=missing_features,
                    row_index=row_index,
                    required_fields=required,
                )
            extra = sorted(set(row) - allowed)
            if extra and policy in self._REJECT_EXTRA_POLICIES:
                raise PredictionInputValidationError(
                    "预测输入包含不允许的额外字段",
                    "EXTRA_FIELDS_NOT_ALLOWED",
                    extra_fields=extra,
                    row_index=row_index,
                )

            normalized: dict[str, Any] = {}
            for field in required:
                normalized[field] = self._validate_value(
                    field, row[field], types[field], time_format
                )
            # Known optional fields (normally only the training target) are
            # checked when supplied, but are never passed to the model frame.
            for field in set(row).intersection(allowed) - set(required):
                self._validate_value(field, row[field], types[field], time_format)
            timestamps.append(normalized[time_column])
            input_rows.append({field: normalized[field] for field in required})

        frame = pd.DataFrame(input_rows, columns=required)
        return PredictionInputValidationResult(
            model_version=version,
            frame=frame,
            timestamps=timestamps,
            time_column=time_column,
            feature_columns=feature_columns,
            target_column=target_column,
            records=input_rows,
        )

    def validate_prediction_input(
        self,
        model_type: ModelType | str,
        model_version: str | Any | None = None,
        data: Any | None = None,
    ) -> PredictionInputValidationResult:
        # The version is optional for the future MCP adapter.  Accepting the
        # two-argument form (model_type, records) keeps default-version
        # selection separate from request-shape validation.
        if data is None and model_version is not None and not isinstance(model_version, str):
            data = model_version
            model_version = None
        version = self.resolve_model_version(model_type, model_version)
        return self.validate_input(version, data)

    def validate(
        self,
        model_type_or_version: ModelType | str | ModelVersionORM,
        model_version_or_data: str | Any | None = None,
        data: Any | None = None,
    ) -> PredictionInputValidationResult:
        """Convenient boundary supporting both resolved and type-scoped calls."""

        if isinstance(model_type_or_version, ModelVersionORM):
            records = model_version_or_data if data is None else data
            return self.validate_input(model_type_or_version, records)
        return self.validate_prediction_input(
            model_type_or_version, model_version_or_data, data
        )

    # Explicit aliases are useful to adapters without adding an HTTP/MCP
    # endpoint at this implementation step.
    validate_prediction_data = validate_prediction_input


# Compatibility vocabulary for integrations that use shorter service names.
PredictionValidationService = PredictionInputValidationService
PredictionInputService = PredictionInputValidationService
PredictionService = PredictionInputValidationService
PredictionValidationError = PredictionInputValidationError
PredictionError = PredictionInputValidationError

__all__ = [
    "PredictionError",
    "PredictionInputService",
    "PredictionInputValidationError",
    "PredictionInputValidationResult",
    "PredictionInputValidationService",
    "PredictionService",
    "PredictionValidationError",
    "PredictionValidationService",
]
