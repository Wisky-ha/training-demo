"""Step 17 acceptance tests for version-scoped prediction input validation."""

from __future__ import annotations

from datetime import datetime

import pytest

from backend.app.core.config import Settings
from backend.app.db.models import ModelTypeORM, ModelVersionORM
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import HealthStatus, ModelType, ModelVersionStatus
from backend.app.services.prediction_validation import (
    PredictionInputValidationError,
    PredictionInputValidationService,
)


@pytest.fixture
def validation_service(tmp_path):
    settings = Settings(database_url=f"sqlite:///{(tmp_path / 'step17.db').as_posix()}")
    engine = initialize_database(settings=settings)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add(ModelTypeORM(code=ModelType.ELECTRIC_LOAD, name="电力负荷"))
        session.commit()
    with factory() as session:
        yield PredictionInputValidationService(session), session
    engine.dispose()


def schema(*, extra_columns: str = "reject", time_format: str | None = None):
    return {
        "columns": ["time", "feature", "target"],
        "required_columns": ["time", "feature"],
        "column_types": {
            "time": "datetime", "feature": "number", "target": "number",
        },
        "time_column": "time",
        "time_format": time_format,
        "target_column": "target",
        "extra_columns": extra_columns,
    }


def add_version(session, *, version="v1", status=ModelVersionStatus.PUBLISHED,
                health=HealthStatus.HEALTHY, input_schema=None, **metadata):
    item = ModelVersionORM(
        model_type=ModelType.ELECTRIC_LOAD,
        version=version,
        model_path=f"legacy/{version}.joblib",
        input_schema=input_schema or schema(),
        time_column=metadata.pop("time_column", "time"),
        feature_columns=metadata.pop("feature_columns", ["feature"]),
        target_column=metadata.pop("target_column", "target"),
        status=status,
        health_status=health,
        **metadata,
    )
    session.add(item)
    session.commit()
    return item


def assert_code(call, code):
    with pytest.raises(PredictionInputValidationError) as caught:
        call()
    assert caught.value.code == code


def test_valid_input_is_name_based_and_returns_saved_feature_order(validation_service):
    service, session = validation_service
    add_version(session)

    result = service.validate_prediction_input("electric_load", "v1", [
        {"feature": 2.5, "time": "2024-01-01"},
        {"time": "2024-01-02", "feature": 3},
    ])

    assert result.time_column == "time"
    assert result.feature_columns == ["feature"]
    assert result.frame.columns.tolist() == ["time", "feature"]
    assert result.frame["feature"].tolist() == [2.5, 3]
    assert result.timestamps[0].year == 2024


def test_missing_fields_and_extra_field_policy(validation_service):
    service, session = validation_service
    add_version(session, version="reject")
    add_version(session, version="ignore", input_schema=schema(extra_columns="ignore"))

    assert_code(lambda: service.validate_prediction_input(
        "electric_load", "reject", [{"feature": 1}]), "MISSING_TIME_FIELD")
    assert_code(lambda: service.validate_prediction_input(
        "electric_load", "reject", [{"time": "2024-01-01"}]), "MISSING_FEATURE")
    assert_code(lambda: service.validate_prediction_input(
        "electric_load", "reject", [{"time": "2024-01-01", "feature": 1, "other": 2}]),
        "EXTRA_FIELDS_NOT_ALLOWED")

    result = service.validate_prediction_input(
        "electric_load", "ignore", [{"other": "not inspected", "time": "2024-01-01", "feature": 1}])
    assert result.frame.columns.tolist() == ["time", "feature"]


def test_type_time_empty_and_null_validation(validation_service):
    service, session = validation_service
    add_version(session, input_schema=schema(time_format="%Y-%m-%d"))

    assert_code(lambda: service.validate_prediction_input(
        "electric_load", "v1", [{"time": "2024-01-01", "feature": "1"}]),
        "INVALID_FIELD_TYPE")
    assert_code(lambda: service.validate_prediction_input(
        "electric_load", "v1", [{"time": "01/01/2024", "feature": 1}]),
        "INVALID_TIME_FORMAT")
    assert_code(lambda: service.validate_prediction_input(
        "electric_load", "v1", []), "PREDICTION_INPUT_EMPTY")
    assert_code(lambda: service.validate_prediction_input(
        "electric_load", "v1", [{"time": "2024-01-01", "feature": None}]),
        "NULL_VALUE_NOT_ALLOWED")
    assert service.validate_prediction_input(
        "electric_load", "v1", [{"time": datetime(2024, 1, 1), "feature": 1}])


def test_model_type_version_and_lifecycle_availability(validation_service):
    service, session = validation_service
    add_version(session, version="healthy")
    for version, status, health in (
        ("abnormal", ModelVersionStatus.ABNORMAL, HealthStatus.ABNORMAL),
        ("training", ModelVersionStatus.TRAINING, HealthStatus.HEALTHY),
        ("ready", ModelVersionStatus.READY, HealthStatus.HEALTHY),
        ("retired", ModelVersionStatus.RETIRED, HealthStatus.HEALTHY),
    ):
        add_version(session, version=version, status=status, health=health)

    assert_code(lambda: service.validate_prediction_input(
        "missing_type", "v1", [{"time": "2024-01-01", "feature": 1}]),
        "MODEL_TYPE_NOT_FOUND")
    assert_code(lambda: service.validate_prediction_input(
        "electric_load", "missing", [{"time": "2024-01-01", "feature": 1}]),
        "MODEL_VERSION_NOT_FOUND")
    for version in ("abnormal", "training", "ready", "retired"):
        assert_code(lambda version=version: service.validate_prediction_input(
            "electric_load", version, [{"time": "2024-01-01", "feature": 1}]),
            "MODEL_VERSION_UNAVAILABLE")


def test_saved_schema_is_authoritative_and_never_inferred_from_input(validation_service):
    service, session = validation_service
    item = add_version(session, input_schema=schema(), feature_columns=["feature"])
    # A new field in the request cannot become a feature merely because it has
    # a numeric value; the persisted version contract remains authoritative.
    assert_code(lambda: service.validate_prediction_input(
        "electric_load", item.id,
        [{"time": "2024-01-01", "feature": 1, "new_feature": 99}]),
        "EXTRA_FIELDS_NOT_ALLOWED")
    # Changing the persisted metadata to disagree with its saved schema is
    # rejected rather than silently re-inferring a new contract.
    item.feature_columns = ["new_feature"]
    session.commit()
    assert_code(lambda: service.validate_prediction_input(
        "electric_load", item.id,
        [{"time": "2024-01-01", "new_feature": 99}]),
        "MODEL_INPUT_SCHEMA_INVALID")
