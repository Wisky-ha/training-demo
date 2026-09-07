from backend.app.domain.enums import (
    AlertStatus,
    HealthStatus,
    ModelLifecycleStatus,
    ModelType,
    ModelVersionStatus,
)
from backend.app.domain.models import ModelAlert, ModelVersion
from backend.app.schemas.models import ModelAlertResponse, ModelSaveRequest


def test_frozen_model_type_and_lifecycle_contract_values():
    assert {item.value for item in ModelType} == {
        "electric_load",
        "heating_cooling_load",
        "integrated_energy",
    }
    assert {item.value for item in ModelLifecycleStatus} == {
        "READY",
        "PUBLISHED",
        "RETIRED",
        "FAILED",
    }
    assert {
        ModelVersionStatus.READY,
        ModelVersionStatus.PUBLISHED,
        ModelVersionStatus.RETIRED,
        ModelVersionStatus.FAILED,
    } <= set(ModelVersionStatus)


def test_missing_health_is_unknown_and_alert_acknowledgement_is_explicit():
    version = ModelVersion(
        model_type=ModelType.ELECTRIC_LOAD,
        version="v1",
        model_path="models/v1.joblib",
    )
    request = ModelSaveRequest(model_type=ModelType.ELECTRIC_LOAD, model_path="models/v1.joblib")
    assert version.health_status is HealthStatus.UNKNOWN
    assert request.health_status is None
    assert {item.value for item in HealthStatus} == {"HEALTHY", "ABNORMAL", "UNKNOWN"}
    assert {item.value for item in AlertStatus} == {
        "ACTIVE",
        "ACKNOWLEDGED",
        "RESOLVED",
    }

    alert = ModelAlert(
        model_type=ModelType.ELECTRIC_LOAD,
        reason="test",
    )
    assert alert.status is AlertStatus.ACTIVE
    acknowledged = alert.model_copy(update={"status": AlertStatus.ACKNOWLEDGED})
    response = ModelAlertResponse.model_validate(acknowledged)
    assert response.status is AlertStatus.ACKNOWLEDGED
    assert response.resolved_at is None
