"""Tests for system baselines and retraining-baseline selection."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import StaticPool

from backend.app.core.config import Settings
from backend.app.db.models import ModelTypeORM, ModelVersionORM
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import HealthStatus, ModelType, ModelVersionStatus
from backend.app.services.model_baseline import (
    ModelBaselineError,
    ModelBaselineService,
)


def test_application_startup_provisions_baselines_for_all_model_types(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'startup.db').as_posix()}",
        storage_root=tmp_path / "storage",
    )
    from backend.app.main import create_app

    with TestClient(create_app(settings)) as client:
        response = client.get("/api/models", params={"status": "READY"})
        assert response.status_code == 200
        baselines = [item for item in response.json() if item["is_baseline"]]
        assert {item["model_type"] for item in baselines} == {item.value for item in ModelType}
        assert {item["version"] for item in baselines} == {"v0-baseline"}


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_database(engine)
    factory = create_session_factory(engine)
    with factory() as db:
        yield db
        db.rollback()
    engine.dispose()


def test_all_model_types_get_idempotent_system_baselines(session):
    service = ModelBaselineService(session)

    first = service.initialize_baselines()
    session.commit()
    second = service.initialize_baselines()
    session.commit()

    assert {item.model_type for item in first} == set(ModelType)
    assert {item.version for item in first} == {"v0-baseline"}
    assert {item.model_type for item in second} == set(ModelType)
    assert {item.id for item in first} == {item.id for item in second}
    assert all(item.is_baseline for item in second)
    assert session.scalar(select(func.count(ModelVersionORM.id))) == 3
    assert session.scalar(select(func.count(ModelTypeORM.code))) == 3


def test_baseline_lookup_is_type_scoped_and_invalid_types_are_rejected(session):
    service = ModelBaselineService(session)
    service.initialize_baselines()
    session.commit()

    electric = service.get_baseline(ModelType.ELECTRIC_LOAD)
    heating = service.get_baseline("heating_cooling_load")
    assert electric.version == heating.version == "v0-baseline"
    assert electric.model_type is ModelType.ELECTRIC_LOAD
    assert heating.model_type is ModelType.HEATING_COOLING_LOAD
    assert electric.id != heating.id

    with pytest.raises(ModelBaselineError) as exc_info:
        service.get_baseline("not-a-model-type")
    assert exc_info.value.code == "MODEL_TYPE_INVALID"


def test_retraining_baseline_prefers_current_healthy_user_production(session):
    service = ModelBaselineService(session)
    service.initialize_baselines()
    user = ModelVersionORM(
        model_type=ModelType.ELECTRIC_LOAD,
        version="v1",
        model_path="models/v1.joblib",
        status=ModelVersionStatus.PUBLISHED,
        health_status=HealthStatus.HEALTHY,
        is_current=True,
    )
    session.add(user)
    session.flush()
    model_type = session.scalar(
        select(ModelTypeORM).where(ModelTypeORM.code == ModelType.ELECTRIC_LOAD)
    )
    model_type.current_version_id = user.id
    session.commit()

    selected = service.get_retraining_baseline(ModelType.ELECTRIC_LOAD)
    assert selected.id == user.id
    assert selected.is_baseline is False


def test_retraining_baseline_falls_back_without_user_production_and_does_not_mix_types(session):
    service = ModelBaselineService(session)
    service.initialize_baselines()
    session.commit()

    selected = service.get_retraining_baseline(ModelType.INTEGRATED_ENERGY)
    assert selected.id == service.get_baseline(ModelType.INTEGRATED_ENERGY).id

    user = ModelVersionORM(
        model_type=ModelType.ELECTRIC_LOAD,
        version="v1",
        model_path="models/v1.joblib",
        status=ModelVersionStatus.PUBLISHED,
        health_status=HealthStatus.HEALTHY,
        is_current=True,
    )
    session.add(user)
    session.flush()
    electric_type = session.scalar(
        select(ModelTypeORM).where(ModelTypeORM.code == ModelType.ELECTRIC_LOAD)
    )
    electric_type.current_version_id = user.id
    session.commit()

    assert service.get_retraining_baseline(ModelType.HEATING_COOLING_LOAD).model_type is ModelType.HEATING_COOLING_LOAD
    assert service.get_retraining_baseline(ModelType.ELECTRIC_LOAD).id == user.id


def test_current_abnormal_or_nonpublished_user_version_does_not_override_baseline(session):
    service = ModelBaselineService(session)
    service.initialize_baselines()
    user = ModelVersionORM(
        model_type=ModelType.INTEGRATED_ENERGY,
        version="v1",
        model_path="models/v1.joblib",
        status=ModelVersionStatus.READY,
        health_status=HealthStatus.HEALTHY,
        is_current=True,
    )
    session.add(user)
    session.flush()
    model_type = session.scalar(
        select(ModelTypeORM).where(ModelTypeORM.code == ModelType.INTEGRATED_ENERGY)
    )
    model_type.current_version_id = user.id
    session.commit()

    selected = service.get_retraining_baseline(ModelType.INTEGRATED_ENERGY)
    assert selected.is_baseline is True

    user.status = ModelVersionStatus.PUBLISHED
    user.health_status = HealthStatus.ABNORMAL
    session.commit()
    assert service.get_retraining_baseline(ModelType.INTEGRATED_ENERGY).is_baseline is True
