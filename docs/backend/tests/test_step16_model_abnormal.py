"""Step 16 acceptance tests for anomaly alerts and automatic failover."""

from __future__ import annotations

import base64
from pathlib import Path

import cloudpickle
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.core.config import Settings
from backend.app.db.models import (
    FileArtifactORM,
    ModelAlertORM,
    ModelTypeORM,
    ModelVersionORM,
    RollbackRecordORM,
)
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import AlertStatus, HealthStatus, ModelType, ModelVersionStatus, RollbackStatus
from backend.app.main import create_app


class LoadedModel:
    def predict(self, values):
        return values


def payload(model_type: str, version: str, **overrides) -> dict:
    body = {
        "model_type": model_type,
        "version": version,
        "model_content_base64": base64.b64encode(cloudpickle.dumps(LoadedModel())).decode(),
        "time_column": "time",
        "feature_columns": ["feature"],
        "target_column": "target",
        "input_schema": {
            "columns": ["time", "feature", "target"],
            "required_columns": ["time", "feature"],
            "column_types": {"time": "datetime", "feature": "number", "target": "number"},
            "time_column": "time",
            "target_column": "target",
            "extra_columns": "reject",
        },
    }
    body.update(overrides)
    return body


class Api:
    def __init__(self, client: TestClient):
        self.client = client

    def save(self, model_type: str, version: str, **overrides) -> dict:
        response = self.client.post("/api/models", json=payload(model_type, version, **overrides))
        assert response.status_code == 201, response.text
        return response.json()

    def publish(self, item: dict):
        return self.client.post(f"/api/models/{item['id']}/publish", json={"confirm": True})

    def abnormal(self, model_type: str, version: str, **body):
        return self.client.post(
            "/api/models/abnormal",
            json={"model_type": model_type, "model_version": version, "reason": "测试异常", **body},
        )


def _fixture(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'step16.db').as_posix()}",
        storage_root=tmp_path / "storage",
    )
    engine = initialize_database(settings=settings)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add(ModelTypeORM(code=ModelType.ELECTRIC_LOAD, name="电力负荷"))
        session.add(ModelTypeORM(code=ModelType.HEATING_COOLING_LOAD, name="冷热负荷"))
        session.commit()
    app = create_app(settings)
    client = TestClient(app)
    client.__enter__()
    return settings, engine, factory, Api(client), client


def test_current_anomaly_rolls_back_records_alert_and_only_publish_resolves(tmp_path):
    settings, engine, factory, api, client = _fixture(tmp_path)
    try:
        first = api.save("electric_load", "v1")
        second = api.save("electric_load", "v2")
        third = api.save("electric_load", "v3")
        for item in (first, second, third):
            assert api.publish(item).status_code == 200

        response = api.abnormal("electric_load", "v3")
        assert response.status_code == 200, response.text
        assert response.json()["version"] == "v2"
        assert response.json()["rollback"]["status"] == "SUCCEEDED"
        assert response.json()["alert"]["status"] == "ACTIVE"

        # Repeating the same command is idempotent and false never resolves it.
        repeated = api.abnormal("electric_load", "v3")
        assert repeated.status_code == 200
        assert api.abnormal("electric_load", "v3", abnormal=False).status_code == 200
        assert api.client.get("/api/alerts", params={"active_only": True}).json()
        with factory() as session:
            assert len(session.scalars(select(RollbackRecordORM)).all()) == 1
            assert session.scalar(select(ModelVersionORM).where(ModelVersionORM.version == "v3")).status is ModelVersionStatus.ABNORMAL
            assert session.scalar(select(ModelVersionORM).where(ModelVersionORM.version == "v2")).is_current

        replacement = api.save("electric_load", "v4")
        assert api.publish(replacement).status_code == 200
        assert api.client.get("/api/alerts", params={"active_only": True}).json() == []
        with factory() as session:
            alert = session.scalar(select(ModelAlertORM).where(ModelAlertORM.model_type == ModelType.ELECTRIC_LOAD))
            assert alert.status is AlertStatus.RESOLVED
    finally:
        client.__exit__(None, None, None)
        engine.dispose()


def test_historical_version_and_other_model_type_are_isolated(tmp_path):
    settings, engine, factory, api, client = _fixture(tmp_path)
    try:
        old = api.save("electric_load", "v1")
        current = api.save("electric_load", "v2")
        assert api.publish(old).status_code == 200
        assert api.publish(current).status_code == 200
        heating = api.save("heating_cooling_load", "v1")
        assert api.publish(heating).status_code == 200

        response = api.abnormal("electric_load", "v1")
        assert response.status_code == 200
        with factory() as session:
            assert session.get(ModelVersionORM, old["id"]).status is ModelVersionStatus.ABNORMAL
            assert session.get(ModelVersionORM, current["id"]).is_current
            assert session.get(ModelVersionORM, heating["id"]).is_current
            assert len(session.scalars(select(ModelAlertORM)).all()) == 1
    finally:
        client.__exit__(None, None, None)
        engine.dispose()


def test_first_user_version_falls_back_to_v0_baseline(tmp_path):
    settings, engine, factory, api, client = _fixture(tmp_path)
    try:
        item = api.save("electric_load", "v1")
        assert api.publish(item).status_code == 200
        response = api.abnormal("electric_load", "v1")
        assert response.status_code == 200, response.text
        assert response.json()["version"] == "v0-baseline"
        with factory() as session:
            baseline = session.scalar(select(ModelVersionORM).where(
                ModelVersionORM.model_type == ModelType.ELECTRIC_LOAD,
                ModelVersionORM.is_baseline.is_(True),
            ))
            assert baseline.is_current
            assert session.scalar(select(ModelTypeORM).where(ModelTypeORM.code == ModelType.ELECTRIC_LOAD)).current_version_id == baseline.id
            rollback = session.scalar(select(RollbackRecordORM))
            assert rollback.rollback_to == baseline.id
    finally:
        client.__exit__(None, None, None)
        engine.dispose()


def test_unusable_backups_are_skipped_and_missing_backup_is_structured(tmp_path):
    settings, engine, factory, api, client = _fixture(tmp_path)
    try:
        first = api.save("electric_load", "v1")
        second = api.save("electric_load", "v2")
        third = api.save("electric_load", "v3")
        for item in (first, second, third):
            assert api.publish(item).status_code == 200
        with factory() as session:
            broken_schema = session.get(ModelVersionORM, second["id"])
            broken_schema.input_schema = {}
            artifact = session.get(FileArtifactORM, broken_schema.model_artifact_id)
            session.commit()
            Path(settings.storage_root, artifact.relative_path).unlink()
        # The newest backup is schema-invalid; the older usable one is selected.
        assert api.abnormal("electric_load", "v3").json()["version"] == "v1"

        # With both user backups unusable, the system baseline is still the safe fallback.
        with factory() as session:
            session.get(ModelVersionORM, first["id"]).input_schema = {}
            session.commit()
        # Publish a fresh current version after the first failover, then break its backup.
        fourth = api.save("electric_load", "v4")
        assert api.publish(fourth).status_code == 200
        response = api.abnormal("electric_load", "v4")
        assert response.status_code == 200
        assert response.json()["version"] == "v0-baseline"

        # Removing the baseline health makes the same operation return the required code.
        with factory() as session:
            baseline = session.scalar(select(ModelVersionORM).where(ModelVersionORM.version == "v0-baseline", ModelVersionORM.model_type == ModelType.ELECTRIC_LOAD))
            baseline.health_status = HealthStatus.ABNORMAL
            session.commit()
        fifth = api.save("electric_load", "v5")
        assert api.publish(fifth).status_code == 200
        response = api.abnormal("electric_load", "v5")
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "NO_HEALTHY_BACKUP"
        assert api.client.get("/api/alerts", params={"active_only": True}).json()
    finally:
        client.__exit__(None, None, None)
        engine.dispose()


def test_type_and_version_are_validated(tmp_path):
    settings, engine, factory, api, client = _fixture(tmp_path)
    try:
        item = api.save("electric_load", "v1")
        assert api.publish(item).status_code == 200
        missing = api.abnormal("electric_load", "does-not-exist")
        assert missing.status_code == 404
        assert missing.json()["detail"]["code"] == "MODEL_VERSION_NOT_FOUND"
        wrong_type = api.client.post("/api/models/abnormal", json={
            "model_type": "heating_cooling_load", "model_version": "v1", "abnormal": True,
        })
        assert wrong_type.status_code == 404
        assert wrong_type.json()["detail"]["code"] == "MODEL_VERSION_NOT_FOUND"
    finally:
        client.__exit__(None, None, None)
        engine.dispose()
