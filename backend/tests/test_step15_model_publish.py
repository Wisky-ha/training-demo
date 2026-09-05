"""Step 15 acceptance tests for validated, atomic model publication."""

from __future__ import annotations

import base64
from pathlib import Path

import cloudpickle
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.core.config import Settings
from backend.app.db.models import FileArtifactORM, ModelAlertORM, ModelTypeORM, ModelVersionORM, PublishRecordORM
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import AlertStatus, ModelType, ModelVersionStatus
from backend.app.main import create_app


class LoadedModel:
    def predict(self, values):
        return values


@pytest.fixture
def publish_api(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'step15.db').as_posix()}",
        storage_root=tmp_path / "storage",
    )
    engine = initialize_database(settings=settings)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add(ModelTypeORM(code=ModelType.ELECTRIC_LOAD, name="电力负荷"))
        session.commit()
    with TestClient(create_app(settings)) as client:
        yield client, factory, settings
    engine.dispose()


def model_payload(version: str = "v1", *, content: bytes | None = None, **overrides):
    content = content or cloudpickle.dumps(LoadedModel())
    payload = {
        "model_type": "electric_load",
        "version": version,
        "model_content_base64": base64.b64encode(content).decode(),
        "time_column": "time",
        "feature_columns": ["feature"],
        "target_column": "target",
        "input_schema": {
            "columns": ["time", "feature", "target"],
            "required_columns": ["time", "feature"],
            "column_types": {"time": "datetime", "feature": "number", "target": "number"},
            "time_column": "time",
            "time_format": None,
            "target_column": "target",
            "extra_columns": "reject",
        },
    }
    payload.update(overrides)
    return payload


def save(client: TestClient, **kwargs) -> dict:
    response = client.post("/api/models", json=model_payload(**kwargs))
    assert response.status_code == 201, response.text
    return response.json()


def publish(client: TestClient, model_id: str):
    return client.post(f"/api/models/{model_id}/publish", json={"confirm": True})


def test_ready_publish_updates_pointer_history_rollback_target_and_alert(publish_api):
    client, factory, _ = publish_api
    first = save(client, version="v1")
    assert publish(client, first["id"]).status_code == 200
    second = save(client, version="v2")

    # Seed a type-scoped alert without changing the production pointer.
    with factory() as session:
        session.add(ModelAlertORM(
            model_type=ModelType.ELECTRIC_LOAD, model_version_id=first["id"],
            reason="test", rollback_from=first["id"], status=AlertStatus.ACTIVE,
        ))
        session.scalar(select(ModelTypeORM).where(ModelTypeORM.code == ModelType.ELECTRIC_LOAD)).alert_status = AlertStatus.ACTIVE
        session.commit()
    response = publish(client, second["id"])
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "PUBLISHED"

    with factory() as session:
        old = session.get(ModelVersionORM, first["id"])
        new = session.get(ModelVersionORM, second["id"])
        assert old.status is ModelVersionStatus.RETIRED and not old.is_current
        assert new.status is ModelVersionStatus.PUBLISHED and new.is_current
        assert session.scalar(select(ModelTypeORM).where(ModelTypeORM.code == ModelType.ELECTRIC_LOAD)).current_version_id == second["id"]
        record = session.scalar(select(PublishRecordORM).where(PublishRecordORM.model_version_id == second["id"]))
        assert record.previous_current_version_id == first["id"]
        assert new.previous_healthy_version_id == first["id"]
        assert session.scalar(select(ModelAlertORM).where(ModelAlertORM.model_type == ModelType.ELECTRIC_LOAD)).status is AlertStatus.RESOLVED


def test_publish_rejects_missing_or_unloadable_artifact_without_touching_current(publish_api):
    client, factory, settings = publish_api
    first = save(client, version="v1")
    assert publish(client, first["id"]).status_code == 200
    candidate = save(client, version="v2")
    with factory() as session:
        artifact = session.get(FileArtifactORM, candidate["model_artifact_id"])
        assert artifact is not None
        (Path(settings.storage_root) / artifact.relative_path).unlink()
    response = publish(client, candidate["id"])
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "MODEL_ARTIFACT_NOT_FOUND"
    assert client.get(f"/api/models/{first['id']}").json()["is_current"] is True

    invalid = save(client, version="v3", content=b"not a serialized model")
    response = publish(client, invalid["id"])
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "MODEL_ARTIFACT_INVALID"
    assert client.get(f"/api/models/{first['id']}").json()["is_current"] is True


def test_publish_rejects_bad_state_preprocessor_association_and_schema(publish_api):
    client, factory, _ = publish_api
    draft = save(client, version="v1", status="DRAFT")
    response = publish(client, draft["id"])
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "PUBLISH_STATE_INVALID"

    bad_preprocessor = save(
        client, version="v2", preprocess_used=True,
        preprocessor_path="preprocessor/state.state",
        preprocessor_state={"fitted": True},
    )
    response = publish(client, bad_preprocessor["id"])
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "PREPROCESSOR_STATE_INVALID"

    bad_schema = save(client, version="v3", input_schema={})
    response = publish(client, bad_schema["id"])
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "MODEL_INPUT_SCHEMA_INVALID"


def test_repeat_publish_is_idempotent_and_types_are_isolated(publish_api):
    client, factory, _ = publish_api
    first = save(client, version="v1")
    assert publish(client, first["id"]).status_code == 200
    repeat = publish(client, first["id"])
    assert repeat.status_code == 200
    with factory() as session:
        assert len(session.scalars(select(PublishRecordORM)).all()) == 1
        assert len(session.scalars(select(ModelVersionORM).where(ModelVersionORM.is_current.is_(True))).all()) == 1
