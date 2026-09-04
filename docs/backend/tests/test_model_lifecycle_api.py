"""Contract tests for model lifecycle state and failover."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import base64

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, func

from backend.app.core.config import Settings
from backend.app.db.models import ModelAlertORM, ModelTypeORM, ModelVersionORM, PublishRecordORM, RollbackRecordORM
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import AlertStatus, ModelType, ModelVersionStatus, RollbackStatus
from backend.app.main import create_app


@pytest.fixture
def lifecycle_api(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'lifecycle.db').as_posix()}",
        storage_root=tmp_path / "storage",
    )
    engine = initialize_database(settings=settings)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add(ModelTypeORM(code=ModelType.ELECTRIC_LOAD, name="电力负荷"))
        session.commit()
    with TestClient(create_app(settings)) as client:
        yield client, factory
    engine.dispose()


def save(client, version: str, *, content: bytes | None = None) -> dict:
    body = {"model_type": "electric_load", "version": version, "model_path": f"legacy/{version}"}
    if content is not None:
        body["model_content_base64"] = base64.b64encode(content).decode()
    response = client.post("/api/models", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def publish(client, model_id: str, **body):
    return client.post(f"/api/models/{model_id}/publish", json={"confirm": True, **body})


def test_publish_requires_confirmation_and_keeps_audit_backup(lifecycle_api):
    client, factory = lifecycle_api
    first = save(client, "v1", content=b"first")
    assert client.post(f"/api/models/{first['id']}/publish", json={}).status_code == 400
    assert client.get(f"/api/models/{first['id']}").json()["status"] == "READY"
    assert publish(client, first["id"]).status_code == 200

    second = save(client, "v2")
    response = publish(client, second["id"], idempotency_key="release-v2")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "PUBLISHED"
    assert response.json()["is_current"] is True
    with factory() as session:
        old = session.get(ModelVersionORM, first["id"])
        new = session.get(ModelVersionORM, second["id"])
        assert old.status is ModelVersionStatus.RETIRED and not old.is_current
        assert new.status is ModelVersionStatus.PUBLISHED and new.is_current
        assert session.scalar(select(func.count(PublishRecordORM.id))) == 2
        assert session.scalar(select(PublishRecordORM).where(PublishRecordORM.model_version_id == second["id"]).order_by(PublishRecordORM.published_at.desc())).previous_current_version_id == first["id"]
    # A retry of an already committed release is idempotent.
    assert publish(client, second["id"], idempotency_key="release-v2").status_code == 200
    with factory() as session:
        assert session.scalar(select(func.count(PublishRecordORM.id))) == 2


def test_offline_and_manual_rollback_validate_target_and_keep_artifacts(lifecycle_api):
    client, factory = lifecycle_api
    first, second, third = save(client, "v1"), save(client, "v2"), save(client, "v3")
    assert publish(client, first["id"]).status_code == 200
    assert publish(client, second["id"]).status_code == 200
    offline = client.post(f"/api/models/{second['id']}/offline")
    assert offline.status_code == 200
    with factory() as session:
        assert session.get(ModelVersionORM, second["id"]).status is ModelVersionStatus.RETIRED
    assert client.post(f"/api/models/{second['id']}/offline").status_code == 409

    # Publish a new candidate, then ask the current path to roll back to the retained v1.
    assert publish(client, third["id"]).status_code == 200
    rollback = client.post(
        f"/api/models/{third['id']}/rollback",
        json={"target_version_id": first["id"], "reason": "回到稳定版本"},
    )
    assert rollback.status_code == 200, rollback.text
    assert rollback.json()["version"] == "v1"
    with factory() as session:
        record = session.scalar(select(RollbackRecordORM).order_by(RollbackRecordORM.created_at.desc()))
        assert record.status is RollbackStatus.SUCCEEDED
        assert record.rollback_from == third["id"] and record.rollback_to == first["id"]
        # The model file is never removed during lifecycle changes.
        assert session.get(ModelVersionORM, first["id"]).model_path

    bad = client.post(f"/api/models/{first['id']}/rollback", json={"target_version_id": first["id"]})
    assert bad.status_code == 409


def test_abnormal_automatically_uses_nearest_backup_and_alert_lasts_until_publish(lifecycle_api):
    client, factory = lifecycle_api
    first, second, third = save(client, "v1"), save(client, "v2"), save(client, "v3")
    for item in (first, second, third):
        assert publish(client, item["id"]).status_code == 200
    abnormal = client.post(f"/api/models/{third['id']}/abnormal", json={"reason": "线上误差超阈值"})
    assert abnormal.status_code == 200, abnormal.text
    assert abnormal.json()["version"] == "v2"
    alerts = client.get("/api/alerts", params={"active_only": "true"})
    assert alerts.status_code == 200 and len(alerts.json()) == 1
    alert_id = alerts.json()[0]["id"]
    acknowledged = client.post(f"/api/alerts/{alert_id}/acknowledge")
    assert acknowledged.status_code == 200
    assert acknowledged.json()["status"] == "ACTIVE"
    with factory() as session:
        broken = session.get(ModelVersionORM, third["id"])
        current = session.get(ModelVersionORM, second["id"])
        assert broken.status is ModelVersionStatus.ABNORMAL and broken.health_status.value == "ABNORMAL"
        assert current.is_current and current.status is ModelVersionStatus.PUBLISHED
        assert session.scalar(select(ModelAlertORM).where(ModelAlertORM.id == alert_id)).status is AlertStatus.ACTIVE
    # Only a successful publication clears the persistent alert.
    response = publish(client, first["id"])
    assert response.status_code == 409  # v1 is a retired backup, not a new draft
    replacement = save(client, "v4")
    assert publish(client, replacement["id"]).status_code == 200
    assert client.get("/api/alerts", params={"active_only": "true"}).json() == []
    assert client.get("/api/alerts").json()[0]["status"] == "RESOLVED"


def test_non_current_abnormal_only_marks_version_and_keeps_current(lifecycle_api):
    client, factory = lifecycle_api
    first, second = save(client, "v1"), save(client, "v2")
    assert publish(client, first["id"]).status_code == 200
    # A non-current published backup is marked abnormal without failover.
    assert publish(client, second["id"]).status_code == 200
    response = client.post(f"/api/models/{first['id']}/abnormal", json={"reason": "历史版本异常"})
    assert response.status_code == 200, response.text
    with factory() as session:
        assert session.get(ModelVersionORM, first["id"]).status is ModelVersionStatus.ABNORMAL
        assert session.get(ModelVersionORM, second["id"]).is_current


def test_abnormal_without_backup_persists_failed_rollback_and_alert(lifecycle_api):
    client, factory = lifecycle_api
    item = save(client, "v1")
    assert publish(client, item["id"]).status_code == 200
    response = client.post(f"/api/models/{item['id']}/abnormal", json={"reason": "制品损坏"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "NO_HEALTHY_ROLLBACK_VERSION"
    with factory() as session:
        rollback = session.scalar(select(RollbackRecordORM))
        assert rollback.status is RollbackStatus.FAILED and rollback.rollback_to is None
        assert session.scalar(select(ModelAlertORM)).status is AlertStatus.ACTIVE
        assert session.scalar(select(ModelTypeORM)).alert_status is AlertStatus.ACTIVE


def test_invalid_states_and_concurrent_publication_leave_one_current(lifecycle_api):
    client, factory = lifecycle_api
    items = [save(client, f"v{i}") for i in (1, 2)]
    assert client.post(f"/api/models/{items[0]['id']}/rollback", json={"target_version_id": items[1]["id"]}).status_code == 409

    def release(item):
        return publish(client, item["id"]).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(release, items))
    assert statuses == [200, 200]
    with factory() as session:
        current = list(session.scalars(select(ModelVersionORM).where(
            ModelVersionORM.model_type == ModelType.ELECTRIC_LOAD,
            ModelVersionORM.is_current.is_(True),
        )))
        assert len(current) == 1
        assert current[0].status is ModelVersionStatus.PUBLISHED
