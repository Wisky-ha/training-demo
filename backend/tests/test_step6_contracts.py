"""Step 6 contract coverage for incremental logs and lifecycle commands."""

from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.core.config import Settings
from backend.app.db.models import AuditEventORM, ModelAlertORM, ModelTypeORM
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import AlertStatus, ModelType
from backend.app.main import create_app


def _app(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'step6.db').as_posix()}",
        storage_root=tmp_path / "storage",
    )
    engine = initialize_database(settings=settings)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add(ModelTypeORM(code=ModelType.ELECTRIC_LOAD, name="电力负荷"))
        session.commit()
    client = TestClient(create_app(settings))
    client.__enter__()
    return engine, factory, client


def _dataset_and_script(client: TestClient) -> tuple[str, str]:
    data = b"time,feature,target\n" + b"\n".join(
        f"2024-01-{day:02d},{day},{day * 10}".encode() for day in range(1, 6)
    )
    dataset = client.post("/api/datasets/upload", files={"file": ("load.csv", data, "text/csv")})
    assert dataset.status_code == 201, dataset.text
    dataset_id = dataset.json()["id"]
    assert client.post(f"/api/datasets/{dataset_id}/split", json={}).status_code == 201
    source = b'''class Model:\n    def predict(self, X): return [0.0] * len(X)\ndef train(X_train, y_train, X_test, y_test, config): return Model()\n'''
    script = client.post(
        "/api/scripts/upload",
        data={"name": "step6-trainer", "script_type": "trainer",
              "supported_model_types": json.dumps(["electric_load"])},
        files={"file": ("trainer.py", source, "text/x-python")},
    )
    assert script.status_code == 201, script.text
    return dataset_id, script.json()["id"]


def _legacy_model(client: TestClient, version: str) -> dict:
    response = client.post("/api/models", json={
        "model_type": "electric_load", "version": version,
        "model_path": f"legacy/{version}", "health_status": "HEALTHY",
    })
    assert response.status_code == 201, response.text
    return response.json()


def test_training_logs_are_incremental_and_cursor_stable(tmp_path):
    engine, _, client = _app(tmp_path)
    try:
        dataset_id, script_id = _dataset_and_script(client)
        created = client.post("/api/training-jobs", json={
            "model_type": "electric_load", "dataset_id": dataset_id,
            "train_script_id": script_id,
        })
        assert created.status_code == 201, created.text
        job_id = created.json()["id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if client.get(f"/api/training-jobs/{job_id}").json()["status"] in {"SUCCEEDED", "FAILED"}:
                break
            time.sleep(0.01)
        first = client.get(f"/api/training-jobs/{job_id}/logs", params={"limit": 2})
        assert first.status_code == 200
        body = first.json()
        assert len(body["items"]) == 2
        assert body["items"][0]["timestamp"] <= body["items"][1]["timestamp"]
        assert body["next_cursor"]
        second = client.get(f"/api/training-jobs/{job_id}/logs", params={
            "limit": 2, "cursor": body["next_cursor"],
        }).json()
        assert second["items"][0]["message"] != body["items"][0]["message"]
        since = body["items"][0]["timestamp"]
        filtered = client.get(f"/api/training-jobs/{job_id}/logs", params={"since": since}).json()
        assert all(item["timestamp"] > since for item in filtered["items"])
        # The no-argument compatibility call still returns the complete page.
        assert len(client.get(f"/api/training-jobs/{job_id}/logs").json()["items"]) >= 2
    finally:
        client.__exit__(None, None, None)
        engine.dispose()


def test_publish_and_rollback_contracts_persist_reason_and_idempotency(tmp_path):
    engine, factory, client = _app(tmp_path)
    try:
        first, second = _legacy_model(client, "v1"), _legacy_model(client, "v2")
        published = client.post(f"/api/models/{first['id']}/publish", json={
            "confirmed": True, "reason": "首版上线", "idempotency_key": "publish-1",
        })
        assert published.status_code == 200, published.text
        assert published.json()["record"]["reason"] == "首版上线"
        assert client.post(f"/api/models/{second['id']}/publish", json={
            "confirmed": True, "reason": "替换版本", "idempotency_key": "publish-2",
        }).status_code == 200
        rollback_body = {
            "target_version_id": first["id"], "reason": "恢复稳定版本",
            "idempotency_key": "rollback-1",
        }
        rollback = client.post(f"/api/models/{second['id']}/rollback", json=rollback_body)
        assert rollback.status_code == 200, rollback.text
        assert rollback.json()["rollback"]["idempotency_key"] == "rollback-1"
        replay = client.post(f"/api/models/{second['id']}/rollback", json=rollback_body)
        assert replay.status_code == 200
        conflict = client.post(f"/api/models/{first['id']}/rollback", json={
            "target_version_id": second["id"], "reason": "冲突", "idempotency_key": "rollback-1",
        })
        assert conflict.status_code == 409
        assert client.post(f"/api/models/{second['id']}/rollback", json={
            "reason": "缺少目标", "idempotency_key": "rollback-missing-target",
        }).status_code == 400
        with factory() as session:
            record = session.scalar(select(AuditEventORM).where(
                AuditEventORM.event_type == "MODEL_PUBLISHED",
                AuditEventORM.result == "SUCCEEDED",
            ))
            assert record is not None
            assert record.event_metadata["reason"] == "首版上线"
    finally:
        client.__exit__(None, None, None)
        engine.dispose()


def test_alert_pagination_statistics_and_acknowledge_remain_open(tmp_path):
    engine, factory, client = _app(tmp_path)
    try:
        first, second = _legacy_model(client, "v1"), _legacy_model(client, "v2")
        assert client.post(f"/api/models/{first['id']}/publish", json={"confirmed": True}).status_code == 200
        assert client.post(f"/api/models/{second['id']}/publish", json={"confirmed": True}).status_code == 200
        abnormal = client.post(f"/api/models/{second['id']}/abnormal", json={"reason": "超阈值"})
        assert abnormal.status_code == 200, abnormal.text
        page = client.get("/api/alerts", params={"limit": 1})
        assert page.status_code == 200
        payload = page.json()
        assert payload["total"] == 1
        assert payload["statistics"]["active"] == 1
        alert_id = payload["items"][0]["id"]
        acknowledged = client.post(f"/api/alerts/{alert_id}/acknowledge", json={"confirmed": True})
        assert acknowledged.status_code == 200
        assert acknowledged.json()["status"] == "ACKNOWLEDGED"
        assert acknowledged.json()["statistics"]["acknowledged"] == 1
        active = client.get("/api/alerts", params={"active_only": True, "limit": 10}).json()
        assert active["total"] == 1
        assert active["items"][0]["status"] == "ACKNOWLEDGED"
        assert client.get("/api/alerts", params={"status": "ACKNOWLEDGED", "limit": 10}).json()["total"] == 1
        with factory() as session:
            assert session.scalar(select(ModelAlertORM).where(ModelAlertORM.id == alert_id)).status in {
                AlertStatus.ACKNOWLEDGED, AlertStatus.ACTIVE,
            }
    finally:
        client.__exit__(None, None, None)
        engine.dispose()
