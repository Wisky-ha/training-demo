"""Acceptance-focused API flows using isolated SQLite databases and local scripts."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.core.config import Settings
from backend.app.db.models import ModelAlertORM, ModelTypeORM, ModelVersionORM, RollbackRecordORM
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import AlertStatus, HealthStatus, ModelType, ModelVersionStatus, RollbackStatus
from backend.app.main import create_app


@pytest.fixture
def acceptance_api(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'acceptance.db').as_posix()}",
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


def upload(client, content: bytes, filename: str = "load.csv"):
    return client.post("/api/datasets/upload", files={"file": (filename, content, "text/csv")})


def script(client, source: bytes, name: str, script_type: str = "trainer") -> str:
    response = client.post(
        "/api/scripts/upload",
        data={
            "name": name,
            "script_type": script_type,
            "supported_model_types": json.dumps(["electric_load"]),
        },
        files={"file": (f"{name}.py", source, "text/x-python")},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def dataset_with_split(client) -> str:
    response = upload(
        client,
        b"time,feature,target\n"
        b"2024-01-01,1,10\n2024-01-02,2,20\n2024-01-03,3,30\n"
        b"2024-01-04,4,40\n2024-01-05,5,50\n",
    )
    assert response.status_code == 201, response.text
    dataset_id = response.json()["id"]
    split = client.post(f"/api/datasets/{dataset_id}/split", json={})
    assert split.status_code == 201, split.text
    return dataset_id


def wait_for(client, job_id: str, status: str, timeout: float = 5):
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        latest = client.get(f"/api/training-jobs/{job_id}").json()
        if latest["status"] == status:
            return latest
        time.sleep(0.02)
    pytest.fail(f"training job did not reach {status}: {latest}")


def save(client, version: str) -> dict:
    response = client.post(
        "/api/models",
        json={"model_type": "electric_load", "version": version, "model_path": f"models/{version}.joblib"},
    )
    assert response.status_code == 201, response.text
    return response.json()


def publish(client, model_id: str, **body):
    return client.post(f"/api/models/{model_id}/publish", json={"confirmed": True, **body})


def test_csv_upload_accepts_valid_csv_and_returns_structured_validation_errors(acceptance_api):
    client, _ = acceptance_api
    valid = upload(client, b"time,feature,target\n2024-01-01,1,10\n2024-01-02,2,20\n")
    assert valid.status_code == 201
    assert valid.json()["status"] == "parsed"
    assert valid.json()["feature_columns"] == ["feature"]

    invalid = upload(client, b"time,feature,target\n2024-01-01,1,10,extra\n2024-01-02,2,20\n")
    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "DATASET_VALIDATION_FAILED"
    assert invalid.json()["errors"]


def test_skip_preprocessing_does_not_execute_selected_script(acceptance_api):
    client, _ = acceptance_api
    dataset_id = upload(client, b"time,feature,target\n2024-01-01,1,10\n2024-01-02,2,20\n").json()["id"]
    raising = script(
        client,
        b"class Preprocessor:\n"
        b"    def fit(self, df, config): raise RuntimeError('must not execute')\n"
        b"    def transform(self, df, config): raise RuntimeError('must not execute')\n",
        "never-execute",
        "preprocessor",
    )
    response = client.post(
        "/api/preprocessing-tasks",
        json={"model_type": "electric_load", "dataset_id": dataset_id, "preprocess_script_id": raising, "mode": "skip", "skip": True},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "SKIPPED"
    assert body["preprocess_used"] is False
    assert body["preprocess_message"] == "未使用预处理"
    assert "未使用预处理" in " ".join(body["logs"])


def test_failed_training_can_retry_without_creating_or_replacing_production(acceptance_api):
    client, factory = acceptance_api
    dataset_id = dataset_with_split(client)
    trainer = script(
        client,
        b"def train(X_train, y_train, X_test, y_test, config):\n"
        b"    raise ValueError('controlled training failure')\n",
        "always-fails",
    )
    current = save(client, "v1")
    assert publish(client, current["id"]).status_code == 200

    created = client.post("/api/training-jobs", json={"model_type": "electric_load", "dataset_id": dataset_id, "train_script_id": trainer})
    assert created.status_code == 201, created.text
    job = wait_for(client, created.json()["id"], "FAILED")
    assert "controlled training failure" in (job["error_message"] or "")
    assert "不会影响生产模型" not in " ".join(job["logs"])  # UI explains this; API keeps execution logs factual.

    retry = client.post(f"/api/training-jobs/{job['id']}/retry")
    assert retry.status_code == 200, retry.text
    retried = wait_for(client, job["id"], "FAILED")
    assert retried["model_version_id"] is None
    production = client.get(f"/api/models/{current['id']}").json()
    assert production["is_current"] is True and production["status"] == "PUBLISHED"
    with factory() as session:
        assert session.scalar(select(ModelVersionORM).where(ModelVersionORM.training_job_id == job["id"])) is None


def test_publish_requires_second_confirmation_and_version_queries_include_detail_and_rollback(acceptance_api):
    client, _ = acceptance_api
    first, second = save(client, "v1"), save(client, "v2")
    refused = client.post(f"/api/models/{first['id']}/publish", json={})
    assert refused.status_code == 400
    assert refused.json()["detail"]["code"] == "PUBLISH_CONFIRMATION_REQUIRED"
    assert client.get("/api/models").status_code == 200
    assert client.get(f"/api/models/{first['id']}").json()["version"] == "v1"
    assert publish(client, first["id"]).status_code == 200
    assert publish(client, second["id"]).status_code == 200

    rollback = client.post(f"/api/models/{second['id']}/rollback", json={"target_version_id": first["id"], "reason": "验收回滚"})
    assert rollback.status_code == 200, rollback.text
    assert rollback.json()["rollback"]["rollback_to"] == first["id"]
    records = client.get(f"/api/models/{second['id']}/rollback-records")
    assert records.status_code == 200
    assert records.json()[0]["reason"] == "验收回滚"


def test_abnormal_current_auto_rolls_back_and_alert_stays_active_until_publish(acceptance_api):
    client, factory = acceptance_api
    first, second = save(client, "v1"), save(client, "v2")
    assert publish(client, first["id"]).status_code == 200
    assert publish(client, second["id"]).status_code == 200
    abnormal = client.post(f"/api/models/{second['id']}/abnormal", json={"reason": "验收异常"})
    assert abnormal.status_code == 200, abnormal.text
    assert abnormal.json()["version"] == "v1"
    active = client.get("/api/alerts", params={"active_only": "true"}).json()
    assert len(active) == 1 and active[0]["status"] == "ACTIVE"
    alert_id = active[0]["id"]
    assert client.post(f"/api/alerts/{alert_id}/acknowledge").json()["status"] == "ACTIVE"
    with factory() as session:
        record = session.scalar(select(RollbackRecordORM).order_by(RollbackRecordORM.created_at.desc()))
        assert record.status is RollbackStatus.SUCCEEDED
        assert session.get(ModelAlertORM, alert_id).status is AlertStatus.ACTIVE
        assert session.get(ModelVersionORM, first["id"]).is_current is True
        assert session.get(ModelVersionORM, second["id"]).health_status is HealthStatus.ABNORMAL
    replacement = save(client, "v3")
    assert publish(client, replacement["id"]).status_code == 200
    assert client.get("/api/alerts", params={"active_only": "true"}).json() == []
    assert client.get(f"/api/alerts/{alert_id}").json()["status"] == "RESOLVED"
