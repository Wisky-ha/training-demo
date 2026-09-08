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
        json={"model_type": "electric_load", "version": version, "model_path": f"models/{version}.joblib", "health_status": "HEALTHY"},
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
    assert client.post(f"/api/alerts/{alert_id}/acknowledge").json()["status"] == "ACKNOWLEDGED"
    with factory() as session:
        record = session.scalar(select(RollbackRecordORM).order_by(RollbackRecordORM.created_at.desc()))
        assert record.status is RollbackStatus.SUCCEEDED
        assert session.get(ModelAlertORM, alert_id).status is AlertStatus.ACKNOWLEDGED
        assert session.get(ModelVersionORM, first["id"]).is_current is True
        assert session.get(ModelVersionORM, second["id"]).health_status is HealthStatus.ABNORMAL
    replacement = save(client, "v3")
    assert publish(client, replacement["id"]).status_code == 200
    assert client.get("/api/alerts", params={"active_only": "true"}).json() == []
    assert client.get(f"/api/alerts/{alert_id}").json()["status"] == "RESOLVED"


def test_lifecycle_failure_audits_keep_resource_links_and_rollback_replay_is_single_shot(acceptance_api):
    client, _ = acceptance_api
    first, second = save(client, "v1"), save(client, "v2")
    assert publish(client, first["id"], idempotency_key="publish-v1").status_code == 200
    assert publish(client, second["id"], idempotency_key="publish-v2").status_code == 200

    failed_save = client.post(
        f"/api/models/{second['id']}/save",
        json={"model_type": "integrated_energy", "status": "READY"},
    )
    assert failed_save.status_code == 400, failed_save.text
    failed_save_events = client.get(
        "/api/audit-events",
        params={
            "event_type": "MODEL_SAVED",
            "object_type": "MODEL_VERSION",
            "result": "FAILED",
            "model_type": "electric_load",
        },
    ).json()
    assert failed_save_events["total"] == 1
    assert failed_save_events["items"][0]["model_version_id"] == second["id"]

    # A failed lifecycle command still refers to the known model version, so
    # the audit page can filter it by model family and link it to the object.
    assert client.post(f"/api/models/{second['id']}/offline").status_code == 200
    failed_publish = publish(client, second["id"], idempotency_key="publish-retired")
    assert failed_publish.status_code == 409, failed_publish.text
    failed_publish_events = client.get(
        "/api/audit-events",
        params={
            "event_type": "MODEL_PUBLISHED",
            "object_type": "MODEL_VERSION",
            "result": "FAILED",
            "model_type": "electric_load",
            "page": 1,
            "page_size": 1,
        },
    ).json()
    assert failed_publish_events["total"] == 1
    assert failed_publish_events["items"][0]["object_id"] == second["id"]
    assert failed_publish_events["items"][0]["model_version_id"] == second["id"]

    third = save(client, "v3")
    assert publish(client, third["id"], idempotency_key="publish-v3").status_code == 200
    failed_rollback = client.post(
        f"/api/models/{third['id']}/rollback",
        json={
            "target_version_id": third["id"],
            "reason": "不能回滚到当前版本",
            "idempotency_key": "rollback-invalid",
        },
    )
    assert failed_rollback.status_code == 409, failed_rollback.text
    failed_rollback_events = client.get(
        "/api/audit-events",
        params={
            "event_type": "MODEL_ROLLBACK_FAILED",
            "object_type": "MODEL_VERSION",
            "result": "FAILED",
            "model_type": "electric_load",
        },
    ).json()
    assert failed_rollback_events["total"] == 1
    assert failed_rollback_events["items"][0]["model_version_id"] == third["id"]

    rollback_body = {
        "target_version_id": second["id"],
        "reason": "恢复稳定版本",
        "idempotency_key": "rollback-v3-to-v2",
    }
    succeeded = client.post(f"/api/models/{third['id']}/rollback", json=rollback_body)
    replay = client.post(f"/api/models/{third['id']}/rollback", json=rollback_body)
    assert succeeded.status_code == replay.status_code == 200
    all_events = client.get("/api/audit-events", params={"page_size": 100}).json()["items"]
    assert sum(item["event_type"] == "MODEL_ROLLBACK_STARTED" for item in all_events) == 2
    assert sum(item["event_type"] == "MODEL_ROLLBACK_SUCCEEDED" for item in all_events) == 1


def test_real_vertical_chain_survives_refresh_and_audit_pagination(acceptance_api):
    client, _ = acceptance_api

    invalid = upload(client, b"time,feature,target\nnot-a-date,1,no\n")
    assert invalid.status_code == 400
    invalid_events = client.get("/api/audit-events", params={
        "event_type": "DATASET_VALIDATED", "result": "FAILED",
    }).json()
    assert invalid_events["total"] == 1
    assert invalid_events["items"][0]["object_id"] is None  # no resource was created

    dataset_response = upload(
        client,
        b"time,feature,target\n"
        b"2024-01-05,5,50\n2024-01-01,1,10\n2024-01-04,4,40\n"
        b"2024-01-02,2,20\n2024-01-03,3,30\n"
        b"2024-01-10,10,100\n2024-01-06,6,60\n2024-01-07,7,70\n"
        b"2024-01-09,9,90\n2024-01-08,8,80\n",
    )
    dataset_id = dataset_response.json()["id"]
    refreshed_dataset = client.get(f"/api/datasets/{dataset_id}")
    assert refreshed_dataset.status_code == 200
    assert refreshed_dataset.json()["id"] == dataset_id

    skipped = client.post("/api/preprocessing-tasks", json={
        "model_type": "electric_load", "dataset_id": dataset_id,
        "mode": "skip", "skip": True,
    })
    assert skipped.status_code == 201, skipped.text
    task = skipped.json()
    task_id = task["id"]
    assert task["status"] == "SKIPPED" and task["preprocess_used"] is False
    assert client.get(f"/api/preprocessing-tasks/{task_id}").json()["dataset_id"] == dataset_id

    split = client.post(
        f"/api/datasets/{dataset_id}/split",
        json={"preprocessing_task_id": task_id},
    )
    assert split.status_code == 201, split.text
    split_body = split.json()
    split_id = split_body["id"]
    assert (split_body["split_ratio"], split_body["test_ratio"]) == (0.8, 0.2)
    assert (split_body["train_row_count"], split_body["test_row_count"]) == (8, 2)
    refreshed_split = client.get(f"/api/datasets/{dataset_id}/split")
    assert refreshed_split.json()["id"] == split_id

    failing_trainer = script(
        client,
        b"def train(X_train, y_train, X_test, y_test, config):\n"
        b"    raise ValueError('vertical failure')\n",
        "vertical-failing-trainer",
    )
    failed_created = client.post("/api/training-jobs", json={
        "model_type": "electric_load", "dataset_id": dataset_id,
        "preprocessing_task_id": task_id, "train_script_id": failing_trainer,
    })
    assert failed_created.status_code == 201, failed_created.text
    failed_job = wait_for(client, failed_created.json()["id"], "FAILED")
    assert failed_job["model_version_id"] is None
    assert "vertical failure" in failed_job["error_message"]

    trainer = script(
        client,
        b"class Model:\n"
        b"    def predict(self, X): return [0.0] * len(X)\n"
        b"def train(X_train, y_train, X_test, y_test, config): return Model()\n",
        "vertical-success-trainer",
    )
    created = client.post("/api/training-jobs", json={
        "model_type": "electric_load", "dataset_id": dataset_id,
        "preprocess_script_id": None, "preprocessing_task_id": task_id,
        "train_script_id": trainer,
    })
    assert created.status_code == 201, created.text
    job_id = created.json()["id"]
    job = wait_for(client, job_id, "SUCCEEDED")
    assert job["dataset_id"] == dataset_id
    assert job["preprocessing_task_id"] == task_id
    assert job["split_ratio"] == 0.8 and job["test_ratio"] == 0.2
    assert (job["train_row_count"], job["test_row_count"]) == (8, 2)
    model_version_id = job["model_version_id"]
    assert model_version_id
    assert client.get(f"/api/training-jobs/{job_id}").json()["id"] == job_id

    first_logs = client.get(f"/api/training-jobs/{job_id}/logs", params={"limit": 2}).json()
    assert first_logs["job_id"] == job_id and len(first_logs["items"]) == 2
    assert first_logs["next_cursor"]
    next_logs = client.get(f"/api/training-jobs/{job_id}/logs", params={
        "limit": 100, "cursor": first_logs["next_cursor"],
    }).json()
    assert next_logs["items"]

    evaluation = client.get(f"/api/training-jobs/{job_id}/evaluation")
    assert evaluation.status_code == 200, evaluation.text
    assert evaluation.json()["job_id"] == job_id
    assert evaluation.json()["model_version_id"] == model_version_id
    assert evaluation.json()["metrics"]["sample_count"] == 2
    candidate = client.get(f"/api/models/{model_version_id}")
    assert candidate.status_code == 200 and candidate.json()["status"] == "READY"

    saved = client.post(f"/api/models/{model_version_id}/save", json={
        "model_type": "electric_load", "status": "READY",
    })
    assert saved.status_code == 200, saved.text
    published = publish(client, model_version_id, reason="首个真实训练版本", idempotency_key="vertical-publish-v1")
    assert published.status_code == 200, published.text
    replay = publish(client, model_version_id, reason="首个真实训练版本", idempotency_key="vertical-publish-v1")
    assert replay.status_code == 200
    assert replay.json()["record"]["id"] == published.json()["record"]["id"]

    second = save(client, "v2")
    assert publish(client, second["id"], reason="替换版本", idempotency_key="vertical-publish-v2").status_code == 200
    assert client.post(f"/api/models/{second['id']}/offline").status_code == 200
    assert client.get(f"/api/models/{second['id']}").json()["status"] == "RETIRED"
    third = save(client, "v3")
    assert publish(client, third["id"], reason="临时版本", idempotency_key="vertical-publish-v3").status_code == 200

    rollback_body = {
        "target_version_id": model_version_id,
        "reason": "回到真实训练版本",
        "idempotency_key": "vertical-rollback-v3",
    }
    rollback = client.post(f"/api/models/{third['id']}/rollback", json=rollback_body)
    assert rollback.status_code == 200, rollback.text
    assert rollback.json()["rollback"]["rollback_to"] == model_version_id
    assert client.post(f"/api/models/{third['id']}/rollback", json=rollback_body).status_code == 200

    abnormal = client.post("/api/models/abnormal", json={
        "model_type": "electric_load", "model_version": model_version_id,
        "reason": "纵向验收异常",
    })
    assert abnormal.status_code == 200, abnormal.text
    alert = client.get("/api/alerts", params={"active_only": True}).json()[0]
    alert_id = alert["id"]
    acknowledged = client.post(f"/api/alerts/{alert_id}/acknowledge", json={"confirmed": True})
    assert acknowledged.status_code == 200
    assert acknowledged.json()["status"] == "ACKNOWLEDGED"
    assert client.get("/api/alerts", params={"active_only": True}).json()[0]["status"] == "ACKNOWLEDGED"
    rejected_ack = client.post(f"/api/alerts/{alert_id}/acknowledge", json={"confirmed": False})
    assert rejected_ack.status_code == 400
    failed_ack_events = client.get("/api/audit-events", params={
        "event_type": "ALERT_ACKNOWLEDGED", "object_type": "ALERT",
        "result": "FAILED", "model_type": "electric_load",
    }).json()
    assert failed_ack_events["total"] == 1
    assert failed_ack_events["items"][0]["model_version_id"] == model_version_id

    replacement = save(client, "v4")
    assert publish(client, replacement["id"], reason="异常后的替换", idempotency_key="vertical-publish-v4").status_code == 200
    assert client.get("/api/alerts", params={"active_only": True}).json() == []
    assert client.get(f"/api/alerts/{alert_id}").json()["status"] == "RESOLVED"

    # Read only the events produced by this test and walk the real pages; no
    # historical rows are inserted to make the audit assertions pass.
    page = 1
    audited: list[dict] = []
    while True:
        response = client.get("/api/audit-events", params={"page": page, "page_size": 5})
        assert response.status_code == 200, response.text
        body = response.json()
        audited.extend(body["items"])
        if not body["has_next"]:
            assert body["total"] == len(audited)
            break
        page += 1
        assert page < 30
    assert len({event["id"] for event in audited}) == len(audited)

    def events_for(event_type: str, result: str = "SUCCEEDED", **filters: str) -> list[dict]:
        params = {"event_type": event_type, "result": result, **filters}
        return client.get("/api/audit-events", params=params).json()["items"]

    assert events_for("DATASET_UPLOADED", object_type="DATASET")[0]["object_id"] == dataset_id
    assert events_for("DATASET_VALIDATED", object_type="DATASET")[0]["object_id"] == dataset_id
    assert events_for("PREPROCESS_SKIPPED", object_type="PREPROCESSING_TASK")[0]["object_id"] == task_id
    assert events_for("SPLIT_CREATED", object_type="DATASET_SPLIT")[0]["object_id"] == split_id
    assert events_for("TRAINING_STARTED", object_type="TRAINING_JOB")[0]["object_id"] == job_id
    assert events_for("TRAINING_SUCCEEDED", object_type="TRAINING_JOB")[0]["model_version_id"] == model_version_id
    assert events_for("EVALUATION_COMPLETED", object_type="TRAINING_JOB")[0]["model_version_id"] == model_version_id
    assert events_for("MODEL_PUBLISHED", object_type="MODEL_VERSION")
    assert events_for("MODEL_OFFLINED", object_type="MODEL_VERSION")
    rollback_events = events_for("MODEL_ROLLBACK_SUCCEEDED", object_type="MODEL_VERSION")
    assert {event["object_id"] for event in rollback_events} >= {model_version_id, third["id"]}
    assert events_for("MODEL_MARKED_ABNORMAL", object_type="MODEL_VERSION")[0]["object_id"] == model_version_id
    assert events_for("ALERT_CREATED", object_type="ALERT")[0]["object_id"] == alert_id
    assert events_for("ALERT_ACKNOWLEDGED", object_type="ALERT")[0]["object_id"] == alert_id
    assert events_for("ALERT_RESOLVED", object_type="ALERT")[0]["object_id"] == alert_id
    failed_training_events = events_for(
        "TRAINING_FAILED", result="FAILED", object_type="TRAINING_JOB", model_type="electric_load",
    )
    assert failed_training_events[0]["object_id"] == failed_job["id"]
