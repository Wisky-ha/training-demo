"""Step 5 coverage for business-operation audit events."""

from __future__ import annotations

import base64
import json
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.core.config import Settings
from backend.app.db.models import AuditEventORM, ModelTypeORM
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import ModelType
from backend.app.schemas.training_jobs import TrainingJobCreate
from backend.app.services.training_jobs import TrainingJobService
from backend.app.main import create_app


@pytest.fixture
def step5_api(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'step5.db').as_posix()}",
        storage_root=tmp_path / "storage",
    )
    engine = initialize_database(settings=settings)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add(ModelTypeORM(code=ModelType.ELECTRIC_LOAD, name="电力"))
        session.commit()
    with TestClient(create_app(settings)) as client:
        yield client, factory
    engine.dispose()


def _events(factory):
    with factory() as session:
        return list(session.scalars(select(AuditEventORM).order_by(AuditEventORM.occurred_at)))


def _upload(client):
    content = b"time,x,target\n" + b"\n".join(
        f"2024-01-{day:02d},{day},{day}".encode() for day in range(1, 11)
    )
    response = client.post(
        "/api/datasets/upload", files={"file": ("load.csv", content, "text/csv")}
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _script(client, source: bytes, name: str, script_type: str = "preprocessor"):
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


def test_dataset_preprocessing_split_and_failures_are_audited(step5_api):
    client, factory = step5_api
    invalid = client.post(
        "/api/datasets/upload",
        files={"file": ("bad.csv", b"time,x,target\nnot-a-date,1,no", "text/csv")},
    )
    assert invalid.status_code == 400
    dataset_id = _upload(client)

    skipped = client.post(
        "/api/preprocessing-tasks",
        json={"model_type": "electric_load", "dataset_id": dataset_id, "skip": True},
    )
    assert skipped.status_code == 201, skipped.text
    good_script = _script(
        client,
        b"class Preprocessor:\n    def fit(self, df, config): return self\n    def transform(self, df, config): return df\n",
        "good-preprocessor",
    )
    succeeded = client.post(
        "/api/preprocessing-tasks",
        json={
            "model_type": "electric_load", "dataset_id": dataset_id,
            "preprocess_script_id": good_script,
        },
    )
    assert succeeded.status_code == 201, succeeded.text
    failed_script = _script(
        client,
        b"class Preprocessor:\n    def fit(self, df, config): raise ValueError('bad fit')\n    def transform(self, df, config): return df\n",
        "bad-preprocessor",
    )
    failed = client.post(
        "/api/preprocessing-tasks",
        json={
            "model_type": "electric_load", "dataset_id": dataset_id,
            "preprocess_script_id": failed_script,
        },
    )
    assert failed.status_code == 400

    split = client.post(f"/api/datasets/{dataset_id}/split", json={})
    assert split.status_code == 201, split.text
    duplicate = client.post(f"/api/datasets/{dataset_id}/split", json={})
    assert duplicate.status_code == 409

    event_results = {(event.event_type, event.result) for event in _events(factory)}
    assert ("DATASET_UPLOADED", "SUCCEEDED") in event_results
    assert ("DATASET_VALIDATED", "SUCCEEDED") in event_results
    assert ("DATASET_VALIDATED", "FAILED") in event_results
    invalid_event = next(event for event in _events(factory) if event.event_type == "DATASET_VALIDATED" and event.result == "FAILED")
    assert invalid_event.object_id is None
    assert ("PREPROCESS_STARTED", "SUCCEEDED") in event_results
    assert ("PREPROCESS_SUCCEEDED", "SUCCEEDED") in event_results
    assert ("PREPROCESS_SKIPPED", "SUCCEEDED") in event_results
    assert ("PREPROCESS_FAILED", "FAILED") in event_results
    assert ("SPLIT_CREATED", "SUCCEEDED") in event_results
    assert ("SPLIT_CREATED", "FAILED") in event_results


def test_training_and_model_lifecycle_audits_include_failure_and_operator_context(step5_api):
    client, factory = step5_api
    dataset_id = _upload(client)
    assert client.post(f"/api/datasets/{dataset_id}/split", json={}).status_code == 201
    trainer = _script(
        client,
        b'''class Model:\n    def predict(self, X): return [0.0] * len(X)\ndef train(X_train, y_train, X_test, y_test, config): return Model()\n''',
        "good-trainer", "trainer",
    )
    headers = {
        "X-Request-ID": "req-step5",
        "X-Correlation-ID": "corr-step5",
        "X-Operator-Id": "operator-1",
        "X-Operator-Name": "step5-user",
    }
    created = client.post(
        "/api/training-jobs", headers=headers,
        json={"model_type": "electric_load", "dataset_id": dataset_id, "train_script_id": trainer},
    )
    assert created.status_code == 201, created.text
    job_id = created.json()["id"]
    for _ in range(200):
        job = client.get(f"/api/training-jobs/{job_id}").json()
        if job["status"] in {"SUCCEEDED", "FAILED"}:
            break
        time.sleep(0.01)
    assert job["status"] == "SUCCEEDED", job
    version_id = job["model_version_id"]

    failed_trainer = _script(
        client, b"def train(X_train, y_train, X_test, y_test, config): return object()",
        "failed-trainer", "trainer",
    )
    failed_created = client.post(
        "/api/training-jobs",
        json={"model_type": "electric_load", "dataset_id": dataset_id, "train_script_id": failed_trainer},
    )
    assert failed_created.status_code == 201
    for _ in range(200):
        failed_job = client.get(f"/api/training-jobs/{failed_created.json()['id']}").json()
        if failed_job["status"] == "FAILED":
            break
        time.sleep(0.01)
    assert failed_job["status"] == "FAILED", failed_job

    # Create without submitting to exercise the durable pending-cancel path.
    with factory() as session:
        pending = TrainingJobService(session).create(
            TrainingJobCreate(
                model_type=ModelType.ELECTRIC_LOAD,
                dataset_id=dataset_id,
                train_script_id=trainer,
            )
        )
        TrainingJobService(session).cancel(pending.id)

    assert client.post(f"/api/models/{version_id}/publish", headers=headers, json={"confirm": True}).status_code == 200
    assert client.post(f"/api/models/{version_id}/offline", headers=headers).status_code == 200

    # A lifecycle rejection is also an auditable failed business operation.
    failed_publish = client.post(f"/api/models/{version_id}/publish", headers=headers, json={"confirm": True})
    assert failed_publish.status_code == 409
    events = _events(factory)
    event_results = {(event.event_type, event.result) for event in events}
    for event_type in (
        "TRAINING_STARTED", "TRAINING_SUCCEEDED", "TRAINING_CANCELLED",
        "MODEL_SAVED", "EVALUATION_COMPLETED", "MODEL_PUBLISHED", "MODEL_OFFLINED",
    ):
        assert (event_type, "SUCCEEDED") in event_results, event_type
    assert ("TRAINING_FAILED", "FAILED") in event_results
    assert ("MODEL_PUBLISHED", "FAILED") in event_results
    evaluation = next(event for event in events if event.event_type == "EVALUATION_COMPLETED" and event.result == "SUCCEEDED")
    assert evaluation.object_type == "TRAINING_JOB"
    assert evaluation.object_id == job_id
    assert evaluation.model_version_id == version_id
    contextual = next(event for event in events if event.event_type == "TRAINING_STARTED")
    assert contextual.request_id == "req-step5"
    assert contextual.correlation_id == "corr-step5"
    assert contextual.operator_id == "operator-1"
    assert contextual.operator_name == "step5-user"


def test_model_save_and_alert_acknowledge_are_audited(step5_api):
    client, factory = step5_api
    body = {
        "model_type": "electric_load", "version": "v1",
        "model_path": "legacy/v1", "health_status": "HEALTHY",
    }
    saved = client.post("/api/models", json=body)
    assert saved.status_code == 201, saved.text
    model_id = saved.json()["id"]
    assert client.post(f"/api/models/{model_id}/publish", json={"confirm": True}).status_code == 200
    abnormal = client.post(f"/api/models/{model_id}/abnormal", json={"reason": "异常"})
    assert abnormal.status_code == 409  # no backup, but the anomaly is committed
    alert = client.get("/api/alerts").json()[0]
    assert client.post(f"/api/alerts/{alert['id']}/acknowledge").status_code == 200
    assert client.post("/api/alerts/missing-alert/acknowledge").status_code == 404
    replacement = client.post("/api/models", json={
        "model_type": "electric_load", "version": "v2", "model_path": "legacy/v2",
        "health_status": "HEALTHY",
    }).json()
    assert client.post(f"/api/models/{replacement['id']}/publish", json={"confirm": True}).status_code == 200
    newer = client.post("/api/models", json={
        "model_type": "electric_load", "version": "v3", "model_path": "legacy/v3",
        "health_status": "HEALTHY",
    }).json()
    assert client.post(f"/api/models/{newer['id']}/publish", json={"confirm": True}).status_code == 200
    assert client.post(
        f"/api/models/{newer['id']}/rollback",
        json={"target_version_id": replacement["id"], "reason": "恢复稳定版本"},
    ).status_code == 200
    event_results = {(event.event_type, event.result) for event in _events(factory)}
    assert ("MODEL_SAVED", "SUCCEEDED") in event_results
    assert ("MODEL_MARKED_ABNORMAL", "SUCCEEDED") in event_results
    assert ("ALERT_CREATED", "SUCCEEDED") in event_results
    assert ("ALERT_ACKNOWLEDGED", "SUCCEEDED") in event_results
    assert ("ALERT_ACKNOWLEDGED", "FAILED") in event_results
    assert ("MODEL_ROLLBACK_STARTED", "SUCCEEDED") in event_results
    assert ("MODEL_ROLLBACK_FAILED", "FAILED") in event_results
    assert ("MODEL_ROLLBACK_SUCCEEDED", "SUCCEEDED") in event_results
    assert ("ALERT_RESOLVED", "SUCCEEDED") in event_results
