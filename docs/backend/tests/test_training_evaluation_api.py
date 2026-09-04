import json
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.core.config import Settings
from backend.app.db.models import ModelTypeORM, ModelVersionORM
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import ModelType, ModelVersionStatus
from backend.app.main import create_app
from backend.app.services.training_jobs import TrainingJobService


@pytest.fixture
def api(tmp_path, monkeypatch):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'evaluation.db').as_posix()}",
        storage_root=tmp_path / "storage",
    )
    engine = initialize_database(settings=settings)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add(ModelTypeORM(code=ModelType.ELECTRIC_LOAD, name="电力"))
        session.commit()
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, factory, monkeypatch
    engine.dispose()


def dataset(client):
    # The final two rows are the complete test set after the fixed 80/20 split.
    content = b"time,x,target\n" + b"\n".join(
        f"2024-01-{n:02d},{n},{0 if n == 9 else n}".encode() for n in range(1, 11)
    ) + b"\n"
    item = client.post("/api/datasets/upload", files={"file": ("x.csv", content, "text/csv")})
    assert item.status_code == 201, item.text
    dataset_id = item.json()["id"]
    split = client.post(f"/api/datasets/{dataset_id}/split", json={})
    assert split.status_code == 201, split.text
    return dataset_id


def script(client, source, name="trainer", script_type="trainer"):
    response = client.post(
        "/api/scripts/upload",
        data={"name": name, "script_type": script_type, "supported_model_types": json.dumps(["electric_load"])},
        files={"file": (f"{name}.py", source, "text/x-python")},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def wait(client, job_id):
    for _ in range(250):
        response = client.get(f"/api/training-jobs/{job_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"SUCCEEDED", "FAILED"}:
            return body
        time.sleep(0.01)
    pytest.fail("training job did not finish")


def test_creation_requires_completed_split_and_valid_script(api):
    client, _, _ = api
    upload = client.post(
        "/api/datasets/upload",
        files={"file": ("small.csv", b"time,x,target\n2024-01-01,1,1\n2024-01-02,2,2\n", "text/csv")},
    )
    dataset_id = upload.json()["id"]
    trainer = script(client, b"def train(X_train, y_train, X_test, y_test, config): return object()")
    missing_split = client.post(
        "/api/training-jobs",
        json={"model_type": "electric_load", "dataset_id": dataset_id, "train_script_id": trainer},
    )
    assert missing_split.status_code == 400
    assert missing_split.json()["detail"]["code"] == "DATASET_SPLIT_NOT_FOUND"
    preprocessor = script(client, b"class Preprocessor: pass", "pre", "preprocessor")
    split = client.post(f"/api/datasets/{dataset_id}/split", json={})
    assert split.status_code == 201
    wrong_type = client.post(
        "/api/training-jobs",
        json={"model_type": "electric_load", "dataset_id": dataset_id, "train_script_id": preprocessor},
    )
    assert wrong_type.status_code == 400
    assert wrong_type.json()["detail"]["code"] == "SCRIPT_TYPE_INVALID"


def test_metrics_use_complete_test_set_mape_zero_note_and_sampled_chart(api):
    client, _, monkeypatch = api
    dataset_id = dataset(client)
    trainer = script(
        client,
        b'''class Model:
    def predict(self, X): return [0.0] * len(X)
def train(X_train, y_train, X_test, y_test, config): return Model()
''',
    )
    monkeypatch.setattr(TrainingJobService, "MAX_CHART_POINTS", 1)
    created = client.post(
        "/api/training-jobs",
        json={"model_type": "electric_load", "dataset_id": dataset_id, "train_script_id": trainer},
    )
    assert created.status_code == 201, created.text
    job = wait(client, created.json()["id"])
    assert job["status"] == "SUCCEEDED", job
    result = client.get(f"/api/training-jobs/{job['id']}/evaluation")
    assert result.status_code == 200, result.text
    body = result.json()
    metrics = body["metrics"]
    assert metrics["sample_count"] == 2  # never reduced to chart sample size
    assert metrics["mae"] == pytest.approx(5.0)
    assert metrics["rmse"] == pytest.approx(50**0.5)
    assert metrics["mape"] == pytest.approx(100.0)
    assert metrics["mape_valid_count"] == 1
    assert metrics["mape_excluded_count"] == 1
    assert "排除 1" in metrics["mape_note"]
    assert body["chart_total_count"] == 2
    assert body["chart_sample_count"] == 1
    assert body["chart_sampled"] is True
    assert body["chart_data"][0]["time"] == "2024-01-09T00:00:00"
    assert body["error_data"][0]["time"] == body["chart_data"][0]["time"]
    assert body["model_comparison"]["candidate"]["model_version_id"] == job["model_version_id"]


def test_retry_and_missing_task_errors(api):
    client, factory, _ = api
    dataset_id = dataset(client)
    trainer = script(
        client,
        b'''def train(X_train, y_train, X_test, y_test, config):
    raise ValueError("retry me")
''',
        "failing",
    )
    created = client.post("/api/training-jobs", json={"model_type": "electric_load", "dataset_id": dataset_id, "train_script_id": trainer})
    job = wait(client, created.json()["id"])
    assert job["status"] == "FAILED"
    retry = client.post(f"/api/training-jobs/{job['id']}/retry")
    assert retry.status_code == 200
    retried = wait(client, job["id"])
    assert retried["status"] == "FAILED"
    assert client.get("/api/training-jobs/no-such/logs").status_code == 404
    assert client.get("/api/training-jobs/no-such/evaluation").status_code == 404
    with factory() as session:
        assert session.scalar(select(ModelVersionORM).where(ModelVersionORM.training_job_id == job["id"])) is None
        # Retry/failure never changes any pre-existing current version pointer.
        assert session.scalar(select(ModelVersionORM).where(ModelVersionORM.status == ModelVersionStatus.PUBLISHED)) is None
