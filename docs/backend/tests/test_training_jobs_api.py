import json
import time
from threading import Event

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.core.config import Settings
from backend.app.db.models import ModelTypeORM, ModelVersionORM
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import HealthStatus, ModelType, ModelVersionStatus
from backend.app.main import create_app
from backend.app.services.training_executor import TrainingExecutionResult, TrainingScriptExecutor


@pytest.fixture
def training_api(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'training.db').as_posix()}",
        storage_root=tmp_path / "storage",
    )
    engine = initialize_database(settings=settings)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add(ModelTypeORM(code=ModelType.ELECTRIC_LOAD, name="电力"))
        session.commit()
    application = create_app(settings)
    with TestClient(application) as client:
        yield client, factory
    engine.dispose()


def _dataset(client, *, split=True):
    response = client.post(
        "/api/datasets/upload",
        files={
            "file": (
                "load.csv",
                b"time,feature,target\n"
                b"2024-01-03,3,30\n"
                b"2024-01-01,1,10\n"
                b"2024-01-02,2,20\n"
                b"2024-01-04,4,40\n"
                b"2024-01-05,5,50\n",
                "text/csv",
            )
        },
    )
    assert response.status_code == 201, response.text
    dataset_id = response.json()["id"]
    if split:
        split_response = client.post(f"/api/datasets/{dataset_id}/split", json={})
        assert split_response.status_code == 201, split_response.text
    return dataset_id


def _script(client, source, *, name, script_type="trainer"):
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


GOOD = b'''class Model:
    def predict(self, X):
        return [0.0] * len(X)

def train(X_train, y_train, X_test, y_test, config):
    print("trainer log")
    return Model()
'''


def _wait_for(client, job_id, expected, timeout=5):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = client.get(f"/api/training-jobs/{job_id}").json()
        if last["status"] == expected:
            return last
        time.sleep(0.02)
    pytest.fail(f"job did not reach {expected}: {last}")


def test_create_returns_without_waiting_and_runs_in_background(training_api, monkeypatch):
    client, _ = training_api
    dataset_id = _dataset(client)
    script_id = _script(client, GOOD, name="slow-trainer")
    started = Event()
    release = Event()

    def slow_execute(*args, **kwargs):
        started.set()
        release.wait(timeout=3)
        return TrainingExecutionResult(True, type("M", (), {"predict": lambda self, X: [0.0] * len(X)})(), ["slow log"])

    monkeypatch.setattr(TrainingScriptExecutor, "execute", slow_execute)
    begin = time.monotonic()
    response = client.post(
        "/api/training-jobs",
        json={"model_type": "electric_load", "dataset_id": dataset_id, "train_script_id": script_id},
    )
    elapsed = time.monotonic() - begin
    assert response.status_code == 201, response.text
    job_id = response.json()["id"]
    assert response.json()["status"] == "PENDING"
    assert elapsed < 1.0
    assert started.wait(timeout=2)
    release.set()
    result = _wait_for(client, job_id, "SUCCEEDED")
    assert result["finished_at"]


def test_success_records_all_states_split_metadata_logs_and_completion(training_api):
    client, factory = training_api
    dataset_id = _dataset(client)
    script_id = _script(client, GOOD, name="successful-trainer")
    response = client.post(
        "/api/training-jobs",
        json={"model_type": "electric_load", "dataset_id": dataset_id, "train_script_id": script_id},
    )
    assert response.status_code == 201, response.text
    job = _wait_for(client, response.json()["id"], "SUCCEEDED")
    assert job["progress_stage"] == "SUCCEEDED"
    assert job["train_row_count"] == 4
    assert job["test_row_count"] == 1
    assert job["stage_started_at"] and job["created_at"] and job["finished_at"]
    assert job["created_at"] <= job["finished_at"]
    logs = " ".join(job["logs"])
    for state in ("RUNNING", "PREPROCESSING", "SPLITTING", "TRAINING", "EVALUATING"):
        assert state.lower() in logs or {
            "RUNNING": "后台任务",
            "PREPROCESSING": "预处理",
            "SPLITTING": "划分",
            "TRAINING": "训练",
            "EVALUATING": "评估",
        }[state] in logs
    with factory() as session:
        assert session.scalar(select(ModelVersionORM).where(ModelVersionORM.training_job_id == response.json()["id"])) is not None


def test_failure_persists_exception_log_and_keeps_production_model(training_api):
    client, factory = training_api
    dataset_id = _dataset(client)
    script_id = _script(
        client,
        b'''def train(X_train, y_train, X_test, y_test, config):
    print("before boom")
    raise ValueError("bad training data")
''',
        name="failing-trainer",
    )
    with factory() as session:
        current = ModelVersionORM(
            model_type=ModelType.ELECTRIC_LOAD,
            version="v1",
            model_path="model/v1.bin",
            status=ModelVersionStatus.PUBLISHED,
            health_status=HealthStatus.HEALTHY,
            is_current=True,
        )
        session.add(current)
        session.flush()
        model_type = session.scalar(
            select(ModelTypeORM).where(ModelTypeORM.code == ModelType.ELECTRIC_LOAD)
        )
        model_type.current_version_id = current.id
        session.commit()
        current_id = current.id

    response = client.post(
        "/api/training-jobs",
        json={"model_type": "electric_load", "dataset_id": dataset_id, "train_script_id": script_id},
    )
    job = _wait_for(client, response.json()["id"], "FAILED")
    assert "bad training data" in (job["error_message"] or "")
    assert job["error_code"] == "TRAIN_EXECUTION_FAILED"
    assert job["error_details"]["exception_type"] == "TrainingJobError"
    assert any("before boom" in item for item in job["logs"])
    assert job["finished_at"] and job["created_at"] <= job["finished_at"]
    assert "traceback" not in str(job).lower()
    with factory() as session:
        assert session.scalar(select(ModelVersionORM).where(ModelVersionORM.training_job_id == response.json()["id"])) is None
        current = session.get(ModelVersionORM, current_id)
        assert current.is_current is True
        assert current.status is ModelVersionStatus.PUBLISHED


def test_cancel_stops_after_current_operation_and_does_not_create_version(training_api, monkeypatch):
    client, factory = training_api
    dataset_id = _dataset(client)
    script_id = _script(client, GOOD, name="cancellable-trainer")
    started = Event()
    release = Event()

    def slow_execute(*args, **kwargs):
        started.set()
        release.wait(timeout=3)
        return TrainingExecutionResult(True, type("M", (), {"predict": lambda self, X: [0.0] * len(X)})(), ["completed work"])

    monkeypatch.setattr(TrainingScriptExecutor, "execute", slow_execute)
    response = client.post(
        "/api/training-jobs",
        json={"model_type": "electric_load", "dataset_id": dataset_id, "train_script_id": script_id},
    )
    job_id = response.json()["id"]
    assert started.wait(timeout=2)
    cancelled = client.post(f"/api/training-jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert "取消" in " ".join(cancelled.json()["logs"])
    release.set()
    job = _wait_for(client, job_id, "CANCELLED")
    assert job["finished_at"]
    with factory() as session:
        assert session.scalar(select(ModelVersionORM).where(ModelVersionORM.training_job_id == job_id)) is None


def test_success_can_use_existing_preprocessing_component(training_api):
    client, _ = training_api
    dataset_id = _dataset(client, split=False)
    preprocess_id = _script(
        client,
        b'''import pandas as pd
class Preprocessor:
    def fit(self, df, config):
        return self
    def transform(self, df, config):
        result = df.copy()
        result["feature"] = result["feature"].astype(float)
        return result
''',
        name="job-preprocessor",
        script_type="preprocessor",
    )
    preprocessing = client.post(
        "/api/preprocessing-tasks",
        json={"model_type": "electric_load", "dataset_id": dataset_id, "preprocess_script_id": preprocess_id},
    )
    assert preprocessing.status_code == 201, preprocessing.text
    split = client.post(
        f"/api/datasets/{dataset_id}/split",
        json={"preprocessing_task_id": preprocessing.json()["id"]},
    )
    assert split.status_code == 201, split.text
    train_id = _script(client, GOOD, name="job-trainer-with-preprocessing")
    response = client.post(
        "/api/training-jobs",
        json={
            "model_type": "electric_load",
            "dataset_id": dataset_id,
            "preprocess_script_id": preprocess_id,
            "train_script_id": train_id,
        },
    )
    assert response.status_code == 201, response.text
    job = _wait_for(client, response.json()["id"], "SUCCEEDED")
    assert "预处理状态" in " ".join(job["logs"])
    assert job["train_row_count"] == 4
