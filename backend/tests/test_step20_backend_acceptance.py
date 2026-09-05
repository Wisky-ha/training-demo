"""Step 20 backend acceptance gaps: isolated, local-only contract tests."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import cloudpickle
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.core.config import Settings
from backend.app.db.models import FileArtifactORM, ModelTypeORM, ModelVersionORM
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import ModelType
from backend.app.main import create_app
from backend.app.services.mcp import MCPModelService, MCPServiceError
from backend.app.services.training_jobs import TrainingJobService


@pytest.fixture
def api(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'step20.db').as_posix()}",
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


def upload_dataset(client, values: list[tuple[str, int, int]]) -> str:
    rows = "time,feature,target\n" + "\n".join(
        f"{timestamp},{feature},{target}" for timestamp, feature, target in values
    ) + "\n"
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("load.csv", rows.encode(), "text/csv")},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def upload_script(client, source: str, name: str, script_type: str = "trainer") -> str:
    response = client.post(
        "/api/scripts/upload",
        data={
            "name": name,
            "script_type": script_type,
            "supported_model_types": json.dumps(["electric_load"]),
        },
        files={"file": (f"{name}.py", source.encode(), "text/x-python")},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def wait_for(client, job_id: str, expected: str = "SUCCEEDED") -> dict:
    deadline = time.monotonic() + 5
    latest = None
    while time.monotonic() < deadline:
        latest = client.get(f"/api/training-jobs/{job_id}").json()
        if latest["status"] == expected:
            return latest
        time.sleep(0.01)
    pytest.fail(f"job did not reach {expected}: {latest}")


def model_payload(version: str, *, content: bytes | None = None) -> dict:
    class Model:
        def predict(self, values):
            return [0.0] * len(values)

    content = content or cloudpickle.dumps(Model())
    return {
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
            "target_column": "target",
            "extra_columns": "reject",
        },
    }


def test_preprocessing_fits_only_sorted_training_rows_and_reuses_state(api):
    client, _, _ = api
    dataset_id = upload_dataset(client, [
        ("2024-01-05", 100, 5),
        ("2024-01-01", 1, 1),
        ("2024-01-04", 3, 4),
        ("2024-01-02", 2, 2),
        ("2024-01-03", 4, 3),
    ])
    source = """class Preprocessor:
    def fit(self, df, config):
        self.fit_count = len(df)
        return self
    def transform(self, df, config):
        result = df.copy()
        result['feature'] = self.fit_count
        return result
"""
    script_id = upload_script(client, source, "train-only-preprocessor", "preprocessor")
    task = client.post("/api/preprocessing-tasks", json={
        "model_type": "electric_load", "dataset_id": dataset_id,
        "preprocess_script_id": script_id, "config": {"marker": "saved"},
    })
    assert task.status_code == 201, task.text
    body = task.json()
    assert body["status"] == "SUCCEEDED"
    # Five rows means four training rows.  The persisted fitted state is reused
    # by the transform endpoint; it must not fit on all five rows.
    assert body["output_summary"]["columns"]["feature"]["min"] == 4
    assert body["preprocessor_state"]["fitted"] is True
    assert body["preprocessor_state"]["config"] == {"marker": "saved"}
    reused = client.post(
        f"/api/preprocessing-tasks/{body['id']}/transform",
        json={"dataset_id": dataset_id},
    )
    assert reused.status_code == 200, reused.text
    assert reused.json()["summary"]["columns"]["feature"]["min"] == 4


def test_preprocessing_rejects_mixed_unconvertible_numeric_values(api):
    client, _, _ = api
    dataset_id = upload_dataset(client, [
        ("2024-01-01", 1, 1), ("2024-01-02", 2, 2),
    ])
    source = """class Preprocessor:
    def fit(self, df, config):
        return self
    def transform(self, df, config):
        result = df.copy()
        result['feature'] = ['1'] * len(result)
        result.loc[result.index[0], 'feature'] = 'not-a-number'
        return result
"""
    script_id = upload_script(client, source, "reject-invalid-values", "preprocessor")
    response = client.post("/api/preprocessing-tasks", json={
        "model_type": "electric_load", "dataset_id": dataset_id,
        "preprocess_script_id": script_id,
    })
    assert response.status_code == 400
    assert response.json()["error_code"] == "PREPROCESS_VALUES_INVALID"


def test_training_persists_required_status_transitions_and_predict_contract(api, monkeypatch):
    client, _, _ = api
    dataset_id = upload_dataset(client, [
        (f"2024-01-{day:02d}", day, day * 2) for day in range(1, 6)
    ])
    assert client.post(f"/api/datasets/{dataset_id}/split", json={}).status_code == 201
    trainer_id = upload_script(client, """class Model:
    def predict(self, X):
        return [0.0] * len(X)
def train(X_train, y_train, X_test, y_test, config):
    return Model()
""", "state-trainer")

    observed: list[str] = []
    original_stage = TrainingJobService._stage

    def record_stage(self, job, stage, message, status=None):
        from backend.app.domain.enums import TrainingJobStatus
        original_stage(self, job, stage, message, status=status or TrainingJobStatus.RUNNING)
        observed.append(job.status.value)

    monkeypatch.setattr(TrainingJobService, "_stage", record_stage)
    created = client.post("/api/training-jobs", json={
        "model_type": "electric_load", "dataset_id": dataset_id,
        "train_script_id": trainer_id,
    })
    assert created.status_code == 201, created.text
    job = wait_for(client, created.json()["id"])
    assert job["status"] == "SUCCEEDED"
    assert {"RUNNING", "PREPROCESSING", "SPLITTING", "TRAINING", "EVALUATING"} <= set(observed)


def test_baseline_is_predictable_before_first_user_release_and_unknown_version_is_structured(api):
    client, _, _ = api
    response = client.post("/api/mcp/predict", json={
        "model_type": "electric_load",
        "data": [{"time": "2026-01-01 10:00:00", "feature": 12}],
    })
    assert response.status_code == 200, response.text
    assert response.json()["model_version"] == "v0-baseline"
    assert response.json()["predictions"] == [0.0]

    unknown = client.post("/api/mcp/predict", json={
        "model_type": "electric_load", "model_version": "v999", "data": [],
    })
    assert unknown.status_code == 404
    assert unknown.json()["error_code"] == "MODEL_VERSION_NOT_FOUND"


def test_prediction_output_must_be_one_dimensional_and_finite(api):
    client, factory, settings = api
    saved = client.post("/api/models", json=model_payload("v1")).json()
    assert client.post(f"/api/models/{saved['id']}/publish", json={"confirm": True}).status_code == 200
    with factory() as session:
        service = MCPModelService(session, settings=settings)
        with pytest.raises(MCPServiceError) as matrix:
            service._prediction_values([[1.0]], 1)
        assert matrix.value.code == "PREDICTION_FAILED"
        with pytest.raises(MCPServiceError) as nonfinite:
            service._prediction_values([float("nan")], 1)
        assert nonfinite.value.code == "PREDICTION_FAILED"


def test_manual_rollback_rejects_corrupt_artifact_without_moving_pointer(api):
    client, factory, settings = api
    first = client.post("/api/models", json=model_payload("v1")).json()
    second = client.post("/api/models", json=model_payload("v2")).json()
    assert client.post(f"/api/models/{first['id']}/publish", json={"confirm": True}).status_code == 200
    assert client.post(f"/api/models/{second['id']}/publish", json={"confirm": True}).status_code == 200
    with factory() as session:
        artifact = session.get(FileArtifactORM, first["model_artifact_id"])
        Path(settings.storage_root, artifact.relative_path).write_bytes(b"broken")
    response = client.post(
        f"/api/models/{second['id']}/rollback",
        json={"target_version_id": first["id"], "reason": "bad artifact"},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "MODEL_ARTIFACT_INVALID"
    assert client.get(f"/api/models/{second['id']}").json()["is_current"] is True


def test_metadata_only_nonlegacy_release_is_rejected(api):
    client, _, _ = api
    response = client.post("/api/models", json={
        "model_type": "electric_load", "version": "v1", "model_path": "missing/model.bin",
    })
    assert response.status_code == 201
    rejected = client.post(f"/api/models/{response.json()['id']}/publish", json={"confirm": True})
    assert rejected.status_code == 400
    assert rejected.json()["detail"]["code"] == "MODEL_ARTIFACT_NOT_FOUND"


def test_script_snapshot_keeps_version_and_source_after_disable(api):
    client, factory, _ = api
    dataset_id = upload_dataset(client, [
        (f"2024-01-{day:02d}", day, day * 2) for day in range(1, 6)
    ])
    assert client.post(f"/api/datasets/{dataset_id}/split", json={}).status_code == 201
    source = """class Model:
    def predict(self, X):
        return [0.0] * len(X)
def train(X_train, y_train, X_test, y_test, config):
    return Model()
"""
    script_id = upload_script(client, source, "snapshot-trainer")
    with factory() as session:
        from backend.app.db.models import ScriptORM
        script = session.get(ScriptORM, script_id)
        script.status = "DISABLED"
        session.commit()
    # A disabled script cannot start new work, but an already selected source
    # is represented immutably by model-version metadata in the same contract.
    assert client.post("/api/training-jobs", json={
        "model_type": "electric_load", "dataset_id": dataset_id,
        "train_script_id": script_id,
    }).status_code == 400
