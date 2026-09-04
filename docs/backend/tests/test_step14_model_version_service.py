"""Step 14 acceptance coverage for immutable model-version persistence."""

from __future__ import annotations

import base64
import json
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from backend.app.core.config import Settings
from backend.app.db.models import FileArtifactORM, ModelTypeORM, ModelVersionORM
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import ModelType
from backend.app.main import create_app
from backend.app.storage import ArtifactType


@pytest.fixture
def api(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'step14.db').as_posix()}",
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


def upload_script(client: TestClient, source: bytes, name: str, kind: str = "trainer") -> str:
    response = client.post(
        "/api/scripts/upload",
        data={
            "name": name,
            "script_type": kind,
            "supported_model_types": json.dumps(["electric_load"]),
        },
        files={"file": (f"{name}.py", source, "text/x-python")},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def make_dataset(client: TestClient) -> str:
    rows = b"\n".join(
        f"2024-01-{day:02d},{day},{day * 10}".encode() for day in range(1, 11)
    )
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("load.csv", b"time,feature,target\n" + rows + b"\n", "text/csv")},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def wait_for_terminal(client: TestClient, job_id: str) -> dict:
    for _ in range(250):
        body = client.get(f"/api/training-jobs/{job_id}").json()
        if body["status"] in {"SUCCEEDED", "FAILED"}:
            return body
        time.sleep(0.01)
    pytest.fail("training job did not finish")


def create_training_job(client: TestClient, dataset_id: str, trainer_id: str, **extra):
    response = client.post(
        "/api/training-jobs",
        json={
            "model_type": "electric_load",
            "dataset_id": dataset_id,
            "train_script_id": trainer_id,
            **extra,
        },
    )
    assert response.status_code == 201, response.text
    return wait_for_terminal(client, response.json()["id"])


def test_successful_training_persists_independent_version_and_all_step14_snapshots(api):
    client, factory = api
    dataset_id = make_dataset(client)
    preprocessor_id = upload_script(
        client,
        b'''class Preprocessor:
    def fit(self, df, config):
        self.offset = 1
        return self
    def transform(self, df, config):
        result = df.copy()
        result["feature"] = result["feature"].astype(float) + self.offset
        return result
''',
        "normalizer",
        "preprocessor",
    )
    task = client.post(
        "/api/preprocessing-tasks",
        json={
            "model_type": "electric_load",
            "dataset_id": dataset_id,
            "preprocess_script_id": preprocessor_id,
            "mode": "use",
            "config": {"fit_marker": "snapshot"},
        },
    )
    assert task.status_code == 201, task.text
    split = client.post(
        f"/api/datasets/{dataset_id}/split",
        json={"preprocessing_task_id": task.json()["id"]},
    )
    assert split.status_code == 201, split.text

    trainer_source = b'''class Model:
    def predict(self, X):
        return [float(value) for value in X["feature"]]
def train(X_train, y_train, X_test, y_test, config):
    return Model()
'''
    trainer_id = upload_script(client, trainer_source, "trainer")
    first = create_training_job(
        client, dataset_id, trainer_id, preprocess_script_id=preprocessor_id,
        preprocessing_task_id=task.json()["id"],
    )
    assert first["status"] == "SUCCEEDED", first
    detail = client.get(f"/api/models/{first['model_version_id']}")
    assert detail.status_code == 200, detail.text
    version = detail.json()
    assert version["version"] == "v1"
    assert version["train_script_source"] == trainer_source.decode()
    assert version["preprocess_script_source"]
    assert version["preprocessor_state"]["fitted"] is True
    assert version["input_schema"]["required_columns"] == ["time", "feature"]
    assert version["feature_columns"] == ["feature"]
    assert version["split_strategy"] == "time_ordered"
    assert version["split_ratio"] == pytest.approx(0.8)
    assert version["test_ratio"] == pytest.approx(0.2)
    assert version["train_data_summary"]["row_count"] == 8
    assert version["test_data_summary"]["row_count"] == 2
    assert version["metrics"]["sample_count"] == 2
    assert version["model_artifact_id"] == version["model_file_metadata"]["id"]
    assert version["model_file_metadata"]["artifact_type"] == ArtifactType.MODEL.value
    assert version["model_file_metadata"]["size_bytes"] > 0
    assert version["preprocessor_artifact_id"] == version["preprocessor_file_metadata"]["id"]
    assert version["preprocessor_file_metadata"]["artifact_type"] == ArtifactType.PREPROCESSOR.value

    # A second successful run gets a new immutable ID/version and leaves the
    # first artifact and snapshots untouched.
    second = create_training_job(
        client, dataset_id, trainer_id, preprocess_script_id=preprocessor_id,
        preprocessing_task_id=task.json()["id"],
    )
    assert second["status"] == "SUCCEEDED"
    assert second["model_version_id"] != first["model_version_id"]
    assert client.get(f"/api/models/{second['model_version_id']}").json()["version"] == "v2"
    with factory() as session:
        versions = list(session.scalars(select(ModelVersionORM).where(
            ModelVersionORM.model_type == ModelType.ELECTRIC_LOAD,
            ModelVersionORM.is_baseline.is_(False),
        )))
        assert {item.version for item in versions} == {"v1", "v2"}
        artifacts = list(session.scalars(select(FileArtifactORM).where(
            FileArtifactORM.artifact_type == ArtifactType.MODEL.value,
        )))
        assert {item.artifact_id for item in artifacts} == {first["model_version_id"], second["model_version_id"]}
        assert all(item.size_bytes > 0 and len(item.checksum_sha256) == 64 for item in artifacts)

    # Library lifecycle changes do not rewrite immutable source snapshots.
    with factory() as session:
        from backend.app.db.models import ScriptORM
        script = session.get(ScriptORM, trainer_id)
        script.source_code = "changed after training"
        session.commit()
    assert client.get(f"/api/models/{first['model_version_id']}").json()["train_script_source"] == trainer_source.decode()


def test_failed_training_creates_no_model_version_or_model_artifact(api):
    client, factory = api
    dataset_id = make_dataset(client)
    split = client.post(f"/api/datasets/{dataset_id}/split", json={})
    assert split.status_code == 201
    trainer_id = upload_script(
        client,
        b"def train(X_train, y_train, X_test, y_test, config):\n    raise RuntimeError('step14 failure')\n",
        "failing",
    )
    job = create_training_job(client, dataset_id, trainer_id)
    assert job["status"] == "FAILED"
    assert job["model_version_id"] is None
    with factory() as session:
        assert session.scalar(select(ModelVersionORM).where(
            ModelVersionORM.training_job_id == job["id"],
        )) is None
        assert session.scalar(select(func.count(FileArtifactORM.id)).where(
            FileArtifactORM.artifact_type == ArtifactType.MODEL.value,
        )) == 0


def test_model_version_number_is_unique_within_model_type(api):
    client, _ = api
    payload = {
        "model_type": "electric_load",
        "version": "v1",
        "model_path": "legacy/v1.joblib",
        "model_content_base64": base64.b64encode(b"artifact").decode(),
    }
    assert client.post("/api/models", json=payload).status_code == 201
    duplicate = client.post("/api/models", json=payload)
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "MODEL_VERSION_ALREADY_EXISTS"
