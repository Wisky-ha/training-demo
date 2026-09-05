import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.core.config import Settings
from backend.app.db.models import DatasetORM, DatasetSplitORM, ModelTypeORM
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import ModelType
from backend.app.main import create_app


@pytest.fixture
def split_api(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'split.db').as_posix()}",
        storage_root=tmp_path / "storage",
    )
    engine = initialize_database(settings=settings)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add(ModelTypeORM(code=ModelType.ELECTRIC_LOAD, name="电力"))
        session.commit()
    application = create_app(settings)
    with TestClient(application) as client:
        yield client, factory, tmp_path / "storage"
    engine.dispose()


def upload(client, content):
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("load.csv", content, "text/csv")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_split_reuses_completed_preprocessing_output_without_changing_source(split_api):
    client, _, _ = split_api
    dataset = upload(
        client,
        b"time,feature,target\n2024-01-02,2,20\n2024-01-01,1,10\n",
    )
    source = b'''import pandas as pd
class Preprocessor:
    def fit(self, df, config):
        return self
    def transform(self, df, config):
        result = df.copy()
        result["feature"] = result["feature"].astype(float) * 2
        return result
'''
    script = client.post(
        "/api/scripts/upload",
        data={
            "name": "split-preprocessor",
            "script_type": "preprocessor",
            "supported_model_types": json.dumps(["electric_load"]),
        },
        files={"file": ("split-preprocessor.py", source, "text/x-python")},
    )
    assert script.status_code == 201, script.text
    task = client.post(
        "/api/preprocessing-tasks",
        json={
            "model_type": "electric_load",
            "dataset_id": dataset["id"],
            "preprocess_script_id": script.json()["id"],
        },
    )
    assert task.status_code == 201, task.text
    assert task.json()["data_source"] == "preprocessed"

    response = client.post(
        f"/api/datasets/{dataset['id']}/split",
        json={"preprocessing_task_id": task.json()["id"]},
    )
    assert response.status_code == 201, response.text
    assert response.json()["data_source"] == "preprocessed"
    assert response.json()["preprocessing_task_id"] == task.json()["id"]
    assert response.json()["train_time_range"]["start"] == "2024-01-01T00:00:00"


def test_split_sorts_unordered_rows_uses_floor_80_20_and_persists_ranges(split_api):
    client, factory, _ = split_api
    dataset = upload(
        client,
        b"time,feature,target\n"
        b"2024-01-05,5,50\n"
        b"2024-01-01,1,10\n"
        b"2024-01-04,4,40\n"
        b"2024-01-02,2,20\n"
        b"2024-01-03,3,30\n",
    )

    response = client.post(f"/api/datasets/{dataset['id']}/split", json={})

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["dataset_id"] == dataset["id"]
    assert body["total_row_count"] == 5
    assert body["train_row_count"] == 4
    assert body["test_row_count"] == 1
    assert body["train_time_range"] == {
        "start": "2024-01-01T00:00:00",
        "end": "2024-01-04T00:00:00",
    }
    assert body["test_time_range"] == {
        "start": "2024-01-05T00:00:00",
        "end": "2024-01-05T00:00:00",
    }
    assert body["split_ratio"] == 0.8
    assert body["test_ratio"] == 0.2
    assert body["sort_order"] == "ascending"
    assert body["rounding_rule"] == "floor(total_row_count * 0.8)"
    assert body["sorted_before_split"] is True

    with factory() as session:
        saved = session.scalar(
            select(DatasetSplitORM).where(DatasetSplitORM.dataset_id == dataset["id"])
        )
        assert saved is not None
        assert saved.train_row_count == 4
        assert saved.test_time_range["start"] == "2024-01-05T00:00:00"

    queried = client.get(f"/api/datasets/{dataset['id']}/split")
    assert queried.status_code == 200
    assert queried.json()["id"] == body["id"]


def test_split_has_one_train_and_one_test_for_two_rows(split_api):
    client, _, _ = split_api
    dataset = upload(
        client,
        b"time,feature,target\n2024-01-02,2,20\n2024-01-01,1,10\n",
    )
    response = client.post(f"/api/datasets/{dataset['id']}/split")
    assert response.status_code == 201
    assert response.json()["train_row_count"] == 1
    assert response.json()["test_row_count"] == 1


def test_split_ratio_is_fixed_and_request_rejects_override(split_api):
    client, _, _ = split_api
    dataset = upload(
        client,
        b"time,feature,target\n2024-01-01,1,10\n2024-01-02,2,20\n",
    )
    response = client.post(
        f"/api/datasets/{dataset['id']}/split", json={"split_ratio": 0.5}
    )
    assert response.status_code == 422


def test_split_rejects_missing_time_column(split_api):
    client, factory, _ = split_api
    dataset = upload(
        client,
        b"time,feature,target\n2024-01-01,1,10\n2024-01-02,2,20\n",
    )
    with factory() as session:
        saved = session.get(DatasetORM, dataset["id"])
        saved.time_column = "missing_time"
        session.commit()
    response = client.post(f"/api/datasets/{dataset['id']}/split")
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "SPLIT_TIME_COLUMN_MISSING"


def test_split_rejects_missing_or_invalid_time_with_structured_errors(split_api):
    client, _, root = split_api
    dataset = upload(
        client,
        b"time,feature,target\n2024-01-01,1,10\n2024-01-02,2,20\n",
    )
    path = root / Path(dataset["file_path"])
    path.write_bytes(b"time,feature,target\nnot-a-date,1,10\n2024-01-02,2,20\n")
    invalid = client.post(f"/api/datasets/{dataset['id']}/split")
    assert invalid.status_code == 400
    assert invalid.json()["detail"]["code"] == "SPLIT_TIME_PARSE_FAILED"

    path.write_bytes(b"time,feature,target\n,1,10\n2024-01-02,2,20\n")
    missing = client.post(f"/api/datasets/{dataset['id']}/split")
    assert missing.status_code == 400
    assert missing.json()["detail"]["code"] == "SPLIT_TIME_VALUE_MISSING"


def test_split_rejects_duplicate_time_and_repeated_request(split_api):
    client, _, root = split_api
    dataset = upload(
        client,
        b"time,feature,target\n2024-01-01,1,10\n2024-01-02,2,20\n",
    )
    path = root / Path(dataset["file_path"])
    path.write_bytes(b"time,feature,target\n2024-01-01,1,10\n2024-01-01,2,20\n")
    duplicate = client.post(f"/api/datasets/{dataset['id']}/split")
    assert duplicate.status_code == 400
    assert duplicate.json()["detail"]["code"] == "SPLIT_TIME_DUPLICATE"

    path.write_bytes(b"time,feature,target\n2024-01-01,1,10\n2024-01-02,2,20\n")
    assert client.post(f"/api/datasets/{dataset['id']}/split").status_code == 201
    repeated = client.post(f"/api/datasets/{dataset['id']}/split")
    assert repeated.status_code == 409
    assert repeated.json()["detail"]["code"] == "DATASET_SPLIT_ALREADY_EXISTS"


def test_split_reports_missing_dataset_and_small_dataset(split_api):
    client, _, _ = split_api
    missing = client.post("/api/datasets/not-found/split")
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "DATASET_NOT_FOUND"

    response = client.post(
        "/api/datasets/upload",
        files={
            "file": (
                "one.csv",
                b"time,feature,target\n2024-01-01,1,10\n",
                "text/csv",
            )
        },
    )
    assert response.status_code == 400
    assert response.json()["errors"][0]["code"] == "INSUFFICIENT_ROWS"
