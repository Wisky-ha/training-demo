from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.core.config import Settings
from backend.app.db.models import DatasetORM, FileArtifactORM
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.main import create_app


@pytest.fixture
def dataset_api(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'datasets.db').as_posix()}",
        storage_root=tmp_path / "storage",
    )
    engine = initialize_database(settings=settings)
    factory = create_session_factory(engine)
    application = create_app(settings)
    with TestClient(application) as client:
        yield client, factory, tmp_path / "storage"
    engine.dispose()


def upload(client, content, filename="load.csv", content_type="text/csv"):
    return client.post(
        "/api/datasets/upload",
        files={"file": (filename, content, content_type)},
    )


def test_valid_csv_returns_roles_types_missing_preview_summary_and_persists(dataset_api):
    client, factory, root = dataset_api
    csv = (
        "time,temperature,load\n"
        "2024-01-01T00:00:00,20.5,100\n"
        "2024-01-01T01:00:00,,101\n"
        "2024-01-01T02:00:00,21.5,102\n"
    ).encode()

    response = upload(client, csv)

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["row_count"] == 3
    assert body["column_count"] == 3
    assert body["time_column"] == "time"
    assert body["feature_columns"] == ["temperature"]
    assert body["target_column"] == "load"
    assert [column["name"] for column in body["columns"]] == ["time", "temperature", "load"]
    assert [column["role"] for column in body["columns"]] == ["time", "feature", "target"]
    assert body["time_parse"]["success"] is True
    assert body["numeric_columns"] == ["temperature", "load"]
    assert body["column_types"] == {
        "time": "datetime",
        "temperature": "number",
        "load": "number",
    }
    assert body["missing_value_counts"] == {"time": 0, "temperature": 1, "load": 0}
    assert body["missing_values"]["temperature"]["missing_ratio"] == 1 / 3
    assert len(body["preview_rows"]) == 3
    assert body["preview_rows"][1]["temperature"] is None
    assert body["summary"]["columns"]["load"]["mean"] == 101.0
    assert body["dataset_id"] == body["id"]
    assert (root / Path(body["file_path"])).read_bytes() == csv

    with factory() as session:
        dataset = session.get(DatasetORM, body["dataset_id"])
        assert dataset is not None
        assert dataset.file_path == body["file_path"]
        artifact = session.scalar(
            select(FileArtifactORM).where(
                FileArtifactORM.artifact_type == "dataset",
                FileArtifactORM.artifact_id == body["dataset_id"],
            )
        )
        assert artifact is not None
        assert artifact.relative_path == dataset.file_path


def test_upload_rejects_fewer_than_three_columns(dataset_api):
    response = upload(dataset_api[0], b"time,target\n2024-01-01,1\n")
    assert response.status_code == 400
    assert "至少需要 3 列" in response.json()["detail"]


def test_upload_rejects_unparseable_time(dataset_api):
    response = upload(
        dataset_api[0], b"time,x,target\nnot-a-date,1,2\n"
    )
    assert response.status_code == 400
    assert "无法解析" in response.json()["detail"]


def test_upload_rejects_non_csv_and_empty_content(dataset_api):
    response = upload(dataset_api[0], b"time,x,target\n2024-01-01,1,2\n", "load.txt")
    assert response.status_code == 415
    response = upload(dataset_api[0], b"", "load.csv")
    assert response.status_code == 400
    assert "为空" in response.json()["detail"]
