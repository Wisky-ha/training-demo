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
        assert dataset.numeric_columns == ["temperature", "load"]
        assert dataset.time_range == {
            "start": "2024-01-01T00:00:00",
            "end": "2024-01-01T02:00:00",
        }
        assert dataset.time_parse["success"] is True
        assert dataset.summary["row_count"] == 3
        assert body["file_storage"]["size_bytes"] == len(csv)
        assert body["file_storage"]["checksum_sha256"] == artifact.checksum_sha256


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


def test_upload_rejects_missing_extension_bad_encoding_and_malformed_csv(dataset_api):
    client = dataset_api[0]
    valid = b"time,x,target\n2024-01-01,1,2\n2024-01-02,2,3\n"

    response = upload(client, valid, "load")
    assert response.status_code == 415
    assert response.json()["errors"][0]["code"] == "UNSUPPORTED_DATASET_FILE"

    response = upload(client, b"\xfftime,x,target\n2024-01-01,1,2\n", "load.csv")
    assert response.status_code == 400
    assert response.json()["errors"][0]["code"] == "ENCODING_INVALID"

    response = upload(client, b'time,x,target\n"2024-01-01,1,2\n', "load.csv")
    assert response.status_code == 400
    assert response.json()["errors"][0]["code"] == "CSV_FORMAT_INVALID"


def test_upload_rejects_inconsistent_csv_row_width(dataset_api):
    response = upload(
        dataset_api[0],
        b"time,x,target\n2024-01-01,1,2,extra\n2024-01-02,2,3\n",
    )
    assert response.status_code == 400
    assert response.json()["errors"][0]["code"] == "COLUMN_COUNT_MISMATCH"


def test_upload_bounds_dataset_size_and_keeps_preview_json_safe(dataset_api):
    client, _, _ = dataset_api
    # The test app's default limit is intentionally overridden on its state;
    # the endpoint reads at most limit + 1 bytes.
    client.app.state.settings.max_dataset_size_bytes = 30
    response = upload(client, b"time,x,target\n2024-01-01,1,2\n2024-01-02,2,3\n")
    assert response.status_code == 400
    assert response.json()["errors"][0]["code"] == "FILE_TOO_LARGE"

    client.app.state.settings.max_dataset_size_bytes = 50 * 1024 * 1024
    response = upload(
        client,
        b"time,x,target\n2024-01-01,Infinity,1\n2024-01-02,NaN,2\n",
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["preview_rows"][0]["x"] == "Infinity"
    assert body["preview_rows"][1]["x"] is None


def test_two_rows_are_the_smallest_valid_80_20_split(dataset_api):
    response = upload(
        dataset_api[0],
        b"time,feature,target\n2024-01-02,1,10\n2024-01-01,2,20\n",
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["row_count"] == 2
    assert body["validation"]["valid"] is True
    assert body["validation"]["checks"]["split"]["train_row_count"] == 1
    assert body["validation"]["checks"]["split"]["test_row_count"] == 1
    assert body["time_parse"]["is_sorted"] is False
    assert body["time_parse"]["out_of_order_count"] == 1
    assert body["validation"]["warnings"][0]["code"] == "TIME_NOT_SORTED"


def test_upload_rejects_a_dataset_without_a_feature_column(dataset_api):
    response = upload(dataset_api[0], b"time,target\n2024-01-01,1\n2024-01-02,2\n")

    assert response.status_code == 400
    assert response.json()["errors"][0]["code"] == "COLUMN_COUNT_TOO_SMALL"
    assert response.json()["errors"][0]["field"] == "columns"


def test_upload_rejects_too_few_rows_for_train_and_test(dataset_api):
    response = upload(dataset_api[0], b"time,feature,target\n2024-01-01,1,1\n")

    assert response.status_code == 400
    assert response.json()["errors"][0]["code"] == "INSUFFICIENT_ROWS"
    assert "训练集和测试集" in response.json()["detail"]


def test_upload_rejects_missing_time_and_target_values(dataset_api):
    response = upload(
        dataset_api[0],
        b"time,feature,target\n,1,10\n2024-01-02,2,\n",
    )

    assert response.status_code == 400
    codes = {error["code"] for error in response.json()["errors"]}
    assert {"TIME_VALUE_MISSING", "TARGET_VALUE_MISSING"} <= codes
    assert all(error["column"] in {"time", "target"} for error in response.json()["errors"])


def test_upload_rejects_non_numeric_or_non_finite_target_values(dataset_api):
    response = upload(
        dataset_api[0],
        b"time,feature,target\n2024-01-01,1,10\n2024-01-02,2,nope\n",
    )

    assert response.status_code == 400
    assert response.json()["errors"][0]["code"] == "TARGET_VALUE_INVALID"
    assert response.json()["errors"][0]["column"] == "target"

    response = upload(
        dataset_api[0],
        b"time,feature,target\n2024-01-01,1,10\n2024-01-02,2,inf\n",
    )
    assert response.status_code == 400
    assert response.json()["errors"][0]["code"] == "TARGET_VALUE_INVALID"


def test_upload_rejects_a_feature_column_that_is_entirely_missing(dataset_api):
    response = upload(
        dataset_api[0],
        b"time,feature,target\n2024-01-01,,10\n2024-01-02,,20\n",
    )

    assert response.status_code == 400
    assert response.json()["errors"][0]["code"] == "FEATURE_VALUES_EMPTY"
    assert response.json()["errors"][0]["column"] == "feature"


def test_upload_rejects_duplicate_times_and_reports_time_order(dataset_api):
    response = upload(
        dataset_api[0],
        b"time,feature,target\n2024-01-01,1,10\n2024-01-01,2,20\n",
    )

    assert response.status_code == 400
    error = response.json()["errors"][0]
    assert error["code"] == "TIME_DUPLICATE"
    assert error["field"] == "time"
