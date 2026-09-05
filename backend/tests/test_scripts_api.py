import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.core.config import Settings
from backend.app.db.models import FileArtifactORM, ModelTypeORM, ScriptORM
from backend.app.db.session import create_session_factory, initialize_database, get_session
from backend.app.domain.enums import ModelType, ScriptStatus, ScriptType
from backend.app.main import create_app
from backend.app.storage.local import FileStorageService


@pytest.fixture
def api_context(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'api.db').as_posix()}",
        storage_root=tmp_path / "storage",
    )
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
    )
    initialize_database(engine)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add_all(
            [
                ModelTypeORM(code=ModelType.ELECTRIC_LOAD, name="电力负荷"),
                ModelTypeORM(code=ModelType.INTEGRATED_ENERGY, name="综合能耗"),
            ]
        )
        session.commit()

    application = create_app(settings)

    def override_session():
        with factory() as session:
            try:
                yield session
            except Exception:
                session.rollback()
                raise

    application.dependency_overrides[get_session] = override_session
    with TestClient(application) as client:
        yield client, factory, tmp_path / "storage"
    application.dependency_overrides.clear()
    engine.dispose()


def _upload(client, *, filename="train.py", name="trainer", version="v1", script_type="trainer", models=None, source=b"pass"):
    return client.post(
        "/api/scripts/upload",
        data={
            "name": name,
            "version": version,
            "script_type": script_type,
            "supported_model_types": json.dumps(
                ["electric_load"] if models is None else models
            ),
        },
        files={"file": (filename, source, "text/x-python")},
    )


def test_upload_returns_display_metadata_and_stores_source(api_context):
    client, factory, storage_root = api_context

    response = _upload(client, source=b"def train():\n    return 1\n")

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "trainer"
    assert body["version"] == "v1"
    assert body["script_type"] == "trainer"
    assert body["supported_model_types"] == ["electric_load"]
    assert body["status"] == "ENABLED"
    assert body["uploaded_at"] == body["created_at"]
    assert body["source_code"] == "def train():\n    return 1\n"
    assert (storage_root / "script" / f"{body['id']}.py").read_bytes() == b"def train():\n    return 1\n"
    with factory() as session:
        record = session.get(ScriptORM, body["id"])
        assert record is not None
        assert record.source_code == body["source_code"]


def test_list_defaults_to_enabled_and_filters_compatibility_and_type(api_context):
    client, factory, _ = api_context
    with factory() as session:
        electric = session.get(ModelTypeORM, "missing")
        session.add_all(
            [
                ScriptORM(name="electric", script_type=ScriptType.TRAINER, version="v1", source_code="pass", status=ScriptStatus.ENABLED),
                ScriptORM(name="disabled", script_type=ScriptType.TRAINER, version="v1", source_code="pass", status=ScriptStatus.DISABLED),
            ]
        )
        # Attach model types through the public relationship after both rows exist.
        session.flush()
        scripts = session.query(ScriptORM).all()
        load = session.query(ModelTypeORM).filter_by(code=ModelType.ELECTRIC_LOAD).one()
        energy = session.query(ModelTypeORM).filter_by(code=ModelType.INTEGRATED_ENERGY).one()
        scripts[0].supported_model_types = [load]
        scripts[1].supported_model_types = [load]
        other = ScriptORM(name="other", script_type=ScriptType.PREPROCESSOR, version="v1", source_code="pass", status=ScriptStatus.ENABLED, supported_model_types=[energy])
        session.add(other)
        session.commit()

    response = client.get("/api/scripts", params={"model_type": "electric_load", "script_type": "trainer"})
    assert response.status_code == 200
    body = response.json()
    assert [item["name"] for item in body["items"]] == ["electric"]
    assert body["pagination"]["total"] == 1


def test_list_explicit_disabled_status_is_available_for_library_management(api_context):
    client, factory, _ = api_context
    with factory() as session:
        load = session.query(ModelTypeORM).filter_by(code=ModelType.ELECTRIC_LOAD).one()
        item = ScriptORM(name="disabled", script_type=ScriptType.PREPROCESSOR, version="v1", source_code="pass", status=ScriptStatus.DISABLED, supported_model_types=[load])
        session.add(item)
        session.commit()

    response = client.get("/api/scripts", params={"model_type": "electric_load", "status": "disabled"})
    assert response.status_code == 200
    assert [item["name"] for item in response.json()["items"]] == ["disabled"]


@pytest.mark.parametrize(
    "kwargs, status_code",
    [
        ({"filename": "train.txt"}, 400),
        ({"filename": "../train.py"}, 400),
        ({"version": ""}, 422),
        ({"script_type": "unknown"}, 422),
        ({"models": ["not-a-model"]}, 422),
        ({"models": []}, 422),
    ],
)
def test_upload_rejects_bad_file_or_metadata(api_context, kwargs, status_code):
    client, _, _ = api_context
    response = _upload(client, **kwargs)
    assert response.status_code == status_code


def test_upload_rejects_python_syntax_errors(api_context):
    client, _, _ = api_context
    response = _upload(client, source=b"def broken(:\n    pass\n")
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_SCRIPT_FILE"


def test_upload_requires_all_metadata_fields(api_context):
    client, _, _ = api_context
    response = client.post(
        "/api/scripts/upload",
        data={"name": "trainer", "script_type": "trainer", "supported_model_types": "[]"},
        files={"file": ("train.py", b"pass", "text/x-python")},
    )
    assert response.status_code == 422


def test_storage_failure_rolls_back_database_and_file(api_context, monkeypatch):
    client, factory, storage_root = api_context
    original_save = FileStorageService.save_script_source

    def save_then_fail(storage, script_id, source):
        original_save(storage, script_id, source)
        raise OSError("disk full")

    monkeypatch.setattr(FileStorageService, "save_script_source", save_then_fail)
    response = _upload(client)

    assert response.status_code == 500
    with factory() as session:
        assert session.query(ScriptORM).count() == 0
    assert not list((storage_root / "script").glob("*.py"))


def test_duplicate_script_version_is_rejected_without_extra_file(api_context):
    client, factory, storage_root = api_context
    first = _upload(client)
    second = _upload(client, source=b"different")

    assert first.status_code == 201
    assert second.status_code == 409
    with factory() as session:
        assert session.query(ScriptORM).filter_by(name="trainer", version="v1").count() == 1
    assert len(list((storage_root / "script").glob("*.py"))) == 1


def test_upload_rejects_unsafe_metadata_name(api_context):
    client, _, _ = api_context
    response = _upload(client, name="../outside")
    assert response.status_code == 422


def test_upload_generates_incrementing_versions_and_keeps_previous_source(api_context):
    client, factory, storage_root = api_context
    first = client.post(
        "/api/scripts/upload",
        data={
            "name": "versioned-trainer",
            "script_type": "trainer",
            "supported_model_types": json.dumps(["electric_load"]),
        },
        files={"file": ("train.py", b"VERSION = 1\n", "text/x-python")},
    )
    second = client.post(
        "/api/scripts/upload",
        data={
            "name": "versioned-trainer",
            "script_type": "trainer",
            "supported_model_types": json.dumps(["electric_load"]),
        },
        files={"file": ("train.py", b"VERSION = 2\n", "text/x-python")},
    )

    assert first.status_code == 201
    assert second.status_code == 201
    first_body, second_body = first.json(), second.json()
    assert first_body["version"] == "v1"
    assert second_body["version"] == "v2"
    assert first_body["source_code"] == "VERSION = 1\n"
    assert second_body["source_code"] == "VERSION = 2\n"
    assert first_body["id"] != second_body["id"]
    with factory() as session:
        assert session.query(ScriptORM).filter_by(name="versioned-trainer").count() == 2
        artifacts = session.query(FileArtifactORM).filter_by(artifact_type="script").all()
        assert {item.artifact_id for item in artifacts} == {first_body["id"], second_body["id"]}
    assert len(list((storage_root / "script").glob("*.py"))) == 2


def test_script_detail_and_enable_disable_endpoints_preserve_source(api_context):
    client, _, _ = api_context
    uploaded = _upload(client, source=b"def train():\n    return 42\n").json()
    script_id = uploaded["id"]

    disabled = client.post(f"/api/scripts/{script_id}/disable")
    assert disabled.status_code == 200
    assert disabled.json()["status"] == "DISABLED"
    assert disabled.json()["source_code"] == uploaded["source_code"]

    detail = client.get(f"/api/scripts/{script_id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == script_id
    assert detail.json()["status"] == "DISABLED"
    assert detail.json()["source_code"] == uploaded["source_code"]

    enabled = client.post(f"/api/scripts/{script_id}/enable")
    assert enabled.status_code == 200
    assert enabled.json()["status"] == "ENABLED"
    assert client.get("/api/scripts", params={"status": "disabled"}).json()["items"] == []


def test_script_management_returns_structured_not_found_error(api_context):
    client, _, _ = api_context
    response = client.get("/api/scripts/not-found")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "SCRIPT_NOT_FOUND"
