"""Tests for explicit evaluation-page save/delete artifact decisions."""

from __future__ import annotations

import base64
import json

from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.core.config import Settings
from backend.app.db.models import (
    AuditEventORM,
    DatasetORM,
    FileArtifactORM,
    ModelTypeORM,
    ModelVersionORM,
    ScriptORM,
    TrainingJobORM,
)
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import DatasetStatus, ModelType
from backend.app.main import create_app


def _context(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'decisions.db').as_posix()}",
        storage_root=tmp_path / "storage",
    )
    engine = initialize_database(settings=settings)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add(ModelTypeORM(code=ModelType.ELECTRIC_LOAD, name="电力负荷"))
        session.commit()
    return settings, engine, factory


def _upload_script(client: TestClient, *, name: str = "trainer") -> dict:
    response = client.post(
        "/api/scripts/upload",
        data={
            "name": name,
            "script_type": "trainer",
            "supported_model_types": json.dumps(["electric_load"]),
        },
        files={"file": (f"{name}.py", b"def train():\n    pass\n", "text/x-python")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_deleting_an_unpublished_model_removes_its_file_and_metadata(tmp_path):
    settings, engine, factory = _context(tmp_path)
    try:
        with TestClient(create_app(settings)) as client:
            response = client.post(
                "/api/models",
                json={
                    "model_type": "electric_load",
                    "version": "v1",
                    "model_content_base64": base64.b64encode(b"candidate-model").decode(),
                    "health_status": "HEALTHY",
                },
            )
            assert response.status_code == 201, response.text
            model = response.json()
            model_id = model["id"]
            model_path = settings.storage_root / model["model_path"]
            assert model_path.is_file()

            deleted = client.delete(f"/api/models/{model_id}")
            assert deleted.status_code == 200, deleted.text
            assert deleted.json() == {
                "operation": "delete",
                "model_version_id": model_id,
                "deleted": True,
                "model_artifact_deleted": True,
            }
            assert not model_path.exists()

            with factory() as session:
                assert session.get(ModelVersionORM, model_id) is None
                assert session.scalar(
                    select(FileArtifactORM).where(FileArtifactORM.artifact_id == model_id)
                ) is None
                event = session.scalar(
                    select(AuditEventORM)
                    .where(AuditEventORM.event_type == "MODEL_DELETED")
                    .order_by(AuditEventORM.occurred_at.desc())
                )
                assert event is not None
                assert event.object_id == model_id
    finally:
        engine.dispose()


def test_deleting_a_script_removes_the_source_file_and_row_when_unreferenced(tmp_path):
    settings, engine, factory = _context(tmp_path)
    try:
        with TestClient(create_app(settings)) as client:
            script = _upload_script(client)
            script_id = script["id"]
            source_path = settings.script_storage_dir or settings.storage_root
            path = source_path / "script" / f"{script_id}.py"
            assert path.is_file()

            deleted = client.delete(f"/api/scripts/{script_id}")
            assert deleted.status_code == 200, deleted.text
            assert deleted.json() == {
                "operation": "delete",
                "script_id": script_id,
                "deleted": True,
                "source_file_deleted": True,
            }
            assert not path.exists()

            with factory() as session:
                assert session.get(ScriptORM, script_id) is None
                assert session.scalar(
                    select(FileArtifactORM).where(FileArtifactORM.artifact_id == script_id)
                ) is None
    finally:
        engine.dispose()


def test_deleting_workflow_scripts_removes_files_but_keeps_referenced_history(tmp_path):
    settings, engine, factory = _context(tmp_path)
    try:
        with TestClient(create_app(settings)) as client:
            script = _upload_script(client)
            script_id = script["id"]
            source_path = settings.script_storage_dir or settings.storage_root
            path = source_path / "script" / f"{script_id}.py"

            # A completed workflow still owns the script through its job.  The
            # source file can be discarded, but the immutable history row must
            # remain so old job/model references are not broken.
            with factory() as session:
                dataset = DatasetORM(
                    file_name="workflow.csv",
                    file_path=None,
                    status=DatasetStatus.PARSED,
                    row_count=2,
                    columns=["time", "feature", "target"],
                    time_column="time",
                    feature_columns=["feature"],
                    target_column="target",
                    column_types={"time": "datetime", "feature": "number", "target": "number"},
                    missing_value_counts={},
                    preview_rows=[],
                    numeric_columns=["feature", "target"],
                    time_parse={},
                    time_range={},
                    summary={},
                )
                session.add(dataset)
                session.flush()
                session.add(TrainingJobORM(
                    model_type=ModelType.ELECTRIC_LOAD,
                    dataset_id=dataset.id,
                    train_script_id=script_id,
                    split_strategy="time_ordered",
                    split_ratio=0.8,
                    test_ratio=0.2,
                ))
                session.commit()

            deleted = client.delete(f"/api/scripts/{script_id}")
            assert deleted.status_code == 200, deleted.text
            assert deleted.json() == {
                "operation": "delete",
                "script_id": script_id,
                "deleted": False,
                "source_file_deleted": True,
            }
            assert not path.exists()
            with factory() as session:
                assert session.get(ScriptORM, script_id).status.value == "DISABLED"
    finally:
        engine.dispose()
