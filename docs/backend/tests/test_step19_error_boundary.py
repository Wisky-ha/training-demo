"""Step 19 coverage for the common safe error boundary."""

from __future__ import annotations

import base64
from pathlib import Path

import cloudpickle
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.core.config import Settings
from backend.app.db.models import FileArtifactORM, ModelTypeORM, ModelVersionORM
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import HealthStatus, ModelType
from backend.app.main import create_app
from backend.app.services.mcp import MCPModelService


class PredictableModel:
    def predict(self, values):
        return (values["feature"] * 2).tolist()


class FailingModel:
    def predict(self, values):
        raise RuntimeError("private prediction diagnostic")


def _model_payload(version: str, model: object = PredictableModel()) -> dict:
    return {
        "model_type": "electric_load",
        "version": version,
        "model_content_base64": base64.b64encode(cloudpickle.dumps(model)).decode(),
        "time_column": "time",
        "feature_columns": ["feature"],
        "target_column": "target",
        "input_schema": {
            "columns": ["time", "feature", "target"],
            "required_columns": ["time", "feature"],
            "column_types": {
                "time": "datetime", "feature": "number", "target": "number"
            },
            "time_column": "time",
            "target_column": "target",
            "extra_columns": "reject",
        },
    }


def _fixture(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'step19.db').as_posix()}",
        storage_root=tmp_path / "storage",
    )
    engine = initialize_database(settings=settings)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add(ModelTypeORM(code=ModelType.ELECTRIC_LOAD, name="电力负荷"))
        session.commit()
    app = create_app(settings)
    return engine, factory, TestClient(app)


def _published(client: TestClient, version: str, model: object = PredictableModel()) -> dict:
    saved = client.post("/api/models", json=_model_payload(version, model)).json()
    response = client.post(f"/api/models/{saved['id']}/publish", json={"confirm": True})
    assert response.status_code == 200, response.text
    return saved


def _predict(client: TestClient, **overrides):
    body = {"model_type": "electric_load", "data": [{"time": "2026-01-01", "feature": 2}]}
    body.update(overrides)
    return client.post("/api/mcp/predict", json=body)


def test_http_and_validation_errors_share_the_safe_envelope(tmp_path):
    engine, _factory, client = _fixture(tmp_path)
    try:
        with client:
            response = client.get("/api/training-jobs/does-not-exist")
            body = response.json()
            assert response.status_code == 404
            assert body["success"] is False
            assert body["error_code"] == "TRAINING_JOB_NOT_FOUND"
            assert isinstance(body["details"], dict)
            assert body["detail"]["code"] == body["error_code"]
            assert "traceback" not in str(body).lower()

            invalid = client.post(
                "/api/mcp/predict",
                json={"model_type": "not-a-model", "data": []},
            )
            invalid_body = invalid.json()
            assert invalid_body["success"] is False
            assert invalid_body["error_code"] == "MODEL_TYPE_NOT_FOUND"
            assert invalid_body["details"]["validation_errors"]
            assert "traceback" not in str(invalid_body).lower()
    finally:
        engine.dispose()


def test_prediction_codes_keep_field_details_and_do_not_leak_failures(tmp_path):
    engine, _factory, client = _fixture(tmp_path)
    try:
        with client:
            _published(client, "v1")
            cases = [
                ({"data": [{"feature": 1}]}, "MISSING_TIME_FIELD", "field"),
                ({"data": [{"time": "2026-01-01"}]}, "MISSING_FEATURE", "fields"),
                (
                    {"data": [{"time": "2026-01-01", "feature": "bad"}]},
                    "INVALID_FIELD_TYPE",
                    "expected",
                ),
                (
                    {"data": [{"time": "not-a-time", "feature": 1}]},
                    "INVALID_TIME_FORMAT",
                    "value",
                ),
            ]
            for overrides, code, detail_key in cases:
                response = _predict(client, **overrides)
                body = response.json()
                assert body["success"] is False
                assert body["error_code"] == code
                assert detail_key in body["details"]
                assert "Traceback" not in str(body)
    finally:
        engine.dispose()


def test_model_preprocess_and_prediction_failures_are_safe(tmp_path, monkeypatch):
    engine, factory, client = _fixture(tmp_path)
    try:
        with client:
            saved = _published(client, "v1")
            with factory() as session:
                artifact = session.get(FileArtifactORM, saved["model_artifact_id"])
                Path(client.app.state.settings.storage_root, artifact.relative_path).write_bytes(
                    b"not a model"
                )
            loaded = _predict(client)
            assert loaded.status_code == 500
            assert loaded.json()["error_code"] == "MODEL_LOAD_FAILED"
            assert "Traceback" not in str(loaded.json())

            _published(client, "v2", FailingModel())
            failed = _predict(client)
            assert failed.status_code == 500
            assert failed.json()["error_code"] == "PREDICTION_FAILED"
            assert "private prediction diagnostic" not in str(failed.json())

            def fail_preprocess(*args, **kwargs):
                raise RuntimeError("private preprocessing diagnostic")

            monkeypatch.setattr(MCPModelService, "_transform_with_version_state", fail_preprocess)
            preprocessed = _predict(client)
            assert preprocessed.status_code == 500
            assert preprocessed.json()["error_code"] == "PREPROCESS_FAILED"
            assert "private preprocessing diagnostic" not in str(preprocessed.json())
    finally:
        engine.dispose()


def test_no_healthy_backup_uses_stable_code_and_details(tmp_path):
    engine, factory, client = _fixture(tmp_path)
    try:
        with client:
            first = _published(client, "v1")
            with factory() as session:
                baseline = session.scalar(select(ModelVersionORM).where(
                    ModelVersionORM.model_type == ModelType.ELECTRIC_LOAD,
                    ModelVersionORM.version == "v0-baseline",
                ))
                baseline.health_status = HealthStatus.ABNORMAL
                session.commit()
            response = client.post(
                "/api/mcp/mark_model_abnormal",
                json={"model_type": "electric_load", "model_version": first["version"]},
            )
            body = response.json()
            assert response.status_code == 409
            assert body["success"] is False
            assert body["error_code"] == "NO_HEALTHY_BACKUP"
            assert body["details"]["rollback_record_id"]
            assert "traceback" not in str(body).lower()
    finally:
        engine.dispose()
