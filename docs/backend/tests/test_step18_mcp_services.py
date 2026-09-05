"""Step 18 acceptance tests for the MCP HTTP model tools."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import cloudpickle
import joblib
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.db.models import FileArtifactORM, ModelTypeORM, ModelVersionORM, ScriptORM
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import ModelType, ScriptStatus, ScriptType
from backend.app.main import create_app
from backend.app.storage import FileStorageService


class PredictableModel:
    def predict(self, values):
        return (values["feature"] * 2).tolist()


class FailingModel:
    def predict(self, values):
        raise RuntimeError("prediction exploded")


class StatefulPreprocessor:
    def transform(self, frame, config):
        result = frame.copy()
        result["feature"] = result["feature"] + 10
        return result


def _payload(version: str, model: object = PredictableModel()) -> dict:
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
        database_url=f"sqlite:///{(tmp_path / 'step18.db').as_posix()}",
        storage_root=tmp_path / "storage",
    )
    engine = initialize_database(settings=settings)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add(ModelTypeORM(code=ModelType.ELECTRIC_LOAD, name="电力负荷"))
        session.commit()
    app = create_app(settings)
    client = TestClient(app)
    client.__enter__()
    return settings, engine, factory, client


def _save_and_publish(client: TestClient, version: str, model: object = PredictableModel()):
    saved = client.post("/api/models", json=_payload(version, model)).json()
    published = client.post(f"/api/models/{saved['id']}/publish", json={"confirm": True})
    assert published.status_code == 200, published.text
    return saved


def _data(feature=2):
    return [{"feature": feature, "time": "2026-01-01 10:00:00"}]


def test_predict_default_and_explicit_unavailable_version_do_not_fallback(tmp_path):
    settings, engine, factory, client = _fixture(tmp_path)
    try:
        first = _save_and_publish(client, "v1")
        second = _save_and_publish(client, "v2")

        response = client.post("/api/mcp/predict", json={
            "model_type": "electric_load", "data": _data(3)
        })
        assert response.status_code == 200
        assert response.json() == {
            "success": True,
            "model_type": "electric_load",
            "model_version": "v2",
            "preprocess_used": False,
            "predictions": [6],
        }

        unavailable = client.post("/mcp/predict", json={
            "model_type": "electric_load", "model_version": "v1", "data": _data()
        })
        assert unavailable.status_code == 409
        body = unavailable.json()
        assert body["success"] is False
        assert body["error_code"] == "MODEL_VERSION_UNAVAILABLE"
        assert body["details"]["model_version"] == "v1"
        assert body["details"]["requested_version"] == "v1"
        assert first["id"] != second["id"]
    finally:
        client.__exit__(None, None, None)
        engine.dispose()


def test_predict_validates_fields_and_returns_structured_failures(tmp_path):
    _settings, engine, _factory, client = _fixture(tmp_path)
    try:
        _save_and_publish(client, "v1")
        missing = client.post("/api/mcp/predict", json={
            "model_type": "electric_load", "data": [{"feature": 1}]
        })
        assert missing.status_code == 400
        assert missing.json()["success"] is False
        assert missing.json()["error_code"] == "MISSING_TIME_FIELD"
        assert "details" in missing.json()

        invalid_type = client.post("/api/mcp/predict", json={
            "model_type": "electric_load", "data": [{
                "time": "2026-01-01", "feature": "bad"
            }]
        })
        assert invalid_type.json()["error_code"] == "INVALID_FIELD_TYPE"

        bad_model_type = client.post("/api/mcp/predict", json={
            "model_type": "not-a-model", "data": _data()
        })
        assert bad_model_type.json()["error_code"] == "MODEL_TYPE_NOT_FOUND"
    finally:
        client.__exit__(None, None, None)
        engine.dispose()


def test_predict_reuses_saved_preprocessor_state_and_direct_mode(tmp_path):
    settings, engine, factory, client = _fixture(tmp_path)
    try:
        saved = client.post("/api/models", json=_payload("v1")).json()
        with factory() as session:
            storage = FileStorageService(settings.file_storage_root, session=session)
            script = ScriptORM(
                id="script-1", name="preprocessor", script_type=ScriptType.PREPROCESSOR,
                version="v1", source_code="class Preprocessor: pass",
                status=ScriptStatus.ENABLED,
            )
            session.add(script)
            buffer = BytesIO()
            joblib.dump(StatefulPreprocessor(), buffer)
            artifact = storage.save_preprocessor_state("preprocess-task-1", buffer.getvalue())
            version = session.get(ModelVersionORM, saved["id"])
            version.preprocess_used = True
            version.preprocessor_path = artifact.relative_path
            version.preprocessor_artifact_id = artifact.id
            version.preprocess_script_id = "script-1"
            version.preprocess_script_source = (
                "class Preprocessor:\n"
                "    def fit(self, df, config): return self\n"
                "    def transform(self, df, config): return df\n"
            )
            version.preprocessor_state = {
                "fitted": True,
                "artifact_type": "preprocessor",
                "relative_path": artifact.relative_path,
                "size_bytes": artifact.size_bytes,
                "checksum_sha256": artifact.checksum_sha256,
            }
            session.commit()
        assert client.post(f"/api/models/{saved['id']}/publish", json={"confirm": True}).status_code == 200
        response = client.post("/api/mcp/predict", json={
            "model_type": "electric_load", "data": _data(2)
        })
        assert response.status_code == 200, response.text
        assert response.json()["preprocess_used"] is True
        assert response.json()["predictions"] == [24]
    finally:
        client.__exit__(None, None, None)
        engine.dispose()


def test_predict_model_load_and_prediction_failures_are_structured(tmp_path):
    settings, engine, factory, client = _fixture(tmp_path)
    try:
        saved = _save_and_publish(client, "v1")
        with factory() as session:
            artifact = session.get(FileArtifactORM, saved["model_artifact_id"])
            Path(settings.file_storage_root, artifact.relative_path).write_bytes(b"not a model")
        loaded = client.post("/api/mcp/predict", json={
            "model_type": "electric_load", "data": _data()
        })
        assert loaded.status_code == 500
        assert loaded.json()["error_code"] == "MODEL_LOAD_FAILED"

        _save_and_publish(client, "v2", FailingModel())
        failed = client.post("/api/mcp/predict", json={
            "model_type": "electric_load", "data": _data()
        })
        assert failed.status_code == 500
        assert failed.json()["error_code"] == "PREDICTION_FAILED"
    finally:
        client.__exit__(None, None, None)
        engine.dispose()


def test_mark_abnormal_rolls_back_and_false_does_not_clear_alert(tmp_path):
    _settings, engine, _factory, client = _fixture(tmp_path)
    try:
        _save_and_publish(client, "v1")
        _save_and_publish(client, "v2")
        abnormal = client.post("/api/mcp/mark_model_abnormal", json={
            "model_type": "electric_load", "model_version": "v2",
            "abnormal": True, "reason": "漂移",
        })
        assert abnormal.status_code == 200, abnormal.text
        assert abnormal.json()["success"] is True
        assert abnormal.json()["current_model_version"] == "v1"
        assert abnormal.json()["rollback"]["status"] == "SUCCEEDED"
        assert abnormal.json()["alert"]["status"] == "ACTIVE"

        recovery = client.post("/mcp/mark_model_abnormal", json={
            "model_type": "electric_load", "model_version": "v2",
            "abnormal": False, "reason": "不应解除",
        })
        assert recovery.status_code == 200
        assert recovery.json()["alert_cleared"] is False
        assert recovery.json()["alert"]["status"] == "ACTIVE"
        assert client.get("/api/alerts", params={"active_only": True}).json()
    finally:
        client.__exit__(None, None, None)
        engine.dispose()
