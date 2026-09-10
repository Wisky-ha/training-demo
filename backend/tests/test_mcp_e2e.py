"""End-to-end HTTP acceptance tests for the two MCP tool routes."""

from __future__ import annotations

import base64
from pathlib import Path

import cloudpickle
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.db.models import ModelTypeORM
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import ModelType
from backend.app.main import create_app


class PredictableModel:
    def predict(self, values):
        return (values["feature"] * 2).tolist()


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{(tmp_path / 'mcp-e2e.db').as_posix()}",
        storage_root=tmp_path / "storage",
    )


def _client(tmp_path: Path) -> TestClient:
    settings = _settings(tmp_path)
    engine = initialize_database(settings=settings)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add(ModelTypeORM(code=ModelType.ELECTRIC_LOAD, name="电力负荷"))
        session.commit()
    engine.dispose()
    client = TestClient(create_app(settings))
    client.__enter__()
    return client


def _model_payload(version: str) -> dict:
    return {
        "model_type": "electric_load",
        "health_status": "HEALTHY",
        "version": version,
        "model_content_base64": base64.b64encode(
            cloudpickle.dumps(PredictableModel())
        ).decode(),
        "time_column": "time",
        "feature_columns": ["feature"],
        "target_column": "target",
        "input_schema": {
            "columns": ["time", "feature", "target"],
            "required_columns": ["time", "feature"],
            "column_types": {
                "time": "datetime",
                "feature": "number",
                "target": "number",
            },
            "time_column": "time",
            "target_column": "target",
            "extra_columns": "reject",
        },
    }


def _publish(client: TestClient, version: str) -> dict:
    saved = client.post("/api/models", json=_model_payload(version))
    assert saved.status_code == 201, saved.text
    model = saved.json()
    published = client.post(
        f"/api/models/{model['id']}/publish",
        json={"confirm": True},
    )
    assert published.status_code == 200, published.text
    return model


def _prediction_data(feature: int = 3) -> list[dict]:
    return [{"feature": feature, "time": "2026-01-01T10:00:00"}]


def test_mcp_predict_e2e_returns_prediction_from_published_model(tmp_path: Path) -> None:
    client = _client(tmp_path)
    try:
        published = _publish(client, "v1")

        response = client.post(
            "/api/mcp/predict",
            json={
                "model_type": "electric_load",
                "model_version": published["version"],
                "data": _prediction_data(),
            },
        )

        assert response.status_code == 200, response.text
        assert response.json() == {
            "success": True,
            "model_type": "electric_load",
            "model_version": "v1",
            "preprocess_used": False,
            "predictions": [6],
        }
    finally:
        client.__exit__(None, None, None)


def test_mcp_mark_model_abnormal_e2e_rolls_back_and_opens_alert(tmp_path: Path) -> None:
    client = _client(tmp_path)
    try:
        backup = _publish(client, "v1")
        current = _publish(client, "v2")

        response = client.post(
            "/api/mcp/mark_model_abnormal",
            json={
                "model_type": "electric_load",
                "model_version": current["version"],
                "abnormal": True,
                "reason": "MCP E2E 异常标记测试",
            },
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["success"] is True
        assert body["model_type"] == "electric_load"
        assert body["model_version"] == "v2"
        assert body["rollback_triggered"] is True
        assert body["current_model_version"] == backup["version"]
        assert body["alert"]["status"] == "ACTIVE"
        assert body["rollback"]["status"] == "SUCCEEDED"
    finally:
        client.__exit__(None, None, None)
