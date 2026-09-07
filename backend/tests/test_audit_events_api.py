from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from backend.app.core.config import Settings
from backend.app.db.models import AuditEventORM, ModelTypeORM
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import ModelType
from backend.app.main import create_app


@pytest.fixture
def audit_api(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'audit.db').as_posix()}",
        storage_root=tmp_path / "storage",
    )
    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    initialize_database(engine)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add_all(
            [
                ModelTypeORM(code=ModelType.ELECTRIC_LOAD, name="电力负荷"),
                ModelTypeORM(code=ModelType.INTEGRATED_ENERGY, name="综合能耗"),
            ]
        )
        occurred_at = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        session.commit()
        session.add_all(
            [
                AuditEventORM(
                    event_id="evt-001",
                    occurred_at=occurred_at,
                    event_type="DATASET_VALIDATED",
                    object_type="DATASET",
                    object_id="dataset-1",
                    model_type=ModelType.ELECTRIC_LOAD,
                    result="SUCCEEDED",
                    message="电力数据校验完成",
                ),
                AuditEventORM(
                    event_id="evt-002",
                    occurred_at=occurred_at,
                    event_type="MODEL_PUBLISHED",
                    object_type="MODEL_VERSION",
                    object_id="model-2",
                    model_type=ModelType.INTEGRATED_ENERGY,
                    result="SUCCEEDED",
                    operator_name="Alice",
                ),
                AuditEventORM(
                    event_id="evt-003",
                    occurred_at=occurred_at,
                    event_type="TRAINING_FAILED",
                    object_type="TRAINING_JOB",
                    object_id="job-1",
                    model_type=ModelType.ELECTRIC_LOAD,
                    result="FAILED",
                    message="network failure",
                ),
                AuditEventORM(
                    event_id="evt-004",
                    occurred_at=datetime(2025, 1, 2, 12, 0, tzinfo=timezone.utc),
                    event_type="MODEL_ROLLBACK_SUCCEEDED",
                    object_type="MODEL_VERSION",
                    object_id="model-3",
                    model_type=ModelType.INTEGRATED_ENERGY,
                    result="SUCCEEDED",
                ),
            ]
        )
        session.commit()

    with TestClient(create_app(settings)) as client:
        yield client
    engine.dispose()


def test_audit_events_default_pagination_and_contract(audit_api):
    response = audit_api.get("/api/audit-events")

    assert response.status_code == 200
    body = response.json()
    assert body["page"] == 1
    assert body["page_size"] == 50
    assert body["total"] == 4
    assert body["has_next"] is False
    assert body["items"][0]["id"] == "evt-004"
    assert body["items"][1]["id"] == "evt-003"
    assert body["items"][2]["id"] == "evt-002"
    assert body["items"][3]["id"] == "evt-001"
    assert body["items"][0]["operator_id"] is None
    assert body["items"][0]["metadata"] == {}


def test_audit_events_total_has_next_and_stable_tie_sort(audit_api):
    first = audit_api.get("/api/audit-events", params={"page": 1, "page_size": 2}).json()
    second = audit_api.get("/api/audit-events", params={"page": 2, "page_size": 2}).json()
    empty = audit_api.get("/api/audit-events", params={"page": 3, "page_size": 2}).json()

    assert first["total"] == second["total"] == empty["total"] == 4
    assert first["has_next"] is True
    assert second["has_next"] is False
    assert empty["items"] == []
    assert [item["id"] for item in first["items"]] == ["evt-004", "evt-003"]
    # evt-003 and evt-002 have the same timestamp; id is the deterministic tie breaker.
    assert [item["id"] for item in second["items"]] == ["evt-002", "evt-001"]


def test_audit_events_event_keyword_and_object_filters(audit_api):
    event_type = audit_api.get(
        "/api/audit-events", params={"event_type": "MODEL_PUBLISHED"}
    ).json()
    keyword = audit_api.get(
        "/api/audit-events", params={"query": "network"}
    ).json()
    object_type = audit_api.get(
        "/api/audit-events", params={"object_type": "TRAINING_JOB"}
    ).json()

    assert [item["id"] for item in event_type["items"]] == ["evt-002"]
    assert [item["id"] for item in keyword["items"]] == ["evt-003"]
    assert [item["id"] for item in object_type["items"]] == ["evt-003"]


def test_audit_events_time_model_and_result_filters(audit_api):
    response = audit_api.get(
        "/api/audit-events",
        params={
            "from": "2025-01-02T00:00:00Z",
            "to": "2025-01-02T23:59:59Z",
            "model_type": "integrated_energy",
            "result": "SUCCEEDED",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["id"] == "evt-004"


def test_audit_events_are_read_only_and_openapi_is_explicit(audit_api):
    assert audit_api.post("/api/audit-events").status_code == 405
    assert audit_api.delete("/api/audit-events").status_code == 405
    assert audit_api.delete("/api/audit-events/evt-001").status_code == 404

    operation = audit_api.get("/openapi.json").json()["paths"]["/api/audit-events"]["get"]
    assert {item["name"] for item in operation["parameters"]} >= {
        "page",
        "page_size",
        "from",
        "to",
        "model_type",
        "event_type",
        "object_type",
        "result",
        "query",
    }
    assert operation["responses"]["200"]["content"]["application/json"]["schema"]
