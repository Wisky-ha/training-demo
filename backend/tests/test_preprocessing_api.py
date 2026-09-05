import json

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.db.models import ModelTypeORM, PreprocessingTaskORM, ScriptORM
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.domain.enums import ModelType, ScriptStatus, ScriptType, PreprocessingTaskStatus
from backend.app.main import create_app


@pytest.fixture
def preprocessing_api(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'preprocess.db').as_posix()}",
        storage_root=tmp_path / "storage",
    )
    engine = initialize_database(settings=settings)
    factory = create_session_factory(engine)
    with factory() as session:
        session.add_all([
            ModelTypeORM(code=ModelType.ELECTRIC_LOAD, name="电力"),
            ModelTypeORM(code=ModelType.INTEGRATED_ENERGY, name="综合"),
        ])
        session.commit()
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, factory
    engine.dispose()


def _dataset(client):
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("load.csv", b"time,x,target\n2024-01-01,1,10\n2024-01-02,2,20\n", "text/csv")},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _script(client, source, *, models=("electric_load",), name="pre"):
    response = client.post(
        "/api/scripts/upload",
        data={
            "name": name,
            "script_type": "preprocessor",
            "supported_model_types": json.dumps(models),
        },
        files={"file": (f"{name}.py", source, "text/x-python")},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


GOOD = b'''import pandas as pd
class Preprocessor:
    def fit(self, df, config):
        self.offset = float(df["x"].astype(float).mean())
        return self
    def transform(self, df, config):
        result = df.copy()
        result["x"] = result["x"].astype(float) - self.offset
        return result
'''


def test_success_has_stage_logs_summaries_and_saved_state(preprocessing_api):
    client, factory = preprocessing_api
    dataset_id = _dataset(client)
    script_id = _script(client, GOOD)

    response = client.post(
        "/api/preprocessing-tasks",
        json={"model_type": "electric_load", "dataset_id": dataset_id, "preprocess_script_id": script_id},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "SUCCEEDED"
    assert body["stage"] == "completed"
    assert body["preprocess_used"] is True
    assert body["input_row_count"] == body["output_row_count"] == 2
    assert body["input_columns"] == body["output_columns"] == ["time", "x", "target"]
    assert body["input_summary"]["row_count"] == 2
    assert body["output_summary"]["row_count"] == 2
    assert body["next_step"] == "dataset_split"
    assert ["数据读取", "执行预处理", "结果校验"] == [
        item for item in body["logs"] if item in {"数据读取", "执行预处理", "结果校验"}
    ]
    assert body["started_at"] and body["finished_at"]
    with factory() as session:
        task = session.get(PreprocessingTaskORM, body["id"])
        assert task.status is PreprocessingTaskStatus.SUCCEEDED
        assert task.preprocessor_path

    reused = client.post(
        f"/api/preprocessing-tasks/{body['id']}/transform",
        json={"dataset_id": dataset_id},
    )
    assert reused.status_code == 200, reused.text
    assert reused.json()["data_source"] == "preprocessed"


def test_skip_persists_unused_and_never_executes_selected_script(preprocessing_api):
    client, _ = preprocessing_api
    dataset_id = _dataset(client)
    raising = b'''class Preprocessor:
    def fit(self, df, config):
        raise RuntimeError("must not run")
    def transform(self, df, config):
        raise RuntimeError("must not run")
'''
    script_id = _script(client, raising)
    response = client.post(
        "/api/preprocessing-tasks",
        json={"model_type": "electric_load", "dataset_id": dataset_id, "mode": "skip", "preprocess_script_id": script_id},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "SKIPPED"
    assert body["preprocess_status"] == "unused"
    assert body["preprocess_used"] is False
    assert body["preprocess_script_id"] is None
    assert body["data_source"] == "raw"
    assert "未使用预处理" in " ".join(body["logs"])


@pytest.mark.parametrize(
    "source,code",
    [
        (b"class Preprocessor:\n    def fit(self, df, config): raise ValueError('fit failed')\n    def transform(self, df, config): return df\n", "PREPROCESS_FIT_FAILED"),
        (b"class Preprocessor:\n    def fit(self, df): return self\n    def transform(self, df, config): return df\n", "INVALID_PREPROCESSOR"),
        (b"class Preprocessor:\n    def fit(self, df, config): return self\n    def transform(self, df, config): return {'x': 1}\n", "PREPROCESS_RESULT_NOT_DATAFRAME"),
        (b"class Preprocessor:\n    def fit(self, df, config): return self\n    def transform(self, df, config): return df.iloc[:-1]\n", "PREPROCESS_ROW_COUNT_INVALID"),
        (b"class Preprocessor:\n    def fit(self, df, config): return self\n    def transform(self, df, config): return df.drop(columns=['target'])\n", "PREPROCESS_FIELDS_INVALID"),
        (b"class Wrong:\n    pass\n", "INVALID_PREPROCESSOR"),
    ],
)
def test_script_execution_and_result_validation_errors_are_persisted(preprocessing_api, source, code):
    client, _ = preprocessing_api
    dataset_id = _dataset(client)
    script_id = _script(client, source, name=code.lower())
    response = client.post(
        "/api/preprocessing-tasks",
        json={"model_type": "electric_load", "dataset_id": dataset_id, "preprocess_script_id": script_id},
    )
    assert response.status_code == 400
    assert response.json()["success"] is False
    assert response.json()["error_code"] == code
    assert response.json()["details"]["task_id"]
    assert response.json()["detail"]["code"] == code
    assert "traceback" not in str(response.json()).lower()
    task_id = response.json()["detail"]["task_id"]
    task = client.get(f"/api/preprocessing-tasks/{task_id}").json()
    assert task["status"] == "FAILED"
    assert task["stage"] == "failed"
    assert task["error_message"]


def test_fit_or_transform_failure_incompatibility_and_unsafe_source(preprocessing_api):
    client, _ = preprocessing_api
    dataset_id = _dataset(client)
    failing = b'''class Preprocessor:
    def fit(self, df, config):
        return self
    def transform(self, df, config):
        raise ValueError("bad data")
'''
    script_id = _script(client, failing, name="failing")
    response = client.post("/api/preprocessing-tasks", json={"model_type": "electric_load", "dataset_id": dataset_id, "preprocess_script_id": script_id})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "PREPROCESS_TRANSFORM_FAILED"

    incompatible_id = _script(client, GOOD, models=("integrated_energy",), name="other-model")
    response = client.post("/api/preprocessing-tasks", json={"model_type": "electric_load", "dataset_id": dataset_id, "preprocess_script_id": incompatible_id})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "MODEL_TYPE_INCOMPATIBLE"

    unsafe_id = _script(client, b"import os\nclass Preprocessor:\n    def fit(self, df, config): return self\n    def transform(self, df, config): return df\n", name="unsafe")
    response = client.post("/api/preprocessing-tasks", json={"model_type": "electric_load", "dataset_id": dataset_id, "preprocess_script_id": unsafe_id})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "UNSAFE_SCRIPT"


def test_disabled_or_unknown_script_is_rejected_without_running_any_code(preprocessing_api):
    client, factory = preprocessing_api
    dataset_id = _dataset(client)
    script_id = _script(client, GOOD, name="disabled")
    with factory() as session:
        script = session.get(ScriptORM, script_id)
        script.status = ScriptStatus.DISABLED
        session.commit()
    response = client.post("/api/preprocessing-tasks", json={"model_type": "electric_load", "dataset_id": dataset_id, "preprocess_script_id": script_id})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "SCRIPT_DISABLED"
    response = client.post("/api/preprocessing-tasks", json={"model_type": "electric_load", "dataset_id": dataset_id, "preprocess_script_id": "../../etc/passwd"})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "SCRIPT_NOT_FOUND"
