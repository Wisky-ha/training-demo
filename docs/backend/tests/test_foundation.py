from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from backend.app.core.config import Settings
from backend.app.db.connection import connect_database
from backend.app.main import app


def test_required_data_science_dependencies_are_importable(tmp_path):
    data = pd.DataFrame({"x": np.array([1.0, 2.0]), "y": [2.0, 4.0]})
    model = LinearRegression().fit(data[["x"]], data["y"])
    model_file = tmp_path / "foundation-model.joblib"
    joblib.dump(model, model_file)
    restored = joblib.load(model_file)
    assert abs(restored.predict([[3.0]])[0] - 6.0) < 1e-9


def test_settings_load_database_configuration(tmp_path):
    database_file = tmp_path / "settings.db"
    settings = Settings(database_url=f"sqlite:///{database_file.as_posix()}")
    assert settings.database_path == database_file
    assert settings.app_name


def test_sqlite_connection_is_available(tmp_path):
    database_file = tmp_path / "nested" / "app.db"
    settings = Settings(database_url=f"sqlite:///{database_file.as_posix()}")
    connection = connect_database(settings)
    try:
        assert connection.execute("SELECT 1").fetchone()[0] == 1
    finally:
        connection.close()
    assert database_file.exists()


def test_fastapi_entrypoint_and_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
