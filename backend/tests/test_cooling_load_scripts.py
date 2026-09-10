"""Contracts for the cooling-load demo scripts."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.app.services.preprocessing import _loaded_module
from backend.app.services.training_executor import TrainingScriptExecutor


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSOR_PATH = (
    BACKEND_ROOT / "demo" / "scripts" / "cooling_load_lightgbm_preprocessor.py"
)
TRAINER_PATH = BACKEND_ROOT / "demo" / "scripts" / "cooling_load_lightgbm_trainer.py"


def _train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2000-01-01 00:00:00", "2000-01-01 01:00:00"],
            "sequence_index": ["1", "2"],
            "dry_bulb_temp_c": ["3.9", "4.5"],
            "all_empty": [None, ""],
            "target_total_cooling_energy_w": ["100.0", "120.0"],
        }
    )


def test_cooling_preprocessor_keeps_platform_roles_and_fills_missing_values():
    source = PREPROCESSOR_PATH.read_text(encoding="utf-8")
    train = _train_frame()
    test = pd.DataFrame(
        {
            "timestamp": ["2000-01-01 02:00:00"],
            "sequence_index": [""],
            "dry_bulb_temp_c": ["not-a-number"],
            "all_empty": [None],
            "target_total_cooling_energy_w": ["140.0"],
        }
    )
    config = {
        "time_column": "timestamp",
        "target_column": "target_total_cooling_energy_w",
    }

    with _loaded_module("cooling-load-preprocessor-test", source) as module:
        instance = module.Preprocessor()
        assert list(inspect.signature(instance.fit).parameters) == ["df", "config"]
        assert list(inspect.signature(instance.transform).parameters) == ["df", "config"]
        assert instance.fit(train, config) is instance
        train_output = instance.transform(train, config)
        test_output = instance.transform(test, config)

    assert isinstance(train_output, pd.DataFrame)
    assert list(train_output.columns) == list(test_output.columns)
    assert train_output.columns[0] == "timestamp"
    assert train_output.columns[-1] == "target_total_cooling_energy_w"
    assert "all_empty" not in train_output.columns
    for output in (train_output, test_output):
        for column in output.columns[1:-1]:
            assert pd.api.types.is_numeric_dtype(output[column])
            assert np.isfinite(output[column].to_numpy(dtype=float)).all()
    assert train_output["target_total_cooling_energy_w"].tolist() == ["100.0", "120.0"]


def test_cooling_preprocessor_transform_accepts_a_prediction_frame_without_target():
    """Prediction requests omit the training target; transform must still run."""

    source = PREPROCESSOR_PATH.read_text(encoding="utf-8")
    train = _train_frame()
    prediction = pd.DataFrame(
        {
            "timestamp": ["2000-01-02 00:00:00"],
            "sequence_index": [3],
            "dry_bulb_temp_c": [10.0],
            "all_empty": [None],
        }
    )
    config = {
        "time_column": "timestamp",
        "target_column": "target_total_cooling_energy_w",
    }

    with _loaded_module("cooling-load-predict-test", source) as module:
        instance = module.Preprocessor().fit(train, config)
        output = instance.transform(prediction, config)

    assert isinstance(output, pd.DataFrame)
    assert len(output) == 1
    assert output.columns[-1] == "target_total_cooling_energy_w"
    assert output["target_total_cooling_energy_w"].isna().all()
    assert np.isfinite(output["dry_bulb_temp_c"].to_numpy(dtype=float)).all()


def test_cooling_trainer_uses_lightgbm_and_runs_through_the_platform_executor():
    source = TRAINER_PATH.read_text(encoding="utf-8")
    assert "lightgbm" in source.lower()
    assert "sklearn" not in source.lower()

    if importlib.util.find_spec("lightgbm") is None:
        pytest.skip("lightgbm is not installed; no substitute model is allowed")

    X_train = pd.DataFrame({"feature": np.arange(20, dtype=float)})
    y_train = pd.Series(2.0 * X_train["feature"] + 1.0)
    X_test = pd.DataFrame({"feature": np.arange(20, 25, dtype=float)})
    y_test = pd.Series(2.0 * X_test["feature"] + 1.0)
    result = TrainingScriptExecutor().execute(
        source_code=source,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        config={
            "n_estimators": 25,
            "early_stopping_rounds": 5,
            "log_period": 0,
            "n_jobs": 1,
        },
    )

    assert result.success, result.error
    assert result.model is not None
    assert callable(result.model.predict)
    predictions = np.asarray(result.model.predict(X_test), dtype=float)
    assert predictions.shape == (len(X_test),)
    assert np.isfinite(predictions).all()
