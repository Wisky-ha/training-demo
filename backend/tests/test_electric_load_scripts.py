"""Contracts for the electric-load demo scripts and CSV adapter."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from backend.app.services.preprocessing import _loaded_module
from backend.app.services.training_executor import TrainingScriptExecutor


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSOR_PATH = (
    BACKEND_ROOT / "demo" / "scripts" / "electric_load_lightgbm_preprocessor.py"
)
TRAINER_PATH = BACKEND_ROOT / "demo" / "scripts" / "electric_load_lightgbm_trainer.py"
PREPARE_PATH = BACKEND_ROOT / "demo" / "electric_load" / "prepare_dataset.py"


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location(
        "electric_load_prepare_dataset", PREPARE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_prepare_output_contract_and_source_is_not_changed(tmp_path):
    prepare = _load_prepare_module()
    source = tmp_path / "source.csv"
    output = tmp_path / "electric_load_platform.csv"
    raw = pd.DataFrame(
        {
            "sequence_index": [0, 1, 2, 3],
            "scenario": ["SMY1", "SMY0", "SMY1", "SMY0"],
            "scenario_row": [0, 1, 2, 3],
            "month": [1, 1, 1, 1],
            "day": [1, 1, 2, 2],
            "hour": [24, 1, 1, 24],
            "numeric_feature": [1.0, 2.0, np.nan, 4.0],
            "liquid_precipitation_depth_mm": [np.nan] * 4,
            "target_total_purchased_electricity_w": [10.0, 11.0, 12.0, 13.0],
        }
    )
    raw.to_csv(source, index=False)
    original = source.read_bytes()

    summary = prepare.prepare_dataset(source, output)
    result = pd.read_csv(output)

    assert source.read_bytes() == original
    assert summary.row_count == len(raw) == len(result)
    assert result.columns[0] == "timestamp"
    assert result.columns[-1] == "target_total_purchased_electricity_w"
    assert "scenario" not in result.columns
    assert "liquid_precipitation_depth_mm" not in result.columns
    assert "scenario_code" in result.columns
    assert pd.api.types.is_numeric_dtype(result["scenario_code"])
    assert all(
        pd.api.types.is_numeric_dtype(result[column])
        for column in result.columns[1:-1]
    )
    timestamps = pd.to_datetime(result["timestamp"], errors="coerce", format="mixed")
    assert timestamps.notna().all()
    assert timestamps.is_unique
    assert timestamps.is_monotonic_increasing
    assert np.isfinite(result[result.columns[-1]].to_numpy(dtype=float)).all()
    assert set(result["scenario_code"].unique()) == {0, 1}
    assert summary.dropped_columns == ("liquid_precipitation_depth_mm",)


def test_preprocessor_uses_restricted_platform_loader_and_keeps_matching_fields():
    source = PREPROCESSOR_PATH.read_text(encoding="utf-8")
    train = pd.DataFrame(
        {
            "timestamp": ["2000-01-01 00:00:00", "2000-01-01 01:00:00"],
            "sequence_index": ["1", "2"],
            "scenario_code": ["0", "0"],
            "all_empty": [None, ""],
            "partly_bad": ["3.0", "not-a-number"],
            "target": ["10", "11"],
        }
    )
    test = pd.DataFrame(
        {
            "timestamp": ["2000-01-01 02:00:00"],
            "sequence_index": [""],
            "scenario_code": ["0"],
            "all_empty": [None],
            "partly_bad": ["not-a-number"],
            "target": ["12"],
        }
    )
    config = {"time_column": "timestamp", "target_column": "target"}

    with _loaded_module("electric-load-preprocessor-test", source) as module:
        instance = module.Preprocessor()
        assert list(inspect.signature(instance.fit).parameters) == ["df", "config"]
        assert list(inspect.signature(instance.transform).parameters) == [
            "df", "config"
        ]
        assert instance.fit(train, config) is instance
        train_output = instance.transform(train, config)
        test_output = instance.transform(test, config)

    assert isinstance(train_output, pd.DataFrame)
    assert list(train_output.columns) == list(test_output.columns)
    assert train_output.columns[0] == "timestamp"
    assert train_output.columns[-1] == "target"
    assert "all_empty" not in train_output.columns
    assert len(train_output.columns) > 2
    for output in (train_output, test_output):
        for column in output.columns[1:-1]:
            assert pd.api.types.is_numeric_dtype(output[column])
            assert np.isfinite(output[column].to_numpy(dtype=float)).all()
    assert train_output["target"].tolist() == ["10", "11"]
    assert test_output["target"].tolist() == ["12"]


def test_lightgbm_trainer_is_explicit_and_runs_through_executor_when_available():
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
