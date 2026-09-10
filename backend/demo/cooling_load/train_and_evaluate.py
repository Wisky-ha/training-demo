"""Train, evaluate and export the cooling-load LightGBM model.

Evaluation design for a single typical-meteorological year:

* A strict chronological 80/20 split makes the test set Oct 19 - Dec 31, so
  the model has never observed winter during training.  That is reported as a
  documented stress test, not as the headline score.
* The headline evaluation uses a season-aware split: every fifth calendar day
  is held out, so both partitions contain all four seasons.
* Early stopping uses an internal validation split, never the evaluation set.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
H2O = Path(r"C:/Users/62001/Documents/Trae/h2o/h2o_gbm_energy_prediction_20260720")
DATA = H2O / "data/processed/cooling_load_platform.csv"
OUT_MODELS = H2O / "models"
OUT_RESULTS = H2O / "results/lightgbm_cooling"
OUT_REPORTS = H2O / "data/reports"

PREPROCESSOR_SCRIPT = ROOT / "backend/demo/scripts/cooling_load_lightgbm_preprocessor.py"
TRAINER_SCRIPT = ROOT / "backend/demo/scripts/cooling_load_lightgbm_trainer.py"
TARGET = "target_total_cooling_energy_w"
VAL_FRACTION = 0.15


def load_platform_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def scores(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - actual
    nonzero = np.abs(actual) > 1e-12
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape": float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100) if nonzero.any() else None,
        "r2": float(1 - np.sum(error ** 2) / np.sum((actual - np.mean(actual)) ** 2)),
        "sample_count": int(len(actual)),
        "mape_valid_count": int(nonzero.sum()),
        "mape_excluded_count": int((~nonzero).sum()),
        "mape_note": f"MAPE 已排除 {int((~nonzero).sum())} 个实际值为 0 的样本",
    }


def fit_and_evaluate(
    preprocess_module, trainer_module, train_df, eval_df, feature_columns,
    time_column, config,
):
    """Fit preprocessing on train only, then train on train and score eval."""

    trained = preprocess_module.Preprocessor().fit(train_df, config)
    train_ready = trained.transform(train_df, config)
    eval_ready = trained.transform(eval_df, config)

    # Internal validation slice for early stopping only.
    val_size = max(1, int(len(train_ready) * VAL_FRACTION))
    inner_train = train_ready.iloc[:-val_size]
    inner_val = train_ready.iloc[-val_size:]

    model = trainer_module.train(
        inner_train.loc[:, feature_columns], inner_train[TARGET],
        inner_val.loc[:, feature_columns], inner_val[TARGET],
        config,
    )
    predicted = np.asarray(model.predict(eval_ready.loc[:, feature_columns]), dtype=float)
    actual = eval_ready[TARGET].to_numpy(dtype=float)
    result = scores(actual, predicted)
    result["best_iteration"] = int(getattr(model, "best_iteration_", 0) or 0)
    return model, result, actual, predicted, eval_ready


def main() -> None:
    frame = pd.read_csv(DATA)
    if frame.columns[-1] != TARGET:
        raise SystemExit(f"目标列必须是最后一列：{frame.columns[-1]}")
    time_column = frame.columns[0]
    feature_columns = [c for c in frame.columns[1:-1]]

    preprocess_module = load_platform_module("cooling_preprocessor", PREPROCESSOR_SCRIPT)
    trainer_module = load_platform_module("cooling_trainer", TRAINER_SCRIPT)
    config = {"time_column": time_column, "target_column": TARGET}

    # ---- headline: season-aware split (every fifth calendar day) ----
    holdout_mask = (frame["day_of_year"] % 5 == 0).to_numpy()
    season_train = frame[~holdout_mask].reset_index(drop=True)
    season_test = frame[holdout_mask].reset_index(drop=True)

    model, season_metrics, actual, predicted, eval_ready = fit_and_evaluate(
        preprocess_module, trainer_module, season_train, season_test,
        feature_columns, time_column, config,
    )
    season_metrics["split"] = "season-aware: every fifth calendar day held out"
    season_metrics["train_rows"] = int(len(season_train))
    season_metrics["test_rows"] = int(len(season_test))

    # ---- baseline: lag_1 persistence on the same holdout ----
    persistence = season_test["lag_1"].to_numpy(dtype=float)
    season_metrics["persistence_lag_1"] = scores(actual, persistence)

    # ---- documented stress test: platform-style chronological 80/20 ----
    split_index = int(len(frame) * 0.8)
    chrono_train = frame.iloc[:split_index].reset_index(drop=True)
    chrono_test = frame.iloc[split_index:].reset_index(drop=True)
    _, chrono_metrics, chrono_actual, _, _ = fit_and_evaluate(
        preprocess_module, trainer_module, chrono_train, chrono_test,
        feature_columns, time_column, config,
    )
    chrono_metrics["split"] = "chronological 80/20 (platform default)"
    chrono_metrics["train_rows"] = int(len(chrono_train))
    chrono_metrics["test_rows"] = int(len(chrono_test))
    chrono_metrics["test_period"] = "last 20% of the derived timeline (~Oct 19 - Dec 31)"
    chrono_metrics["limitation"] = (
        "单年 TMY 数据在时序切分下训练期不含冬季，测试期恰为冬季，"
        "存在结构性分布偏移，不能代表模型能力"
    )
    chrono_metrics["persistence_lag_1"] = scores(
        chrono_actual, chrono_test["lag_1"].to_numpy(dtype=float)
    )

    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)

    model_path = OUT_MODELS / "lightgbm_cooling_load_v1.joblib"
    joblib.dump(model, model_path)

    predictions = pd.DataFrame({
        "timestamp": season_test[time_column].to_numpy(),
        "actual": actual,
        "predicted": predicted,
        "error": predicted - actual,
    })
    predictions["absolute_error"] = predictions["error"].abs()
    predictions["percentage_error"] = np.where(
        actual == 0, np.nan, np.abs(predictions["error"] / actual) * 100
    )
    prediction_path = OUT_RESULTS / "test_predictions.csv"
    predictions.to_csv(prediction_path, index=False)

    booster = model.booster_
    importance = pd.DataFrame({
        "feature": feature_columns,
        "split": booster.feature_importance(importance_type="split"),
        "gain": booster.feature_importance(importance_type="gain"),
    }).sort_values(["gain", "split"], ascending=False, kind="stable").reset_index(drop=True)
    importance.insert(0, "rank", range(1, len(importance) + 1))
    importance_path = OUT_RESULTS / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    report = {
        "model_type": "cooling_load",
        "algorithm": "LightGBM LGBMRegressor",
        "dataset": str(DATA),
        "location": "Anqing, Anhui, China (WMO 584240)",
        "building_model": "ASHRAE901 OfficeLarge STD2019 Chiller205 Detailed",
        "weather_source": "CHN_AH_Anqing.584240_TMYx.2011-2025.epw",
        "preprocessor_script": str(PREPROCESSOR_SCRIPT),
        "trainer_script": str(TRAINER_SCRIPT),
        "target": TARGET,
        "target_unit": "W",
        "target_definition": "all air-system cooling coils (building + datacenter)",
        "validation_note": "early stopping uses an internal 15% validation slice, never the evaluation set",
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "metrics": season_metrics,
        "chronological_stress_test": chrono_metrics,
        "artifacts": {
            "model": str(model_path),
            "test_predictions": str(prediction_path),
            "feature_importance": str(importance_path),
        },
    }
    report_path = OUT_REPORTS / "lightgbm_cooling_metrics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
