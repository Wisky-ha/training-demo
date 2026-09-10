"""Train and evaluate the cooling-load model on the multi-site weather panel.

Evaluation design:

* Primary: leave-one-site-out. Each of the nine climate-consistent sites is
  held out in turn as a complete unseen "year", so the model is always scored
  on a full annual cycle it never saw.  This is what makes a multi-year panel
  useful in the first place.
* Secondary: the platform's fixed chronological 80/20 split, reported because
  that is what the platform itself will apply.
* Baseline: ``lag_1`` persistence scored on the same folds.
* Early stopping uses an internal validation slice of the training folds, never
  the evaluation fold.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
H2O = Path(r"C:/Users/62001/Documents/Trae/h2o/h2o_gbm_energy_prediction_20260720")
DEFAULT_DATA = H2O / "data/processed/cooling_load_climate_years_platform.csv"
OUT_MODELS = H2O / "models"
OUT_RESULTS = H2O / "results/lightgbm_cooling_climate_years"
OUT_REPORTS = H2O / "data/reports"

PREPROCESSOR_SCRIPT = ROOT / "backend/demo/scripts/cooling_load_lightgbm_preprocessor.py"
TRAINER_SCRIPT = ROOT / "backend/demo/scripts/cooling_load_lightgbm_trainer.py"
TARGET = "target_total_cooling_energy_w"
VAL_FRACTION = 0.15
SEED = 20260720


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
    }


def fit_predict(preprocess_module, trainer_module, train_df, eval_df, features, config):
    trained = preprocess_module.Preprocessor().fit(train_df, config)
    train_ready = trained.transform(train_df, config)
    eval_ready = trained.transform(eval_df, config)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(train_ready))
    val_size = max(1, int(len(train_ready) * VAL_FRACTION))
    val_index, fit_index = order[:val_size], order[val_size:]
    inner_fit = train_ready.iloc[fit_index]
    inner_val = train_ready.iloc[val_index]

    model = trainer_module.train(
        inner_fit.loc[:, features], inner_fit[TARGET],
        inner_val.loc[:, features], inner_val[TARGET],
        config,
    )
    predicted = np.asarray(model.predict(eval_ready.loc[:, features]), dtype=float)
    actual = eval_ready[TARGET].to_numpy(dtype=float)
    result = scores(actual, predicted)
    result["best_iteration"] = int(getattr(model, "best_iteration_", 0) or 0)
    return model, result, actual, predicted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.data)
    if frame.columns[-1] != TARGET:
        raise SystemExit(f"目标列必须是最后一列：{frame.columns[-1]}")
    time_column = frame.columns[0]
    features = [c for c in frame.columns[1:-1]]

    preprocess_module = load_platform_module("cooling_preprocessor_cy", PREPROCESSOR_SCRIPT)
    trainer_module = load_platform_module("cooling_trainer_cy", TRAINER_SCRIPT)
    config = {"time_column": time_column, "target_column": TARGET}

    # ---- primary: leave-one-site-out (fit once per fold) ----
    folds = []
    prediction_blocks = []
    for code in sorted(frame["scenario_code"].unique()):
        train_df = frame[frame["scenario_code"] != code].reset_index(drop=True)
        test_df = frame[frame["scenario_code"] == code].reset_index(drop=True)
        _, metrics, actual, predicted = fit_predict(
            preprocess_module, trainer_module, train_df, test_df, features, config
        )
        metrics["scenario_code"] = int(code)
        metrics["persistence_lag_1"] = scores(
            actual, test_df["lag_1"].to_numpy(dtype=float)
        )
        folds.append(metrics)

        block = pd.DataFrame({
            "scenario_code": int(code),
            "timestamp": test_df[time_column].to_numpy(),
            "actual": actual,
            "predicted": predicted,
        })
        block["error"] = block["predicted"] - block["actual"]
        block["absolute_error"] = block["error"].abs()
        block["percentage_error"] = np.where(
            block["actual"] == 0, np.nan, np.abs(block["error"] / block["actual"]) * 100
        )
        prediction_blocks.append(block)
        print(
            f"  site {code}: R2={metrics['r2']:.4f} MAE={metrics['mae']:>8,.0f} "
            f"MAPE={metrics['mape']:>5.2f}% | persistence MAE={metrics['persistence_lag_1']['mae']:>8,.0f}",
            flush=True,
        )

    loo_summary = {
        "folds": len(folds),
        "mae": float(np.mean([f["mae"] for f in folds])),
        "rmse": float(np.mean([f["rmse"] for f in folds])),
        "mape": float(np.mean([f["mape"] for f in folds])),
        "r2": float(np.mean([f["r2"] for f in folds])),
        "persistence_mae": float(np.mean([f["persistence_lag_1"]["mae"] for f in folds])),
        "persistence_rmse": float(np.mean([f["persistence_lag_1"]["rmse"] for f in folds])),
        "persistence_mape": float(np.mean([f["persistence_lag_1"]["mape"] for f in folds])),
        "persistence_r2": float(np.mean([f["persistence_lag_1"]["r2"] for f in folds])),
        "per_site": folds,
    }

    # ---- secondary: platform chronological 80/20 ----
    split_index = int(len(frame) * 0.8)
    chrono_train = frame.iloc[:split_index].reset_index(drop=True)
    chrono_test = frame.iloc[split_index:].reset_index(drop=True)
    final_model, chrono_metrics, chrono_actual, chrono_predicted = fit_predict(
        preprocess_module, trainer_module, chrono_train, chrono_test, features, config
    )
    chrono_metrics["train_rows"] = int(len(chrono_train))
    chrono_metrics["test_rows"] = int(len(chrono_test))
    chrono_metrics["persistence_lag_1"] = scores(
        chrono_actual, chrono_test["lag_1"].to_numpy(dtype=float)
    )

    # ---- export: refit on every site for the delivered artifact ----
    full_model, full_metrics, _, _ = fit_predict(
        preprocess_module, trainer_module, frame, frame, features, config
    )

    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)

    model_path = OUT_MODELS / "lightgbm_cooling_load_climate_years_v1.joblib"
    joblib.dump(full_model, model_path)

    # Deliver per-fold predictions for the held-out years.
    prediction_path = OUT_RESULTS / "leave_one_site_out_predictions.csv"
    pd.concat(prediction_blocks, ignore_index=True).to_csv(prediction_path, index=False)

    booster = full_model.booster_
    importance = pd.DataFrame({
        "feature": features,
        "split": booster.feature_importance(importance_type="split"),
        "gain": booster.feature_importance(importance_type="gain"),
    }).sort_values(["gain", "split"], ascending=False, kind="stable").reset_index(drop=True)
    importance.insert(0, "rank", range(1, len(importance) + 1))
    importance_path = OUT_RESULTS / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    summary_path = Path(args.data).parent.parent / "reports/cooling_climate_years_summary.json"
    report = {
        "model_type": "cooling_load",
        "dataset": str(args.data),
        "dataset_kind": "multi-site climate-consistent weather years",
        "caveat": (
            "nine locations with a similar cooling profile, not nine historical "
            "years of one station; the panel is synthetic"
        ),
        "algorithm": "LightGBM LGBMRegressor",
        "preprocessor_script": str(PREPROCESSOR_SCRIPT),
        "trainer_script": str(TRAINER_SCRIPT),
        "target": TARGET,
        "target_unit": "W",
        "validation_note": "early stopping uses an internal 15% slice of the training folds",
        "row_count": int(len(frame)),
        "sites": int(frame["scenario_code"].nunique()),
        "feature_count": len(features),
        "features": features,
        "leave_one_site_out": loo_summary,
        "chronological_80_20": chrono_metrics,
        "refit_on_all_sites": full_metrics,
        "artifacts": {
            "model": str(model_path),
            "leave_one_site_out_predictions": str(prediction_path),
            "feature_importance": str(importance_path),
        },
    }
    report_path = OUT_REPORTS / "lightgbm_cooling_climate_years_metrics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"leave-one-site-out: R2={loo_summary['r2']:.4f} MAE={loo_summary['mae']:,.0f} "
          f"RMSE={loo_summary['rmse']:,.0f} MAPE={loo_summary['mape']:.2f}%")
    print(f"  persistence   : R2={loo_summary['persistence_r2']:.4f} "
          f"MAE={loo_summary['persistence_mae']:,.0f} MAPE={loo_summary['persistence_mape']:.2f}%")
    print(f"chronological 80/20: R2={chrono_metrics['r2']:.4f} MAE={chrono_metrics['mae']:,.0f} "
          f"MAPE={chrono_metrics['mape']:.2f}%")
    print(f"  persistence   : R2={chrono_metrics['persistence_lag_1']['r2']:.4f} "
          f"MAE={chrono_metrics['persistence_lag_1']['mae']:,.0f}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
