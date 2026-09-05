"""Complete, side-by-side evaluation of a candidate and baseline model."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class ModelEvaluationError(ValueError):
    """Raised when an evaluation input cannot produce a trustworthy result."""

    def __init__(self, message: str, code: str = "MODEL_EVALUATION_INVALID_INPUT") -> None:
        super().__init__(message)
        self.code = code


class ModelEvaluationService:
    """Calculate metrics without sampling and retain every test-set row."""

    METRIC_NAMES = ("mae", "rmse", "mape", "r2")

    @staticmethod
    def _numeric(values: Sequence[Any], label: str) -> np.ndarray:
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ModelEvaluationError(f"{label}必须是数值序列", "EVALUATION_VALUES_INVALID") from exc
        if array.ndim != 1:
            raise ModelEvaluationError(f"{label}必须是一维序列", "EVALUATION_VALUES_INVALID")
        if not np.isfinite(array).all():
            raise ModelEvaluationError(f"{label}包含无效数值", "EVALUATION_VALUES_INVALID")
        return array

    @classmethod
    def metrics(cls, actual: Sequence[Any], predicted: Sequence[Any]) -> dict[str, Any]:
        """Return all four metrics over the supplied complete test set."""

        y_true = cls._numeric(actual, "实际值")
        y_pred = cls._numeric(predicted, "预测值")
        if len(y_true) == 0:
            raise ModelEvaluationError("评估测试集不能为空", "EVALUATION_TEST_SET_EMPTY")
        if len(y_true) != len(y_pred):
            raise ModelEvaluationError("实际值和预测值长度必须一致", "EVALUATION_LENGTH_MISMATCH")

        error = y_true - y_pred
        absolute_error = np.abs(error)
        nonzero = y_true != 0
        valid_count = int(nonzero.sum())
        if valid_count:
            mape: float | None = float(np.mean(np.abs(error[nonzero] / y_true[nonzero])) * 100.0)
            note = f"MAPE 已排除 {len(y_true) - valid_count} 个实际值为 0 的样本"
        else:
            mape = None
            note = "MAPE 没有有效样本：所有实际值均为 0"

        denominator = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
        if denominator:
            r2 = float(1.0 - np.sum(error * error) / denominator)
        else:
            # This is the same finite convention used by sklearn's r2_score
            # for constant targets: a perfect prediction scores one, otherwise
            # zero.  It also keeps one-row test sets usable.
            r2 = 1.0 if not np.any(error) else 0.0
        return {
            "mae": float(np.mean(absolute_error)),
            "rmse": float(np.sqrt(np.mean(error * error))),
            "mape": mape,
            "r2": r2,
            "sample_count": int(len(y_true)),
            "mape_valid_count": valid_count,
            "mape_valid_sample_count": valid_count,
            "mape_excluded_count": int(len(y_true) - valid_count),
            "mape_note": note,
        }

    @classmethod
    def evaluate(
        cls,
        *,
        times: Sequence[Any],
        actual: Sequence[Any],
        candidate_predictions: Sequence[Any],
        baseline_predictions: Sequence[Any],
        candidate_id: str | None = None,
        baseline_id: str | None = None,
        candidate_version: str | None = None,
        baseline_version: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate both models against exactly the same complete test set.

        The returned arrays are deliberately not chart-sampled.  A caller may
        derive a smaller display series separately, while persisted metrics and
        audit data always cover every test row.
        """

        try:
            time_values = list(times)
        except TypeError as exc:
            raise ModelEvaluationError("测试集时间序列必须可迭代", "EVALUATION_TIME_INVALID") from exc
        y_true = cls._numeric(actual, "实际值")
        candidate = cls._numeric(candidate_predictions, "候选模型预测值")
        baseline = cls._numeric(baseline_predictions, "基线模型预测值")
        expected = len(y_true)
        if expected == 0:
            raise ModelEvaluationError("评估测试集不能为空", "EVALUATION_TEST_SET_EMPTY")
        if len(time_values) != expected:
            raise ModelEvaluationError("时间序列和测试集长度必须一致", "EVALUATION_LENGTH_MISMATCH")
        if len(candidate) != expected or len(baseline) != expected:
            raise ModelEvaluationError("候选模型、基线模型必须使用同一测试集", "EVALUATION_LENGTH_MISMATCH")

        candidate_metrics = cls.metrics(y_true, candidate)
        baseline_metrics = cls.metrics(y_true, baseline)
        candidate_errors = (y_true - candidate).astype(float).tolist()
        baseline_errors = (y_true - baseline).astype(float).tolist()
        differences = {
            name: (
                candidate_metrics[name] - baseline_metrics[name]
                if candidate_metrics[name] is not None and baseline_metrics[name] is not None
                else None
            )
            for name in cls.METRIC_NAMES
        }
        candidate_model = {
            "model_version_id": candidate_id,
            "version": candidate_version,
            "metrics": candidate_metrics,
        }
        baseline_model = {
            "model_version_id": baseline_id,
            "version": baseline_version,
            "metrics": baseline_metrics,
        }
        error_series = [
            {
                "time": time_values[index],
                "timestamp": time_values[index],
                "candidate_error": candidate_errors[index],
                "baseline_error": baseline_errors[index],
                # ``error`` remains the candidate error for existing clients.
                "error": candidate_errors[index],
                "absolute_error": abs(candidate_errors[index]),
                "percentage_error": (
                    None
                    if y_true[index] == 0
                    else float(abs(candidate_errors[index] / y_true[index]) * 100.0)
                ),
            }
            for index in range(expected)
        ]
        return {
            # Top-level candidate metrics retain the previous API contract.
            **candidate_metrics,
            "sample_count": expected,
            "test_time_series": time_values,
            "timestamps": time_values,
            "actual_values": y_true.tolist(),
            "candidate_predictions": candidate.tolist(),
            "baseline_predictions": baseline.tolist(),
            "candidate_errors": candidate_errors,
            "baseline_errors": baseline_errors,
            "error_series": candidate_errors,
            "metrics": {
                "candidate": candidate_metrics,
                "baseline": baseline_metrics,
                "difference": differences,
                "differences": differences,
            },
            "metric_differences": differences,
            "metrics_difference": differences,
            "error_data": error_series,
            "model_comparison": {
                "candidate": candidate_model,
                "new_model": candidate_model,
                "baseline": baseline_model,
                "production": baseline_model,
                "current_model": baseline_model,
                "changes": differences,
            },
        }


__all__ = ["ModelEvaluationError", "ModelEvaluationService"]
