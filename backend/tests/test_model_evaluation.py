"""Unit tests for complete, side-by-side model evaluation."""

import numpy as np
import pytest

from backend.app.services.model_evaluation import ModelEvaluationError, ModelEvaluationService


def test_evaluates_all_metrics_and_persists_full_series_and_differences():
    result = ModelEvaluationService.evaluate(
        times=["t1", "t2", "t3"],
        actual=[2.0, 0.0, 4.0],
        candidate_predictions=[1.0, 3.0, 5.0],
        baseline_predictions=[2.0, 2.0, 2.0],
        candidate_id="candidate",
        baseline_id="baseline",
    )

    assert result["sample_count"] == 3
    assert result["metrics"]["candidate"]["mae"] == pytest.approx(5 / 3)
    assert result["metrics"]["candidate"]["rmse"] == pytest.approx(np.sqrt(11 / 3))
    assert result["metrics"]["candidate"]["mape"] == pytest.approx(37.5)
    assert result["metrics"]["candidate"]["r2"] == pytest.approx(-0.375)
    assert result["metrics"]["candidate"]["mape_valid_count"] == 2
    assert result["metrics"]["baseline"]["mae"] == pytest.approx(4 / 3)
    assert result["metric_differences"]["mae"] == pytest.approx(1 / 3)
    assert result["metric_differences"]["mape"] == pytest.approx(12.5)

    assert result["test_time_series"] == ["t1", "t2", "t3"]
    assert result["actual_values"] == [2.0, 0.0, 4.0]
    assert result["candidate_predictions"] == [1.0, 3.0, 5.0]
    assert result["baseline_predictions"] == [2.0, 2.0, 2.0]
    assert result["candidate_errors"] == [1.0, -3.0, -1.0]
    assert result["baseline_errors"] == [0.0, -2.0, 2.0]
    assert result["error_series"] == result["candidate_errors"]
    assert result["model_comparison"]["candidate"]["model_version_id"] == "candidate"
    assert result["model_comparison"]["baseline"]["model_version_id"] == "baseline"


def test_mape_excludes_zero_values_and_all_zero_returns_null_with_explanation():
    result = ModelEvaluationService.evaluate(
        times=[1, 2], actual=[0.0, 0.0], candidate_predictions=[1.0, 2.0], baseline_predictions=[3.0, 4.0]
    )

    for metrics in (result["metrics"]["candidate"], result["metrics"]["baseline"]):
        assert metrics["mape"] is None
        assert metrics["mape_valid_count"] == 0
        assert metrics["mape_excluded_count"] == 2
        assert "没有有效样本" in metrics["mape_note"]
    assert result["metric_differences"]["mape"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"times": [1], "actual": [], "candidate_predictions": [], "baseline_predictions": []},
        {"times": [1, 2], "actual": [1], "candidate_predictions": [1, 2], "baseline_predictions": [1, 2]},
        {"times": [1], "actual": [1], "candidate_predictions": [np.nan], "baseline_predictions": [1]},
        {"times": [1], "actual": [1], "candidate_predictions": [1], "baseline_predictions": [np.inf]},
    ],
)
def test_rejects_invalid_input(kwargs):
    with pytest.raises(ModelEvaluationError):
        ModelEvaluationService.evaluate(**kwargs)
