import logging

import pandas as pd
import pytest

from backend.app.services.training_executor import TrainingExecutionResult, TrainingScriptExecutor


@pytest.fixture
def executor():
    return TrainingScriptExecutor()


def _data():
    X_train = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y_train = pd.Series([2.0, 4.0, 6.0])
    X_test = pd.DataFrame({"feature": [4.0]})
    y_test = pd.Series([8.0])
    return X_train, y_train, X_test, y_test


def test_valid_script_receives_all_inputs_and_returns_predictable_model(executor):
    source = """
class Model:
    def __init__(self, received):
        self.received = received
    def predict(self, X):
        return [42] * len(X)

def train(X_train, y_train, X_test, y_test, config):
    print('training started')
    return Model((len(X_train), len(y_train), len(X_test), len(y_test), config['marker']))
"""
    X_train, y_train, X_test, y_test = _data()
    config = {"marker": "passed"}
    result = executor.execute(source, X_train, y_train, X_test, y_test, config)

    assert isinstance(result, TrainingExecutionResult)
    assert result.success is True
    assert result.status == "SUCCEEDED"
    assert result.model.predict(X_test) == [42]
    assert result.model.received == (3, 3, 1, 1, "passed")
    assert "training started" in result.logs
    assert result.error is None
    assert result.to_dict()["result"] is result.model


def test_missing_predict_is_reported_as_a_structured_failure(executor):
    result = executor.execute(
        "def train(X_train, y_train, X_test, y_test, config):\n    return object()\n",
        *_data(),
        {},
    )

    assert result.success is False
    assert result.status == "FAILED"
    assert result.error_code == "MODEL_PREDICT_INVALID"
    assert "predict" in result.error
    assert result.model is None


def test_non_callable_predict_is_reported(executor):
    source = """
class Model:
    predict = None

def train(X_train, y_train, X_test, y_test, config):
    return Model()
"""
    result = executor.execute(source, *_data(), {})

    assert result.success is False
    assert result.error_code == "MODEL_PREDICT_INVALID"
    assert "callable" in result.error.lower()


def test_training_exception_and_logs_are_collected(executor):
    source = """
import logging

def train(X_train, y_train, X_test, y_test, config):
    print('before failure')
    logging.getLogger('trainer').warning('logged warning')
    raise ValueError('bad training data')
"""
    result = executor.execute(source, *_data(), {})

    assert result.success is False
    assert result.error_code == "TRAIN_EXECUTION_FAILED"
    assert result.exception_type == "ValueError"
    assert "bad training data" in result.error
    assert "before failure" in result.logs
    assert any("logged warning" in item for item in result.logs)
    assert result.traceback and "ValueError" in result.traceback


def test_logs_and_logging_state_are_restored(executor):
    root = logging.getLogger()
    old_level = root.level
    result = executor.execute(
        """
import logging

def train(X_train, y_train, X_test, y_test, config):
    logging.getLogger('trainer').info('info log')
    print('print log')
    return type('Model', (), {'predict': lambda self, X: []})()
""",
        *_data(),
        {},
    )

    assert result.success
    assert "print log" in result.logs
    assert any("info log" in item for item in result.logs)
    assert root.level == old_level


def test_source_code_keyword_and_mapping_result_access_are_supported(executor):
    result = executor.execute(
        source_code="def train(X_train, y_train, X_test, y_test, config):\n    return type('M', (), {'predict': lambda self, X: X})()\n",
        X_train=_data()[0],
        y_train=_data()[1],
        X_test=_data()[2],
        y_test=_data()[3],
        config={},
    )

    assert result["success"] is True
    assert result["model"] is result.model
    assert result["logs"] == result.logs
