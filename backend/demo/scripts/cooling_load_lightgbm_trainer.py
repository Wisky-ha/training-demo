"""LightGBM regression trainer for the platform cooling-load demo.

The platform supplies already-partitioned feature frames and target series to
``train``.  This script deliberately imports and uses LightGBM directly; a
missing LightGBM installation is an execution prerequisite failure, not a
reason to substitute another estimator.

Unlike the electric-load trainer, the defaults here were selected for the
single-year cooling-load dataset: a smaller tree (31 leaves) with a slightly
higher learning rate generalised better than the 63-leaf default during
model selection on an internal validation split.
"""

import lightgbm as lgb


DEFAULT_SEED = 20260720
DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.1,
    "random_state": DEFAULT_SEED,
    "bagging_seed": DEFAULT_SEED,
    "feature_fraction_seed": DEFAULT_SEED,
    "data_random_seed": DEFAULT_SEED,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": -1,
    "verbosity": -1,
}


def _parameters(config):
    """Merge supported flat or ``lightgbm_params`` overrides only."""

    options = config if isinstance(config, dict) else {}
    nested = options.get("lightgbm_params", {})
    nested = nested if isinstance(nested, dict) else {}
    params = dict(DEFAULT_PARAMS)
    for name in DEFAULT_PARAMS:
        if name in options:
            params[name] = options[name]
        if name in nested:
            params[name] = nested[name]

    # Keep every LightGBM seed aligned unless a caller explicitly supplies a
    # seed parameter.  The default remains deterministic across executions.
    seed = params["random_state"]
    if (
        "seed" in options
        and "random_state" not in options
        and "random_state" not in nested
    ):
        seed = options["seed"]
        params["random_state"] = seed
    for name in ("bagging_seed", "feature_fraction_seed", "data_random_seed"):
        if name not in options and name not in nested:
            params[name] = seed
    return params


def train(X_train, y_train, X_test, y_test, config):
    """Fit and return a LightGBM regressor implementing ``predict(X)``."""

    options = config if isinstance(config, dict) else {}
    stopping_rounds = int(options.get("early_stopping_rounds", 100))
    log_period = int(options.get("log_period", 100))
    if stopping_rounds < 0:
        raise ValueError("early_stopping_rounds 不能为负数")
    if log_period < 0:
        raise ValueError("log_period 不能为负数")

    model = lgb.LGBMRegressor(**_parameters(options))
    callbacks = []
    if stopping_rounds:
        callbacks.append(lgb.early_stopping(stopping_rounds, verbose=False))
    if log_period:
        callbacks.append(lgb.log_evaluation(period=log_period))
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=options.get("eval_metric", "rmse"),
        callbacks=callbacks,
    )
    return model
