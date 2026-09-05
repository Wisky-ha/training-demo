"""Minimal scikit-learn trainer for the internal demonstration."""

import sklearn.linear_model


def train(X_train, y_train, X_test, y_test, config):
    """Return any object implementing predict(X); test data is supplied for API compatibility."""
    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)
    return model
