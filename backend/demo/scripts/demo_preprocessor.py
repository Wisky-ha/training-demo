"""Minimal preprocessing script for the internal demonstration."""

import pandas as pd


class Preprocessor:
    """Fill numeric gaps while preserving the platform time/target columns."""

    def fit(self, df, config):
        self.numeric_columns = [
            column for column in df.columns if column != "timestamp"
        ]
        self.fill_values = {}
        for column in self.numeric_columns:
            values = pd.to_numeric(df[column], errors="coerce")
            self.fill_values[column] = float(values.mean())
        return self

    def transform(self, df, config):
        output = df.copy()
        for column in self.numeric_columns:
            # Prediction rows do not contain the training target column.
            if column in output.columns:
                output[column] = pd.to_numeric(
                    output[column], errors="coerce"
                ).fillna(self.fill_values[column])
        return output
