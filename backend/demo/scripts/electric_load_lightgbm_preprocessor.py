"""Platform preprocessor for the prepared electric-load dataset.

The platform calls exactly one ``Preprocessor`` with ``fit(df, config)`` and
then calls ``transform(df, config)`` on each partition.  Fitted medians are
learned from the training partition only; no source files or global state are
read here.  The first and last dataset columns are retained as the platform
 time and target columns (or explicit ``config`` names when supplied).
"""

import numpy as np
import pandas as pd


class Preprocessor:
    """Convert usable feature columns to numeric and fill them consistently."""

    def fit(self, df, config):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("预处理输入必须是 pandas.DataFrame")
        if len(df.columns) < 3:
            raise ValueError("预处理输入至少需要时间、特征和目标列")

        options = config if isinstance(config, dict) else {}
        self.time_column = str(options.get("time_column", df.columns[0]))
        self.target_column = str(options.get("target_column", df.columns[-1]))
        if self.time_column not in df.columns:
            raise ValueError("预处理输入缺少平台时间列：" + self.time_column)
        if self.target_column not in df.columns:
            raise ValueError("预处理输入缺少平台目标列：" + self.target_column)
        if self.time_column == self.target_column:
            raise ValueError("平台时间列和目标列不能相同")

        self.feature_columns = []
        self.fill_values = {}
        for column in df.columns:
            column = str(column)
            if column in {self.time_column, self.target_column}:
                continue
            numeric = pd.to_numeric(df[column], errors="coerce")
            finite = numeric.notna() & np.isfinite(numeric.to_numpy(dtype=float))
            if not finite.any():
                # This includes source features that are entirely empty or
                # entirely non-numeric.  They cannot provide a trainable value.
                continue
            value = float(numeric.loc[finite].median())
            if not np.isfinite(value):
                continue
            self.feature_columns.append(column)
            self.fill_values[column] = value

        if not self.feature_columns:
            raise ValueError("预处理后没有可用数值特征")
        return self

    def transform(self, df, config):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("预处理输入必须是 pandas.DataFrame")
        if not hasattr(self, "feature_columns"):
            raise ValueError("预处理器必须先执行 fit")
        if self.time_column not in df.columns:
            raise ValueError("预处理输入缺少平台时间列：" + self.time_column)
        if self.target_column not in df.columns:
            raise ValueError("预处理输入缺少平台目标列：" + self.target_column)

        output = pd.DataFrame(index=df.index)
        # Keep the positional platform roles explicit and unchanged.
        output[self.time_column] = df[self.time_column]
        for column in self.feature_columns:
            if column in df.columns:
                values = pd.to_numeric(df[column], errors="coerce")
            else:
                values = pd.Series(np.nan, index=df.index)
            values = values.replace([np.inf, -np.inf], np.nan)
            output[column] = values.fillna(self.fill_values[column]).astype(float)
        output[self.target_column] = df[self.target_column]
        return output
