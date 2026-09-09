"""Prepare the raw electric-load CSV for the platform upload contract.

The platform treats the first CSV column as time and the last column as the
regression target.  The source file has no timestamp column, so this adapter
creates a deterministic *platform sorting time axis* from ``scenario`` and
``month/day/hour``.  These timestamps are ordering keys only; they do not claim
that the source scenarios occurred in the represented calendar years.

This script never writes to the input path.  Pass both paths explicitly, for
example::

    python prepare_dataset.py --input raw.csv --output electric_load_platform.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEQUENCE_COLUMN = "sequence_index"
SCENARIO_COLUMN = "scenario"
SCENARIO_CODE_COLUMN = "scenario_code"
TARGET_COLUMN = "target_total_purchased_electricity_w"
TIMESTAMP_COLUMN = "timestamp"
TIME_COMPONENTS = ("month", "day", "hour")
REQUIRED_COLUMNS = {
    SEQUENCE_COLUMN,
    SCENARIO_COLUMN,
    *TIME_COMPONENTS,
    TARGET_COLUMN,
}
# A non-leap reference year makes day/month validation deterministic.  The
# resulting date is not an assertion about the source data's real year.
REFERENCE_YEAR = 2001
PLATFORM_EPOCH = pd.Timestamp("2000-01-01 00:00:00")


class DatasetPreparationError(ValueError):
    """Raised when the fixed raw electric-load schema cannot be adapted."""


@dataclass(frozen=True)
class PreparationSummary:
    """Facts about one successfully generated platform CSV."""

    input_path: str
    output_path: str
    row_count: int
    input_column_count: int
    output_columns: tuple[str, ...]
    dropped_columns: tuple[str, ...]
    scenario_mapping: dict[str, int]
    timestamp_start: str
    timestamp_end: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "row_count": self.row_count,
            "input_column_count": self.input_column_count,
            "output_columns": list(self.output_columns),
            "dropped_columns": list(self.dropped_columns),
            "scenario_mapping": dict(self.scenario_mapping),
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
        }


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stripped, unique string headers."""

    output = frame.copy(deep=True)
    columns = [str(column).strip() for column in output.columns]
    if any(not column for column in columns):
        raise DatasetPreparationError("原始 CSV 表头不能包含空字段名")
    if len(set(columns)) != len(columns):
        raise DatasetPreparationError("原始 CSV 表头不能包含重复字段名")
    output.columns = columns
    return output


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Convert one source column to numeric and turn infinities into missing."""

    values = pd.to_numeric(frame[column], errors="coerce")
    finite = values.notna() & np.isfinite(values.to_numpy(dtype=float))
    return values.where(finite, np.nan)


def _integer_component(frame: pd.DataFrame, column: str) -> pd.Series:
    values = _numeric_series(frame, column)
    array = values.to_numpy(dtype=float)
    is_integer = np.equal(array, np.floor(array), where=np.isfinite(array)).all()
    if values.isna().any() or not is_integer:
        raise DatasetPreparationError(f"时间组件“{column}”必须是有限整数")
    return values.astype("int64")


def _scenario_codes(frame: pd.DataFrame) -> tuple[pd.Series, dict[str, int]]:
    values = frame[SCENARIO_COLUMN].astype("string").str.strip()
    missing = values.isna() | values.eq("")
    if missing.any():
        raise DatasetPreparationError("scenario 不能包含空值")
    names = sorted(str(value) for value in values.unique())
    mapping = {name: index for index, name in enumerate(names)}
    return values.map(mapping).astype("int64"), mapping


def _platform_timestamps(
    frame: pd.DataFrame, scenario_code: pd.Series
) -> tuple[pd.Series, np.ndarray]:
    """Build unique timestamps and an integer key used for stable sorting."""

    month = _integer_component(frame, "month")
    day = _integer_component(frame, "day")
    hour = _integer_component(frame, "hour")
    if ((month < 1) | (month > 12)).any():
        raise DatasetPreparationError("month 必须在 1 到 12 之间")
    if ((day < 1) | (day > 31)).any():
        raise DatasetPreparationError("day 必须在 1 到 31 之间")
    if ((hour < 1) | (hour > 24)).any():
        raise DatasetPreparationError("hour 必须在 1 到 24 之间")

    keys = pd.DataFrame(
        {
            SCENARIO_COLUMN: frame[SCENARIO_COLUMN].astype("string").str.strip(),
            "month": month,
            "day": day,
            "hour": hour,
        },
        index=frame.index,
    )
    if keys.duplicated().any():
        raise DatasetPreparationError(
            "同一 scenario 的 month/day/hour 组合必须唯一"
        )

    calendar_dates = pd.to_datetime(
        pd.DataFrame(
            {
                "year": pd.Series(REFERENCE_YEAR, index=frame.index),
                "month": month,
                "day": day,
            }
        ),
        errors="coerce",
    )
    if calendar_dates.isna().any():
        raise DatasetPreparationError("month/day 组合包含无效日期")

    # Source hours are 1-based.  Adding 24 hours maps hour=24 to the next
    # day's 00:00, rather than creating a duplicate at 24:00.
    local_hours = (
        (calendar_dates - pd.Timestamp(f"{REFERENCE_YEAR}-01-01"))
        / pd.Timedelta(hours=1)
    ).astype("int64") + hour
    max_local_hour = int(local_hours.max())
    # Leave one unused tick between scenario blocks.  This prevents equal
    # local calendar coordinates in different scenarios from sharing a time.
    scenario_span = max_local_hour + 1
    platform_key = (
        scenario_code.to_numpy(dtype="int64") * scenario_span
        + local_hours.to_numpy(dtype="int64")
    )
    if len(np.unique(platform_key)) != len(platform_key):
        raise DatasetPreparationError("派生平台时间轴包含重复值")

    timestamps = PLATFORM_EPOCH + pd.to_timedelta(platform_key, unit="h")
    if not timestamps.is_unique:
        raise DatasetPreparationError("派生 timestamp 必须唯一")
    return pd.Series(timestamps, index=frame.index), platform_key


def _validate_platform_frame(frame: pd.DataFrame) -> None:
    """Fail closed if the output no longer matches positional platform roles."""

    if frame.empty:
        raise DatasetPreparationError("原始 CSV 至少需要一行数据")
    if frame.columns[0] != TIMESTAMP_COLUMN:
        raise DatasetPreparationError("输出首列必须是 timestamp")
    if frame.columns[-1] != TARGET_COLUMN:
        raise DatasetPreparationError("输出末列必须是目标列")
    if len(set(frame.columns)) != len(frame.columns):
        raise DatasetPreparationError("输出字段名不能重复")

    parsed = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce", format="mixed")
    if (
        parsed.isna().any()
        or not parsed.is_unique
        or not parsed.is_monotonic_increasing
    ):
        raise DatasetPreparationError("输出 timestamp 必须可解析、唯一且递增")
    if SCENARIO_CODE_COLUMN not in frame.columns:
        raise DatasetPreparationError("输出必须包含数值 scenario_code")
    scenario_codes = frame[SCENARIO_CODE_COLUMN].to_numpy(dtype=float)
    if (
        not np.isfinite(scenario_codes).all()
        or not np.equal(scenario_codes, np.floor(scenario_codes)).all()
    ):
        raise DatasetPreparationError("scenario_code 必须是有限整数编码")
    for column in frame.columns[1:]:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise DatasetPreparationError(f"输出特征/目标不是数值列：{column}")
    target = frame[TARGET_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise DatasetPreparationError("目标列必须全部是有限数值")


def build_platform_frame(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    """Convert an in-memory raw frame without mutating ``source``.

    The accepted raw layout intentionally mirrors the supplied file: no
    timestamp column, ``sequence_index`` first, ``scenario`` as a string, and
    the named target column last.  Numeric columns with no finite value after
    conversion are omitted; partial missing values remain available for the
    uploaded preprocessor to impute from training statistics.
    """

    frame = _normalise_columns(source)
    if frame.columns[0] != SEQUENCE_COLUMN:
        raise DatasetPreparationError(
            f"原始 CSV 首列必须是 {SEQUENCE_COLUMN}，不能把派生 timestamp 当作原始数据"
        )
    if TIMESTAMP_COLUMN in frame.columns:
        raise DatasetPreparationError("原始 CSV 不应包含 timestamp 列")
    if frame.columns[-1] != TARGET_COLUMN:
        raise DatasetPreparationError(f"原始 CSV 末列必须是 {TARGET_COLUMN}")
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise DatasetPreparationError(
            "原始 CSV 缺少必需字段：" + ", ".join(missing_columns)
        )
    if frame.empty:
        raise DatasetPreparationError("原始 CSV 至少需要一行数据")

    scenario_code, mapping = _scenario_codes(frame)
    timestamps, platform_key = _platform_timestamps(frame, scenario_code)
    target = _numeric_series(frame, TARGET_COLUMN)
    if target.isna().any():
        raise DatasetPreparationError("目标列必须全部是有限数值")

    output = pd.DataFrame(index=frame.index)
    output[TIMESTAMP_COLUMN] = timestamps
    dropped: list[str] = []
    for column in frame.columns:
        if column == TARGET_COLUMN:
            continue
        if column == SCENARIO_COLUMN:
            output[SCENARIO_CODE_COLUMN] = scenario_code
            continue
        values = _numeric_series(frame, column)
        if not values.notna().any():
            dropped.append(column)
            continue
        output[column] = values
    output[TARGET_COLUMN] = target

    order = pd.Series(platform_key, index=frame.index).sort_values(
        kind="mergesort"
    ).index
    output = output.loc[order].reset_index(drop=True)
    if len(output) != len(frame):
        raise DatasetPreparationError("派生 CSV 行数必须与原始 CSV 相同")
    _validate_platform_frame(output)
    return output, mapping, dropped


def prepare_dataset(
    input_path: str | Path, output_path: str | Path
) -> PreparationSummary:
    """Read, adapt, validate, and write one derived platform CSV."""

    source_path = Path(input_path)
    destination_path = Path(output_path)
    if source_path.resolve() == destination_path.resolve():
        raise DatasetPreparationError("输出路径不能覆盖原始 CSV")
    if not source_path.is_file():
        raise FileNotFoundError(f"原始 CSV 不存在：{source_path}")

    source = pd.read_csv(source_path, dtype=object, keep_default_na=True)
    output, mapping, dropped = build_platform_frame(source)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination_path, index=False)
    # Re-read the written file so the reported contract covers the actual CSV
    # representation consumed by the platform, not only the in-memory frame.
    written = pd.read_csv(destination_path)
    if len(written) != len(source):
        raise DatasetPreparationError("写出后的 CSV 行数与原始 CSV 不一致")
    _validate_platform_frame(written)
    return PreparationSummary(
        input_path=str(source_path),
        output_path=str(destination_path),
        row_count=len(written),
        input_column_count=len(source.columns),
        output_columns=tuple(str(column) for column in written.columns),
        dropped_columns=tuple(dropped),
        scenario_mapping=mapping,
        timestamp_start=str(written[TIMESTAMP_COLUMN].iloc[0]),
        timestamp_end=str(written[TIMESTAMP_COLUMN].iloc[-1]),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把无 timestamp 的电力负荷原始 CSV 转成平台可上传 CSV"
    )
    parser.add_argument("--input", required=True, type=Path, help="原始 CSV 路径")
    parser.add_argument("--output", required=True, type=Path, help="派生 CSV 路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = prepare_dataset(args.input, args.output)
    except (DatasetPreparationError, OSError, pd.errors.ParserError, ValueError) as exc:
        _parser().error(str(exc))
    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
