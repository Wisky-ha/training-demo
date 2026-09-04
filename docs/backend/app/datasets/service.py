"""CSV parsing, inspection, and persistence for uploaded datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from sqlalchemy.orm import Session

from ..core.config import Settings, get_settings
from ..db.repositories import DatasetRepository
from ..domain.models import DatasetRecord
from ..storage import FileStorageService, StoredArtifact


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One machine-readable dataset validation failure or warning."""

    code: str
    field: str
    message: str
    column: str | None = None
    row_numbers: list[int] = field(default_factory=list)
    count: int | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "field": self.field,
            "message": self.message,
        }
        if self.column is not None:
            result["column"] = self.column
        if self.row_numbers:
            result["row_numbers"] = self.row_numbers
        if self.count is not None:
            result["count"] = self.count
        return result


class CSVParseError(ValueError):
    """Raised when an uploaded file is not a valid supported dataset CSV."""

    default_code = "CSV_INVALID"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        field: str = "file",
        column: str | None = None,
        row_numbers: list[int] | None = None,
        count: int | None = None,
        issues: list[ValidationIssue] | None = None,
    ) -> None:
        super().__init__(message)
        self.issues = issues or [
            ValidationIssue(
                code=code or self.default_code,
                field=field,
                message=message,
                column=column,
                row_numbers=row_numbers or [],
                count=count,
            )
        ]


class DatasetValidationError(CSVParseError):
    """Raised when one or more parsed dataset values violate a rule."""

    def __init__(self, issues: list[ValidationIssue]) -> None:
        if not issues:
            raise ValueError("DatasetValidationError requires at least one issue")
        super().__init__(
            "；".join(issue.message for issue in issues),
            code="DATASET_VALIDATION_FAILED",
            field="dataset",
            issues=issues,
        )


class UnsupportedDatasetFileError(CSVParseError):
    """Raised when the uploaded filename or media type is not CSV."""

    default_code = "UNSUPPORTED_DATASET_FILE"


@dataclass(frozen=True, slots=True)
class ParsedDataset:
    """The inspection result produced before a dataset is persisted."""

    file_name: str
    row_count: int
    columns: list[str]
    time_column: str
    feature_columns: list[str]
    target_column: str
    column_types: dict[str, str]
    missing_value_counts: dict[str, int]
    preview_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    time_parse: dict[str, Any]
    time_range: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None

    @property
    def column_count(self) -> int:
        return len(self.columns)

    @property
    def preview(self) -> list[dict[str, Any]]:
        return self.preview_rows

    def as_dict(self) -> dict[str, Any]:
        """Return the stable, JSON-compatible inspection response."""

        missing_values = {
            column: {
                "missing_count": count,
                "missing_ratio": count / self.row_count if self.row_count else 0.0,
            }
            for column, count in self.missing_value_counts.items()
        }
        roles = {
            self.time_column: "time",
            self.target_column: "target",
            **{column: "feature" for column in self.feature_columns},
        }
        columns = [
            {
                "name": column,
                "role": roles[column],
                "data_type": self.column_types[column],
                "nullable": self.missing_value_counts[column] > 0,
                "missing_count": self.missing_value_counts[column],
                "missing_ratio": missing_values[column]["missing_ratio"],
            }
            for column in self.columns
        ]
        result = {
            "file_name": self.file_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": columns,
            "column_names": self.columns,
            "time_column": self.time_column,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "field_roles": roles,
            "time_parse": self.time_parse,
            "time_range": self.time_range,
            "validation": self.validation,
            "numeric_columns": [
                column for column in self.columns if self.column_types[column] == "number"
            ],
            "column_types": self.column_types,
            "missing_values": missing_values,
            "missing_value_counts": self.missing_value_counts,
            "preview_rows": self.preview_rows,
            "preview": self.preview_rows,
            "summary": self.summary,
            "data_summary": self.summary,
            "status": "parsed",
        }
        if self.created_at is not None:
            result["created_at"] = self.created_at.isoformat()
        return result


class DatasetService:
    """Parse CSV data and optionally save it with its dataset metadata.

    The parser intentionally uses positional roles: first column is time, last
    column is target, and every column between them is a feature.  No later
    training or prediction behaviour is performed here.
    """

    DEFAULT_PREVIEW_ROWS = 5
    CSV_CONTENT_TYPES = {
        "text/csv",
        "text/plain",
        "application/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
    }

    def __init__(
        self,
        session: Session | None = None,
        storage: FileStorageService | None = None,
        *,
        settings: Settings | None = None,
        preview_rows: int = DEFAULT_PREVIEW_ROWS,
    ) -> None:
        if preview_rows < 1:
            raise ValueError("preview_rows must be at least 1")
        self.session = session
        self.settings = settings or get_settings()
        # An explicit shared storage root from step 4 wins; otherwise dataset
        # uploads live in the configured upload directory rather than model
        # artifacts.
        dataset_root = self.settings.storage_root or self.settings.upload_storage_dir
        self.storage = storage or FileStorageService(dataset_root, session=session)
        self.preview_rows = preview_rows

    @staticmethod
    def _filename(file_name: str | None) -> str:
        if not file_name or not isinstance(file_name, str):
            raise UnsupportedDatasetFileError("必须提供 CSV 文件名")
        # Upload clients can send a client-side path.  Store only its final
        # component; the path is never used to select a storage location.
        name = file_name.replace("\\", "/").rsplit("/", 1)[-1].strip()
        if not name or Path(name).suffix.lower() != ".csv":
            raise UnsupportedDatasetFileError("仅支持 .csv 文件")
        return name

    @classmethod
    def _decode(cls, content: bytes) -> str:
        if not content or not content.strip():
            raise CSVParseError("CSV 文件内容为空", code="FILE_EMPTY")
        # UTF-8 (including its BOM) is the API default.  GB18030 is accepted
        # for common Chinese desktop exports while still rejecting binary data.
        for encoding in ("utf-8-sig", "gb18030"):
            try:
                text = content.decode(encoding)
                if not text.strip().lstrip("\ufeff"):
                    raise CSVParseError("CSV 文件内容为空", code="FILE_EMPTY")
                return text
            except UnicodeDecodeError:
                continue
        raise CSVParseError(
            "CSV 文件必须使用 UTF-8 或 GB18030 编码", code="ENCODING_INVALID"
        )

    @staticmethod
    def _json_value(value: Any) -> Any:
        """Convert pandas/numpy values to strict-JSON-safe primitives."""

        if value is None:
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat() if not pd.isna(value) else None
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value

    @staticmethod
    def _missing_mask(series: pd.Series) -> pd.Series:
        """Treat whitespace-only CSV cells as missing as well as CSV nulls."""

        return series.isna() | series.astype("string").str.strip().eq("")

    @classmethod
    def _non_missing(cls, series: pd.Series) -> pd.Series:
        return series.loc[~cls._missing_mask(series)]

    @classmethod
    def _numeric_values(cls, series: pd.Series) -> pd.Series | None:
        non_missing = cls._non_missing(series)
        if non_missing.empty:
            return None
        converted = pd.to_numeric(non_missing.astype(str).str.strip(), errors="coerce")
        if converted.isna().any() or not np.isfinite(converted.to_numpy(dtype=float)).all():
            return None
        return converted

    @classmethod
    def _is_datetime(cls, series: pd.Series) -> bool:
        non_missing = cls._non_missing(series)
        if non_missing.empty:
            return False
        converted = pd.to_datetime(
            non_missing.astype(str).str.strip(), errors="coerce", format="mixed"
        )
        return not converted.isna().any()

    @classmethod
    def _column_type(cls, series: pd.Series, *, time: bool = False) -> str:
        if time:
            return "datetime"
        non_missing = cls._non_missing(series)
        if non_missing.empty:
            return "unknown"
        if cls._numeric_values(series) is not None:
            return "number"
        boolean_values = non_missing.astype(str).str.strip().str.lower()
        if boolean_values.isin({"true", "false"}).all():
            return "boolean"
        if cls._is_datetime(series):
            return "datetime"
        return "string"

    @classmethod
    def _summary_for_column(
        cls, series: pd.Series, column_type: str, missing_count: int
    ) -> dict[str, Any]:
        non_missing = cls._non_missing(series)
        summary: dict[str, Any] = {
            "type": column_type,
            "missing_count": missing_count,
            "non_missing_count": int(non_missing.size),
            "unique_count": int(non_missing.nunique(dropna=True)),
        }
        if non_missing.empty:
            return summary
        if column_type == "number":
            numbers = cls._numeric_values(series)
            assert numbers is not None
            summary.update(
                {
                    "min": cls._json_value(numbers.min()),
                    "max": cls._json_value(numbers.max()),
                    "mean": cls._json_value(numbers.mean()),
                }
            )
        elif column_type == "datetime":
            dates = pd.to_datetime(
                non_missing.astype(str).str.strip(), errors="coerce", format="mixed"
            )
            if not dates.isna().any():
                summary.update(
                    {
                        "min": cls._json_value(dates.min()),
                        "max": cls._json_value(dates.max()),
                    }
                )
        return summary

    @classmethod
    def _raw_header_and_validate_rows(cls, text: str) -> list[str]:
        """Validate CSV record widths before pandas can infer an index.

        Pandas may silently use the first field as an index for some malformed
        rows.  The standard-library reader gives us a strict, comma-delimited
        record check while preserving empty fields as legitimate missing data.
        """

        reader = csv.reader(StringIO(text), strict=True)
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise CSVParseError("CSV 文件缺少表头", code="HEADER_MISSING") from exc
        expected = len(raw_header)
        for row in reader:
            if not row:  # blank lines are ignored by pandas as well
                continue
            if len(row) != expected:
                raise CSVParseError(
                    f"CSV 第 {reader.line_num} 行包含 {len(row)} 列，预期 {expected} 列",
                    code="COLUMN_COUNT_MISMATCH",
                    field="columns",
                    row_numbers=[reader.line_num],
                )
        return [str(column).strip() for column in raw_header]

    @classmethod
    def parse_csv(
        cls,
        file_name: str,
        content: bytes,
        *,
        preview_rows: int = DEFAULT_PREVIEW_ROWS,
    ) -> ParsedDataset:
        """Validate and inspect CSV bytes without touching the database."""

        name = cls._filename(file_name)
        if preview_rows < 1:
            raise ValueError("preview_rows must be at least 1")
        text = cls._decode(content)
        try:
            raw_columns = cls._raw_header_and_validate_rows(text)
        except csv.Error as exc:
            raise CSVParseError(
                f"CSV 内容格式错误：{exc}", code="CSV_FORMAT_INVALID"
            ) from exc
        try:
            frame = pd.read_csv(
                StringIO(text),
                header=0,
                dtype=object,
                keep_default_na=True,
                na_filter=True,
                on_bad_lines="error",
                index_col=False,
            )
        except EmptyDataError as exc:
            raise CSVParseError("CSV 文件缺少表头") from exc
        except (ParserError, ValueError) as exc:
            raise CSVParseError(f"CSV 内容无法解析：{exc}") from exc

        # Duplicate headers make role metadata ambiguous.
        columns = raw_columns
        if len(columns) < 3:
            raise CSVParseError(
                "CSV 至少需要 3 列：时间列、特征列和目标列",
                code="COLUMN_COUNT_TOO_SMALL",
                field="columns",
                count=len(columns),
            )
        if any(not column for column in columns):
            raise CSVParseError(
                "CSV 表头不能包含空字段名",
                code="COLUMN_NAME_EMPTY",
                field="columns",
            )
        if len(set(columns)) != len(columns):
            raise CSVParseError(
                "CSV 表头不能包含重复字段名",
                code="COLUMN_NAME_DUPLICATE",
                field="columns",
            )
        if frame.empty:
            raise CSVParseError(
                "CSV 至少需要一行数据",
                code="NO_DATA_ROWS",
                field="rows",
            )
        # ``raw_header`` and the normal parse should have the same width.  If
        # a malformed parser edge case disagrees, fail closed rather than
        # assigning role metadata to the wrong values.
        if len(columns) != len(frame.columns):
            raise CSVParseError(
                "CSV 表头和数据列数不一致",
                code="COLUMN_COUNT_MISMATCH",
                field="columns",
            )
        frame.columns = columns

        row_count = int(len(frame))
        feature_columns = columns[1:-1]
        target_column = columns[-1]
        issues: list[ValidationIssue] = []
        if not feature_columns:
            issues.append(
                ValidationIssue(
                    code="FEATURE_COLUMNS_EMPTY",
                    field="feature_columns",
                    message="CSV 至少需要一个特征字段",
                )
            )
        split_index = int(row_count * 0.8)
        train_rows = split_index
        test_rows = row_count - split_index
        if train_rows < 1 or test_rows < 1:
            issues.append(
                ValidationIssue(
                    code="INSUFFICIENT_ROWS",
                    field="rows",
                    message="数据量不足以完成 80%/20% 划分，训练集和测试集都必须非空",
                    count=row_count,
                )
            )

        time_column = columns[0]
        time_values = frame[time_column]
        time_missing = cls._missing_mask(time_values)
        if time_missing.any():
            rows = [int(index) + 2 for index in frame.index[time_missing][:10]]
            issues.append(
                ValidationIssue(
                    code="TIME_VALUE_MISSING",
                    field="time",
                    column=time_column,
                    message=f"时间字段“{time_column}”包含缺失值，无法解析",
                    row_numbers=rows,
                    count=int(time_missing.sum()),
                )
            )
        parsed_times = pd.to_datetime(
            time_values.where(~time_missing, None).astype("string").str.strip(),
            errors="coerce",
            format="mixed",
        )
        invalid_times = parsed_times.isna() & ~time_missing
        if invalid_times.any():
            rows = [int(index) + 2 for index in frame.index[invalid_times][:10]]
            issues.append(
                ValidationIssue(
                    code="TIME_PARSE_FAILED",
                    field="time",
                    column=time_column,
                    message=f"时间字段“{time_column}”存在无法解析的值（第 {rows[0]} 行）",
                    row_numbers=rows,
                    count=int(invalid_times.sum()),
                )
            )

        # Ordering is a warning because the documented split operation sorts
        # by time.  Duplicate timestamps, however, make a time boundary
        # ambiguous and are rejected at upload time.
        duplicate_count = 0
        out_of_order_count = 0
        is_sorted = True
        if not time_missing.any() and not invalid_times.any():
            duplicate_count = int(parsed_times.duplicated().sum())
            is_sorted = bool(parsed_times.is_monotonic_increasing)
            out_of_order_count = int(
                (parsed_times.iloc[1:].to_numpy() < parsed_times.iloc[:-1].to_numpy()).sum()
            )
            if duplicate_count:
                issues.append(
                    ValidationIssue(
                        code="TIME_DUPLICATE",
                        field="time",
                        column=time_column,
                        message=f"时间字段“{time_column}”包含重复时间值",
                        count=duplicate_count,
                    )
                )

        target_values = frame[target_column]
        target_missing = cls._missing_mask(target_values)
        if target_missing.any():
            rows = [int(index) + 2 for index in frame.index[target_missing][:10]]
            issues.append(
                ValidationIssue(
                    code="TARGET_VALUE_MISSING",
                    field="target",
                    column=target_column,
                    message=f"目标字段“{target_column}”包含空值",
                    row_numbers=rows,
                    count=int(target_missing.sum()),
                )
            )
        target_non_missing = cls._non_missing(target_values)
        target_numbers = pd.to_numeric(
            target_non_missing.astype(str).str.strip(), errors="coerce"
        )
        invalid_targets = target_numbers.isna()
        if not target_numbers.empty:
            invalid_targets = invalid_targets | ~np.isfinite(target_numbers.to_numpy(dtype=float))
            if invalid_targets.any():
                bad_indexes = target_non_missing.index[invalid_targets]
                rows = [int(index) + 2 for index in bad_indexes[:10]]
                issues.append(
                    ValidationIssue(
                        code="TARGET_VALUE_INVALID",
                        field="target",
                        column=target_column,
                        message=f"目标字段“{target_column}”必须为有限数值（第 {rows[0]} 行）",
                        row_numbers=rows,
                        count=int(invalid_targets.sum()),
                    )
                )

        missing_value_counts = {
            column: int(cls._missing_mask(frame[column]).sum()) for column in columns
        }
        for column in feature_columns:
            if missing_value_counts[column] == row_count:
                issues.append(
                    ValidationIssue(
                        code="FEATURE_VALUES_EMPTY",
                        field="feature",
                        column=column,
                        message=f"特征字段“{column}”不能全部为空",
                        count=row_count,
                    )
                )

        if issues:
            raise DatasetValidationError(issues)

        if not is_sorted:
            warning = ValidationIssue(
                code="TIME_NOT_SORTED",
                field="time",
                column=time_column,
                message=f"时间字段“{time_column}”不是升序，后续划分时将按时间升序排序",
                count=out_of_order_count,
            )
            warnings = [warning.as_dict()]
        else:
            warnings = []
        column_types = {
            column: cls._column_type(frame[column], time=(column == time_column))
            for column in columns
        }
        column_summary = {
            column: cls._summary_for_column(
                frame[column], column_types[column], missing_value_counts[column]
            )
            for column in columns
        }
        summary = {
            "row_count": row_count,
            "column_count": len(columns),
            "columns": column_summary,
            "column_types": column_types,
            "missing_value_counts": missing_value_counts,
        }
        preview = [
            {column: cls._json_value(value) for column, value in row.items()}
            for row in frame.head(preview_rows).to_dict(orient="records")
        ]
        time_min = cls._json_value(parsed_times.min())
        time_max = cls._json_value(parsed_times.max())
        time_range = {"start": time_min, "end": time_max}
        time_parse = {
            "success": True,
            "format": None,
            "invalid_count": 0,
            "min": time_min,
            "max": time_max,
            "is_sorted": is_sorted,
            "out_of_order_count": out_of_order_count,
            "duplicate_count": duplicate_count,
            "message": None,
        }
        validation = {
            "valid": True,
            "errors": [],
            "warnings": warnings,
            "checks": {
                "column_count": {"valid": True, "actual": len(columns), "minimum": 3},
                "feature_columns": {"valid": True, "count": len(feature_columns)},
                "time": {"valid": True, "column": time_column, "missing_count": 0},
                "target": {
                    "valid": True,
                    "column": target_column,
                    "missing_count": 0,
                    "data_type": "number",
                },
                "split": {
                    "valid": True,
                    "split_ratio": 0.8,
                    "test_ratio": 0.2,
                    "train_row_count": train_rows,
                    "test_row_count": test_rows,
                },
                "missing_values": {
                    "valid": True,
                    "counts": missing_value_counts,
                    "feature_values_may_be_missing": True,
                },
                "time_order": {
                    "valid": True,
                    "is_sorted": is_sorted,
                    "duplicate_count": duplicate_count,
                    "out_of_order_count": out_of_order_count,
                    "sorted_before_split": not is_sorted,
                },
            },
        }
        return ParsedDataset(
            file_name=name,
            row_count=row_count,
            columns=columns,
            time_column=time_column,
            feature_columns=feature_columns,
            target_column=target_column,
            column_types=column_types,
            missing_value_counts=missing_value_counts,
            preview_rows=preview,
            summary=summary,
            time_parse=time_parse,
            time_range=time_range,
            validation=validation,
        )

    def upload(
        self,
        file_name: str,
        content: bytes,
        *,
        content_type: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        """Parse, persist, and return one uploaded dataset inspection result."""

        if content_type and content_type.lower().split(";", 1)[0].strip() not in self.CSV_CONTENT_TYPES:
            raise UnsupportedDatasetFileError(
                f"不支持的文件类型“{content_type}”，请上传 CSV 文件"
            )
        if len(content) > self.settings.max_dataset_size_bytes:
            raise CSVParseError(
                f"CSV 文件超过大小限制（最大 {self.settings.max_dataset_size_bytes} 字节）",
                code="FILE_TOO_LARGE",
                field="file",
            )
        parsed = self.parse_csv(file_name, content, preview_rows=self.preview_rows)
        if self.session is None:
            raise RuntimeError("保存数据集需要提供数据库 session")

        dataset_id = str(uuid4())
        artifact: StoredArtifact | None = None
        try:
            artifact = self.storage.save_dataset(dataset_id, content)
            created_at = datetime.now(timezone.utc)
            record = DatasetRecord(
                id=dataset_id,
                file_name=parsed.file_name,
                file_path=artifact.relative_path,
                row_count=parsed.row_count,
                columns=parsed.columns,
                time_column=parsed.time_column,
                feature_columns=parsed.feature_columns,
                target_column=parsed.target_column,
                column_types=parsed.column_types,
                missing_value_counts=parsed.missing_value_counts,
                preview_rows=parsed.preview_rows,
                numeric_columns=[
                    column
                    for column in parsed.columns
                    if parsed.column_types[column] == "number"
                ],
                time_parse=parsed.time_parse,
                time_range=parsed.time_range,
                summary=parsed.summary,
            )
            DatasetRepository(self.session).create(record)
            if commit:
                self.session.commit()
        except Exception:
            self.session.rollback()
            # File writes and database commits cannot share one transaction;
            # compensate an already-written file when metadata persistence
            # fails so failed uploads do not leave orphaned CSVs.
            if artifact is not None:
                try:
                    self.storage.remove_dataset(dataset_id)
                except Exception:
                    pass
            raise

        assert artifact is not None
        result = ParsedDataset(
            file_name=parsed.file_name,
            row_count=parsed.row_count,
            columns=parsed.columns,
            time_column=parsed.time_column,
            feature_columns=parsed.feature_columns,
            target_column=parsed.target_column,
            column_types=parsed.column_types,
            missing_value_counts=parsed.missing_value_counts,
            preview_rows=parsed.preview_rows,
            summary=parsed.summary,
            time_parse=parsed.time_parse,
            time_range=parsed.time_range,
            validation=parsed.validation,
            created_at=created_at,
        ).as_dict()
        file_storage = {
            "artifact_type": artifact.artifact_type.value,
            "relative_path": artifact.relative_path,
            "size_bytes": artifact.size_bytes,
            "checksum_sha256": artifact.checksum_sha256,
        }
        result.update(
            {
                "dataset_id": dataset_id,
                "id": dataset_id,
                "file_path": artifact.relative_path,
                "file_storage": file_storage,
                # Flat aliases keep the response convenient for simple UIs.
                "file_size_bytes": artifact.size_bytes,
                "checksum_sha256": artifact.checksum_sha256,
            }
        )
        return result

    # Explicit aliases help callers distinguish parsing-only from persistence.
    parse = parse_csv
    upload_csv = upload
