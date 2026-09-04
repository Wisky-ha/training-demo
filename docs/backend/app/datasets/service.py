"""CSV parsing, inspection, and persistence for uploaded datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from sqlalchemy.orm import Session

from ..core.config import Settings, get_settings
from ..db.repositories import DatasetRepository
from ..domain.models import DatasetRecord
from ..storage import FileStorageService, StoredArtifact


class CSVParseError(ValueError):
    """Raised when an uploaded file is not a valid supported dataset CSV."""


class UnsupportedDatasetFileError(CSVParseError):
    """Raised when the uploaded filename is not a CSV filename."""


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
            "time_parse": self.time_parse,
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
            raise CSVParseError("CSV 文件内容为空")
        # UTF-8 (including its BOM) is the API default.  GB18030 is accepted
        # for common Chinese desktop exports while still rejecting binary data.
        for encoding in ("utf-8-sig", "gb18030"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise CSVParseError("CSV 文件必须使用 UTF-8 或 GB18030 编码")

    @staticmethod
    def _json_value(value: Any) -> Any:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return value

    @classmethod
    def _numeric_values(cls, series: pd.Series) -> pd.Series | None:
        non_missing = series.dropna()
        if non_missing.empty:
            return None
        converted = pd.to_numeric(non_missing.astype(str).str.strip(), errors="coerce")
        if converted.isna().any():
            return None
        return converted

    @classmethod
    def _is_datetime(cls, series: pd.Series) -> bool:
        non_missing = series.dropna()
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
        non_missing = series.dropna()
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
        non_missing = series.dropna()
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
            frame = pd.read_csv(
                StringIO(text),
                header=0,
                dtype=object,
                keep_default_na=True,
                na_filter=True,
                on_bad_lines="error",
            )
        except EmptyDataError as exc:
            raise CSVParseError("CSV 文件缺少表头") from exc
        except (ParserError, ValueError) as exc:
            raise CSVParseError(f"CSV 内容无法解析：{exc}") from exc

        columns = [str(column).strip() for column in frame.columns]
        if len(columns) < 3:
            raise CSVParseError("CSV 至少需要 3 列：时间列、特征列和目标列")
        if any(not column for column in columns):
            raise CSVParseError("CSV 表头不能包含空字段名")
        if len(set(columns)) != len(columns):
            raise CSVParseError("CSV 表头不能包含重复字段名")
        if frame.empty:
            raise CSVParseError("CSV 至少需要一行数据")
        frame.columns = columns

        time_column = columns[0]
        time_values = frame[time_column]
        if time_values.isna().any():
            raise CSVParseError(f"时间字段“{time_column}”包含缺失值，无法解析")
        parsed_times = pd.to_datetime(
            time_values.astype(str).str.strip(), errors="coerce", format="mixed"
        )
        if parsed_times.isna().any():
            bad_row = int(parsed_times[parsed_times.isna()].index[0]) + 2
            raise CSVParseError(
                f"时间字段“{time_column}”存在无法解析的值（第 {bad_row} 行）"
            )

        feature_columns = columns[1:-1]
        target_column = columns[-1]
        column_types = {
            column: cls._column_type(frame[column], time=(column == time_column))
            for column in columns
        }
        missing_value_counts = {
            column: int(frame[column].isna().sum()) for column in columns
        }
        column_summary = {
            column: cls._summary_for_column(
                frame[column], column_types[column], missing_value_counts[column]
            )
            for column in columns
        }
        summary = {
            "row_count": int(len(frame)),
            "column_count": len(columns),
            "columns": column_summary,
            "column_types": column_types,
            "missing_value_counts": missing_value_counts,
        }
        preview = [
            {column: cls._json_value(value) for column, value in row.items()}
            for row in frame.head(preview_rows).to_dict(orient="records")
        ]
        return ParsedDataset(
            file_name=name,
            row_count=int(len(frame)),
            columns=columns,
            time_column=time_column,
            feature_columns=feature_columns,
            target_column=target_column,
            column_types=column_types,
            missing_value_counts=missing_value_counts,
            preview_rows=preview,
            summary=summary,
            time_parse={
                "success": True,
                "format": None,
                "invalid_count": 0,
                "min": cls._json_value(parsed_times.min()),
                "max": cls._json_value(parsed_times.max()),
                "message": None,
            },
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
        parsed = self.parse_csv(file_name, content, preview_rows=self.preview_rows)
        if self.session is None:
            raise RuntimeError("保存数据集需要提供数据库 session")

        dataset_id = str(uuid4())
        try:
            artifact: StoredArtifact = self.storage.save_dataset(dataset_id, content)
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
            )
            DatasetRepository(self.session).create(record)
            if commit:
                self.session.commit()
        except Exception:
            self.session.rollback()
            raise

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
            created_at=created_at,
        ).as_dict()
        result.update({"dataset_id": dataset_id, "id": dataset_id, "file_path": artifact.relative_path})
        return result

    # Explicit aliases help callers distinguish parsing-only from persistence.
    parse = parse_csv
    upload_csv = upload
