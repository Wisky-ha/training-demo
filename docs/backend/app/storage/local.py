"""Atomic, path-safe storage for files owned by the training platform.

This module intentionally contains no upload, training, or publication logic.
It stores bytes and their database metadata only; transaction ownership remains
with the caller-provided SQLAlchemy session.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import BinaryIO

from sqlalchemy.orm import Session

from ..core.config import Settings, get_settings
from ..db.models import FileArtifactORM
from ..db.repositories import FileArtifactRepository


_artifact_locks: dict[tuple[str, str], threading.Lock] = {}
_artifact_locks_guard = threading.Lock()


def _lock_for(artifact_type: str, artifact_id: str) -> threading.Lock:
    """Get a process-local lock for one logical artifact key."""

    key = (artifact_type, artifact_id)
    with _artifact_locks_guard:
        return _artifact_locks.setdefault(key, threading.Lock())


class ArtifactType(str, Enum):
    """Logical kinds of files managed by this service."""

    MODEL = "model"
    PREPROCESSOR = "preprocessor"
    SCRIPT = "script"
    DATASET = "dataset"

    # Descriptive aliases keep the API readable without adding new values.
    MODEL_FILE = "model"
    PREPROCESSOR_STATE = "preprocessor"
    SCRIPT_SOURCE = "script"
    CSV = "dataset"
    DATASET_FILE = "dataset"


class InvalidStoragePathError(ValueError):
    """Raised when an artifact identifier could escape the storage root."""


class ArtifactNotFoundError(FileNotFoundError):
    """Raised when metadata exists or is addressable but its file is absent."""


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """Metadata returned after an artifact has been atomically stored."""

    id: str | None
    artifact_type: ArtifactType
    artifact_id: str
    relative_path: str
    size_bytes: int
    checksum_sha256: str

    @property
    def file_size(self) -> int:
        return self.size_bytes

    @property
    def checksum(self) -> str:
        return self.checksum_sha256

    @property
    def path(self) -> str:
        return self.relative_path

    @property
    def size(self) -> int:
        return self.size_bytes

    @property
    def sha256(self) -> str:
        return self.checksum_sha256

    @classmethod
    def from_orm(cls, record: FileArtifactORM) -> "StoredArtifact":
        return cls(
            id=record.id,
            artifact_type=ArtifactType(record.artifact_type),
            artifact_id=record.artifact_id,
            relative_path=record.relative_path,
            size_bytes=record.size_bytes,
            checksum_sha256=record.checksum_sha256,
        )


class FileStorageService:
    """Store model, preprocessor-state, and script-source artifacts locally.

    ``root_dir`` is explicit in tests and deployments.  If omitted, the
    configured ``APP_STORAGE_ROOT``/``APP_FILE_STORAGE_ROOT`` is used, falling
    back to the existing ``APP_MODEL_STORAGE_DIR`` setting.  A session is
    optional for file-only use; when supplied, metadata is upserted and flushed
    but never committed by this service.
    """

    _suffixes = {
        ArtifactType.MODEL: ".bin",
        ArtifactType.PREPROCESSOR: ".state",
        ArtifactType.SCRIPT: ".py",
        ArtifactType.DATASET: ".csv",
    }

    def __init__(
        self,
        root_dir: str | os.PathLike[str] | Settings | None = None,
        session: Session | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        # Accepting Settings positionally mirrors the existing database helper
        # conventions while keeping the explicit root_dir form convenient.
        if isinstance(root_dir, Settings):
            if settings is not None:
                raise ValueError("provide root_dir or settings, not both")
            settings = root_dir
            root_dir = None
        if root_dir is not None and settings is not None:
            raise ValueError("provide root_dir or settings, not both")
        active_settings = settings or get_settings()
        configured_root = (
            Path(root_dir)
            if root_dir is not None
            else active_settings.file_storage_root
        )
        self.root_dir = configured_root.expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.session = session
        self._repository = (
            FileArtifactRepository(session) if session is not None else None
        )

    @staticmethod
    def _kind(artifact_type: ArtifactType | str) -> ArtifactType:
        try:
            return (
                artifact_type
                if isinstance(artifact_type, ArtifactType)
                else ArtifactType(artifact_type)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported artifact type: {artifact_type!r}") from exc

    @staticmethod
    def _component(value: str, label: str = "artifact identifier") -> str:
        if not isinstance(value, str) or not value or value in {".", ".."}:
            raise InvalidStoragePathError(f"invalid {label}")
        # Reject both POSIX and Windows separators, including Windows drive
        # paths even when this application happens to run on POSIX.
        if "\x00" in value or "/" in value or "\\" in value or ":" in value:
            raise InvalidStoragePathError(f"invalid {label}")
        if len(value) > 255:
            raise InvalidStoragePathError(f"{label} is too long")
        return value

    def _relative_path(self, kind: ArtifactType, artifact_id: str) -> str:
        identifier = self._component(artifact_id)
        filename = f"{identifier}{self._suffixes[kind]}"
        # Common filesystems cap one filename component at 255 bytes, not
        # characters (which matters for non-ASCII identifiers).
        if len(filename.encode("utf-8")) > 255:
            raise InvalidStoragePathError("artifact identifier is too long")
        return (Path(kind.value) / filename).as_posix()

    def _safe_path(self, relative_path: str) -> Path:
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or "\x00" in relative_path
        ):
            raise InvalidStoragePathError("invalid relative storage path")
        candidate = Path(relative_path)
        if (
            candidate.is_absolute()
            or "\\" in relative_path
            or any(part in {".", ".."} for part in candidate.parts)
        ):
            raise InvalidStoragePathError(
                "storage path must be relative and normalized"
            )
        resolved = (self.root_dir / candidate).resolve()
        try:
            resolved.relative_to(self.root_dir)
        except ValueError as exc:
            raise InvalidStoragePathError("storage path escapes the storage root") from exc
        return resolved

    @staticmethod
    def _bytes(content: bytes | bytearray | memoryview | str | BinaryIO) -> bytes:
        if isinstance(content, str):
            return content.encode("utf-8")
        if isinstance(content, (bytes, bytearray, memoryview)):
            return bytes(content)
        if hasattr(content, "read"):
            value = content.read()
            if isinstance(value, str):
                return value.encode("utf-8")
            if isinstance(value, (bytes, bytearray, memoryview)):
                return bytes(value)
        raise TypeError("content must be bytes, text, or a binary file-like object")

    def _atomic_write(self, path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # The resolved parent check also protects against a category directory
        # being replaced with a symlink after the root was initialized.
        try:
            path.parent.resolve().relative_to(self.root_dir)
        except ValueError as exc:
            raise InvalidStoragePathError("storage path escapes the storage root") from exc

        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "wb") as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, path)
            # Best effort directory fsync: Windows does not support opening a
            # directory this way, while os.replace already provides atomicity.
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def save(
        self,
        artifact_type: ArtifactType | str,
        artifact_id: str,
        content: bytes | bytearray | memoryview | str | BinaryIO,
    ) -> StoredArtifact:
        """Atomically save an artifact and upsert its integrity metadata."""

        kind = self._kind(artifact_type)
        # Validate before indexing the lock map; invalid input never creates
        # an unbounded in-memory key.
        self._component(artifact_id)
        with _lock_for(kind.value, artifact_id):
            relative_path = self._relative_path(kind, artifact_id)
            path = self._safe_path(relative_path)
            raw = self._bytes(content)
            self._atomic_write(path, raw)
            checksum = hashlib.sha256(raw).hexdigest()
            record: FileArtifactORM | None = None
            if self._repository is not None:
                record = self._repository.upsert(
                    artifact_type=kind.value,
                    artifact_id=artifact_id,
                    relative_path=relative_path,
                    size_bytes=len(raw),
                    checksum_sha256=checksum,
                )
            return StoredArtifact(
                id=record.id if record is not None else None,
                artifact_type=kind,
                artifact_id=artifact_id,
                relative_path=relative_path,
                size_bytes=len(raw),
                checksum_sha256=checksum,
            )

    def _path_for(self, kind: ArtifactType, artifact_id: str) -> Path:
        """Resolve the canonical path for an artifact.

        A database row is metadata, not a permission to redirect reads.  The
        path is derived again from the validated type and identifier so a
        corrupted or manually edited row cannot make one artifact read another.
        """

        self._component(artifact_id)
        return self._safe_path(self._relative_path(kind, artifact_id))

    def read(self, artifact_type: ArtifactType | str, artifact_id: str) -> bytes:
        kind = self._kind(artifact_type)
        self._component(artifact_id)
        with _lock_for(kind.value, artifact_id):
            path = self._path_for(kind, artifact_id)
            try:
                return path.read_bytes()
            except FileNotFoundError as exc:
                raise ArtifactNotFoundError(
                    f"artifact not found: {kind.value}/{artifact_id}"
                ) from exc

    def exists(self, artifact_type: ArtifactType | str, artifact_id: str) -> bool:
        kind = self._kind(artifact_type)
        self._component(artifact_id)
        with _lock_for(kind.value, artifact_id):
            return self._path_for(kind, artifact_id).is_file()

    def remove(self, artifact_type: ArtifactType | str, artifact_id: str) -> bool:
        """Remove one artifact file and its pending metadata, if present.

        This is primarily a compensation operation for services that have to
        roll back a database transaction after a file has already been saved.
        It never follows a path supplied by a caller; the canonical path is
        derived from the validated artifact key.
        """

        kind = self._kind(artifact_type)
        self._component(artifact_id)
        with _lock_for(kind.value, artifact_id):
            path = self._path_for(kind, artifact_id)
            existed = path.is_file()
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            if self._repository is not None:
                record = self._repository.get_for_artifact(kind.value, artifact_id)
                if record is not None:
                    self._repository.delete(record)
                    existed = True
            return existed

    delete = remove

    def metadata(
        self, artifact_type: ArtifactType | str, artifact_id: str
    ) -> StoredArtifact | None:
        kind = self._kind(artifact_type)
        self._component(artifact_id)
        if self._repository is None:
            if not self.exists(kind, artifact_id):
                return None
            raw = self.read(kind, artifact_id)
            return StoredArtifact(
                None,
                kind,
                artifact_id,
                self._relative_path(kind, artifact_id),
                len(raw),
                hashlib.sha256(raw).hexdigest(),
            )
        record = self._repository.get_for_artifact(kind.value, artifact_id)
        if record is None:
            return None
        expected_path = self._relative_path(kind, artifact_id)
        if record.relative_path != expected_path:
            raise InvalidStoragePathError("artifact metadata contains a non-canonical path")
        return StoredArtifact.from_orm(record)

    def save_model(
        self, model_id: str, content: bytes | bytearray | memoryview | BinaryIO
    ) -> StoredArtifact:
        return self.save(ArtifactType.MODEL, model_id, content)

    def read_model(self, model_id: str) -> bytes:
        return self.read(ArtifactType.MODEL, model_id)

    def model_exists(self, model_id: str) -> bool:
        return self.exists(ArtifactType.MODEL, model_id)

    def save_preprocessor_state(
        self,
        artifact_id: str,
        content: bytes | bytearray | memoryview | BinaryIO,
    ) -> StoredArtifact:
        return self.save(ArtifactType.PREPROCESSOR, artifact_id, content)

    def read_preprocessor_state(self, artifact_id: str) -> bytes:
        return self.read(ArtifactType.PREPROCESSOR, artifact_id)

    def preprocessor_exists(self, artifact_id: str) -> bool:
        return self.exists(ArtifactType.PREPROCESSOR, artifact_id)

    def save_script_source(
        self, script_id: str, source: str | bytes | BinaryIO
    ) -> StoredArtifact:
        return self.save(ArtifactType.SCRIPT, script_id, source)

    def save_dataset(
        self, dataset_id: str, content: bytes | bytearray | memoryview | BinaryIO
    ) -> StoredArtifact:
        """Save the source CSV under the dataset's logical identifier."""

        return self.save(ArtifactType.DATASET, dataset_id, content)

    def read_dataset(self, dataset_id: str) -> bytes:
        return self.read(ArtifactType.DATASET, dataset_id)

    def dataset_exists(self, dataset_id: str) -> bool:
        return self.exists(ArtifactType.DATASET, dataset_id)

    def read_script_source(self, script_id: str) -> str:
        try:
            return self.read(ArtifactType.SCRIPT, script_id).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("stored script source is not valid UTF-8") from exc

    def script_exists(self, script_id: str) -> bool:
        return self.exists(ArtifactType.SCRIPT, script_id)

    # Generic aliases make the service usable by infrastructure code without
    # coupling it to the three convenience methods above.
    save_artifact = save
    read_artifact = read
    artifact_exists = exists
    get_metadata = metadata
    remove_artifact = remove

    # Short aliases for callers that already know the artifact category.
    save_preprocessor = save_preprocessor_state
    read_preprocessor = read_preprocessor_state
    preprocessor_state_exists = preprocessor_exists
    save_script = save_script_source
    read_script = read_script_source
    script_source_exists = script_exists
    save_csv = save_dataset
    read_csv = read_dataset
    csv_exists = dataset_exists


LocalFileStorage = FileStorageService

__all__ = [
    "ArtifactType",
    "ArtifactNotFoundError",
    "FileStorageService",
    "InvalidStoragePathError",
    "LocalFileStorage",
    "StoredArtifact",
]
