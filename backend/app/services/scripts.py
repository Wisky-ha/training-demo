"""Business logic for the global Python script library."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..core.config import Settings, get_settings
from ..db.models import ModelTypeORM, ScriptORM
from ..db.repositories import ScriptRepository
from ..domain.enums import ModelType, ScriptStatus, ScriptType
from ..schemas.scripts import ScriptResponse, ScriptUploadMetadata
from ..storage import ArtifactType, FileStorageService


class InvalidScriptFileError(ValueError):
    """Raised when an upload is not a safe, valid Python source file."""


class DuplicateScriptVersionError(ValueError):
    """Raised when an immutable script version already exists."""


class InvalidScriptMetadataError(ValueError):
    """Raised when metadata refers to an unregistered model family."""


class ScriptStorageError(RuntimeError):
    """Raised when source storage cannot complete."""


class ScriptPersistenceError(RuntimeError):
    """Raised when a script-library state change cannot be committed."""


_FILENAME_DISALLOWED = re.compile(r'[<>:"|?*]')


class ScriptService:
    """Coordinate validation, database persistence, and source-file storage."""

    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.repository = ScriptRepository(session)

    @staticmethod
    def validate_filename(filename: str | None) -> str:
        """Validate the client filename without ever using it as a path."""

        if not isinstance(filename, str) or not filename:
            raise InvalidScriptFileError("a Python filename is required")
        if "\x00" in filename or "/" in filename or "\\" in filename:
            raise InvalidScriptFileError("filename must not contain path separators")
        if any(ord(character) < 32 for character in filename):
            raise InvalidScriptFileError("filename contains a control character")
        if _FILENAME_DISALLOWED.search(filename):
            raise InvalidScriptFileError("filename contains an unsafe character")
        if len(filename.encode("utf-8")) > 255:
            raise InvalidScriptFileError("filename is too long")
        if not filename.lower().endswith(".py"):
            raise InvalidScriptFileError("only .py files are accepted")
        if filename[:-3].strip(" .") == "":
            raise InvalidScriptFileError("filename must have a name")
        return filename

    def validate_source(self, filename: str | None, source: bytes) -> str:
        """Decode and parse source before creating any database or file state."""

        self.validate_filename(filename)
        if not source:
            raise InvalidScriptFileError("script file must not be empty")
        if len(source) > self.settings.max_script_size_bytes:
            raise InvalidScriptFileError("script file exceeds the configured size limit")
        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidScriptFileError("script file must be UTF-8 encoded Python") from exc
        try:
            ast.parse(text, filename=filename or "script.py")
        except (SyntaxError, ValueError, TypeError) as exc:
            raise InvalidScriptFileError("script file contains invalid Python syntax") from exc
        return text

    def upload(
        self,
        *,
        metadata: ScriptUploadMetadata,
        filename: str | None,
        source: bytes,
    ) -> ScriptORM:
        """Persist one script and compensate the file if the transaction fails."""

        source_code = self.validate_source(filename, source)
        version = metadata.version or self.repository.next_version(
            metadata.name, metadata.script_type
        )
        if self.repository.version_exists(
            metadata.name, metadata.script_type, version
        ):
            raise DuplicateScriptVersionError(
                "a script with the same name, type, and version already exists"
            )
        registered_types = set(
            self.session.scalars(
                select(ModelTypeORM.code).where(
                    ModelTypeORM.code.in_(metadata.supported_model_types)
                )
            ).all()
        )
        missing_types = set(metadata.supported_model_types) - registered_types
        if missing_types:
            missing = ", ".join(sorted(item.value for item in missing_types))
            raise InvalidScriptMetadataError(
                f"model type is not registered: {missing}"
            )

        script_id = str(uuid4())
        storage = FileStorageService(
            self.settings.script_file_storage_root,
            session=self.session,
        )
        saved = False
        try:
            script = self.repository.create(
                id=script_id,
                name=metadata.name,
                script_type=metadata.script_type,
                version=version,
                source_code=source_code,
                supported_model_types=metadata.supported_model_types,
                status=ScriptStatus.ENABLED,
            )
            # Mark this before save: the storage helper may have written the
            # file before a metadata flush raises an exception.
            saved = True
            storage.save_script_source(script.id, source_code)
            self.session.commit()
            return script
        except IntegrityError as exc:
            self.session.rollback()
            if saved:
                storage.remove(ArtifactType.SCRIPT, script_id)
            raise DuplicateScriptVersionError(
                "a script with the same name, type, and version already exists"
            ) from exc
        except Exception as exc:
            self.session.rollback()
            if saved:
                try:
                    storage.remove(ArtifactType.SCRIPT, script_id)
                except Exception:
                    # Preserve the original API error; the orphan can be
                    # detected from the artifact directory and cleaned later.
                    pass
            if isinstance(exc, (InvalidScriptFileError, DuplicateScriptVersionError)):
                raise
            raise ScriptStorageError("could not persist the script source") from exc

    def get(self, script_id: str) -> ScriptORM | None:
        """Return one script version, including its immutable source."""

        return self.repository.get(script_id)

    def set_status(
        self, script_id: str, status: ScriptStatus
    ) -> ScriptORM | None:
        """Enable or disable a version without changing its source/history."""

        script = self.repository.get(script_id)
        if script is None:
            return None
        try:
            script.status = status
            self.session.commit()
        except Exception as exc:
            self.session.rollback()
            raise ScriptPersistenceError("could not update script status") from exc
        return script

    def list(
        self,
        *,
        model_type: ModelType | None,
        script_type: ScriptType | None,
        status: ScriptStatus,
        page: int,
        page_size: int,
    ) -> tuple[list[ScriptResponse], int]:
        total = self.repository.count_library(
            model_type=model_type, script_type=script_type, status=status
        )
        records = self.repository.list_library(
            model_type=model_type, script_type=script_type, status=status
        )
        offset = (page - 1) * page_size
        return [self.to_response(item) for item in records[offset : offset + page_size]], total

    @staticmethod
    def to_response(script: ScriptORM) -> ScriptResponse:
        created_at = script.created_at
        return ScriptResponse(
            id=script.id,
            name=script.name,
            script_type=script.script_type,
            version=script.version,
            source_code=script.source_code,
            supported_model_types=[item.code for item in script.supported_model_types],
            status=script.status,
            created_at=created_at,
            uploaded_at=created_at,
        )
