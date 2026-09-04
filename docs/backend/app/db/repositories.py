"""Small transaction-neutral CRUD repositories for the persistence models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any, Generic, TypeVar

from pydantic import BaseModel as PydanticModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..domain.enums import (
    AlertStatus,
    DatasetStatus,
    HealthStatus,
    ModelType,
    ModelVersionStatus,
    RollbackStatus,
    ScriptStatus,
    ScriptType,
    SplitStrategy,
    TrainingJobStatus,
)
from .models import (
    DatasetORM,
    FileArtifactORM,
    ModelAlertORM,
    ModelTypeORM,
    ModelVersionORM,
    PublishRecordORM,
    RollbackRecordORM,
    ScriptORM,
    TrainingJobORM,
)

ORMModel = TypeVar("ORMModel")


class Repository(Generic[ORMModel]):
    """Generic CRUD operations that deliberately do not commit transactions.

    A repository flushes writes so callers immediately receive generated/default
    values and constraint errors, but the session owner decides whether to
    commit or roll back the unit of work.
    """

    model: type[ORMModel]

    def __init__(self, session: Session, model: type[ORMModel] | None = None):
        self.session = session
        if model is not None:
            self.model = model

    @staticmethod
    def _payload(entity: Any, values: Mapping[str, Any]) -> dict[str, Any]:
        if entity is None:
            return dict(values)
        if isinstance(entity, PydanticModel):
            payload = entity.model_dump(mode="python")
            payload.update(values)
            return payload
        if isinstance(entity, Mapping):
            payload = dict(entity)
            payload.update(values)
            return payload
        if values:
            return dict(values)
        return {
            key: value
            for key, value in vars(entity).items()
            if not key.startswith("_")
        }

    def _normalise(self, payload: dict[str, Any]) -> dict[str, Any]:
        enum_fields = {
            "code": ModelType,
            "model_type": ModelType,
            "script_type": ScriptType,
            "status": {
                DatasetORM: DatasetStatus,
                ModelTypeORM: AlertStatus,
                ScriptORM: ScriptStatus,
                TrainingJobORM: TrainingJobStatus,
                ModelVersionORM: ModelVersionStatus,
                ModelAlertORM: AlertStatus,
                RollbackRecordORM: RollbackStatus,
                PublishRecordORM: None,
            }.get(self.model),
            "health_status": HealthStatus,
            "script_type": ScriptType,
            "script_status": ScriptStatus,
            "split_strategy": SplitStrategy,
        }
        for field, enum_class in enum_fields.items():
            if field in payload and enum_class and not isinstance(enum_class, dict):
                payload[field] = enum_class(payload[field])
        if "supported_model_types" in payload:
            codes = payload.pop("supported_model_types") or []
            resolved = []
            for value in codes:
                code = value.code if isinstance(value, ModelTypeORM) else ModelType(value)
                item = self.session.scalar(
                    select(ModelTypeORM).where(ModelTypeORM.code == code)
                )
                if item is None:
                    raise ValueError(f"model type does not exist: {code.value}")
                resolved.append(item)
            payload["_supported_model_types"] = resolved
        return payload

    def create(self, entity: Any = None, **values: Any) -> ORMModel:
        payload = self._normalise(self._payload(entity, values))
        related_model_types = payload.pop("_supported_model_types", None)
        if isinstance(entity, self.model) and not values:
            instance = entity
        else:
            instance = self.model(**payload)
        if related_model_types is not None:
            instance.supported_model_types = related_model_types
        self.session.add(instance)
        self.session.flush()
        return instance

    def get(self, entity_id: str) -> ORMModel | None:
        return self.session.get(self.model, entity_id)

    def get_by(self, **filters: Any) -> ORMModel | None:
        statement = select(self.model).filter_by(**self._normalise(dict(filters)))
        return self.session.scalars(statement).first()

    def list(self, **filters: Any) -> list[ORMModel]:
        statement = select(self.model)
        if filters:
            statement = statement.filter_by(**self._normalise(dict(filters)))
        return list(self.session.scalars(statement).all())

    def update(self, entity_or_id: str | ORMModel | PydanticModel, values: Mapping[str, Any] | None = None, **changes: Any) -> ORMModel | None:
        if isinstance(entity_or_id, str):
            instance = self.get(entity_or_id)
        else:
            instance = entity_or_id
            if not isinstance(instance, self.model):
                instance = self.get(getattr(entity_or_id, "id"))
        if instance is None:
            return None
        if isinstance(entity_or_id, PydanticModel):
            changes = {**entity_or_id.model_dump(mode="python"), **(dict(values or {})), **changes}
        elif values:
            changes = {**dict(values), **changes}
        changes.pop("id", None)
        changes = self._normalise(changes)
        related_model_types = changes.pop("_supported_model_types", None)
        for field, value in changes.items():
            if hasattr(instance, field):
                setattr(instance, field, value)
        if related_model_types is not None:
            instance.supported_model_types = related_model_types
        self.session.flush()
        return instance

    def delete(self, entity_or_id: str | ORMModel) -> bool:
        instance = self.get(entity_or_id) if isinstance(entity_or_id, str) else entity_or_id
        if instance is None:
            return False
        self.session.delete(instance)
        self.session.flush()
        return True


class ModelTypeRepository(Repository[ModelTypeORM]):
    model = ModelTypeORM


class ScriptRepository(Repository[ScriptORM]):
    model = ScriptORM

    def _library_statement(
        self,
        *,
        model_type: ModelType | None = None,
        script_type: ScriptType | None = None,
        status: ScriptStatus | None = None,
    ):
        statement = select(ScriptORM)
        if model_type is not None:
            statement = statement.join(ScriptORM.supported_model_types).where(
                ModelTypeORM.code == model_type
            )
        if script_type is not None:
            statement = statement.where(ScriptORM.script_type == script_type)
        if status is not None:
            statement = statement.where(ScriptORM.status == status)
        return statement.order_by(ScriptORM.created_at.desc(), ScriptORM.id.desc())

    def list_library(
        self,
        *,
        model_type: ModelType | None = None,
        script_type: ScriptType | None = None,
        status: ScriptStatus | None = None,
    ) -> list[ScriptORM]:
        """List scripts with optional compatibility and library filters."""

        return list(self.session.scalars(self._library_statement(
            model_type=model_type, script_type=script_type, status=status
        )).unique().all())

    def count_library(
        self,
        *,
        model_type: ModelType | None = None,
        script_type: ScriptType | None = None,
        status: ScriptStatus | None = None,
    ) -> int:
        """Count scripts matching the same filters as :meth:`list_library`."""

        statement = select(func.count(ScriptORM.id))
        if model_type is not None:
            statement = statement.join(ScriptORM.supported_model_types).where(
                ModelTypeORM.code == model_type
            )
        if script_type is not None:
            statement = statement.where(ScriptORM.script_type == script_type)
        if status is not None:
            statement = statement.where(ScriptORM.status == status)
        return int(self.session.scalar(statement) or 0)

    def version_exists(self, name: str, script_type: ScriptType, version: str) -> bool:
        """Check the immutable version key before writing an upload."""

        statement = select(ScriptORM.id).where(
            ScriptORM.name == name,
            ScriptORM.script_type == script_type,
            ScriptORM.version == version,
        )
        return self.session.scalar(statement) is not None

    def next_version(self, name: str, script_type: ScriptType) -> str:
        """Return the next generated version for one script identity.

        Versions supplied by callers may use any non-empty label.  Generated
        versions use the stable ``vN`` form and advance past existing numeric
        versions, so explicit ``v10`` uploads cannot later be overwritten by
        an automatically generated version.
        """

        statement = select(ScriptORM.version).where(
            ScriptORM.name == name,
            ScriptORM.script_type == script_type,
        )
        highest = 0
        for version in self.session.scalars(statement):
            match = re.fullmatch(r"v(\d+)", version.strip(), flags=re.IGNORECASE)
            if match:
                highest = max(highest, int(match.group(1)))
        return f"v{highest + 1}"


class DatasetRepository(Repository[DatasetORM]):
    model = DatasetORM


class TrainingJobRepository(Repository[TrainingJobORM]):
    model = TrainingJobORM


class FileArtifactRepository(Repository[FileArtifactORM]):
    model = FileArtifactORM

    def get_for_artifact(self, artifact_type: str, artifact_id: str) -> FileArtifactORM | None:
        """Return metadata for one logical artifact."""

        return self.get_by(artifact_type=artifact_type, artifact_id=artifact_id)

    def upsert(
        self,
        *,
        artifact_type: str,
        artifact_id: str,
        relative_path: str,
        size_bytes: int,
        checksum_sha256: str,
    ) -> FileArtifactORM:
        """Create or replace metadata without committing the caller's transaction."""

        item = self.get_for_artifact(artifact_type, artifact_id)
        values = {
            "artifact_type": artifact_type,
            "artifact_id": artifact_id,
            "relative_path": relative_path,
            "size_bytes": size_bytes,
            "checksum_sha256": checksum_sha256,
        }
        if item is None:
            return self.create(**values)
        self.update(item, values)
        return item


class ModelVersionRepository(Repository[ModelVersionORM]):
    model = ModelVersionORM


class PublishRecordRepository(Repository[PublishRecordORM]):
    model = PublishRecordORM


class ModelAlertRepository(Repository[ModelAlertORM]):
    model = ModelAlertORM


class RollbackRepository(Repository[RollbackRecordORM]):
    model = RollbackRecordORM


# Plural aliases are harmless and make imports read naturally in service code.
ModelTypesRepository = ModelTypeRepository
ScriptsRepository = ScriptRepository
DatasetsRepository = DatasetRepository
TrainingJobsRepository = TrainingJobRepository
FileArtifactsRepository = FileArtifactRepository
ModelVersionsRepository = ModelVersionRepository
PublishRecordsRepository = PublishRecordRepository
ModelAlertsRepository = ModelAlertRepository
RollbackRecordsRepository = RollbackRepository

__all__ = [
    "Repository",
    "ModelTypeRepository",
    "ScriptRepository",
    "DatasetRepository",
    "TrainingJobRepository",
    "FileArtifactRepository",
    "ModelVersionRepository",
    "PublishRecordRepository",
    "ModelAlertRepository",
    "RollbackRepository",
    "ModelTypesRepository",
    "ScriptsRepository",
    "DatasetsRepository",
    "TrainingJobsRepository",
    "FileArtifactsRepository",
    "ModelVersionsRepository",
    "PublishRecordsRepository",
    "ModelAlertsRepository",
    "RollbackRecordsRepository",
]
