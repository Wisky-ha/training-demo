"""Transactional model-version lifecycle and production failover service."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..core.config import Settings, get_settings
from ..db.models import (
    ModelAlertORM,
    ModelTypeORM,
    ModelVersionORM,
    PublishRecordORM,
    RollbackRecordORM,
)
from ..domain.enums import AlertStatus, HealthStatus, ModelType, ModelVersionStatus, RollbackStatus
from ..schemas.models import AbnormalRequest, ModelSaveRequest
from ..storage import ArtifactType, FileStorageService


class ModelLifecycleError(ValueError):
    """A stable, safe error returned by lifecycle endpoints."""

    def __init__(self, message: str, code: str = "MODEL_LIFECYCLE_FAILED", **details: Any):
        super().__init__(message)
        self.code = code
        self.details = details


class ModelNotFoundError(ModelLifecycleError):
    def __init__(self, message: str = "模型版本不存在"):
        super().__init__(message, "MODEL_NOT_FOUND")


class NoHealthyRollbackError(ModelLifecycleError):
    def __init__(self, message: str, **details: Any):
        super().__init__(message, "NO_HEALTHY_ROLLBACK_VERSION", **details)


class ModelLifecycleService:
    """Own all state transitions that affect a model type's production pointer.

    The process lock prevents two API workers in this process from observing
    the same current version.  The partial unique database index is the final
    guard and makes a failed concurrent transaction harmless.
    """

    _type_locks: dict[ModelType, RLock] = {}
    _locks_guard = RLock()

    def __init__(self, session: Session, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()
        self.storage = FileStorageService(self.settings.file_storage_root, session=session)

    @classmethod
    def _lock_for(cls, model_type: ModelType) -> RLock:
        with cls._locks_guard:
            return cls._type_locks.setdefault(model_type, RLock())

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def get(self, version_id: str) -> ModelVersionORM | None:
        return self.session.get(ModelVersionORM, version_id)

    def list(self, *, model_type: ModelType | None = None,
             status: ModelVersionStatus | None = None,
             health_status: HealthStatus | None = None) -> list[ModelVersionORM]:
        statement = select(ModelVersionORM)
        if model_type is not None:
            statement = statement.where(ModelVersionORM.model_type == model_type)
        if status is not None:
            statement = statement.where(ModelVersionORM.status == status)
        if health_status is not None:
            statement = statement.where(ModelVersionORM.health_status == health_status)
        return list(self.session.scalars(statement.order_by(ModelVersionORM.created_at.desc(), ModelVersionORM.id.desc())).all())

    def _model_type(self, code: ModelType) -> ModelTypeORM:
        record = self.session.scalar(select(ModelTypeORM).where(ModelTypeORM.code == code))
        if record is None:
            raise ModelLifecycleError("模型类型不存在", "MODEL_TYPE_NOT_FOUND")
        return record

    def _lock_type(self, code: ModelType) -> ModelTypeORM:
        # SELECT FOR UPDATE is effective on server databases; the process lock
        # and SQLite's write lock provide the equivalent for the local backend.
        return self.session.scalar(
            select(ModelTypeORM).where(ModelTypeORM.code == code).with_for_update()
        ) or self._model_type(code)

    def _current(self, record: ModelTypeORM, *, include_unhealthy: bool = False) -> ModelVersionORM | None:
        version = self.session.get(ModelVersionORM, record.current_version_id) if record.current_version_id else None
        if version is None:
            version = self.session.scalar(
                select(ModelVersionORM).where(
                    ModelVersionORM.model_type == record.code,
                    ModelVersionORM.is_current.is_(True),
                )
            )
        if version is None or not version.is_current:
            return None
        if not include_unhealthy and (
            version.status is not ModelVersionStatus.PUBLISHED
            or version.health_status is not HealthStatus.HEALTHY
        ):
            return None
        return version

    @staticmethod
    def _next_version(session: Session, model_type: ModelType) -> str:
        import re
        highest = 0
        for value in session.scalars(select(ModelVersionORM.version).where(ModelVersionORM.model_type == model_type)):
            match = re.fullmatch(r"v(\d+)", value or "", flags=re.IGNORECASE)
            if match:
                highest = max(highest, int(match.group(1)))
        return f"v{highest + 1}"

    def save(self, request: ModelSaveRequest) -> ModelVersionORM:
        """Register metadata and optionally persist a model file as a READY row."""
        self._model_type(request.model_type)
        version_id = request.id or str(uuid4())
        if self.session.get(ModelVersionORM, version_id) is not None:
            raise ModelLifecycleError("模型版本 ID 已存在", "MODEL_VERSION_ALREADY_EXISTS")
        version_label = request.version or self._next_version(self.session, request.model_type)
        if self.session.scalar(select(ModelVersionORM.id).where(
            ModelVersionORM.model_type == request.model_type,
            ModelVersionORM.version == version_label,
        )) is not None:
            raise ModelLifecycleError("模型版本号已存在", "MODEL_VERSION_ALREADY_EXISTS")

        artifact = None
        if request.model_content_base64 is not None:
            try:
                content = base64.b64decode(request.model_content_base64, validate=True)
            except Exception as exc:
                raise ModelLifecycleError("模型制品不是有效的 base64", "MODEL_ARTIFACT_INVALID") from exc
            if not content:
                raise ModelLifecycleError("模型制品不能为空", "MODEL_ARTIFACT_INVALID")
            artifact = self.storage.save_model(version_id, content)
        path = artifact.relative_path if artifact else (request.model_path or f"model/{version_id}.joblib")
        try:
            model = ModelVersionORM(
                id=version_id, model_type=request.model_type, version=version_label,
                model_path=path, preprocessor_path=request.preprocessor_path,
                training_job_id=request.training_job_id, train_script_id=request.train_script_id,
                train_script_version=request.train_script_version,
                preprocess_script_id=request.preprocess_script_id,
                preprocess_script_version=request.preprocess_script_version,
                preprocess_used=request.preprocess_used, preprocessor_state=request.preprocessor_state,
                input_schema=request.input_schema, time_column=request.time_column,
                feature_columns=request.feature_columns, target_column=request.target_column,
                metrics=request.metrics, status=request.status,
            )
            self.session.add(model)
            self.session.flush()
            self.session.commit()
            return model
        except IntegrityError as exc:
            self.session.rollback()
            if artifact:
                self.storage.remove(ArtifactType.MODEL, version_id)
                self.session.commit()
            raise ModelLifecycleError("模型版本号已存在", "MODEL_VERSION_ALREADY_EXISTS") from exc
        except Exception:
            self.session.rollback()
            if artifact:
                try:
                    self.storage.remove(ArtifactType.MODEL, version_id)
                    self.session.commit()
                except Exception:
                    self.session.rollback()
            raise

    def _clear_current(self, record: ModelTypeORM) -> list[ModelVersionORM]:
        versions = list(self.session.scalars(select(ModelVersionORM).where(
            ModelVersionORM.model_type == record.code,
            ModelVersionORM.is_current.is_(True),
        )))
        record.current_version_id = None
        for item in versions:
            item.is_current = False
        # The target unique index must see the old current cleared first.
        self.session.flush()
        return versions

    def _make_current(self, record: ModelTypeORM, target: ModelVersionORM) -> None:
        if target.health_status is not HealthStatus.HEALTHY:
            raise ModelLifecycleError("异常模型不能成为当前有效版本", "MODEL_HEALTH_INVALID")
        if target.status not in {ModelVersionStatus.PUBLISHED, ModelVersionStatus.RETIRED}:
            raise ModelLifecycleError("目标模型必须是已发布版本", "MODEL_STATE_INVALID")
        self._clear_current(record)
        target.status = ModelVersionStatus.PUBLISHED
        target.is_current = True
        record.current_version_id = target.id
        self.session.flush()

    def _active_alert(self, model_type: ModelType) -> ModelAlertORM | None:
        return self.session.scalar(select(ModelAlertORM).where(
            ModelAlertORM.model_type == model_type,
            ModelAlertORM.status == AlertStatus.ACTIVE,
        ))

    def _resolve_alerts(self, model_type: ModelType, now: datetime) -> None:
        for alert in self.session.scalars(select(ModelAlertORM).where(
            ModelAlertORM.model_type == model_type,
            ModelAlertORM.status == AlertStatus.ACTIVE,
        )):
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = now
        record = self._model_type(model_type)
        record.alert_status = AlertStatus.RESOLVED

    def _published_healthy_candidates(self, model_type: ModelType, exclude_id: str) -> list[ModelVersionORM]:
        # RETIRED is a historical production version: it was published and is
        # deliberately retained as a rollback backup, so it remains eligible.
        return list(self.session.scalars(select(ModelVersionORM).where(
            ModelVersionORM.model_type == model_type,
            ModelVersionORM.id != exclude_id,
            ModelVersionORM.health_status == HealthStatus.HEALTHY,
            ModelVersionORM.status.in_([ModelVersionStatus.PUBLISHED, ModelVersionStatus.RETIRED]),
            ModelVersionORM.published_at.is_not(None),
        ).order_by(ModelVersionORM.published_at.desc(), ModelVersionORM.created_at.desc(), ModelVersionORM.id.desc())).all())

    def publish(self, version_id: str, *, confirmed: bool = False,
                message: str | None = None, idempotency_key: str | None = None) -> tuple[ModelVersionORM, PublishRecordORM | None]:
        if not confirmed:
            raise ModelLifecycleError("发布生产模型需要二次确认", "PUBLISH_CONFIRMATION_REQUIRED")
        version = self.get(version_id)
        if version is None:
            raise ModelNotFoundError()
        with self._lock_for(version.model_type):
            record = self._lock_type(version.model_type)
            if idempotency_key:
                prior = self.session.scalar(select(PublishRecordORM).where(
                    PublishRecordORM.idempotency_key == idempotency_key
                ))
                if prior is not None:
                    if prior.model_version_id != version.id:
                        raise ModelLifecycleError("幂等键已用于其他模型发布", "IDEMPOTENCY_KEY_CONFLICT")
                    return version, prior
            current = self._current(record)
            if current is version:
                # A retry after a committed request is a safe no-op and does
                # not create duplicate audit history.
                existing = self.session.scalar(select(PublishRecordORM).where(
                    PublishRecordORM.model_version_id == version.id
                ).order_by(PublishRecordORM.published_at.desc()))
                return version, existing
            if version.health_status is not HealthStatus.HEALTHY:
                raise ModelLifecycleError("只有健康模型才能发布", "MODEL_HEALTH_INVALID")
            if version.status is not ModelVersionStatus.READY:
                raise ModelLifecycleError("只有就绪模型才能发布", "PUBLISH_STATE_INVALID")
            now = self._now()
            previous_id = current.id if current else None
            cleared = self._clear_current(record)
            for old in cleared:
                if old.id != version.id and old.status is ModelVersionStatus.PUBLISHED:
                    old.status = ModelVersionStatus.RETIRED
            version.previous_healthy_version_id = previous_id if current and current.health_status is HealthStatus.HEALTHY else None
            version.status = ModelVersionStatus.PUBLISHED
            version.is_current = True
            version.published_at = now
            record.current_version_id = version.id
            record.alert_status = AlertStatus.RESOLVED
            self._resolve_alerts(version.model_type, now)
            release = PublishRecordORM(
                model_version_id=version.id, published_version=version.version,
                previous_current_version_id=previous_id, published_at=now, message=message,
                idempotency_key=idempotency_key,
            )
            self.session.add(release)
            try:
                self.session.commit()
            except IntegrityError as exc:
                self.session.rollback()
                raise ModelLifecycleError("模型发布并发冲突，请重试", "MODEL_LIFECYCLE_CONFLICT") from exc
            return version, release

    def retire(self, version_id: str) -> ModelVersionORM:
        version = self.get(version_id)
        if version is None:
            raise ModelNotFoundError()
        with self._lock_for(version.model_type):
            record = self._lock_type(version.model_type)
            if version.health_status is HealthStatus.ABNORMAL or version.status is ModelVersionStatus.ABNORMAL:
                raise ModelLifecycleError("异常模型不能直接下线", "OFFLINE_STATE_INVALID")
            if version.status is not ModelVersionStatus.PUBLISHED:
                raise ModelLifecycleError("只有已发布模型才能下线", "OFFLINE_STATE_INVALID")
            if version.is_current or record.current_version_id == version.id:
                self._clear_current(record)
            version.is_current = False
            version.status = ModelVersionStatus.RETIRED
            self.session.commit()
            return version

    def rollback(self, model_id: str, *, target_version_id: str | None = None,
                 target_version: str | None = None, reason: str = "手动回滚") -> tuple[RollbackRecordORM, ModelVersionORM]:
        requested = self.get(model_id)
        if requested is None:
            raise ModelNotFoundError()
        with self._lock_for(requested.model_type):
            record = self._lock_type(requested.model_type)
            current = self._current(record)
            target = None
            if target_version_id:
                target = self.get(target_version_id)
            elif target_version:
                target = self.session.scalar(select(ModelVersionORM).where(
                    ModelVersionORM.model_type == requested.model_type,
                    ModelVersionORM.version == target_version,
                ))
            else:
                # The path may identify the target version, which makes the
                # endpoint convenient for a UI's rollback button.
                target = requested
            if target is None:
                raise ModelLifecycleError("回滚目标版本不存在", "ROLLBACK_TARGET_NOT_FOUND")
            if target.model_type != requested.model_type:
                raise ModelLifecycleError("回滚目标模型类型不匹配", "ROLLBACK_TARGET_INVALID")
            source = current
            if requested is not current and requested.id != target.id and requested.is_current:
                source = requested
            if source is None:
                raise ModelLifecycleError("当前没有可回滚的生产版本", "ROLLBACK_STATE_INVALID")
            if target.id == source.id:
                raise ModelLifecycleError("不能回滚到当前版本", "ROLLBACK_TARGET_INVALID")
            if source.status is not ModelVersionStatus.PUBLISHED or source.health_status is not HealthStatus.HEALTHY:
                raise ModelLifecycleError("当前生产版本状态不可回滚", "ROLLBACK_STATE_INVALID")
            if target.status not in {ModelVersionStatus.PUBLISHED, ModelVersionStatus.RETIRED} or \
                    target.health_status is not HealthStatus.HEALTHY or target.published_at is None:
                raise ModelLifecycleError("目标必须是健康且已发布的历史版本", "ROLLBACK_TARGET_INVALID")
            now = self._now()
            source_id = source.id
            cleared = self._clear_current(record)
            for old in cleared:
                if old.id != target.id and old.status is ModelVersionStatus.PUBLISHED:
                    old.status = ModelVersionStatus.RETIRED
            source.status = ModelVersionStatus.RETIRED
            target.status = ModelVersionStatus.PUBLISHED
            target.is_current = True
            record.current_version_id = target.id
            rollback = RollbackRecordORM(
                model_type=requested.model_type, rollback_from=source_id,
                rollback_to=target.id, reason=reason, status=RollbackStatus.SUCCEEDED,
                created_at=now, finished_at=now,
            )
            self.session.add(rollback)
            try:
                self.session.commit()
            except IntegrityError as exc:
                self.session.rollback()
                raise ModelLifecycleError("回滚并发冲突，请重试", "MODEL_LIFECYCLE_CONFLICT") from exc
            return rollback, target

    def mark_abnormal(self, version_id: str, reason: str) -> tuple[ModelAlertORM, RollbackRecordORM | None, ModelVersionORM | None]:
        version = self.get(version_id)
        if version is None:
            raise ModelNotFoundError()
        if not reason.strip():
            raise ModelLifecycleError("异常原因不能为空", "ABNORMAL_REASON_REQUIRED")
        with self._lock_for(version.model_type):
            record = self._lock_type(version.model_type)
            current = self._current(record, include_unhealthy=True)
            if version.status is ModelVersionStatus.ABNORMAL and version.health_status is HealthStatus.ABNORMAL:
                alert = self._active_alert(version.model_type)
                return alert, None, self._current(record)
            now = self._now()
            # A historical anomaly is still recorded, but must not disturb a
            # healthy production pointer.  This is important when monitoring
            # a retired version or a published canary.
            if current is None or current.id != version.id:
                if version.status not in {ModelVersionStatus.PUBLISHED, ModelVersionStatus.RETIRED,
                                           ModelVersionStatus.READY}:
                    raise ModelLifecycleError("该版本状态不可标记异常", "ABNORMAL_STATE_INVALID")
                version.health_status = HealthStatus.ABNORMAL
                version.status = ModelVersionStatus.ABNORMAL
                alert = self._active_alert(version.model_type)
                if alert is None:
                    alert = ModelAlertORM(
                        model_type=version.model_type, model_version_id=version.id,
                        reason=reason, rollback_from=version.id,
                        rollback_to=current.id if current else None,
                        status=AlertStatus.ACTIVE, created_at=now,
                    )
                    self.session.add(alert)
                else:
                    alert.model_version_id = version.id
                    alert.reason = reason
                    alert.rollback_from = version.id
                    alert.rollback_to = current.id if current else None
                    alert.acknowledged_at = None
                record.alert_status = AlertStatus.ACTIVE
                self.session.commit()
                return alert, None, current
            if current.status is not ModelVersionStatus.PUBLISHED:
                raise ModelLifecycleError("只有当前已发布模型才能标记异常", "ABNORMAL_STATE_INVALID")
            version.health_status = HealthStatus.ABNORMAL
            version.status = ModelVersionStatus.ABNORMAL
            self._clear_current(record)
            alert = self._active_alert(version.model_type)
            if alert is None:
                alert = ModelAlertORM(
                    model_type=version.model_type, model_version_id=version.id,
                    reason=reason, rollback_from=version.id, status=AlertStatus.ACTIVE,
                    created_at=now,
                )
                self.session.add(alert)
                self.session.flush()
            else:
                alert.model_version_id = version.id
                alert.reason = reason
                alert.rollback_from = version.id
                alert.rollback_to = None
                alert.acknowledged_at = None
            record.alert_status = AlertStatus.ACTIVE
            candidates = self._published_healthy_candidates(version.model_type, version.id)
            target = candidates[0] if candidates else None
            rollback = RollbackRecordORM(
                model_type=version.model_type, rollback_from=version.id,
                rollback_to=target.id if target else None, alert=alert,
                reason=f"自动切换：{reason}", status=RollbackStatus.PENDING, created_at=now,
            )
            self.session.add(rollback)
            if target is not None:
                self._make_current(record, target)
                rollback.status = RollbackStatus.SUCCEEDED
                rollback.finished_at = self._now()
                alert.rollback_to = target.id
                self.session.commit()
                return alert, rollback, target
            rollback.status = RollbackStatus.FAILED
            rollback.finished_at = self._now()
            self.session.commit()
            raise NoHealthyRollbackError(
                "当前模型异常，未找到健康且已发布的回滚版本；告警仍保持 ACTIVE",
                alert_id=alert.id, rollback_record_id=rollback.id,
            )

    def alerts(self, *, model_type: ModelType | None = None,
               active_only: bool = False) -> list[ModelAlertORM]:
        statement = select(ModelAlertORM)
        if model_type is not None:
            statement = statement.where(ModelAlertORM.model_type == model_type)
        if active_only:
            statement = statement.where(ModelAlertORM.status == AlertStatus.ACTIVE)
        return list(self.session.scalars(statement.order_by(ModelAlertORM.created_at.desc(), ModelAlertORM.id.desc())).all())

    def acknowledge_alert(self, alert_id: str) -> ModelAlertORM:
        alert = self.session.get(ModelAlertORM, alert_id)
        if alert is None:
            raise ModelLifecycleError("告警不存在", "ALERT_NOT_FOUND")
        if alert.status is AlertStatus.RESOLVED:
            return alert
        alert.acknowledged_at = self._now()
        # Do not resolve here.  The alert must remain on the home page until a
        # successful publication explicitly resolves it.
        self.session.commit()
        return alert

    # Explicit operation aliases keep this boundary convenient for callers
    # that use the endpoint vocabulary rather than the short method names.
    save_model = save
    publish_model = publish
    offline = retire
    unpublish = retire
    mark_model_abnormal = mark_abnormal
    acknowledge = acknowledge_alert

    @staticmethod
    def to_model_response(version: ModelVersionORM) -> dict[str, Any]:
        return {
            "id": version.id, "model_type": version.model_type.value, "version": version.version,
            "model_path": version.model_path, "preprocessor_path": version.preprocessor_path,
            "training_job_id": version.training_job_id, "train_script_id": version.train_script_id,
            "train_script_version": version.train_script_version,
            "preprocess_script_id": version.preprocess_script_id,
            "preprocess_script_version": version.preprocess_script_version,
            "preprocess_used": version.preprocess_used, "input_schema": dict(version.input_schema or {}),
            "time_column": version.time_column, "feature_columns": list(version.feature_columns or []),
            "target_column": version.target_column, "metrics": dict(version.metrics or {}),
            "status": version.status.value, "health_status": version.health_status.value,
            "is_baseline": version.is_baseline, "is_current": version.is_current,
            "previous_healthy_version_id": version.previous_healthy_version_id,
            "created_at": version.created_at, "published_at": version.published_at,
        }

    @staticmethod
    def to_alert_response(alert: ModelAlertORM) -> dict[str, Any]:
        return {"id": alert.id, "model_type": alert.model_type.value,
                "model_version_id": alert.model_version_id, "reason": alert.reason,
                "rollback_from": alert.rollback_from, "rollback_to": alert.rollback_to,
                "status": alert.status.value, "created_at": alert.created_at,
                "acknowledged_at": alert.acknowledged_at, "resolved_at": alert.resolved_at}

    @staticmethod
    def to_rollback_response(item: RollbackRecordORM) -> dict[str, Any]:
        return {"id": item.id, "model_type": item.model_type.value,
                "rollback_from": item.rollback_from, "rollback_to": item.rollback_to,
                "alert_id": item.alert_id, "reason": item.reason,
                "status": item.status.value, "created_at": item.created_at,
                "finished_at": item.finished_at}


# Compatibility names for callers that refer to this boundary as the model
# service rather than the more explicit lifecycle service.
ModelService = ModelLifecycleService
ModelVersionService = ModelLifecycleService

__all__ = [
    "ModelLifecycleError", "ModelLifecycleService", "ModelNotFoundError",
    "NoHealthyRollbackError", "ModelService", "ModelVersionService",
]
