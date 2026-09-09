"""Model version, release, failover, and alert endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db.models import ModelAlertORM, ModelVersionORM, PublishRecordORM, RollbackRecordORM
from .db.session import get_session
from .domain.enums import AlertStatus, HealthStatus, ModelType, ModelVersionStatus
from .schemas.models import (
    AbnormalRequest,
    LifecycleOperationResponse,
    ModelAbnormalRequest,
    ModelAlertResponse,
    ModelDeletionResponse,
    ModelSaveRequest,
    ModelVersionResponse,
    PublishRequest,
    RollbackRequest,
    RollbackResponse,
    AlertAcknowledgeRequest,
    AlertAcknowledgeResponse,
    AlertListResponse,
    PublishRecordResponse,
)
from .services.audit_events import context_from_request, record_audit_event, record_failure_audit_event
from .services.model_lifecycle import (
    ModelLifecycleError,
    ModelLifecycleService,
    ModelNotFoundError,
    ModelVersionNotFoundError,
    NoHealthyRollbackError,
)

router = APIRouter(prefix="/api/models", tags=["models"])
alerts_router = APIRouter(prefix="/api/alerts", tags=["alerts"])


def _service(request: Request, session: Session) -> ModelLifecycleService:
    return ModelLifecycleService(session, settings=getattr(request.app.state, "settings", None))


def _error(exc: ModelLifecycleError) -> HTTPException:
    if isinstance(exc, (ModelNotFoundError, ModelVersionNotFoundError)) or exc.code == "ALERT_NOT_FOUND":
        response_status = status.HTTP_404_NOT_FOUND
    elif exc.code in {
        "PUBLISH_CONFIRMATION_REQUIRED", "MODEL_ARTIFACT_INVALID",
        "MODEL_ARTIFACT_NOT_FOUND", "PREPROCESSOR_STATE_INVALID",
        "MODEL_INPUT_SCHEMA_INVALID", "ABNORMAL_REASON_REQUIRED", "MODEL_TYPE_NOT_FOUND",
        "ROLLBACK_TARGET_REQUIRED",
    }:
        response_status = status.HTTP_400_BAD_REQUEST
    elif exc.code in {
        "MODEL_VERSION_ALREADY_EXISTS", "MODEL_LIFECYCLE_CONFLICT", "IDEMPOTENCY_KEY_CONFLICT",
        "PUBLISH_STATE_INVALID", "OFFLINE_STATE_INVALID", "ROLLBACK_STATE_INVALID",
        "ROLLBACK_TARGET_INVALID", "ROLLBACK_TARGET_NOT_FOUND", "MODEL_HEALTH_INVALID",
        "ABNORMAL_STATE_INVALID", "NO_HEALTHY_BACKUP", "NO_HEALTHY_ROLLBACK_VERSION",
        "MODEL_BASELINE_IMMUTABLE", "MODEL_BASELINE_INVALID", "MODEL_DELETE_REFERENCED",
        "MODEL_ARTIFACT_DELETE_FAILED", "MODEL_DELETE_STATE_INVALID",
    }:
        response_status = status.HTTP_409_CONFLICT
    else:
        response_status = status.HTTP_400_BAD_REQUEST
    return HTTPException(
        status_code=response_status,
        detail={"code": exc.code, "message": str(exc), **exc.details},
    )


def _operation(operation: str, service: ModelLifecycleService, version: ModelVersionORM,
               *, rollback=None, alert=None, record=None) -> dict[str, Any]:
    model = service.to_model_response(version)
    result: dict[str, Any] = {"operation": operation, "model": model, **model}
    if record is not None:
        result["record"] = {
            "id": record.id, "model_version_id": record.model_version_id,
            "published_version": record.published_version,
            "previous_current_version_id": record.previous_current_version_id,
            "published_at": record.published_at,
            "reason": record.reason if record.reason is not None else record.message,
            "message": record.message,
            "idempotency_key": record.idempotency_key,
        }
    if rollback is not None:
        result["rollback"] = service.to_rollback_response(rollback)
    if alert is not None:
        result["alert"] = service.to_alert_response(alert)
    return result


@router.post("", status_code=status.HTTP_201_CREATED, response_model=ModelVersionResponse)
@router.post("/", status_code=status.HTTP_201_CREATED, response_model=ModelVersionResponse, include_in_schema=False)
def save_model(body: ModelSaveRequest, request: Request, session: Session = Depends(get_session)):
    service = _service(request, session)
    version_for_audit = service.get(body.id) if body.id else None
    try:
        return service.to_model_response(service.save(body, audit_context=context_from_request(request)))
    except ModelLifecycleError as exc:
        record_failure_audit_event(
            session, event_type="MODEL_SAVED", object_type="MODEL_VERSION",
            object_id=body.id,
            model_type=version_for_audit.model_type if version_for_audit is not None else body.model_type,
            model_version_id=version_for_audit.id if version_for_audit is not None else None,
            message=str(exc), metadata={"error_code": exc.code}, context=context_from_request(request),
        )
        raise _error(exc) from exc


@router.get("", response_model=list[ModelVersionResponse])
@router.get("/", response_model=list[ModelVersionResponse], include_in_schema=False)
def list_models(
    request: Request,
    model_type: ModelType | None = None,
    model_status: ModelVersionStatus | None = Query(default=None, alias="status"),
    health_status: HealthStatus | None = None,
    session: Session = Depends(get_session),
):
    service = _service(request, session)
    return [service.to_model_response(item) for item in service.list(
        model_type=model_type, status=model_status, health_status=health_status
    )]


@router.get("/{model_id}", response_model=ModelVersionResponse)
def get_model(model_id: str, request: Request, session: Session = Depends(get_session)):
    service = _service(request, session)
    version = service.get(model_id)
    if version is None:
        raise _error(ModelNotFoundError())
    return service.to_model_response(version)


@router.post("/{model_id}/save", response_model=ModelVersionResponse)
def save_existing_model(model_id: str, body: ModelSaveRequest, request: Request,
                        session: Session = Depends(get_session)):
    """Complete registration of a training-created draft without publishing it."""
    service = _service(request, session)
    existing = service.get(model_id)
    if existing is None:
        exc = ModelNotFoundError()
        record_failure_audit_event(
            session, event_type="MODEL_SAVED", object_type="MODEL_VERSION", object_id=model_id,
            model_type=body.model_type,
            message=str(exc), metadata={"error_code": exc.code}, context=context_from_request(request),
        )
        raise _error(exc)
    if existing.model_type is not body.model_type:
        exc = ModelLifecycleError("模型类型不匹配", "MODEL_TYPE_INVALID")
        record_failure_audit_event(
            session, event_type="MODEL_SAVED", object_type="MODEL_VERSION", object_id=model_id,
            model_type=existing.model_type, model_version_id=existing.id,
            message=str(exc), metadata={"error_code": exc.code}, context=context_from_request(request),
        )
        raise _error(exc)
    if existing.is_baseline:
        exc = ModelLifecycleError("系统基线不能修改", "MODEL_BASELINE_IMMUTABLE")
        record_failure_audit_event(
            session, event_type="MODEL_SAVED", object_type="MODEL_VERSION", object_id=model_id,
            model_type=existing.model_type, model_version_id=existing.id,
            message=str(exc), metadata={"error_code": exc.code}, context=context_from_request(request),
        )
        raise _error(exc)
    if existing.status not in {ModelVersionStatus.DRAFT, ModelVersionStatus.READY}:
        exc = ModelLifecycleError("当前状态不能再次保存", "MODEL_SAVE_STATE_INVALID")
        record_failure_audit_event(
            session, event_type="MODEL_SAVED", object_type="MODEL_VERSION", object_id=model_id,
            model_type=existing.model_type, model_version_id=existing.id,
            message=str(exc), metadata={"error_code": exc.code}, context=context_from_request(request),
        )
        raise _error(exc)
    try:
        # The normal save contract remains immutable.  This small adapter only
        # fills the optional artifact/metadata fields on an existing draft.
        if body.model_content_base64 is not None:
            import base64
            try:
                raw = base64.b64decode(body.model_content_base64, validate=True)
            except Exception as exc:
                raise ModelLifecycleError("模型制品不是有效的 base64", "MODEL_ARTIFACT_INVALID") from exc
            artifact = service.storage.save_model(model_id, raw)
            existing.model_path = artifact.relative_path
            existing.model_artifact_id = artifact.id
        for field in ("preprocessor_path", "training_job_id", "train_script_id",
                      "train_script_version", "train_script_source", "preprocess_script_id",
                      "preprocess_script_version", "preprocess_script_source", "input_schema",
                      "feature_columns", "time_column", "target_column", "split_strategy",
                      "split_ratio", "test_ratio", "train_data_summary", "test_data_summary",
                      "metrics", "preprocess_used", "preprocessor_state", "health_status"):
            value = getattr(body, field)
            if value is not None and value != [] and value != {}:
                if field == "metrics":
                    # A legacy caller may submit only the flat metric set;
                    # preserve the immutable evaluation chart/comparison
                    # envelope already produced by training.
                    value = {**dict(existing.metrics or {}), **value}
                setattr(existing, field, value)
        existing.status = body.status
        record_audit_event(
            session, event_type="MODEL_SAVED", object_type="MODEL_VERSION",
            object_id=existing.id, model_type=existing.model_type,
            model_version_id=existing.id, training_job_id=existing.training_job_id,
            message="已有模型版本保存成功", metadata={"version": existing.version},
            context=context_from_request(request),
        )
        session.commit()
        return service.to_model_response(existing)
    except ModelLifecycleError as exc:
        session.rollback()
        record_failure_audit_event(
            session, event_type="MODEL_SAVED", object_type="MODEL_VERSION",
            object_id=model_id, model_type=existing.model_type, model_version_id=existing.id,
            message=str(exc), metadata={"error_code": exc.code},
            context=context_from_request(request),
        )
        raise _error(exc) from exc


@router.post("/{model_id}/publish", response_model=LifecycleOperationResponse)
def publish_model(model_id: str, request: Request, body: PublishRequest | None = None,
                  session: Session = Depends(get_session)):
    service = _service(request, session)
    body = body or PublishRequest()
    version_for_audit = service.get(model_id)
    try:
        version, record = service.publish(
            model_id, confirmed=body.confirmed, reason=body.reason,
            idempotency_key=body.idempotency_key,
            audit_context=context_from_request(request),
        )
        return _operation("publish", service, version, record=record)
    except ModelLifecycleError as exc:
        record_failure_audit_event(
            session, event_type="MODEL_PUBLISHED", object_type="MODEL_VERSION",
            object_id=model_id,
            model_type=version_for_audit.model_type if version_for_audit is not None else None,
            model_version_id=version_for_audit.id if version_for_audit is not None else None,
            message=str(exc),
            metadata={
                "error_code": exc.code,
                "reason": body.reason,
                "idempotency_key": body.idempotency_key,
            },
            context=context_from_request(request),
        )
        raise _error(exc) from exc


def _retire(model_id: str, request: Request, session: Session):
    service = _service(request, session)
    version_for_audit = service.get(model_id)
    try:
        return _operation("offline", service, service.retire(model_id, context_from_request(request)))
    except ModelLifecycleError as exc:
        record_failure_audit_event(
            session, event_type="MODEL_OFFLINED", object_type="MODEL_VERSION",
            object_id=model_id,
            model_type=version_for_audit.model_type if version_for_audit is not None else None,
            model_version_id=version_for_audit.id if version_for_audit is not None else None,
            message=str(exc), metadata={"error_code": exc.code},
            context=context_from_request(request),
        )
        raise _error(exc) from exc


@router.post("/{model_id}/offline", response_model=LifecycleOperationResponse)
def offline_model(model_id: str, request: Request, session: Session = Depends(get_session)):
    return _retire(model_id, request, session)


@router.post("/{model_id}/unpublish", response_model=LifecycleOperationResponse, include_in_schema=False)
def unpublish_model(model_id: str, request: Request, session: Session = Depends(get_session)):
    return _retire(model_id, request, session)


@router.post("/{model_id}/retire", response_model=LifecycleOperationResponse, include_in_schema=False)
def retire_model(model_id: str, request: Request, session: Session = Depends(get_session)):
    return _retire(model_id, request, session)


def _delete_candidate(model_id: str, request: Request, session: Session):
    service = _service(request, session)
    version_for_audit = service.get(model_id)
    try:
        return service.delete_candidate(model_id, context_from_request(request))
    except ModelLifecycleError as exc:
        record_failure_audit_event(
            session,
            event_type="MODEL_DELETED",
            object_type="MODEL_VERSION",
            object_id=model_id,
            model_type=version_for_audit.model_type if version_for_audit is not None else None,
            model_version_id=version_for_audit.id if version_for_audit is not None else None,
            training_job_id=version_for_audit.training_job_id if version_for_audit is not None else None,
            message=str(exc),
            metadata={"error_code": exc.code},
            context=context_from_request(request),
        )
        raise _error(exc) from exc


@router.delete("/{model_id}/artifacts", response_model=ModelDeletionResponse)
def delete_model_artifacts(model_id: str, request: Request, session: Session = Depends(get_session)):
    """Hard-delete the unpublished model artifact for an evaluation decision."""
    return _delete_candidate(model_id, request, session)


@router.delete("/{model_id}/discard", response_model=None, include_in_schema=False)
def discard_model_candidate(model_id: str, request: Request, session: Session = Depends(get_session)):
    """Compatibility spelling for discarding an unpublished candidate."""
    return _delete_candidate(model_id, request, session)


@router.delete("/{model_id}", response_model=None, include_in_schema=False)
def delete_model_as_offline(model_id: str, request: Request, session: Session = Depends(get_session)):
    """Delete an unpublished candidate; retain offline behavior for history."""
    service = _service(request, session)
    version = service.get(model_id)
    if version is not None and not version.is_baseline and not version.is_current and version.status in {
        ModelVersionStatus.DRAFT,
        ModelVersionStatus.READY,
        ModelVersionStatus.FAILED,
    }:
        return _delete_candidate(model_id, request, session)
    return _retire(model_id, request, session)


@router.post("/{model_id}/rollback", response_model=LifecycleOperationResponse)
def rollback_model(model_id: str, request: Request, body: RollbackRequest | None = None,
                   session: Session = Depends(get_session)):
    service = _service(request, session)
    body = body or RollbackRequest()
    version_for_audit = service.get(model_id)
    try:
        if not body.target_version_id:
            raise ModelLifecycleError(
                "回滚必须显式提供 target_version_id", "ROLLBACK_TARGET_REQUIRED"
            )
        rollback, target = service.rollback(
            model_id, target_version_id=body.target_version_id, reason=body.reason,
            idempotency_key=body.idempotency_key,
            audit_context=context_from_request(request),
        )
        return _operation("rollback", service, target, rollback=rollback)
    except ModelLifecycleError as exc:
        record_failure_audit_event(
            session, event_type="MODEL_ROLLBACK_FAILED", object_type="MODEL_VERSION",
            object_id=model_id,
            model_type=version_for_audit.model_type if version_for_audit is not None else None,
            model_version_id=version_for_audit.id if version_for_audit is not None else None,
            message=str(exc),
            metadata={
                "error_code": exc.code,
                "target_version_id": body.target_version_id,
                "reason": body.reason,
                "idempotency_key": body.idempotency_key,
            },
            context=context_from_request(request),
        )
        raise _error(exc) from exc


@router.post("/abnormal", response_model=LifecycleOperationResponse)
def abnormal_model_by_type(body: ModelAbnormalRequest, request: Request,
                           session: Session = Depends(get_session)):
    """Mark a type-scoped version abnormal and fail over atomically."""
    service = _service(request, session)
    version_for_audit = None
    try:
        version_for_audit = service._resolve_model_version(body.model_type, body.model_version)
        alert, rollback, target = service.mark_model_abnormal(
            body.model_type, body.model_version,
            reason=body.reason, abnormal=body.abnormal,
            audit_context=context_from_request(request),
        )
        response_version = target or version_for_audit
        return _operation("abnormal", service, response_version, rollback=rollback, alert=alert)
    except ModelLifecycleError as exc:
        # NoHealthyRollbackError is raised after the abnormal state and the
        # failed rollback have already committed; do not add a false failure
        # for the successfully applied abnormal transition.
        if not isinstance(exc, NoHealthyRollbackError):
            record_failure_audit_event(
                session, event_type="MODEL_MARKED_ABNORMAL", object_type="MODEL_VERSION",
                object_id=version_for_audit.id if version_for_audit is not None else body.model_version,
                model_type=version_for_audit.model_type if version_for_audit is not None else body.model_type,
                model_version_id=version_for_audit.id if version_for_audit is not None else None,
                message=str(exc), metadata={"error_code": exc.code}, context=context_from_request(request),
            )
        raise _error(exc) from exc


@router.post("/{model_id}/abnormal", response_model=LifecycleOperationResponse)
def abnormal_model(model_id: str, request: Request, body: AbnormalRequest | None = None,
                   session: Session = Depends(get_session)):
    """Compatibility form retained for clients that already have a version id."""
    service = _service(request, session)
    body = body or AbnormalRequest()
    version_for_audit = service.get(model_id)
    try:
        alert, rollback, target = service.mark_abnormal(
            model_id, body.reason, abnormal=body.abnormal,
            no_backup_code="NO_HEALTHY_ROLLBACK_VERSION",
            audit_context=context_from_request(request),
        )
        # A repeated request can find the existing alert and have no target;
        # return the abnormal source where possible rather than failing a
        # harmless retry.
        response_version = target or service.get(model_id)
        if response_version is None:
            raise ModelNotFoundError()
        return _operation("abnormal", service, response_version, rollback=rollback, alert=alert)
    except ModelLifecycleError as exc:
        if not isinstance(exc, NoHealthyRollbackError):
            record_failure_audit_event(
                session, event_type="MODEL_MARKED_ABNORMAL", object_type="MODEL_VERSION",
                object_id=model_id,
                model_type=version_for_audit.model_type if version_for_audit is not None else None,
                model_version_id=version_for_audit.id if version_for_audit is not None else None,
                message=str(exc), metadata={"error_code": exc.code},
                context=context_from_request(request),
            )
        raise _error(exc) from exc


@router.get("/{model_id}/publish-records", response_model=list[PublishRecordResponse])
def model_publish_records(model_id: str, request: Request, session: Session = Depends(get_session)):
    service = _service(request, session)
    if service.get(model_id) is None:
        raise _error(ModelNotFoundError())
    records = session.scalars(select(PublishRecordORM).where(
        PublishRecordORM.model_version_id == model_id
    ).order_by(PublishRecordORM.published_at.desc())).all()
    return [{"id": item.id, "model_version_id": item.model_version_id,
             "published_version": item.published_version,
             "previous_current_version_id": item.previous_current_version_id,
             "published_at": item.published_at,
             "reason": item.reason if item.reason is not None else item.message,
             "message": item.message,
             "idempotency_key": item.idempotency_key} for item in records]


@router.get("/{model_id}/rollback-records", response_model=list[RollbackResponse])
def model_rollback_records(model_id: str, request: Request, session: Session = Depends(get_session)):
    service = _service(request, session)
    version = service.get(model_id)
    if version is None:
        raise _error(ModelNotFoundError())
    records = session.scalars(select(RollbackRecordORM).where(
        (RollbackRecordORM.rollback_from == model_id) | (RollbackRecordORM.rollback_to == model_id)
    ).order_by(RollbackRecordORM.created_at.desc())).all()
    return [service.to_rollback_response(item) for item in records]


@alerts_router.get("", response_model=AlertListResponse | list[ModelAlertResponse])
@alerts_router.get("/", response_model=AlertListResponse | list[ModelAlertResponse], include_in_schema=False)
def list_alerts(
    request: Request,
    response: Response,
    model_type: ModelType | None = None,
    status: AlertStatus | None = None,
    active_only: bool = False,
    page: int | None = Query(default=None, ge=1),
    page_size: int | None = Query(default=None, ge=1, le=1000),
    limit: int | None = Query(default=None, ge=1, le=1000),
    cursor: str | None = Query(default=None, description="上一次响应的 next_cursor"),
    session: Session = Depends(get_session),
):
    import base64
    import json
    service = _service(request, session)
    rows = service.alerts(model_type=model_type, status=status, active_only=active_only)
    total = len(rows)
    statistics = service.alert_statistics(rows)
    start = 0
    if cursor:
        try:
            token = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
            key = (token["created_at"], token["id"])
            start = next((index + 1 for index, item in enumerate(rows)
                          if (item.created_at.isoformat(), item.id) == key), total)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            raise _error(ModelLifecycleError("告警游标无效", "ALERT_CURSOR_INVALID"))
    requested_size = limit if limit is not None else page_size
    page_number = page or 1
    if cursor is None and page is not None and requested_size is not None:
        start = (page_number - 1) * requested_size
    page_items = rows[start:start + requested_size] if requested_size is not None else rows[start:]
    next_page = None
    if requested_size is not None and start + requested_size < total and page_items:
        last = page_items[-1]
        next_page = base64.urlsafe_b64encode(json.dumps({
            "created_at": last.created_at.isoformat(), "id": last.id,
        }, separators=(",", ":")).encode()).decode()
    items = [service.to_alert_response(item) for item in page_items]
    if page is None and page_size is None and limit is None and cursor is None:
        # Compatibility: existing clients receive the original array shape;
        # X-Total-Count supplies the same total without a second query.
        response.headers["X-Total-Count"] = str(total)
        return items
    return {
        "items": items, "total": total,
        "page": page_number,
        "page_size": requested_size,
        "limit": limit,
        "next_cursor": next_page, "statistics": statistics,
    }


@alerts_router.get("/{alert_id}", response_model=ModelAlertResponse)
def get_alert(alert_id: str, request: Request, session: Session = Depends(get_session)):
    service = _service(request, session)
    alert = session.get(ModelAlertORM, alert_id)
    if alert is None:
        raise _error(ModelLifecycleError("告警不存在", "ALERT_NOT_FOUND"))
    return service.to_alert_response(alert)


@alerts_router.post("/{alert_id}/acknowledge", response_model=AlertAcknowledgeResponse)
@alerts_router.post("/{alert_id}/ack", response_model=AlertAcknowledgeResponse, include_in_schema=False)
@alerts_router.post("/{alert_id}/confirm", response_model=AlertAcknowledgeResponse, include_in_schema=False)
def acknowledge_alert(alert_id: str, request: Request, body: AlertAcknowledgeRequest | None = None,
                      session: Session = Depends(get_session)):
    service = _service(request, session)
    alert_for_audit = session.get(ModelAlertORM, alert_id)
    try:
        if body is not None and not body.confirmed:
            raise ModelLifecycleError("确认告警必须 confirmed=true", "ALERT_CONFIRMATION_REQUIRED")
        alert = service.acknowledge_alert(alert_id, context_from_request(request))
        rows = service.alerts(model_type=alert.model_type)
        return {
            **service.to_alert_response(alert),
            "statistics": service.alert_statistics(rows),
        }
    except ModelLifecycleError as exc:
        record_failure_audit_event(
            session, event_type="ALERT_ACKNOWLEDGED", object_type="ALERT",
            object_id=alert_id,
            model_type=alert_for_audit.model_type if alert_for_audit is not None else None,
            model_version_id=alert_for_audit.model_version_id if alert_for_audit is not None else None,
            message=str(exc), metadata={"error_code": exc.code},
            context=context_from_request(request),
        )
        raise _error(exc) from exc


__all__ = ["alerts_router", "router"]
