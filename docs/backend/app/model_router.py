"""Model version, release, failover, and alert endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db.models import ModelAlertORM, ModelVersionORM, PublishRecordORM, RollbackRecordORM
from .db.session import get_session
from .domain.enums import HealthStatus, ModelType, ModelVersionStatus
from .schemas.models import (
    AbnormalRequest,
    LifecycleOperationResponse,
    ModelAbnormalRequest,
    ModelAlertResponse,
    ModelSaveRequest,
    ModelVersionResponse,
    PublishRequest,
    RollbackRequest,
    RollbackResponse,
)
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
    }:
        response_status = status.HTTP_400_BAD_REQUEST
    elif exc.code in {
        "MODEL_VERSION_ALREADY_EXISTS", "MODEL_LIFECYCLE_CONFLICT", "IDEMPOTENCY_KEY_CONFLICT",
        "PUBLISH_STATE_INVALID", "OFFLINE_STATE_INVALID", "ROLLBACK_STATE_INVALID",
        "ROLLBACK_TARGET_INVALID", "ROLLBACK_TARGET_NOT_FOUND", "MODEL_HEALTH_INVALID",
        "ABNORMAL_STATE_INVALID", "NO_HEALTHY_BACKUP", "NO_HEALTHY_ROLLBACK_VERSION",
        "MODEL_BASELINE_IMMUTABLE", "MODEL_BASELINE_INVALID",
    }:
        response_status = status.HTTP_409_CONFLICT
    else:
        response_status = status.HTTP_400_BAD_REQUEST
    return HTTPException(
        status_code=response_status,
        detail={"code": exc.code, "message": str(exc), **exc.details},
    )


def _operation(operation: str, service: ModelLifecycleService, version: ModelVersionORM,
               *, rollback=None, alert=None) -> dict[str, Any]:
    model = service.to_model_response(version)
    result: dict[str, Any] = {"operation": operation, "model": model, **model}
    if rollback is not None:
        result["rollback"] = service.to_rollback_response(rollback)
    if alert is not None:
        result["alert"] = service.to_alert_response(alert)
    return result


@router.post("", status_code=status.HTTP_201_CREATED, response_model=ModelVersionResponse)
@router.post("/", status_code=status.HTTP_201_CREATED, response_model=ModelVersionResponse, include_in_schema=False)
def save_model(body: ModelSaveRequest, request: Request, session: Session = Depends(get_session)):
    service = _service(request, session)
    try:
        return service.to_model_response(service.save(body))
    except ModelLifecycleError as exc:
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
        raise _error(ModelNotFoundError())
    if existing.model_type is not body.model_type:
        raise _error(ModelLifecycleError("模型类型不匹配", "MODEL_TYPE_INVALID"))
    if existing.is_baseline:
        raise _error(ModelLifecycleError("系统基线不能修改", "MODEL_BASELINE_IMMUTABLE"))
    if existing.status not in {ModelVersionStatus.DRAFT, ModelVersionStatus.READY}:
        raise _error(ModelLifecycleError("当前状态不能再次保存", "MODEL_SAVE_STATE_INVALID"))
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
                      "metrics", "preprocess_used", "preprocessor_state"):
            value = getattr(body, field)
            if value is not None and value != [] and value != {}:
                setattr(existing, field, value)
        existing.status = body.status
        session.commit()
        return service.to_model_response(existing)
    except ModelLifecycleError as exc:
        session.rollback()
        raise _error(exc) from exc


@router.post("/{model_id}/publish", response_model=LifecycleOperationResponse)
def publish_model(model_id: str, request: Request, body: PublishRequest | None = None,
                  session: Session = Depends(get_session)):
    service = _service(request, session)
    body = body or PublishRequest()
    try:
        version, _record = service.publish(
            model_id, confirmed=body.is_confirmed, message=body.message or body.reason,
            idempotency_key=body.idempotency_key,
        )
        return _operation("publish", service, version)
    except ModelLifecycleError as exc:
        raise _error(exc) from exc


def _retire(model_id: str, request: Request, session: Session):
    service = _service(request, session)
    try:
        return _operation("offline", service, service.retire(model_id))
    except ModelLifecycleError as exc:
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


@router.delete("/{model_id}", response_model=LifecycleOperationResponse, include_in_schema=False)
def delete_model_as_offline(model_id: str, request: Request, session: Session = Depends(get_session)):
    """Compatibility alias: deleting an API resource never deletes its artifact."""
    return _retire(model_id, request, session)


@router.post("/{model_id}/rollback", response_model=LifecycleOperationResponse)
def rollback_model(model_id: str, request: Request, body: RollbackRequest | None = None,
                   session: Session = Depends(get_session)):
    service = _service(request, session)
    body = body or RollbackRequest()
    try:
        rollback, target = service.rollback(
            model_id, target_version_id=body.target_version_id or body.version_id,
            target_version=body.target_version or body.version, reason=body.reason,
        )
        return _operation("rollback", service, target, rollback=rollback)
    except ModelLifecycleError as exc:
        raise _error(exc) from exc


@router.post("/abnormal", response_model=LifecycleOperationResponse)
def abnormal_model_by_type(body: ModelAbnormalRequest, request: Request,
                           session: Session = Depends(get_session)):
    """Mark a type-scoped version abnormal and fail over atomically."""
    service = _service(request, session)
    try:
        version = service._resolve_model_version(body.model_type, body.model_version)
        alert, rollback, target = service.mark_model_abnormal(
            body.model_type, body.model_version,
            reason=body.reason, abnormal=body.abnormal,
        )
        response_version = target or version
        return _operation("abnormal", service, response_version, rollback=rollback, alert=alert)
    except ModelLifecycleError as exc:
        raise _error(exc) from exc


@router.post("/{model_id}/abnormal", response_model=LifecycleOperationResponse)
def abnormal_model(model_id: str, request: Request, body: AbnormalRequest | None = None,
                   session: Session = Depends(get_session)):
    """Compatibility form retained for clients that already have a version id."""
    service = _service(request, session)
    body = body or AbnormalRequest()
    try:
        alert, rollback, target = service.mark_abnormal(
            model_id, body.reason, abnormal=body.abnormal,
            no_backup_code="NO_HEALTHY_ROLLBACK_VERSION",
        )
        # A repeated request can find the existing alert and have no target;
        # return the abnormal source where possible rather than failing a
        # harmless retry.
        response_version = target or service.get(model_id)
        if response_version is None:
            raise ModelNotFoundError()
        return _operation("abnormal", service, response_version, rollback=rollback, alert=alert)
    except ModelLifecycleError as exc:
        raise _error(exc) from exc


@router.get("/{model_id}/publish-records")
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
             "published_at": item.published_at, "message": item.message,
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


@alerts_router.get("", response_model=list[ModelAlertResponse])
@alerts_router.get("/", response_model=list[ModelAlertResponse], include_in_schema=False)
def list_alerts(request: Request, model_type: ModelType | None = None,
                active_only: bool = False, session: Session = Depends(get_session)):
    service = _service(request, session)
    return [service.to_alert_response(item) for item in service.alerts(
        model_type=model_type, active_only=active_only
    )]


@alerts_router.get("/{alert_id}", response_model=ModelAlertResponse)
def get_alert(alert_id: str, request: Request, session: Session = Depends(get_session)):
    service = _service(request, session)
    alert = session.get(ModelAlertORM, alert_id)
    if alert is None:
        raise _error(ModelLifecycleError("告警不存在", "ALERT_NOT_FOUND"))
    return service.to_alert_response(alert)


@alerts_router.post("/{alert_id}/acknowledge", response_model=ModelAlertResponse)
@alerts_router.post("/{alert_id}/ack", response_model=ModelAlertResponse, include_in_schema=False)
@alerts_router.post("/{alert_id}/confirm", response_model=ModelAlertResponse, include_in_schema=False)
def acknowledge_alert(alert_id: str, request: Request, session: Session = Depends(get_session)):
    service = _service(request, session)
    try:
        return service.to_alert_response(service.acknowledge_alert(alert_id))
    except ModelLifecycleError as exc:
        raise _error(exc) from exc


__all__ = ["alerts_router", "router"]
