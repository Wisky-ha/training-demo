"""HTTP endpoints for training creation, polling, retry, logs, and evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from ..db.session import get_session
from ..services.audit_events import context_from_request, record_failure_audit_event
from ..schemas.training_jobs import EvaluationResponse, TrainingJobCreate, TrainingJobLogsResponse, TrainingJobResponse
from ..services.training_jobs import TrainingJobError, TrainingJobNotFoundError, TrainingJobService

router = APIRouter(prefix="/api/training-jobs", tags=["training-jobs"])


def _service(request: Request, session: Session) -> TrainingJobService:
    return TrainingJobService(session, settings=getattr(request.app.state, "settings", None))


def _error(exc: TrainingJobError) -> HTTPException:
    code = getattr(exc, "code", "TRAINING_JOB_FAILED")
    if isinstance(exc, TrainingJobNotFoundError):
        response_status = status.HTTP_404_NOT_FOUND
    elif code == "TRAINING_RETRY_NOT_ALLOWED":
        response_status = status.HTTP_409_CONFLICT
    elif code == "EVALUATION_NOT_READY":
        response_status = status.HTTP_409_CONFLICT
    elif code == "TRAINING_LOG_CURSOR_INVALID":
        response_status = status.HTTP_400_BAD_REQUEST
    else:
        response_status = status.HTTP_400_BAD_REQUEST
    details = dict(getattr(exc, "details", {}) or {})
    if exc.job_id:
        details["job_id"] = exc.job_id
    return HTTPException(
        status_code=response_status,
        detail={"code": code, "message": str(exc), **details},
    )


def _response(job: Any) -> dict[str, Any]:
    return TrainingJobService.to_response(job)


@router.post("", status_code=status.HTTP_201_CREATED, response_model=TrainingJobResponse)
@router.post("/", status_code=status.HTTP_201_CREATED, response_model=TrainingJobResponse, include_in_schema=False)
def create_training_job(body: TrainingJobCreate, request: Request, session: Session = Depends(get_session)) -> dict[str, Any]:
    service = _service(request, session)
    try:
        job = service.create(body, audit_context=context_from_request(request))
        executor = getattr(request.app.state, "training_job_executor", None)
        if executor is None:
            raise TrainingJobError("后台任务执行器不可用", "TRAINING_EXECUTOR_UNAVAILABLE")
        TrainingJobService.submit(
            executor, request.app.state.session_factory, job.id, body.config,
            service.settings, context_from_request(request),
        )
    except TrainingJobError as exc:
        record_failure_audit_event(
            session, event_type="TRAINING_FAILED", object_type="TRAINING_JOB",
            object_id=job.id if "job" in locals() else None,
            model_type=body.model_type,
            training_job_id=job.id if "job" in locals() else None,
            message=str(exc),
            metadata={"error_code": exc.code}, context=context_from_request(request),
        )
        raise _error(exc) from exc
    except Exception:
        record_failure_audit_event(
            session, event_type="TRAINING_FAILED", object_type="TRAINING_JOB",
            object_id=job.id if "job" in locals() else None,
            model_type=body.model_type,
            training_job_id=job.id if "job" in locals() else None,
            message="训练任务提交失败",
            metadata={"error_code": "TRAINING_SUBMIT_FAILED"}, context=context_from_request(request),
        )
        raise
    return _response(job)


@router.get("/{job_id}", response_model=TrainingJobResponse)
def get_training_job(job_id: str, request: Request, session: Session = Depends(get_session)) -> dict[str, Any]:
    job = _service(request, session).get(job_id)
    if job is None:
        raise _error(TrainingJobNotFoundError(f"训练任务不存在：{job_id}"))
    return _response(job)


@router.get("/{job_id}/logs", response_model=TrainingJobLogsResponse)
def get_training_job_logs(
    job_id: str,
    request: Request,
    since: datetime | None = Query(default=None, description="只返回该时间之后新增的日志"),
    limit: int | None = Query(default=None, ge=1, le=1000),
    cursor: str | None = Query(default=None, description="上一次响应的 next_cursor"),
    next_cursor: str | None = Query(default=None, description="cursor 的兼容查询名称"),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    service = _service(request, session)
    job = service.get(job_id)
    if job is None:
        raise _error(TrainingJobNotFoundError(f"训练任务不存在：{job_id}"))
    try:
        items, page_cursor = service.log_page(
            job, since=since, limit=limit, cursor=cursor or next_cursor,
        )
    except TrainingJobError as exc:
        raise _error(exc) from exc
    return {"job_id": job.id, "items": items, "next_cursor": page_cursor}


@router.post("/{job_id}/retry", response_model=TrainingJobResponse)
def retry_training_job(job_id: str, request: Request, session: Session = Depends(get_session)) -> dict[str, Any]:
    service = _service(request, session)
    try:
        job = service.retry(job_id, audit_context=context_from_request(request))
        executor = getattr(request.app.state, "training_job_executor", None)
        if executor is None:
            raise TrainingJobError("后台任务执行器不可用", "TRAINING_EXECUTOR_UNAVAILABLE")
        TrainingJobService.submit(
            executor, request.app.state.session_factory, job.id, job.config,
            service.settings, context_from_request(request),
        )
    except TrainingJobError as exc:
        record_failure_audit_event(
            session, event_type="TRAINING_FAILED", object_type="TRAINING_JOB",
            object_id=job_id, training_job_id=job_id, message=str(exc),
            metadata={"error_code": exc.code},
            context=context_from_request(request),
        )
        raise _error(exc) from exc
    except Exception:
        record_failure_audit_event(
            session, event_type="TRAINING_FAILED", object_type="TRAINING_JOB",
            object_id=job_id, training_job_id=job_id, message="训练任务重试提交失败",
            metadata={"error_code": "TRAINING_RETRY_SUBMIT_FAILED"}, context=context_from_request(request),
        )
        raise
    return _response(job)


@router.get("/{job_id}/evaluation", response_model=EvaluationResponse)
def get_training_evaluation(job_id: str, request: Request, session: Session = Depends(get_session)) -> dict[str, Any]:
    service = _service(request, session)
    job = service.get(job_id)
    if job is None:
        raise _error(TrainingJobNotFoundError(f"训练任务不存在：{job_id}"))
    try:
        return service.evaluation_response(job)
    except TrainingJobError as exc:
        raise _error(exc) from exc


@router.post("/{job_id}/cancel", response_model=TrainingJobResponse)
def cancel_training_job(job_id: str, request: Request, session: Session = Depends(get_session)) -> dict[str, Any]:
    try:
        job = _service(request, session).cancel(job_id, context_from_request(request))
    except TrainingJobError as exc:
        record_failure_audit_event(
            session, event_type="TRAINING_CANCELLED", object_type="TRAINING_JOB",
            object_id=job_id, training_job_id=job_id, message=str(exc),
            metadata={"error_code": exc.code},
            context=context_from_request(request),
        )
        raise _error(exc) from exc
    return _response(job)


@router.delete("/{job_id}", response_model=TrainingJobResponse, include_in_schema=False)
def delete_training_job(job_id: str, request: Request, session: Session = Depends(get_session)) -> dict[str, Any]:
    return cancel_training_job(job_id, request, session)


__all__ = ["router"]
