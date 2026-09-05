"""HTTP endpoints for training creation, polling, retry, logs, and evaluation."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from ..db.session import get_session
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
        job = service.create(body)
        executor = getattr(request.app.state, "training_job_executor", None)
        if executor is None:
            raise TrainingJobError("后台任务执行器不可用", "TRAINING_EXECUTOR_UNAVAILABLE")
        TrainingJobService.submit(executor, request.app.state.session_factory, job.id, body.config, service.settings)
    except TrainingJobError as exc:
        raise _error(exc) from exc
    return _response(job)


@router.get("/{job_id}", response_model=TrainingJobResponse)
def get_training_job(job_id: str, request: Request, session: Session = Depends(get_session)) -> dict[str, Any]:
    job = _service(request, session).get(job_id)
    if job is None:
        raise _error(TrainingJobNotFoundError(f"训练任务不存在：{job_id}"))
    return _response(job)


@router.get("/{job_id}/logs", response_model=TrainingJobLogsResponse)
def get_training_job_logs(job_id: str, request: Request, session: Session = Depends(get_session)) -> dict[str, Any]:
    job = _service(request, session).get(job_id)
    if job is None:
        raise _error(TrainingJobNotFoundError(f"训练任务不存在：{job_id}"))
    return {"job_id": job.id, "items": list(job.logs or [])}


@router.post("/{job_id}/retry", response_model=TrainingJobResponse)
def retry_training_job(job_id: str, request: Request, session: Session = Depends(get_session)) -> dict[str, Any]:
    service = _service(request, session)
    try:
        job = service.retry(job_id)
        executor = getattr(request.app.state, "training_job_executor", None)
        if executor is None:
            raise TrainingJobError("后台任务执行器不可用", "TRAINING_EXECUTOR_UNAVAILABLE")
        TrainingJobService.submit(executor, request.app.state.session_factory, job.id, job.config, service.settings)
    except TrainingJobError as exc:
        raise _error(exc) from exc
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
        job = _service(request, session).cancel(job_id)
    except TrainingJobError as exc:
        raise _error(exc) from exc
    return _response(job)


@router.delete("/{job_id}", response_model=TrainingJobResponse, include_in_schema=False)
def delete_training_job(job_id: str, request: Request, session: Session = Depends(get_session)) -> dict[str, Any]:
    return cancel_training_job(job_id, request, session)


__all__ = ["router"]
