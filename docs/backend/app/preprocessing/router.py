"""HTTP adapter for creating and inspecting preprocessing tasks."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from ..db.session import get_session
from ..schemas.preprocessing import (
    PreprocessingTaskCreate,
    PreprocessingTaskResponse,
    PreprocessingTransformRequest,
    PreprocessingTransformResponse,
)
from ..services.preprocessing import (
    PreprocessingError,
    PreprocessingNotFoundError,
    PreprocessingService,
)

router = APIRouter(prefix="/api/preprocessing-tasks", tags=["preprocessing"])


def _service(request: Request, session: Session) -> PreprocessingService:
    return PreprocessingService(session, settings=getattr(request.app.state, "settings", None))


def _error(exc: PreprocessingError, *, not_found: bool = False) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND if not_found else status.HTTP_400_BAD_REQUEST,
        detail={
            "code": exc.code,
            "message": str(exc),
            **({"task_id": exc.task_id} if exc.task_id else {}),
        },
    )


def _response(service: PreprocessingService, task: Any) -> dict[str, Any]:
    return service.to_response(task).model_dump(mode="json")


@router.post("", status_code=status.HTTP_201_CREATED, response_model=PreprocessingTaskResponse)
@router.post("/", status_code=status.HTTP_201_CREATED, response_model=PreprocessingTaskResponse, include_in_schema=False)
@router.post("/execute", status_code=status.HTTP_201_CREATED, response_model=PreprocessingTaskResponse, include_in_schema=False)
def create_preprocessing_task(
    request_body: PreprocessingTaskCreate,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Create and synchronously execute one preprocessing task.

    The synchronous implementation is intentional for the current demo.  The
    persisted stage record makes a later background executor possible without
    changing the client contract.
    """

    service = _service(request, session)
    try:
        task = service.create_and_execute(request_body)
    except PreprocessingNotFoundError as exc:
        raise _error(exc, not_found=True) from exc
    except PreprocessingError as exc:
        raise _error(exc) from exc
    return _response(service, task)


@router.post("/{task_id}/execute", response_model=PreprocessingTaskResponse)
def execute_preprocessing_task(
    task_id: str,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    service = _service(request, session)
    try:
        task = service.execute(task_id)
    except PreprocessingNotFoundError as exc:
        raise _error(exc, not_found=True) from exc
    except PreprocessingError as exc:
        raise _error(exc) from exc
    return _response(service, task)


@router.get("/{task_id}", response_model=PreprocessingTaskResponse)
def get_preprocessing_task(
    task_id: str,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    service = _service(request, session)
    task = service.get(task_id)
    if task is None:
        raise _error(PreprocessingNotFoundError(f"预处理任务不存在：{task_id}"), not_found=True)
    return _response(service, task)


@router.post("/{task_id}/transform", response_model=PreprocessingTransformResponse)
def transform_with_saved_preprocessor(
    task_id: str,
    body: PreprocessingTransformRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Use a fitted state for test/prediction data; never calls ``fit``."""

    service = _service(request, session)
    try:
        return service.transform_dataset(
            task_id, body.dataset_id, config=body.config
        ).model_dump(mode="json")
    except PreprocessingNotFoundError as exc:
        raise _error(exc, not_found=True) from exc
    except PreprocessingError as exc:
        raise _error(exc) from exc


__all__ = ["router"]
