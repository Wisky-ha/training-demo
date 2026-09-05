"""HTTP endpoints for CSV dataset upload and inspection."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from ..db.session import SessionLocal, get_session
from ..schemas.dataset_split import DatasetSplitRequest, DatasetSplitResponse
from ..services.dataset_split import (
    DatasetSplitError,
    DatasetSplitNotFoundError,
    DatasetSplitService,
)
from .service import CSVParseError, DatasetService, UnsupportedDatasetFileError

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def get_request_session(request: Request) -> Generator[Session, None, None]:
    """Use the engine owned by the application instance (including test apps)."""

    factory = getattr(request.app.state, "session_factory", SessionLocal)
    session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _validation_error(exc: CSVParseError) -> JSONResponse:
    """Keep the legacy string ``detail`` and add machine-readable issues."""

    if isinstance(exc, UnsupportedDatasetFileError):
        response_status = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        error_code = "UNSUPPORTED_DATASET_FILE"
    else:
        response_status = status.HTTP_400_BAD_REQUEST
        error_code = "DATASET_VALIDATION_FAILED"
    issues = [issue.as_dict() for issue in exc.issues]
    return JSONResponse(
        status_code=response_status,
        content={
            "success": False,
            "error_code": error_code,
            "message": str(exc),
            "details": {"issues": issues},
            # Keep the original upload response fields for existing clients.
            "detail": str(exc),
            "error": {"code": error_code, "message": str(exc), "issues": issues},
            "errors": issues,
        },
    )


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    request: Request,
    file: UploadFile = File(..., description="CSV 数据文件"),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Upload a CSV, inspect its schema, and persist its metadata."""

    settings = getattr(request.app.state, "settings", None)
    max_size = getattr(settings, "max_dataset_size_bytes", 50 * 1024 * 1024)
    try:
        # Reading one byte beyond the configured limit bounds memory used by a
        # rejected request while still allowing the service to validate direct
        # callers that bypass HTTP.
        content = await file.read(max_size + 1)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无法读取上传的 CSV 文件",
        ) from exc

    try:
        service = DatasetService(session, settings=settings)
        return service.upload(
            file.filename or "",
            content,
            content_type=file.content_type,
        )
    except CSVParseError as exc:
        return _validation_error(exc)
    except UnicodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"CSV 内容编码无效：{exc}",
        ) from exc


def _split_error(exc: DatasetSplitError) -> HTTPException:
    if exc.code == "DATASET_SPLIT_ALREADY_EXISTS":
        response_status = status.HTTP_409_CONFLICT
    elif exc.code == "DATASET_NOT_FOUND" or isinstance(exc, DatasetSplitNotFoundError):
        response_status = status.HTTP_404_NOT_FOUND
    else:
        response_status = status.HTTP_400_BAD_REQUEST
    return HTTPException(
        status_code=response_status,
        detail={"code": exc.code, "message": str(exc), **exc.details},
    )


@router.post(
    "/{dataset_id}/split",
    status_code=status.HTTP_201_CREATED,
    response_model=DatasetSplitResponse,
)
def split_dataset(
    dataset_id: str,
    request: Request,
    body: DatasetSplitRequest | None = None,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Persist the fixed ascending-time 80/20 split metadata.

    The request accepts only an optional completed preprocessing task ID.  The
    split strategy and ratios are server constants and cannot be overridden.
    """

    service = DatasetSplitService(
        session, settings=getattr(request.app.state, "settings", None)
    )
    try:
        split = service.split(
            dataset_id,
            preprocessing_task_id=body.preprocessing_task_id if body else None,
        )
    except DatasetSplitError as exc:
        raise _split_error(exc) from exc
    return service.to_response(split).model_dump(mode="json")


@router.get("/{dataset_id}/split", response_model=DatasetSplitResponse)
def get_dataset_split(
    dataset_id: str,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Return the persisted split without rereading or changing source rows."""

    service = DatasetSplitService(
        session, settings=getattr(request.app.state, "settings", None)
    )
    try:
        split = service.get(dataset_id)
    except DatasetSplitError as exc:
        raise _split_error(exc) from exc
    if split is None:
        raise _split_error(
            DatasetSplitNotFoundError("该数据集尚未完成划分", dataset_id=dataset_id)
        )
    return service.to_response(split).model_dump(mode="json")


__all__ = ["get_request_session", "get_dataset_split", "router", "split_dataset", "upload_dataset"]
