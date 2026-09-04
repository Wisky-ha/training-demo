"""HTTP endpoints for CSV dataset upload and inspection."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from ..db.session import SessionLocal, get_session
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


def _validation_error(exc: CSVParseError) -> HTTPException:
    if isinstance(exc, UnsupportedDatasetFileError):
        return HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc))
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    request: Request,
    file: UploadFile = File(..., description="CSV 数据文件"),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Upload a CSV, inspect its schema, and persist its metadata."""

    try:
        content = await file.read()
        service = DatasetService(
            session,
            settings=getattr(request.app.state, "settings", None),
        )
        return service.upload(
            file.filename or "",
            content,
            content_type=file.content_type,
        )
    except CSVParseError as exc:
        raise _validation_error(exc) from exc
    except UnicodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"CSV 内容编码无效：{exc}",
        ) from exc


__all__ = ["get_request_session", "router", "upload_dataset"]
