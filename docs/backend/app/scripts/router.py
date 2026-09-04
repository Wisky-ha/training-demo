"""Small HTTP adapter for the step-4 global script library.

This router is kept here so the dataset step can run alongside the completed
foundation without coupling CSV code to script workflow logic.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError
from sqlalchemy.orm import Session

from ..db.session import get_session
from ..domain.enums import ModelType, ScriptStatus, ScriptType
from ..schemas.scripts import (
    PaginatedScriptsResponse,
    ScriptResponse,
    ScriptUploadMetadata,
)
from ..services.scripts import (
    DuplicateScriptVersionError,
    InvalidScriptFileError,
    InvalidScriptMetadataError,
    ScriptService,
    ScriptStorageError,
)

router = APIRouter(prefix="/api/scripts", tags=["scripts"])


def _response(item: ScriptORM) -> dict[str, Any]:
    created_at = item.created_at.isoformat() if item.created_at else None
    return {
        "id": item.id,
        "name": item.name,
        "script_type": item.script_type.value,
        "version": item.version,
        "source_code": item.source_code,
        "supported_model_types": [model.code.value for model in item.supported_model_types],
        "status": item.status.value,
        "created_at": created_at,
        "uploaded_at": created_at,
    }


def _metadata_or_422(
    name: str, script_type: str, version: str, supported_model_types: str
) -> ScriptUploadMetadata:
    try:
        models = json.loads(supported_model_types)
        if not isinstance(models, list):
            raise ValueError("supported_model_types must be a JSON array")
        return ScriptUploadMetadata(
            name=name,
            script_type=script_type,
            version=version,
            supported_model_types=models,
        )
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
        if isinstance(exc, ValidationError):
            detail = [
                {key: value for key, value in error.items() if key != "ctx"}
                for error in exc.errors()
            ]
        else:
            detail = str(exc)
        raise HTTPException(status_code=422, detail=detail) from exc


@router.post("/upload", status_code=201, response_model=ScriptResponse)
async def upload_script(
    request: Request,
    name: str = Form(...),
    version: str = Form(...),
    script_type: str = Form(...),
    supported_model_types: str = Form(...),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    metadata = _metadata_or_422(name, script_type, version, supported_model_types)
    source_bytes = await file.read()
    service = ScriptService(
        session,
        settings=getattr(request.app.state, "settings", None),
    )
    try:
        item = service.upload(
            metadata=metadata,
            filename=file.filename,
            source=source_bytes,
        )
    except InvalidScriptFileError as exc:
        raise HTTPException(status_code=400, detail={
            "code": "INVALID_SCRIPT_FILE", "message": str(exc)
        }) from exc
    except InvalidScriptMetadataError as exc:
        raise HTTPException(status_code=422, detail={
            "code": "INVALID_SCRIPT_METADATA", "message": str(exc)
        }) from exc
    except DuplicateScriptVersionError as exc:
        raise HTTPException(status_code=409, detail={
            "code": "SCRIPT_VERSION_EXISTS", "message": str(exc)
        }) from exc
    except ScriptStorageError as exc:
        raise HTTPException(status_code=500, detail={
            "code": "SCRIPT_STORAGE_ERROR", "message": str(exc)
        }) from exc
    return _response(item)


@router.get("", response_model=PaginatedScriptsResponse)
@router.get("/", response_model=PaginatedScriptsResponse)
def list_scripts(
    model_type: str | None = None,
    script_type: ScriptType | None = None,
    status: str = ScriptStatus.ENABLED.value,
    page: int = 1,
    page_size: int = 20,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """List enabled compatible selections by default.

    ``status`` defaults to ``ENABLED``.  An explicit ``status=disabled`` is
    reserved for global-library management and returns disabled entries;
    ``model_type`` filters every result to scripts declaring compatibility.
    Omitting ``model_type`` returns the enabled global library.
    """

    if page < 1 or page_size < 1 or page_size > 100:
        raise HTTPException(status_code=422, detail="page 和 page_size 参数无效")
    try:
        status_value = ScriptStatus(status.upper())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="status 参数无效") from exc
    try:
        model_type_value = ModelType(model_type) if model_type is not None else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="model_type 参数无效") from exc

    items, total = ScriptService(session).list(
        model_type=model_type_value,
        script_type=script_type,
        status=status_value,
        page=page,
        page_size=page_size,
    )
    return {
        "items": [item.model_dump(mode="json") for item in items],
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size if total else 0,
        },
    }


__all__ = ["router"]
