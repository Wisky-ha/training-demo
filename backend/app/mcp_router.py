"""HTTP transport for the protocol-neutral MCP model tools.

There is intentionally no third-party MCP server dependency in this project.
These routes provide a stable HTTP boundary that can be called by an MCP
bridge without coupling the backend to a particular MCP SDK.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy.orm import Session

from .db.session import get_session
from .schemas.mcp import MCPModelAbnormalRequest, PredictRequest
from .services.mcp import MCPModelService, MCPServiceError


router = APIRouter(tags=["mcp"])


def _safe(value: Any) -> Any:
    return MCPModelService._json_safe(value)


def _error(code: str, message: str, *, details: dict[str, Any] | None = None,
           status_code: int = status.HTTP_400_BAD_REQUEST) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error_code": code,
            "message": message,
            "details": _safe(details or {}),
        },
    )


def _service_error(exc: MCPServiceError) -> JSONResponse:
    not_found = {"MODEL_TYPE_NOT_FOUND", "MODEL_VERSION_NOT_FOUND"}
    conflict = {"MODEL_VERSION_UNAVAILABLE", "NO_HEALTHY_BACKUP", "NO_HEALTHY_ROLLBACK_VERSION"}
    server = {"MODEL_LOAD_FAILED", "PREDICTION_FAILED", "PREPROCESS_FAILED"}
    if exc.code in not_found:
        response_status = status.HTTP_404_NOT_FOUND
    elif exc.code in conflict:
        response_status = status.HTTP_409_CONFLICT
    elif exc.code in server:
        response_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    else:
        response_status = status.HTTP_400_BAD_REQUEST
    return _error(exc.code, str(exc), details=exc.details, status_code=response_status)


async def _body(request: Request) -> tuple[Any | None, JSONResponse | None]:
    try:
        return await request.json(), None
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        return None, _error("INVALID_REQUEST", "请求体必须是有效的 JSON", details={"exception_type": type(exc).__name__})


def _validation_error(exc: ValidationError, *, abnormal: bool) -> JSONResponse:
    errors = _safe(exc.errors())
    code = "INVALID_REQUEST"
    message = "请求参数无效"
    for item in exc.errors():
        location = item.get("loc", ())
        if "model_type" in location:
            code = "MODEL_TYPE_NOT_FOUND"
            message = "模型类型不存在"
            break
        if not abnormal and "data" in location:
            code = "INVALID_INPUT_DATA"
            message = "预测输入必须是记录数组"
            break
    return _error(code, message, details={"validation_errors": errors})


async def _predict(request: Request, session: Session) -> JSONResponse:
    payload, failure = await _body(request)
    if failure is not None:
        return failure
    try:
        body = PredictRequest.model_validate(payload)
    except ValidationError as exc:
        return _validation_error(exc, abnormal=False)
    try:
        result = MCPModelService(
            session, settings=getattr(request.app.state, "settings", None)
        ).predict(body.model_type, body.data, body.model_version)
        return JSONResponse(status_code=status.HTTP_200_OK, content=_safe(result))
    except MCPServiceError as exc:
        return _service_error(exc)
    except Exception:
        # Do not expose a traceback or implementation detail through a tool
        # boundary.  The application logger can record the underlying error.
        return _error("INTERNAL_ERROR", "服务内部错误", status_code=500)


async def _mark_abnormal(request: Request, session: Session) -> JSONResponse:
    payload, failure = await _body(request)
    if failure is not None:
        return failure
    try:
        body = MCPModelAbnormalRequest.model_validate(payload)
    except ValidationError as exc:
        return _validation_error(exc, abnormal=True)
    try:
        result = MCPModelService(
            session, settings=getattr(request.app.state, "settings", None)
        ).mark_model_abnormal(
            body.model_type,
            body.model_version,
            abnormal=body.abnormal,
            reason=body.reason,
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=_safe(result))
    except MCPServiceError as exc:
        return _service_error(exc)
    except Exception:
        return _error("INTERNAL_ERROR", "服务内部错误", status_code=500)


# The /api/mcp paths are the documented HTTP transport.  The shorter /mcp
# aliases make the same transport easy to mount behind an MCP gateway.
@router.post("/api/mcp/predict")
@router.post("/mcp/predict", include_in_schema=False)
async def mcp_predict(request: Request, session: Session = Depends(get_session)):
    return await _predict(request, session)


@router.post("/api/mcp/mark_model_abnormal")
@router.post("/mcp/mark_model_abnormal", include_in_schema=False)
async def mcp_mark_model_abnormal(request: Request, session: Session = Depends(get_session)):
    return await _mark_abnormal(request, session)


__all__ = ["router"]
