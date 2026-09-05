"""Shared HTTP error response helpers.

Every API adapter exposes the same small error envelope.  ``detail`` is kept as
an additive compatibility alias for clients from the earlier steps; new clients
should use ``error_code``, ``message`` and ``details``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import date, datetime
from enum import Enum
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def _safe(value: Any, *, key: str | None = None) -> Any:
    """Convert details to JSON without returning internal tracebacks."""

    if key and key.lower() in {"traceback", "stack_trace", "exception_trace"}:
        return None
    if isinstance(value, Mapping):
        return {
            str(item_key): converted
            for item_key, item_value in value.items()
            if (converted := _safe(item_value, key=str(item_key))) is not None
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        # Pydantic/SQLAlchemy values commonly provide a JSON-friendly dict.
        return _safe(value.model_dump(mode="json"))
    except AttributeError:
        return str(value)
    except Exception:
        return f"<{type(value).__name__}>"


def envelope(
    error_code: str,
    message: str,
    details: Mapping[str, Any] | None = None,
    *,
    status_code: int = status.HTTP_400_BAD_REQUEST,
    compatibility_detail: bool = True,
) -> JSONResponse:
    safe_details = _safe(dict(details or {}))
    content: dict[str, Any] = {
        "success": False,
        "error_code": error_code,
        "message": message,
        "details": safe_details,
    }
    if compatibility_detail:
        # Existing consumers read ``detail.code``.  Keep it while making the
        # top-level envelope the canonical contract.
        content["detail"] = {
            "code": error_code,
            "message": message,
            **(safe_details if isinstance(safe_details, dict) else {}),
        }
    return JSONResponse(status_code=status_code, content=content)


def _validation_code(exc: RequestValidationError) -> tuple[str, str]:
    for item in exc.errors():
        location = item.get("loc", ())
        if "model_type" in location:
            return "MODEL_TYPE_NOT_FOUND", "模型类型不存在"
        if "data" in location:
            return "INVALID_INPUT_DATA", "预测输入必须是记录数组"
    return "INVALID_REQUEST", "请求参数无效"


def http_exception_response(exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, Mapping):
        code = str(detail.get("code") or detail.get("error_code") or "HTTP_ERROR")
        message = str(detail.get("message") or "请求处理失败")
        details = {
            str(key): value
            for key, value in detail.items()
            if key not in {"code", "error_code", "message"}
        }
    else:
        code = "HTTP_ERROR"
        message = str(detail) if detail else "请求处理失败"
        details = {}
    # A server-side HTTPException may still carry a diagnostic message.  Keep
    # the stable domain code but use a safe generic message for unknown errors.
    if exc.status_code >= 500 and code in {"HTTP_ERROR", "INTERNAL_ERROR"}:
        code, message = "INTERNAL_ERROR", "服务内部错误"
    return envelope(code, message, details, status_code=exc.status_code)


async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    code, message = _validation_code(exc)
    logger.info("request validation failed path=%s errors=%s", request.url.path, exc.errors())
    return envelope(
        code,
        message,
        {"validation_errors": _safe(exc.errors())},
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger.info("http error path=%s status=%s detail=%s", request.url.path, exc.status_code, exc.detail)
    return http_exception_response(exc)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Diagnostics stay in server logs, never in the public response.
    logger.exception("unhandled API error path=%s", request.url.path, exc_info=exc)
    return envelope(
        "INTERNAL_ERROR",
        "服务内部错误",
        {},
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


__all__ = [
    "envelope",
    "http_exception_handler",
    "http_exception_response",
    "request_validation_exception_handler",
    "unhandled_exception_handler",
]
