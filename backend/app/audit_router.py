"""HTTP adapter for the read-only audit-event query API."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from .db.session import get_session
from .domain.enums import ModelType
from .schemas.audit_events import AuditEventsResponse
from .services.audit_events import AuditEventService

router = APIRouter(prefix="/api/audit-events", tags=["audit"])


@router.get(
    "",
    response_model=AuditEventsResponse,
    summary="查询审计事件",
    description=(
        "按时间、模型类型、事件类型、对象类型、结果和关键词查询追加式审计事件。"
        "事件仅支持读取，按 occurred_at DESC、id DESC 稳定排序。"
    ),
)
def list_audit_events(
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(50, ge=1, le=100, description="每页条数，最大 100"),
    from_: datetime | None = Query(
        None, alias="from", description="起始时间（含），ISO 8601"
    ),
    to: datetime | None = Query(
        None, description="结束时间（含），ISO 8601"
    ),
    model_type: ModelType | None = Query(None, description="模型类型"),
    event_type: str | None = Query(None, description="事件类型，精确匹配"),
    object_type: str | None = Query(None, description="对象类型，精确匹配"),
    result: str | None = Query(None, description="事件结果，精确匹配"),
    query: str | None = Query(None, description="关键词，匹配事件和对象相关文本"),
    session: Session = Depends(get_session),
) -> AuditEventsResponse:
    return AuditEventService(session).list(
        page=page,
        page_size=page_size,
        from_=from_,
        to=to,
        model_type=model_type,
        event_type=event_type,
        object_type=object_type,
        result=result,
        query=query,
    )


__all__ = ["router"]
