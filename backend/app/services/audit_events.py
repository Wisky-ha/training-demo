"""Read-only application service for audit-event queries."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from ..db.models import AuditEventORM
from ..db.repositories import AuditEventRepository
from ..domain.enums import ModelType
from ..schemas.audit_events import AuditEventResponse, AuditEventsResponse


class AuditEventService:
    """Apply the audit query contract without owning a write transaction."""

    def __init__(self, session: Session) -> None:
        self.repository = AuditEventRepository(session)

    def list(
        self,
        *,
        page: int,
        page_size: int,
        from_: datetime | None = None,
        to: datetime | None = None,
        model_type: ModelType | None = None,
        event_type: str | None = None,
        object_type: str | None = None,
        result: str | None = None,
        query: str | None = None,
    ) -> AuditEventsResponse:
        records, total = self.repository.list_page(
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
        return AuditEventsResponse(
            items=[self.to_response(item) for item in records],
            page=page,
            page_size=page_size,
            total=total,
            has_next=page * page_size < total,
        )

    @staticmethod
    def to_response(event: AuditEventORM) -> AuditEventResponse:
        return AuditEventResponse.model_validate(event)


__all__ = ["AuditEventService"]
