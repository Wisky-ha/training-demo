"""Audit event query and append-only write helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session, sessionmaker

from ..db.models import AuditEventORM
from ..db.repositories import AuditEventRepository
from ..domain.enums import ModelType
from ..schemas.audit_events import AuditEventResponse, AuditEventsResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AuditContext:
    """Request/operator information carried into a domain audit event.

    IDs are copied only from request metadata.  A worker or an old caller has
    no request context, so those fields intentionally remain ``None``.
    """

    operator_type: str = "SYSTEM"
    operator_id: str | None = None
    operator_name: str | None = None
    request_id: str | None = None
    correlation_id: str | None = None


def context_from_request(request: Any) -> AuditContext:
    """Build context from explicit headers without inventing identifiers."""

    headers = getattr(request, "headers", {})
    return AuditContext(
        operator_type=headers.get("X-Operator-Type", "USER" if headers.get("X-Operator-Id") else "SYSTEM"),
        operator_id=headers.get("X-Operator-Id"),
        operator_name=headers.get("X-Operator-Name"),
        request_id=headers.get("X-Request-ID"),
        correlation_id=headers.get("X-Correlation-ID"),
    )


def _event_values(
    *,
    event_type: str,
    object_type: str,
    object_id: str | None,
    result: str,
    model_type: ModelType | str | None = None,
    model_version_id: str | None = None,
    training_job_id: str | None = None,
    message: str | None = None,
    metadata: dict[str, Any] | None = None,
    context: AuditContext | None = None,
) -> dict[str, Any]:
    active = context or AuditContext()
    return {
        "event_id": str(uuid4()),
        "occurred_at": datetime.now(timezone.utc),
        "event_type": event_type,
        "object_type": object_type,
        "object_id": str(object_id) if object_id is not None else None,
        "model_type": ModelType(model_type) if model_type is not None else None,
        "model_version_id": model_version_id,
        "training_job_id": training_job_id,
        "operator_type": active.operator_type or "SYSTEM",
        "operator_id": active.operator_id,
        "operator_name": active.operator_name,
        "result": result,
        "message": message,
        "request_id": active.request_id,
        "correlation_id": active.correlation_id,
        "event_metadata": dict(metadata or {}),
    }


def record_audit_event(
    session: Session,
    *,
    event_type: str,
    object_type: str,
    object_id: str | None,
    result: str = "SUCCEEDED",
    model_type: ModelType | str | None = None,
    model_version_id: str | None = None,
    training_job_id: str | None = None,
    message: str | None = None,
    metadata: dict[str, Any] | None = None,
    context: AuditContext | None = None,
    independent: bool = False,
) -> AuditEventORM:
    """Append one event, either to the caller transaction or a new one.

    ``independent`` is used only after a business rollback (or for an
    operation-start marker that must survive a later rollback).  The normal
    success path uses the caller session so the event commits with the state
    transition.
    """

    values = _event_values(
        event_type=event_type,
        object_type=object_type,
        object_id=object_id,
        result=result,
        model_type=model_type,
        model_version_id=model_version_id,
        training_job_id=training_job_id,
        message=message,
        metadata=metadata,
        context=context,
    )
    if not independent:
        event = AuditEventORM(**values)
        session.add(event)
        session.flush()
        return event

    factory = sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    with factory() as audit_session:
        event = AuditEventORM(**values)
        audit_session.add(event)
        audit_session.commit()
        return event


def record_failure_audit_event(session: Session, **kwargs: Any) -> None:
    """Best-effort independent failure audit that never masks the root error."""

    kwargs["result"] = "FAILED"
    kwargs["independent"] = True
    try:
        record_audit_event(session, **kwargs)
    except Exception:
        logger.exception(
            "could not persist failure audit event event_type=%s object_id=%s",
            kwargs.get("event_type"), kwargs.get("object_id"),
        )


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


__all__ = [
    "AuditContext",
    "AuditEventService",
    "context_from_request",
    "record_audit_event",
    "record_failure_audit_event",
]
