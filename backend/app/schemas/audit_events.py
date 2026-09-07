"""HTTP contracts for read-only audit-event queries."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..domain.enums import ModelType
from ..domain.models import ResourceId


class AuditEventResponse(BaseModel):
    """One immutable server-side audit event."""

    model_config = ConfigDict(from_attributes=True)

    id: ResourceId
    occurred_at: datetime
    event_type: str
    object_type: str
    object_id: ResourceId
    model_type: ModelType | None
    model_version_id: ResourceId | None
    training_job_id: ResourceId | None
    operator_type: str | None
    operator_id: str | None
    operator_name: str | None
    result: str
    message: str | None
    request_id: str | None
    correlation_id: str | None
    metadata: dict[str, Any]


class AuditEventsResponse(BaseModel):
    """A page of audit events and the metadata needed to fetch the next page."""

    items: list[AuditEventResponse]
    page: int
    page_size: int
    total: int
    has_next: bool


__all__ = ["AuditEventResponse", "AuditEventsResponse"]
