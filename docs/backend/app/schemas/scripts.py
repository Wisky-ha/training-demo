"""Pydantic contracts for the global script-library API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..domain.enums import ModelType, ScriptStatus, ScriptType


def _safe_text(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be blank")
    if value in {".", ".."}:
        raise ValueError(f"{field_name} must be a meaningful value")
    if any(character in value for character in ("\x00", "/", "\\")):
        raise ValueError(f"{field_name} contains an unsafe path character")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{field_name} contains a control character")
    return value


class ScriptUploadMetadata(BaseModel):
    """Validated multipart metadata for one immutable script version."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    script_type: ScriptType
    version: str = Field(min_length=1, max_length=100)
    supported_model_types: list[ModelType] = Field(min_length=1, max_length=20)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _safe_text(value, "name")

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _safe_text(value, "version")

    @field_validator("supported_model_types")
    @classmethod
    def validate_model_types(cls, value: list[ModelType]) -> list[ModelType]:
        if len(set(value)) != len(value):
            raise ValueError("supported_model_types must not contain duplicates")
        return value


class ScriptResponse(BaseModel):
    """Script data returned to selection and library-management screens."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    script_type: ScriptType
    version: str
    source_code: str
    supported_model_types: list[ModelType]
    status: ScriptStatus
    created_at: datetime
    uploaded_at: datetime


class PaginationMeta(BaseModel):
    page: int
    page_size: int
    total: int
    total_pages: int


class PaginatedScriptsResponse(BaseModel):
    items: list[ScriptResponse]
    pagination: PaginationMeta
