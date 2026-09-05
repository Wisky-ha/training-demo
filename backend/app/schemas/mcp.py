"""Request contracts for the HTTP adapter of the MCP model tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..domain.enums import ModelType


class PredictRequest(BaseModel):
    """JSON rows and optional version supplied to the ``predict`` tool."""

    model_config = ConfigDict(extra="forbid")

    model_type: ModelType
    model_version: str | None = Field(default=None, min_length=1, max_length=100)
    data: list[dict[str, Any]]


class MCPModelAbnormalRequest(BaseModel):
    """Payload for the type/version scoped anomaly tool."""

    model_config = ConfigDict(extra="forbid")

    model_type: ModelType
    model_version: str = Field(min_length=1, max_length=100)
    abnormal: bool = True
    reason: str = Field(default="健康检查异常", min_length=1, max_length=2000)


__all__ = ["MCPModelAbnormalRequest", "PredictRequest"]
