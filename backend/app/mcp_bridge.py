"""Minimal MCP bridge for the model-training platform.

The bridge exposes standard MCP tools and forwards their calls to the existing
FastAPI HTTP transport. It deliberately keeps model business logic in the
backend service instead of duplicating it here.

Run from the repository root:

    python -m backend.app.mcp_bridge --transport stdio

The upstream API defaults to http://127.0.0.1:8000 and can be changed with
MODEL_PLATFORM_BASE_URL.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Literal

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

ModelTypeCode = Literal[
    "electric_load",
    "heating_cooling_load",
    "integrated_energy",
]

_UPSTREAM_BASE_URL = os.getenv(
    "MODEL_PLATFORM_BASE_URL", "http://127.0.0.1:8000"
).rstrip("/")
_UPSTREAM_TIMEOUT_SECONDS = float(os.getenv("MODEL_PLATFORM_TIMEOUT_SECONDS", "60"))


class UpstreamToolError(RuntimeError):
    """An upstream HTTP failure that should become an MCP tool error."""


def _format_upstream_error(status_code: int, payload: Any) -> str:
    if isinstance(payload, dict):
        code = payload.get("error_code") or payload.get("code") or "UPSTREAM_ERROR"
        message = payload.get("message") or payload.get("detail") or "模型平台请求失败"
        details = payload.get("details")
        suffix = (
            f" details={json.dumps(details, ensure_ascii=False)}" if details else ""
        )
        return f"[{code}] {message}{suffix} (HTTP {status_code})"
    return f"模型平台请求失败 (HTTP {status_code}): {payload}"


async def _post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Forward one tool call to the platform's documented HTTP transport."""

    try:
        async with httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT_SECONDS) as client:
            response = await client.post(f"{_UPSTREAM_BASE_URL}{path}", json=payload)
    except httpx.HTTPError as exc:
        raise UpstreamToolError(
            f"无法连接模型训练平台 {_UPSTREAM_BASE_URL}: {exc}"
        ) from exc

    try:
        body: Any = response.json()
    except ValueError:
        body = response.text
    if response.is_error:
        raise UpstreamToolError(_format_upstream_error(response.status_code, body))
    if not isinstance(body, dict):
        raise UpstreamToolError("模型训练平台返回的工具结果不是 JSON 对象")
    return body


mcp = FastMCP(
    "model-training-platform",
    host=os.getenv("MCP_BRIDGE_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_BRIDGE_PORT", "8001")),
    instructions=(
        "模型训练平台工具。predict 是只读预测工具；"
        "mark_model_abnormal 会修改模型健康状态并可能触发告警和回滚，"
        "只能在用户明确要求时调用。"
    ),
)


@mcp.tool(
    name="predict",
    description=(
        "使用模型训练平台当前健康生产模型进行预测。"
        "model_version 省略时使用当前生产版本；data 必须符合已发布模型的输入 schema。"
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    structured_output=True,
)
async def predict(
    model_type: ModelTypeCode,
    data: list[dict[str, Any]],
    model_version: str | None = None,
) -> dict[str, Any]:
    """Call the platform prediction HTTP endpoint."""

    payload: dict[str, Any] = {
        "model_type": model_type,
        "data": data,
    }
    if model_version is not None:
        payload["model_version"] = model_version
    return await _post_json("/api/mcp/predict", payload)


@mcp.tool(
    name="mark_model_abnormal",
    description=(
        "标记指定模型版本异常，并由平台创建告警、寻找健康备份并执行必要回滚。"
        "这是有副作用的操作，只能在用户明确要求后调用。"
    ),
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=False,
    ),
    structured_output=True,
)
async def mark_model_abnormal(
    model_type: ModelTypeCode,
    model_version: str,
    abnormal: bool = True,
    reason: str = "健康检查异常",
) -> dict[str, Any]:
    """Call the platform anomaly-marking HTTP endpoint."""

    return await _post_json(
        "/api/mcp/mark_model_abnormal",
        {
            "model_type": model_type,
            "model_version": model_version,
            "abnormal": abnormal,
            "reason": reason,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
        help="MCP transport; stdio is the default for desktop clients",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
