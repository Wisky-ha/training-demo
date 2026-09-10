"""Contract tests for the standard MCP bridge wrapper."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from backend.app import mcp_bridge


def _run(awaitable):
    return asyncio.run(awaitable)


def test_bridge_stdio_transport_supports_standard_tool_discovery() -> None:
    async def discover():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "backend.app.mcp_bridge", "--transport", "stdio"],
            cwd=Path(__file__).resolve().parents[2],
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                initialized = await session.initialize()
                discovered = await session.list_tools()
                return initialized, discovered.tools

    initialized, tools = _run(discover())
    assert initialized.serverInfo.name == "model-training-platform"
    assert {tool.name for tool in tools} == {"predict", "mark_model_abnormal"}


def test_bridge_discovers_the_two_platform_tools() -> None:
    tools = _run(mcp_bridge.mcp.list_tools())
    by_name = {tool.name: tool for tool in tools}

    assert set(by_name) == {"predict", "mark_model_abnormal"}
    predict_schema = by_name["predict"].inputSchema
    assert set(predict_schema["required"]) == {"model_type", "data"}
    assert {"model_type", "data", "model_version"} <= set(
        predict_schema["properties"]
    )
    abnormal_schema = by_name["mark_model_abnormal"].inputSchema
    assert {"model_type", "model_version"} <= set(abnormal_schema["required"])
    assert {"abnormal", "reason"} <= set(abnormal_schema["properties"])
    assert by_name["mark_model_abnormal"].annotations.destructiveHint is True


def test_bridge_forwards_predict_call_to_platform(monkeypatch) -> None:
    calls = []

    async def fake_post(path, payload):
        calls.append((path, payload))
        return {"success": True, "predictions": [6]}

    monkeypatch.setattr(mcp_bridge, "_post_json", fake_post)
    result = _run(
        mcp_bridge.predict(
            model_type="electric_load",
            model_version="v5",
            data=[{"timestamp": "2026-01-01T00:00:00", "feature": 3}],
        )
    )

    assert result == {"success": True, "predictions": [6]}
    assert calls == [
        (
            "/api/mcp/predict",
            {
                "model_type": "electric_load",
                "model_version": "v5",
                "data": [{"timestamp": "2026-01-01T00:00:00", "feature": 3}],
            },
        )
    ]


def test_bridge_forwards_abnormal_call_to_platform(monkeypatch) -> None:
    calls = []

    async def fake_post(path, payload):
        calls.append((path, payload))
        return {"success": True, "rollback_triggered": True}

    monkeypatch.setattr(mcp_bridge, "_post_json", fake_post)
    result = _run(
        mcp_bridge.mark_model_abnormal(
            model_type="electric_load",
            model_version="v5",
            reason="MCP bridge E2E",
        )
    )

    assert result == {"success": True, "rollback_triggered": True}
    assert calls == [
        (
            "/api/mcp/mark_model_abnormal",
            {
                "model_type": "electric_load",
                "model_version": "v5",
                "abnormal": True,
                "reason": "MCP bridge E2E",
            },
        )
    ]
