"""
FastMCP：内置 MCP 服务 + Client，导出为 LangChain StructuredTool（同步 + 异步均可调用）。

环境变量：MCP_TOOLS=0 关闭；MCP_CLIENT_URL= 连远端；MCP_TOOL_NAME_PREFIX= 工具名前缀（默认 mcp_）。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import re
from collections.abc import Coroutine
from typing import Annotated, Any

from fastmcp import Client, FastMCP
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, create_model

logger = logging.getLogger(__name__)


def _trace(msg: str) -> None:
    if os.environ.get("RAG_TRACE", "").strip() in {"1", "true", "yes", "on"}:
        logger.info("[MCP] %s", msg)


def build_embedded_fastmcp_server() -> FastMCP:
    mcp = FastMCP("EmbeddedAgentTools")

    @mcp.tool()
    def ping() -> str:
        """健康检查：确认 MCP 工具链可用。"""
        return "pong"

    @mcp.tool()
    def echo(message: str) -> str:
        """原样返回 message，用于调试或占位。"""
        return message

    @mcp.tool()
    def json_pretty(text: str) -> str:
        """若 text 为合法 JSON，则格式化缩进后返回；否则返回原文本。"""
        try:
            return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, TypeError):
            return text

    return mcp


def _lc_tool_name(raw: str, prefix: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", f"{prefix}{raw}")
    if s and s[0].isdigit():
        s = "t_" + s
    return s or f"{prefix}tool"


def _json_type(spec: dict[str, Any]) -> Any:
    t = (spec or {}).get("type")
    if t == "string":
        return str
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "boolean":
        return bool
    if t == "array":
        return list[Any]
    if t == "object":
        return dict[str, Any]
    return Any


def _args_model_for_tool(tool_name: str, input_schema: dict[str, Any] | None) -> type[BaseModel]:
    schema = input_schema or {"type": "object", "properties": {}}
    props: dict[str, Any] = schema.get("properties") or {}
    required: set[str] = set(schema.get("required") or [])
    safe = re.sub(r"[^A-Za-z0-9_]", "_", tool_name)[:60] or "Tool"
    if safe[0].isdigit():
        safe = "T_" + safe

    if not props:
        return create_model(f"MCPArgs_{safe}", __config__=ConfigDict(extra="forbid"))

    fields: dict[str, Any] = {}
    for key, spec in props.items():
        py = _json_type(spec if isinstance(spec, dict) else {})
        desc = (spec or {}).get("description") if isinstance(spec, dict) else None
        if key in required:
            fields[key] = (Annotated[py, Field(description=desc)] if desc else py, ...)
        else:
            fields[key] = (
                (py | None, Field(default=None, description=desc))
                if desc
                else (py | None, Field(default=None))
            )
    return create_model(f"MCPArgs_{safe}", **fields)


def _tool_result_text(result: Any) -> str:
    if getattr(result, "is_error", False):
        parts = [
            getattr(b, "text", "")
            for b in (getattr(result, "content", None) or [])
            if getattr(b, "text", None)
        ]
        return "MCP tool error: " + ("\n".join(parts) if parts else "unknown error")
    data = getattr(result, "data", None)
    if data is not None:
        return (
            json.dumps(data, ensure_ascii=False, indent=2)
            if isinstance(data, (dict, list))
            else str(data)
        )
    lines = [
        getattr(b, "text", "")
        for b in (getattr(result, "content", None) or [])
        if getattr(b, "text", None)
    ]
    return "\n".join(lines) if lines else "(empty MCP result)"


def _run_async_in_new_loop(coro: Coroutine[Any, Any, str]) -> str:
    """在无 loop 的线程里 asyncio.run；若当前线程已有 loop（少见），换线程跑。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _model_to_args(model: BaseModel) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in model.model_dump().items():
        if v is None and k not in model.model_fields_set:
            continue
        out[k] = v
    return out


def _mcp_structured_tool(
    *,
    mcp_name: str,
    lc_name: str,
    description: str,
    input_schema: dict[str, Any] | None,
    client: Client[Any],
) -> StructuredTool:
    ArgsModel = _args_model_for_tool(mcp_name, input_schema)

    async def _arun(**kwargs: Any) -> str:
        args = _model_to_args(ArgsModel(**kwargs))
        async with client:
            res = await client.call_tool(mcp_name, args)
        return _tool_result_text(res)

    def _run(**kwargs: Any) -> str:
        return _run_async_in_new_loop(_arun(**kwargs))

    return StructuredTool.from_function(
        name=lc_name,
        description=description or f"MCP tool `{mcp_name}`.",
        args_schema=ArgsModel,
        func=_run,
        coroutine=_arun,
    )


async def _discover_tools(client: Client[Any], prefix: str) -> list[StructuredTool]:
    async with client:
        specs = await client.list_tools()
    out: list[StructuredTool] = []
    seen: set[str] = set()
    for t in specs:
        lc = _lc_tool_name(t.name, prefix)
        if lc in seen:
            lc = _lc_tool_name(f"{t.name}_{len(seen)}", prefix)
        seen.add(lc)
        raw = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)
        out.append(
            _mcp_structured_tool(
                mcp_name=t.name,
                lc_name=lc,
                description=(t.description or "").strip(),
                input_schema=raw if isinstance(raw, dict) else None,
                client=client,
            )
        )
    _trace(f"loaded {len(out)} MCP tools")
    return out


def load_mcp_extra_tools() -> list[StructuredTool]:
    if os.environ.get("MCP_TOOLS", "1").strip().lower() in {"0", "false", "no", "off"}:
        _trace("MCP_TOOLS disabled")
        return []

    prefix = os.environ.get("MCP_TOOL_NAME_PREFIX", "mcp_").strip() or "mcp_"
    url = os.environ.get("MCP_CLIENT_URL", "").strip()
    client: Client[Any] = Client(url) if url else Client(build_embedded_fastmcp_server())
    _trace(f"using remote MCP url={url!r}" if url else "using embedded FastMCP")

    try:
        return asyncio.run(_discover_tools(client, prefix))
    except Exception as e:
        _trace(f"load failed: {e!r}")
        raise
