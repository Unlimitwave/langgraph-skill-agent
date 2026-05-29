"""对话流式输出（CLI / Streamlit 共用）。"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage

from langgraph_skill_agent.utility.messages import stringify_message_content


def tool_names_from_message_chunk(message_chunk: object) -> list[str]:
    names: list[str] = []
    for tc in getattr(message_chunk, "tool_calls", None) or []:
        n = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        if isinstance(n, str) and n.strip():
            names.append(n.strip())
    for part in getattr(message_chunk, "tool_call_chunks", None) or []:
        n = part.get("name") if isinstance(part, dict) else getattr(part, "name", None)
        if isinstance(n, str) and n.strip() and n.strip() not in names:
            names.append(n.strip())
    return names


def format_status_line(
    *,
    pending_tool_names: list[str],
    tools_node_running: bool,
    agent_node_running: bool,
    has_text: bool,
) -> str | None:
    if pending_tool_names:
        joined = "`, `".join(pending_tool_names)
        return f"🔄 **正在调用工具** `{joined}` …（执行中）"
    if tools_node_running:
        return "🔄 **工具** 正在运行（子进程 / 网络可能较慢）…"
    if agent_node_running and not has_text:
        return "⏳ **模型** 推理中…"
    return None


async def stream_assistant_text(
    graph: Any,
    *,
    user_text: str,
    config: dict,
    on_update: Callable[..., None] | None = None,
) -> str:
    buf: list[str] = []
    pending_tool_names: list[str] = []
    tools_depth = 0
    agent_depth = 0

    def redraw(*, cursor: bool = True) -> None:
        body = "".join(buf)
        status = format_status_line(
            pending_tool_names=pending_tool_names,
            tools_node_running=tools_depth > 0,
            agent_node_running=agent_depth > 0,
            has_text=bool(body.strip()),
        )
        if on_update is not None:
            on_update(status=status, text=body, cursor=cursor)

    redraw()

    async for chunk in graph.astream(
        {"messages": [HumanMessage(content=user_text)]},
        config=config,
        stream_mode=["messages", "tasks"],
        version="v2",
    ):
        kind = chunk.get("type")
        if kind == "tasks":
            data = chunk.get("data")
            if not isinstance(data, dict):
                continue
            name = data.get("name")
            is_tools_node = name == "tools"
            is_model_node = name in ("agent", "model")
            if "result" in data:
                if is_tools_node:
                    tools_depth = max(0, tools_depth - 1)
                    if tools_depth == 0:
                        pending_tool_names = []
                elif is_model_node:
                    agent_depth = max(0, agent_depth - 1)
                redraw()
            elif "triggers" in data and "input" in data:
                if is_tools_node:
                    tools_depth += 1
                elif is_model_node:
                    agent_depth += 1
                redraw()
            continue

        if kind != "messages":
            continue

        message_chunk, _meta = chunk["data"]
        names = tool_names_from_message_chunk(message_chunk)
        if names:
            pending_tool_names = names
        piece = stringify_message_content(getattr(message_chunk, "content", None))
        if piece:
            buf.append(piece)
        redraw()

    text = "".join(buf)
    if on_update is not None:
        on_update(
            status=format_status_line(
                pending_tool_names=pending_tool_names,
                tools_node_running=tools_depth > 0,
                agent_node_running=agent_depth > 0,
                has_text=bool(text.strip()),
            ),
            text=text,
            cursor=False,
        )
    return text


def iter_assistant_text_sync(
    graph: Any,
    *,
    user_text: str,
    config: dict,
    on_token: Callable[[str], None] | None = None,
) -> str:
    tokens: list[str] = []

    def _on_update(*, status: str | None, text: str, cursor: bool) -> None:
        if on_token is None:
            return
        delta = text[len("".join(tokens)) :]
        if delta:
            tokens.append(delta)
            on_token(delta)

    return asyncio.run(
        stream_assistant_text(
            graph,
            user_text=user_text,
            config=config,
            on_update=_on_update,
        )
    )


def stream_assistant_reply(agent: Any, user_text: str, config: dict) -> None:
    sys.stdout.write("助手: ")
    sys.stdout.flush()
    iter_assistant_text_sync(
        agent,
        user_text=user_text,
        config=config,
        on_token=lambda t: (sys.stdout.write(t), sys.stdout.flush()),
    )
    sys.stdout.write("\n")
    sys.stdout.flush()
