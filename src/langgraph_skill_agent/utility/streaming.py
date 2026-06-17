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

    # 从消息中提取工具名称
    for tc in getattr(message_chunk, "tool_calls", None) or []:
        """
        message_chunk 中的toolcall 有2种，一种是完整的生成的toolcall，一种是分段的toolcall_chunks

        完整的toolcall 形如：
                message_chunk = {
            "type": "AIMessageChunk",
            "content": "",
            "tool_calls": [
                {
                    "name": "search_web",
                    "args": {"query": "LangGraph docs"},
                    "id": "call_abc123",
                    "type": "tool_call",
                }
            ],
            "tool_call_chunks": [
                {
                    "name": None,
                    "args": ' "LangGraph docs"}',
                    "id": None,
                    "index": 0,
                    "type": "tool_call_chunk",
                }
            ],
            "invalid_tool_calls": [],
            "additional_kwargs": {},
            "response_metadata": {
                "model_name": "deepseek-chat",
                "finish_reason": "tool_calls",
            },
            "id": "run-xxx-chunk-3",
        }
        """
        n = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        if isinstance(n, str) and n.strip():
            names.append(n.strip())

    """
    分段的toolcall_chunks 形如：
    （1）可能有name
    message_chunk = {
    "type": "AIMessageChunk",
    "content": "",
    "tool_calls": [],
    "tool_call_chunks": [
        {
            "name": "search_web",
            "args": "",
            "id": "call_abc123",
            "index": 0,
            "type": "tool_call_chunk",
        }
    ],
    "invalid_tool_calls": [],
    "additional_kwargs": {},
    "response_metadata": {},
    "id": "run-xxx-chunk-1",
    }

    （2）可能没有name
        message_chunk = {
        "type": "AIMessageChunk",
        "content": "",
        "tool_calls": [],
        "tool_call_chunks": [
            {
                "name": None,
                "args": '{"query":',
                "id": None,
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
        "invalid_tool_calls": [],
        "additional_kwargs": {},
        "response_metadata": {},
        "id": "run-xxx-chunk-2",
    }
    """

    # 从分段的toolcall_chunks中提取工具名称
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

        """

        stream_mode=["messages", "tasks"]：同时订阅两类事件
        messages：模型输出的 token/chunk
        tasks：图中各节点（agent、tools 等）的开始/结束
        version="v2"：使用 LangGraph v2 流式 API 的 chunk 格式

        每个 chunk 大致形如：
        message_chunk = {
            "type": "AIMessageChunk",
            "content": "",
            "tool_calls": [],
            "tool_call_chunks": [
                {
                    "name": "search_web",
                    "args": "",
                    "id": "call_abc123",
                    "index": 0,
                    "type": "tool_call_chunk",
                }
            ],
            "invalid_tool_calls": [],
            "additional_kwargs": {},
            "response_metadata": {},
            "id": "run-xxx-chunk-1",
        }
        """

        # 处理任务事件
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

        # 其他类型的事件，直接跳过
        if kind != "messages":
            continue

        # 处理消息事件
        message_chunk, _meta = chunk["data"]

        names = tool_names_from_message_chunk(message_chunk)
        if names:
            pending_tool_names = names
        # 将消息内容拼接成字符串
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
    """
    tokens 是记录已经通过 on_token 发出去的内容

    """
    tokens: list[str] = []

    def _on_update(*, status: str | None, text: str, cursor: bool) -> None:
        if on_token is None:
            return

        """
        text 是模型输出的完整内容
        delta 是模型输出的增量内容，用切片 [已发长度:] 做 diff：只取还没发过的尾巴。
        """
        delta = text[len("".join(tokens)) :]
        if delta:
            tokens.append(delta)
            on_token(delta)

    """
    1.第一次 redraw()（还没内容）

    text = ""
    tokens = [] → delta = "" → 不调用 on_token

    2.收到 "你"
    text = "你"
    已发 ""，delta = "你"
    on_token("你")，tokens = ["你"]

    3.收到 "好"

    text = "你好"
    已发 "你"，长度 1，delta = "好"
    on_token("好")，tokens = ["你", "好"]
    """

    """
    执行顺序：

    stream_assistant_text 收到 chunk，调用 redraw()
    redraw() 调用 _on_update(...)
    _on_update 调用 on_token("你")
    on_token 写 stdout，返回
    _on_update 返回
    redraw() 返回
    stream_assistant_text 继续等下一个 chunk

    """
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
