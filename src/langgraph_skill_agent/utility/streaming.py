"""对话流式输出（CLI / Streamlit 共用）。

阅读实现前建议先看同目录说明文档：streaming.md
（messages / tasks 两层 type、tool call vs ToolMessage、chunk 示例与时间线）
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Callable
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from langgraph_skill_agent.utility.hitl import (
    AssistantTurnResult,
    HitlRequest,
    get_pending_hitl,
    prompt_hitl_decisions_cli,
)
from langgraph_skill_agent.utility.logging_config import env_truthy
from langgraph_skill_agent.utility.messages import stringify_message_content

logger = logging.getLogger(__name__)

_TOOL_TRACE_RESULT_MAX = 800


class ToolResult(TypedDict):
    name: str
    content: str


def _is_tool_message_chunk(message_chunk: object) -> bool:
    return getattr(message_chunk, "type", None) == "tool"


def _tool_trace(msg: str, *args: object) -> None:
    if env_truthy("AGENT_TOOL_TRACE"):
        logger.info("[TOOL_TRACE] " + msg, *args)


def _log_tool_calls(message_chunk: object, seen: set[str]) -> None:
    for tc in getattr(message_chunk, "tool_calls", None) or []:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
        if not isinstance(name, str) or not name.strip() or not isinstance(args, dict):
            continue
        key = json.dumps({"name": name.strip(), "args": args}, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        _tool_trace("call %s args=%s", name.strip(), args)


def _log_tool_result(message_chunk: object) -> None:
    if getattr(message_chunk, "type", None) != "tool":
        return
    name = getattr(message_chunk, "name", None) or "?"
    content = stringify_message_content(getattr(message_chunk, "content", None))
    if len(content) > _TOOL_TRACE_RESULT_MAX:
        content = content[:_TOOL_TRACE_RESULT_MAX] + "...(truncated)"
    _tool_trace("result %s: %s", name, content)


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
    user_text: str = "",
    graph_input: Any | None = None,
    config: dict,
    on_update: Callable[..., None] | None = None,
    text_prefix: str = "",
    tool_results_prefix: list[ToolResult] | None = None,
) -> tuple[str, list[ToolResult]]:
    """
    消费 LangGraph 流式输出，累积助手正文与工具结果，并通过 on_update 推送 UI 快照。

    Streamlit：on_update 里整页重绘气泡（status + 全文 + tool_results + 光标）。
    CLI：由 iter_assistant_text_sync 包装 on_update，只取 text 增量写 stdout。
    """
    buf: list[str] = [text_prefix] if text_prefix else []
    tool_results: list[ToolResult] = list(tool_results_prefix or [])
    pending_tool_names: list[
        str
    ] = []  # 模型刚解析出的待执行工具名（来自 tool_calls / tool_call_chunks）
    tools_depth = 0  # tools 节点嵌套深度；>0 表示工具子图正在执行
    agent_depth = 0  # agent/model 节点嵌套深度；>0 表示模型正在推理
    seen_tool_calls: set[str] = set()  # 去重，仅用于 AGENT_TOOL_TRACE 日志

    def redraw(*, cursor: bool = True) -> None:
        """
        把当前内部状态合成一帧「快照」，推给 on_update。

        快照字段：
          - text: buf 拼接的助手全文（非增量）
          - status: 由 format_status_line 根据工具名 / 节点深度推导的状态行
          - tool_results: 已完成的工具返回列表副本
          - cursor: 流式进行中为 True（UI 显示 ▌），结束时由外层单独传 False

        调用时机（凡状态可能变化处都应 redraw）：
          ① 流开始前初始化（下方 redraw()）
          ② tasks 事件：agent/tools 节点开始或结束
          ③ messages 事件：工具名出现、正文 token 到达、工具结果到达
          ④ messages 事件：content 为空但 pending_tool_names 等已更新（仅刷新 status）
        循环结束后不再走 redraw，而是直接 on_update(..., cursor=False) 收尾。
        """
        body = "".join(buf)
        status = format_status_line(
            pending_tool_names=pending_tool_names,
            tools_node_running=tools_depth > 0,
            agent_node_running=agent_depth > 0,
            has_text=bool(body.strip()),
        )
        if on_update is not None:
            on_update(
                status=status,
                text=body,
                tool_results=list(tool_results),
                cursor=cursor,
            )

    # ① 初始化：在首个 chunk 到达前通知 UI（空正文，cursor=True）
    redraw()

    payload = (
        graph_input if graph_input is not None else {"messages": [HumanMessage(content=user_text)]}
    )
    async for chunk in graph.astream(
        payload,
        config=config,
        stream_mode=["messages", "tasks"],
        version="v2",
    ):
        kind = chunk.get("type")

        """
        stream_mode=["messages", "tasks"]：同时订阅两类事件
        messages：模型输出的 token/chunk、tool_call 片段、ToolMessage 工具结果
        tasks：图中各节点（agent、tools 等）的开始/结束（需 checkpointer，本项目用 MemorySaver）
        version="v2"：统一 StreamPart 外壳，不论哪种 mode 都是 {type, ns, data}

        完整时间线、两层 type 说明、更多 chunk 示例见同目录 streaming.md

        ── 公共外壳（每个 chunk 都有）──
        chunk = {
            "type": "messages" | "tasks" | ...,
            "ns": (),                    # 子图时非空，如 ("agent:abc123",)
            "data": <因 type 而异>,
        }

        ── type == "messages"：data 是 (message_chunk, metadata) 二元组 ──
        # 代码里：message_chunk, _meta = chunk["data"]

        # (1) 助手正文 token 流式片段
        {
            "type": "messages",
            "ns": (),
            "data": (
                AIMessageChunk(content="你", tool_calls=[], tool_call_chunks=[], ...),
                {"langgraph_node": "agent", "langgraph_step": 2, ...},
            ),
        }

        # (2) 仅有 tool_call、尚无正文（content 为空）→ 用于 pending_tool_names / status
        # chunk["data"][0]（message_chunk）形如：
        {
            "type": "messages",
            "ns": (),
            "data": (
                {   # message_chunk = AIMessageChunk(...)
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
                },
                {"langgraph_node": "agent", ...},
            ),
        }

        # (3) 工具执行完毕返回（type=="tool" 的 ToolMessage，不进 buf，进 tool_results）
        {
            "type": "messages",
            "ns": (),
            "data": (
                ToolMessage(type="tool", name="search_web", content="检索结果...", tool_call_id="call_abc123"),
                {"langgraph_node": "tools", ...},
            ),
        }

        ── type == "tasks"：data 是 dict，开始与结束互斥字段 ──

        # (4) 节点开始（TaskPayload）：有 input + triggers，无 result
        {
            "type": "tasks",
            "ns": (),
            "data": {
                "id": "6f3a2b1c-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                "name": "agent",              # 或 "tools"、"model"（与图中节点名一致）
                "input": {"messages": [...]}, # 传入该节点的状态
                "triggers": ["messages"],     # 触发该节点的 channel 写入
            },
        }

        # (5) 节点结束（TaskResultPayload）：有 result，无 input / triggers
        {
            "type": "tasks",
            "ns": (),
            "data": {
                "id": "6f3a2b1c-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                "name": "tools",
                "result": {"messages": [ToolMessage(...)]},
                "error": None,
                "interrupts": [],
            },
        }

        下方 if kind == "tasks" 用 "result" in data 判断结束，用 triggers+input 判断开始。
        """

        # 处理任务事件（kind == "tasks"：图中节点生命周期，与 token 内容无关）
        if kind == "tasks":
            data = chunk.get("data")
            if not isinstance(data, dict):
                continue
            name = data.get("name")
            is_tools_node = name == "tools"
            is_model_node = name in ("agent", "model")
            # ②a 节点执行结束：depth--，tools 全部结束时清空 pending_tool_names
            if "result" in data:
                task_interrupts = data.get("interrupts")
                if isinstance(task_interrupts, list) and task_interrupts:
                    _tool_trace("interrupt in tasks: %s", task_interrupts)
                if is_tools_node:
                    tools_depth = max(0, tools_depth - 1)
                    if tools_depth == 0:
                        _tool_trace("tools node finished")
                        pending_tool_names = []
                elif is_model_node:
                    agent_depth = max(0, agent_depth - 1)
                redraw()  # 状态行可能从「工具运行中」变为「推理中」或消失
            # ②b 节点开始执行：depth++，随后 tools 节点会真正跑工具
            elif "triggers" in data and "input" in data:
                if is_tools_node:
                    tools_depth += 1
                    if pending_tool_names:
                        _tool_trace("executing %s", ", ".join(pending_tool_names))
                    else:
                        _tool_trace("tools node started")
                elif is_model_node:
                    agent_depth += 1
                redraw()  # 状态行可能变为「⏳ 模型推理中」或「🔄 工具正在运行」
            continue

        # 其他类型的事件，直接跳过
        if kind != "messages":
            continue

        # 处理消息事件（kind == "messages"：模型 token、tool_call 片段、ToolMessage 结果）
        message_chunk, _meta = chunk["data"]

        _log_tool_calls(message_chunk, seen_tool_calls)
        _log_tool_result(message_chunk)

        # 从 AIMessageChunk 里尽早解析工具名，用于 status「正在调用工具 xxx」
        names = tool_names_from_message_chunk(message_chunk)
        if names:
            pending_tool_names = names

        piece = stringify_message_content(getattr(message_chunk, "content", None))
        if not piece:
            # ③a 尚无正文 token，但 pending_tool_names / depth 可能已变 → 只刷新 status
            redraw()
            continue

        # ③b 有 content：ToolMessage 进 tool_results，AIMessage 进 buf（不进 stdout 的正文分离）
        if _is_tool_message_chunk(message_chunk):
            name = getattr(message_chunk, "name", None) or "tool"
            tool_results.append({"name": str(name), "content": piece})
        else:
            buf.append(piece)
        redraw()

    text = "".join(buf)
    # 收尾：去掉流式光标 ▌，并返回最终正文与工具结果供调用方写历史
    if on_update is not None:
        on_update(
            status=format_status_line(
                pending_tool_names=pending_tool_names,
                tools_node_running=tools_depth > 0,
                agent_node_running=agent_depth > 0,
                has_text=bool(text.strip()),
            ),
            text=text,
            tool_results=list(tool_results),
            cursor=False,
        )
    return text, tool_results


async def run_assistant_turn(
    graph: Any,
    *,
    user_text: str = "",
    graph_input: Any | None = None,
    config: dict,
    on_update: Callable[..., None] | None = None,
    decide: Callable[[HitlRequest], list[dict[str, Any]]] | None = None,
    text_prefix: str = "",
    tool_results_prefix: list[ToolResult] | None = None,
) -> AssistantTurnResult:
    """
    单次用户轮次：流式输出 + HITL interrupt 循环（Command(resume=...) 标准 resume）。
    decide 为 None 时遇到 interrupt 即停止并返回 pending_hitl（供 Streamlit 跨 rerun 审批）。
    """
    payload = graph_input
    accumulated_text = text_prefix
    tool_results: list[ToolResult] = list(tool_results_prefix or [])

    while True:
        segment_text, segment_tools = await stream_assistant_text(
            graph,
            user_text=user_text if payload is None else "",
            graph_input=payload,
            config=config,
            on_update=on_update,
            text_prefix=accumulated_text,
            tool_results_prefix=tool_results,
        )
        accumulated_text = segment_text
        tool_results = segment_tools

        pending = get_pending_hitl(graph, config)
        if pending is None:
            return AssistantTurnResult(text=accumulated_text, tool_results=tool_results)

        if decide is None:
            return AssistantTurnResult(
                text=accumulated_text,
                tool_results=tool_results,
                pending_hitl=pending,
            )

        decisions = decide(pending)
        payload = Command(resume={"decisions": decisions})
        user_text = ""


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

    def _on_update(
        *,
        status: str | None,
        text: str,
        cursor: bool,
        tool_results: list[ToolResult] | None = None,
    ) -> None:
        del status, tool_results
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
    turn = asyncio.run(
        run_assistant_turn(
            graph,
            user_text=user_text,
            config=config,
            on_update=_on_update,
            decide=prompt_hitl_decisions_cli,
        )
    )
    return turn.text


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
