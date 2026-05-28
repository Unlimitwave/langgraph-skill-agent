from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessageChunk, ToolMessage
import json
from datetime import datetime, timezone
from langchain_core.messages import BaseMessage, message_to_dict



PROJECT_ROOT = Path(__file__).parent

logger = logging.getLogger(__name__)


def _write_stdout(s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.flush()

def _normalize_skill_sources(sources: list[str]) -> list[str]:
    """将技能目录路径转为绝对路径字符串，供 deepagents 加载。"""
    return [str(Path(s).expanduser().resolve()) for s in sources]


def _stringify_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _preview(text: str, max_len: int = 3000) -> str:
    text = text.strip()
    if len(text) > max_len:
        return text[:max_len] + "\n...(truncated)"
    return text


def _log_tool_message(msg: ToolMessage) -> None:
    name = getattr(msg, "name", None) or "(unknown tool)"
    call_id = getattr(msg, "tool_call_id", None) or ""
    body = _stringify_message_content(msg.content)
    suffix = f" tool_call_id={call_id}" if call_id else ""
    logger.info("\n[ToolMessage] %s%s\n%s\n", name, suffix, _preview(body))


def _log_tool_message_once(msg: ToolMessage, seen_tool: set[str]) -> None:
    """同一 tool_call_id 只记一次（v2 流里 messages 与 updates 会各带一条）。"""
    tid = str(getattr(msg, "tool_call_id", None) or id(msg))
    if tid in seen_tool:
        return
    seen_tool.add(tid)
    _log_tool_message(msg)


def _extract_messages_from_update(data: Any) -> list[Any]:
    """updates 里各节点的 payload 可能是 dict（含 messages）或已是消息列表。"""
    if data is None:
        return []
    if isinstance(data, dict) and "messages" in data:
        raw = data["messages"]
        return list(raw) if isinstance(raw, (list, tuple)) else [raw]
    if isinstance(data, (list, tuple)):
        return list(data)
    return []


def _stream_v2(agent: Any, payload: dict[str, Any], config: dict) -> None:
    _write_stdout("助手: ")
    stream = agent.stream(
        payload,
        config=config,
        stream_mode=["messages", "updates"],
        subgraphs=True,
        version="v2",
    )
    seen_tool: set[str] = set()
    for chunk in stream:
        if not isinstance(chunk, dict):
            continue
        kind = chunk.get("type")
        if kind == "messages":
            pair = chunk.get("data")
            if not isinstance(pair, (list, tuple)) or len(pair) < 1:
                continue
            token, _meta = pair[0], pair[1] if len(pair) > 1 else None

            if isinstance(token, ToolMessage):
                _log_tool_message_once(token, seen_tool)
                continue

            ttype = getattr(token, "type", None)
            if ttype == "tool" and not isinstance(token, ToolMessage):
                content = _stringify_message_content(getattr(token, "content", ""))
                name = getattr(token, "name", None) or "tool"
                logger.info("\n[ToolMessage] %s\n%s\n", name, _preview(content))
                continue

            if isinstance(token, AIMessageChunk):
                tcc = getattr(token, "tool_call_chunks", None) or []
                for tc in tcc:
                    if not isinstance(tc, dict):
                        continue
                    if tc.get("name"):
                        _write_stdout(f"\n[Tool call] {tc['name']}")
                    if tc.get("args"):
                        _write_stdout(str(tc["args"]))
                text = _stringify_message_content(getattr(token, "content", None))
                if text and not tcc:
                    _write_stdout(text)
        elif kind == "updates":
            data = chunk.get("data")
            if not isinstance(data, dict):
                continue
            for _node, node_data in data.items():
                for msg in _extract_messages_from_update(node_data):
                    if isinstance(msg, ToolMessage):
                        _log_tool_message_once(msg, seen_tool)


def _consume_messages_stream(stream: Any) -> None:
    """消费 stream_mode='messages' 的迭代器，打印文本与工具相关块。"""
    seen_tool: set[str] = set()
    for item in stream:
        token = item[0] if isinstance(item, (list, tuple)) and len(item) >= 1 else item
        if isinstance(token, ToolMessage):
            _log_tool_message_once(token, seen_tool)
            continue
        if isinstance(token, AIMessageChunk):
            tcc = getattr(token, "tool_call_chunks", None) or []
            for tc in tcc:
                if not isinstance(tc, dict):
                    continue
                if tc.get("name"):
                    _write_stdout(f"\n[Tool call] {tc['name']}")
                if tc.get("args"):
                    _write_stdout(str(tc["args"]))
            text = _stringify_message_content(getattr(token, "content", None))
            if text and not tcc:
                _write_stdout(text)


def _stream_simple(agent: Any, payload: dict[str, Any], config: dict) -> None:
    """简化流式打印：仅 stream_mode='messages'，无 version=v2、无 updates 双通道。

    仍能打印助手文本流、ToolMessage、AIMessageChunk 中的 tool_call_chunks。
    若 graph 不接受该 stream 调用方式，向外抛出 TypeError，便于上层改用 _stream_v2。
    """
    stream = agent.stream(payload, config=config, stream_mode="messages")
    _write_stdout("助手: ")
    _consume_messages_stream(stream)
    _write_stdout("\n")


def _stream_fallback(agent: Any, payload: dict[str, Any], config: dict) -> None:
    """不支持 version='v2' 且简化流也失败时：再尝试 messages 流；失败则提示并返回。"""
    try:
        stream = agent.stream(payload, config=config, stream_mode="messages")
    except TypeError:
        logger.warning("当前 graph 不支持 stream_mode='messages'")
        return
    _write_stdout("助手: ")
    _consume_messages_stream(stream)
    _write_stdout("\n")


def stream_assistant_reply(agent: Any, user_text: str, config: dict) -> None:
    payload: dict[str, Any] = {"messages": [{"role": "user", "content": user_text}]}
    try:
        _stream_simple(agent, payload, config)
    except TypeError:
        try:
            _stream_v2(agent, payload, config)
            _write_stdout("\n")
        except TypeError:
            _stream_fallback(agent, payload, config)



def _messages_to_jsonable(messages: list[BaseMessage]) -> list[dict]:
    return [message_to_dict(m) for m in messages]


def save_conversation_snapshot(
    agent,
    config: dict,
    *,
    project_root: Path | None = None,
) -> Path:
    """把当前 thread 的完整 messages 快照写入 conversation_history/。"""
    root = project_root or PROJECT_ROOT
    hist_dir = root / "conversation_history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    thread_id = str((config.get("configurable") or {}).get("thread_id") or "default")
    safe_tid = "".join(c if c.isalnum() or c in "-_" else "_" for c in thread_id)[:200]
    snap = agent.get_state(config)
    values = getattr(snap, "values", None) or {}
    messages = values.get("messages") or []
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = hist_dir / f"{safe_tid}_{ts}.json"
    payload = {
        "thread_id": thread_id,
        "saved_at_utc": ts,
        "messages": _messages_to_jsonable(messages),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path



def _load_agent_memory_blocks(root: Path) -> str:
    """按 OpenClaw 风格加载可选 Markdown，不存在则跳过。"""
    mem_dir = root / "agent_memory"  # 或 .openclaw / memory 等你喜欢的目录名
    parts: list[str] = []
    for name, title in [
        ("soul.md", "## Agent soul (persona)"),
        ("user.md", "## User profile"),
        ("Memory.md", "## Long-term memory"),
    ]:
        p = mem_dir / name
        if p.is_file():
            text = p.read_text(encoding="utf-8").strip()
            if text:
                parts.append(f"{title}\n{text}")
    return "\n\n".join(parts).strip()