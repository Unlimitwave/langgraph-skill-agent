"""Token estimation helpers shared by context budgeting and compaction."""

from __future__ import annotations

import os
from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from langgraph_skill_agent.utility.messages import stringify_message_content


def context_reserve_tokens() -> int:
    """预留给本轮回复 + 工具输出（compactor / rolling window 共用）。"""
    for name in ("CONTEXT_RESERVE_TOKENS", "COMPACT_RESERVE_TOKENS"):
        raw = os.environ.get(name)
        if raw is not None and str(raw).strip():
            try:
                return int(str(raw).strip(), 10)
            except ValueError:
                break
    return 8_000


def _load_tiktoken_encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


_ENC = None


def estimate_tokens(text: str) -> int:
    """粗略 token 数：优先 tiktoken，否则 chars/4。"""
    global _ENC
    if not text:
        return 0
    if _ENC is None:
        _ENC = _load_tiktoken_encoder()
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, len(text) // 4)


def _message_kind(msg: BaseMessage) -> str:
    t = getattr(msg, "type", "") or ""
    return str(t)


def message_to_plain_text(msg: BaseMessage) -> str:
    role = _message_kind(msg)
    if role in ("human", "user"):
        label = "User"
    elif role in ("ai", "assistant"):
        label = "Assistant"
    elif role == "system":
        label = "System"
    elif role == "tool":
        name = getattr(msg, "name", None) or "tool"
        label = f"Tool({name})"
    else:
        label = role or "Unknown"
    body = stringify_message_content(getattr(msg, "content", None)).strip()
    return f"### {label}\n{body}".strip()


def estimate_messages_tokens(messages: Sequence[BaseMessage]) -> int:
    return sum(estimate_tokens(message_to_plain_text(m)) for m in messages)


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    *,
    suffix: str = "\n...(truncated for context budget)",
) -> str:
    """
    将文本截断到不超过 max_tokens 的近似长度（二分查找字符前缀）。
    """
    if max_tokens <= 0 or not text:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    room = max(0, max_tokens - estimate_tokens(suffix))
    if room <= 0:
        return suffix.strip()
    return text[:lo] + suffix


__all__ = [
    "context_reserve_tokens",
    "estimate_tokens",
    "estimate_messages_tokens",
    "message_to_plain_text",
    "truncate_to_tokens",
]
