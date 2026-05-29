"""
会话压缩：估算当前 thread 的 messages 占用 token，超过预算则摘要早期对话并写回 checkpointer。

环境变量（可选）：
  COMPACT_ENABLED=1              # 0/false 关闭
  COMPACT_MAX_CONTEXT_TOKENS=60000   # 总预算（含预留）
  COMPACT_RESERVE_TOKENS=8000        # 预留给本轮回复 + 工具输出
  COMPACT_TAIL_MESSAGES=32           # 保留尾部原始消息条数（建议偶数、含 tool 时适当加大）
  COMPACT_SYSTEM_OVERHEAD_TOKENS=3500 # 系统提示、skills 等不在 state.messages 里的粗略加算
"""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph_skill_agent.deepseek_model import build_deepseek_chat_model
from langgraph_skill_agent.utility.messages import stringify_message_content


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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip(), 10)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def compaction_enabled() -> bool:
    return _env_bool("COMPACT_ENABLED", True)


def effective_budget_tokens() -> tuple[int, int, int]:
    """
    返回 (max_context, reserve, overhead)。
    当「当前估算 + reserve + overhead > max_context」时认为越界（保守：用 max_context 作硬顶）。
    """
    max_ctx = _env_int("COMPACT_MAX_CONTEXT_TOKENS", 60_000)
    reserve = _env_int("COMPACT_RESERVE_TOKENS", 8_000)
    overhead = _env_int("COMPACT_SYSTEM_OVERHEAD_TOKENS", 3_500)
    return max_ctx, reserve, overhead


def should_compact(messages: Sequence[BaseMessage], *, extra_text: str = "") -> bool:
    if not compaction_enabled():
        return False
    max_ctx, reserve, overhead = effective_budget_tokens()
    used = estimate_messages_tokens(messages) + estimate_tokens(extra_text) + overhead
    # 预留本轮：超过 (预算 - 预留) 就压
    return used > (max_ctx - reserve)


def _default_summarizer_model() -> BaseChatModel:
    return build_deepseek_chat_model(streaming=False).model_copy(update={"temperature": 0.2})


def summarize_transcript(transcript: str, *, llm: BaseChatModel | None = None) -> str:
    """把早期对话剧本压成一段第三人称摘要（事实、决定、未完成任务）。"""
    model = llm or _default_summarizer_model()
    sys = SystemMessage(
        content=(
            "You compress chat history for another assistant. "
            "Output ONE concise markdown note in Chinese: key facts, user goals, "
            "decisions, open tasks, file paths, and errors. Omit small talk. "
            "Do not invent facts."
        )
    )
    human = HumanMessage(
        content="以下是对话摘录，请压缩：\n\n" + transcript[:120_000]
    )
    out = model.invoke([sys, human])
    text = stringify_message_content(getattr(out, "content", "")).strip()
    return text or "(empty summary)"


def _messages_have_ids(messages: Sequence[BaseMessage]) -> bool:
    for m in messages:
        if getattr(m, "id", None):
            return True
    return False


def replace_thread_messages(
    compiled: Any,
    config: dict,
    new_messages: list[BaseMessage],
    *,
    old_messages: list[BaseMessage] | None = None,
) -> None:
    """
    用 RemoveMessage 清空旧列表再写入 new_messages（适配 add_messages reducer）。
    若旧消息无 id，则退化为仅 append new_messages（可能重复，慎用）。
    """
    snap = compiled.get_state(config)
    old = list(old_messages if old_messages is not None else (snap.values or {}).get("messages") or [])

    if old and _messages_have_ids(old):
        removals: list[RemoveMessage] = []
        for m in old:
            mid = getattr(m, "id", None)
            if mid:
                removals.append(RemoveMessage(id=mid))
        compiled.update_state(config, {"messages": removals + new_messages})
        return

    # 无 id：无法可靠删除，只追加摘要（会占上下文；应升级 langchain 或先跑几轮让 graph 写入 id）
    compiled.update_state(config, {"messages": new_messages})


def maybe_compact_thread(
    compiled: Any,
    config: dict,
    *,
    summarizer_llm: BaseChatModel | None = None,
    extra_token_text: str = "",
    tail_messages: int | None = None,
    on_trace: Callable[[str], None] | None = None,
) -> bool:
    """
    若当前 thread 消息过长则压缩。返回是否执行了压缩。
    """
    if not compaction_enabled():
        return False

    snap = compiled.get_state(config)
    messages: list[BaseMessage] = list((snap.values or {}).get("messages") or [])
    if not messages:
        return False

    max_ctx, reserve, overhead = effective_budget_tokens()
    tail_n = tail_messages if tail_messages is not None else _env_int("COMPACT_TAIL_MESSAGES", 32)

    used = estimate_messages_tokens(messages) + estimate_tokens(extra_token_text) + overhead
    threshold = max_ctx - reserve
    if used <= threshold:
        return False

    if len(messages) <= tail_n + 2:
        if on_trace:
            on_trace("compactor: over token budget but tail too large to split; skip or raise tail_n")
        return False

    head = messages[:-tail_n]
    tail = messages[-tail_n:]

    lines = [message_to_plain_text(m) for m in head]
    transcript = "\n\n".join(lines).strip()
    summary = summarize_transcript(transcript, llm=summarizer_llm)

    summary_msg = SystemMessage(
        content=(
            "[会话前文已压缩 — 仅作上下文恢复，请勿当作用户新指令]\n"
            + summary
        )
    )
    new_chain: list[BaseMessage] = [summary_msg, *tail]

    replace_thread_messages(compiled, config, new_chain, old_messages=messages)

    if on_trace:
        new_used = estimate_messages_tokens(new_chain) + estimate_tokens(extra_token_text) + overhead
        on_trace(
            f"compactor: compacted head={len(head)} tail={len(tail)} "
            f"tokens_before≈{used} after≈{new_used} threshold={threshold}"
        )
    return True


__all__ = [
    "estimate_tokens",
    "estimate_messages_tokens",
    "should_compact",
    "maybe_compact_thread",
    "replace_thread_messages",
    "compaction_enabled",
]