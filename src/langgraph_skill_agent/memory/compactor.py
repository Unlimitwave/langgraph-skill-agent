"""
会话压缩：估算当前 thread 的 messages 占用 token，超过预算则摘要早期对话并写回 checkpointer。

预算与 ContextBudget 对齐（见 context.py）：
  COMPACT_ENABLED=1              # 0/false 关闭
  COMPACT_TAIL_MESSAGES=32       # 保留尾部原始消息条数（建议偶数、含 tool 时适当加大）

总预算 / System 开销 / 预留分别由 CONTEXT_WINDOW、CONTEXT_*_RATIO、
CONTEXT_RESERVE_TOKENS（或已废弃别名 COMPACT_RESERVE_TOKENS）控制。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)

from langgraph_skill_agent.deepseek_model import build_deepseek_chat_model
from langgraph_skill_agent.memory.context import COMPACT_SUMMARY_PREFIX, ContextBudget
from langgraph_skill_agent.memory.tokens import (
    context_reserve_tokens,
    estimate_messages_tokens,
    estimate_tokens,
    message_to_plain_text,
)
from langgraph_skill_agent.utility.messages import stringify_message_content


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
    返回 (max_context, reserve, overhead)，与 ContextBudget 对齐。

    max_context = input_budget；overhead = system_budget（每轮注入 System，不在 messages 内）；
    reserve = context_reserve_tokens()。当 messages + overhead + reserve > max_context 时触发压缩。
    """
    budget = ContextBudget.from_env()
    return budget.input_budget, context_reserve_tokens(), budget.system_budget


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
    human = HumanMessage(content="以下是对话摘录，请压缩：\n\n" + transcript[:120_000])
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
    old = list(
        old_messages if old_messages is not None else (snap.values or {}).get("messages") or []
    )

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
            on_trace(
                "compactor: over token budget but tail too large to split; skip or raise tail_n"
            )
        return False

    head = messages[:-tail_n]
    tail = messages[-tail_n:]

    lines = [message_to_plain_text(m) for m in head]
    transcript = "\n\n".join(lines).strip()
    summary = summarize_transcript(transcript, llm=summarizer_llm)

    summary_msg = SystemMessage(
        content=(f"{COMPACT_SUMMARY_PREFIX} — 仅作上下文恢复，请勿当作用户新指令]\n" + summary)
    )
    new_chain: list[BaseMessage] = [summary_msg, *tail]

    replace_thread_messages(compiled, config, new_chain, old_messages=messages)

    if on_trace:
        new_used = (
            estimate_messages_tokens(new_chain) + estimate_tokens(extra_token_text) + overhead
        )
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
