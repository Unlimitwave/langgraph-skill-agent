"""
企业级工作记忆裁剪：工具瘦身、滚动窗口、裁剪瀑布（管线编排）。

裁剪瀑布（自外而内、由软到硬）：
  1. 工具输出瘦身 — 截断历史 ToolMessage 与实时工具返回
  2. 工具参数瘦身 — 截断较早 AIMessage.tool_calls 中的大参数
  3. 滚动窗口 — 按轮次丢弃旧对话，保留最近 N 轮 / token 预算内尾部
  4. System 分层瀑布 — 见 context.build_system_content（P2→P1→P0）
  5. 会话压缩 — 见 compactor.maybe_compact_thread（跨轮 LLM 摘要，调用前执行）
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.types import Command

from langgraph_skill_agent.memory.tokens import (
    context_reserve_tokens,
    estimate_messages_tokens,
    estimate_tokens,
    truncate_to_tokens,
)
from langgraph_skill_agent.utility.messages import stringify_message_content

logger = logging.getLogger(__name__)

TOOL_OUTPUT_SUFFIX = "\n...(tool output truncated for context budget)"
TOOL_ARGS_SUFFIX = "\n...(tool args truncated for context budget)"


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


@dataclass(frozen=True)
class PruningConfig:
    """工作记忆裁剪配置（环境变量驱动）。"""

    enabled: bool = True
    tool_output_max_tokens: int = 2_000
    tool_args_max_tokens: int = 800
    tool_args_keep_recent_messages: int = 12
    rolling_max_turns: int = 0
    rolling_reserve_tokens: int = 8_000

    @classmethod
    def from_env(cls) -> PruningConfig:
        return cls(
            enabled=_env_bool("CONTEXT_PRUNE_ENABLED", True),
            tool_output_max_tokens=_env_int("TOOL_OUTPUT_MAX_TOKENS", 2_000),
            tool_args_max_tokens=_env_int("TOOL_ARGS_MAX_TOKENS", 800),
            tool_args_keep_recent_messages=_env_int("TOOL_ARGS_KEEP_RECENT_MESSAGES", 12),
            rolling_max_turns=_env_int("ROLLING_WINDOW_MAX_TURNS", 0),
            rolling_reserve_tokens=context_reserve_tokens(),
        )


@dataclass
class PruningStats:
    """单次裁剪统计，便于 trace / 调试。"""

    tool_outputs_slimmed: int = 0
    tool_args_slimmed: int = 0
    turns_dropped: int = 0
    messages_dropped: int = 0

    def summary(self) -> str:
        return (
            f"tool_outputs={self.tool_outputs_slimmed} "
            f"tool_args={self.tool_args_slimmed} "
            f"turns_dropped={self.turns_dropped} "
            f"msgs_dropped={self.messages_dropped}"
        )


def _prune_trace_enabled() -> bool:
    return _env_bool("CONTEXT_PRUNE_TRACE", False)


def _message_content_str(msg: BaseMessage) -> str:
    return stringify_message_content(getattr(msg, "content", None))


def _copy_message(msg: BaseMessage, **updates: Any) -> BaseMessage:
    if hasattr(msg, "model_copy"):
        return msg.model_copy(update=updates)
    for key, val in updates.items():
        setattr(msg, key, val)
    return msg


def slim_tool_output_text(text: str, *, max_tokens: int) -> tuple[str, bool]:
    """截断工具返回文本；返回 (新文本, 是否发生裁剪)。"""
    if max_tokens <= 0 or not text:
        return text, False
    if estimate_tokens(text) <= max_tokens:
        return text, False
    return truncate_to_tokens(text, max_tokens, suffix=TOOL_OUTPUT_SUFFIX), True


def slim_tool_message(msg: ToolMessage, *, max_tokens: int) -> tuple[ToolMessage, bool]:
    raw = _message_content_str(msg)
    slimmed, changed = slim_tool_output_text(raw, max_tokens=max_tokens)
    if not changed:
        return msg, False
    return _copy_message(msg, content=slimmed), True


def _slim_tool_call_arg_value(value: Any, *, max_tokens: int) -> tuple[Any, bool]:
    if isinstance(value, str):
        slimmed, changed = slim_tool_output_text(value, max_tokens=max_tokens)
        return slimmed, changed
    if isinstance(value, (dict, list)):
        encoded = json.dumps(value, ensure_ascii=False)
        if estimate_tokens(encoded) <= max_tokens:
            return value, False
        slimmed, _ = slim_tool_output_text(encoded, max_tokens=max_tokens)
        return slimmed, True
    return value, False


def _slim_tool_calls(
    tool_calls: list[dict[str, Any]], *, max_tokens: int
) -> tuple[list[dict[str, Any]], bool]:
    changed = False
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        args = tc.get("args") or {}
        if not isinstance(args, dict):
            out.append(tc)
            continue
        new_args: dict[str, Any] = {}
        for key, val in args.items():
            slim_val, arg_changed = _slim_tool_call_arg_value(val, max_tokens=max_tokens)
            new_args[key] = slim_val
            changed = changed or arg_changed
        new_tc = dict(tc)
        new_tc["args"] = new_args
        out.append(new_tc)
    return out, changed


def slim_tool_messages_in_history(
    messages: list[BaseMessage],
    *,
    config: PruningConfig | None = None,
    stats: PruningStats | None = None,
) -> list[BaseMessage]:
    """工具瘦身：截断历史中过大的 ToolMessage 正文。"""
    config = config or PruningConfig.from_env()
    if not config.enabled or config.tool_output_max_tokens <= 0:
        return messages

    out: list[BaseMessage] = []
    for msg in messages:
        if getattr(msg, "type", None) != "tool":
            out.append(msg)
            continue
        slimmed, changed = slim_tool_message(msg, max_tokens=config.tool_output_max_tokens)
        if changed and stats is not None:
            stats.tool_outputs_slimmed += 1
        out.append(slimmed)
    return out


def slim_tool_call_args_in_history(
    messages: list[BaseMessage],
    *,
    config: PruningConfig | None = None,
    stats: PruningStats | None = None,
) -> list[BaseMessage]:
    """
    工具参数瘦身：对「较早」的 AIMessage.tool_calls 中大参数做截断。

    最近 tool_args_keep_recent_messages 条消息不处理，避免影响当前轮推理。
    """
    config = config or PruningConfig.from_env()
    if not config.enabled or config.tool_args_max_tokens <= 0:
        return messages

    keep_from = max(0, len(messages) - config.tool_args_keep_recent_messages)
    out: list[BaseMessage] = []
    for i, msg in enumerate(messages):
        if i >= keep_from or getattr(msg, "type", None) not in ("ai", "assistant"):
            out.append(msg)
            continue
        tool_calls = list(getattr(msg, "tool_calls", None) or [])
        if not tool_calls:
            out.append(msg)
            continue
        new_calls, changed = _slim_tool_calls(tool_calls, max_tokens=config.tool_args_max_tokens)
        if changed:
            if stats is not None:
                stats.tool_args_slimmed += 1
            out.append(_copy_message(msg, tool_calls=new_calls))
        else:
            out.append(msg)
    return out


def group_conversation_turns(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    """
    按 Human 消息切分对话轮次。

    每轮包含：Human + 后续 AI/Tool，直到下一条 Human。
    非 Human 开头的消息归入首轮前缀（如压缩摘要后的 AI 回复）。
    """
    turns: list[list[BaseMessage]] = []
    current: list[BaseMessage] = []
    for msg in messages:
        if getattr(msg, "type", None) == "human" and current:
            turns.append(current)
            current = []
        current.append(msg)
    if current:
        turns.append(current)
    return turns


def apply_rolling_window(
    messages: list[BaseMessage],
    *,
    messages_budget: int,
    config: PruningConfig | None = None,
    stats: PruningStats | None = None,
    reserve_tokens: int | None = None,
) -> list[BaseMessage]:
    """
    滚动窗口截断：在 token 预算内保留最近轮次；可选 ROLLING_WINDOW_MAX_TURNS 硬顶。

    messages_budget 通常 = ContextBudget.messages_budget - reserve。
    """
    config = config or PruningConfig.from_env()
    if not messages:
        return messages

    reserve = reserve_tokens if reserve_tokens is not None else config.rolling_reserve_tokens
    cap = max(512, messages_budget - reserve)
    turns = group_conversation_turns(messages)

    # 轮次硬顶：先丢掉最旧整轮
    if config.rolling_max_turns > 0 and len(turns) > config.rolling_max_turns:
        drop = len(turns) - config.rolling_max_turns
        if stats is not None:
            stats.turns_dropped += drop
            stats.messages_dropped += sum(len(t) for t in turns[:drop])
        turns = turns[drop:]

    def _flatten(ts: list[list[BaseMessage]]) -> list[BaseMessage]:
        flat: list[BaseMessage] = []
        for t in ts:
            flat.extend(t)
        return flat

    flat = _flatten(turns)
    if estimate_messages_tokens(flat) <= cap:
        return flat

    # token 仍超限：从头部逐轮丢弃（滚动窗口）
    while len(turns) > 1 and estimate_messages_tokens(_flatten(turns)) > cap:
        dropped = turns.pop(0)
        if stats is not None:
            stats.turns_dropped += 1
            stats.messages_dropped += len(dropped)

    return _flatten(turns)


def apply_working_memory_pruning(
    messages: list[BaseMessage],
    *,
    messages_budget: int,
    config: PruningConfig | None = None,
    on_trace: Callable[[str], None] | None = None,
    reserve_tokens: int | None = None,
) -> list[BaseMessage]:
    """
    工作记忆裁剪管线（工具瘦身 → 参数瘦身 → 滚动窗口）。

    System 分层瀑布与 compactor 在 apply_context_layers / prepare_thread_for_turn 中执行。
    """
    config = config or PruningConfig.from_env()
    if not config.enabled or not messages:
        return messages

    stats = PruningStats()
    msgs = slim_tool_messages_in_history(messages, config=config, stats=stats)
    msgs = slim_tool_call_args_in_history(msgs, config=config, stats=stats)
    msgs = apply_rolling_window(
        msgs,
        messages_budget=messages_budget,
        config=config,
        stats=stats,
        reserve_tokens=reserve_tokens,
    )

    if on_trace:
        on_trace(f"pruning: {stats.summary()}")
    elif _prune_trace_enabled() and stats.summary() != (
        "tool_outputs=0 tool_args=0 turns_dropped=0 msgs_dropped=0"
    ):
        logger.info("[PRUNE] %s", stats.summary())

    return msgs


def _slim_tool_result_object(result: Any, *, max_tokens: int) -> Any:
    if isinstance(result, ToolMessage):
        slimmed, _ = slim_tool_message(result, max_tokens=max_tokens)
        return slimmed
    if isinstance(result, Command):
        update = getattr(result, "update", None) or {}
        msgs = update.get("messages")
        if not isinstance(msgs, list):
            return result
        new_msgs: list[Any] = []
        for item in msgs:
            if isinstance(item, ToolMessage):
                slimmed, _ = slim_tool_message(item, max_tokens=max_tokens)
                new_msgs.append(slimmed)
            else:
                new_msgs.append(item)
        return Command(update={**update, "messages": new_msgs})
    return result


@wrap_tool_call(name="enterprise_tool_output_slim")
def slim_tool_output_middleware(request: Any, handler: Callable[..., Any]) -> Any:
    """实时工具瘦身：工具执行返回写入 state 前截断过大输出。"""
    config = PruningConfig.from_env()
    result = handler(request)
    if not config.enabled or config.tool_output_max_tokens <= 0:
        return result
    return _slim_tool_result_object(result, max_tokens=config.tool_output_max_tokens)


__all__ = [
    "PruningConfig",
    "PruningStats",
    "apply_rolling_window",
    "apply_working_memory_pruning",
    "group_conversation_turns",
    "slim_tool_call_args_in_history",
    "slim_tool_messages_in_history",
    "slim_tool_output_middleware",
    "slim_tool_output_text",
]
