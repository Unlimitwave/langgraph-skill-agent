"""
企业级上下文工程：分层 System 注入 + ContextBudget 预算分配 + 工作记忆裁剪瀑布。

设计原则：
- 稳定事实 / SOP / 任务进度 → System（每轮重建，不占对话轮次）
- 交互过程 → messages（Human / AI / Tool）
- task_state 不进 messages，序列化进 System P1

裁剪瀑布（每轮 before_model，自外而内）：
  1. 工具输出瘦身（历史 ToolMessage + wrap_tool_call 实时截断）
  2. 工具参数瘦身（较早 AIMessage.tool_calls 大参数）
  3. 滚动窗口（按轮次 / token 预算保留最近对话）
  4. System 分层瀑布（P2 长期记忆/RAG → P1 任务态 → P0 角色/SOP）
  跨轮：compactor.maybe_compact_thread（LLM 摘要，调用前执行）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import before_model
from langchain.agents.middleware.types import AgentState
from langchain_core.messages import BaseMessage, RemoveMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from langgraph_skill_agent.memory.blocks import MemorySections, load_memory_sections
from langgraph_skill_agent.memory.pruning import apply_working_memory_pruning
from langgraph_skill_agent.memory.tokens import (
    context_reserve_tokens,
    estimate_tokens,
    truncate_to_tokens,
)

# 注入式 System 消息标记，便于每轮替换
CONTEXT_SYSTEM_MARKER = "[CTX-SYSTEM]"

# 压缩摘要前缀（与 compactor 保持一致）
COMPACT_SUMMARY_PREFIX = "[会话前文已压缩"


@dataclass(frozen=True)
class ContextBudget:
    """上下文 token 预算分配。"""

    context_window: int = 128_000
    input_budget_ratio: float = 0.70
    system_budget_ratio: float = 0.35

    @classmethod
    def from_env(cls) -> ContextBudget:
        return cls(
            context_window=_env_int("CONTEXT_WINDOW", 128_000),
            input_budget_ratio=_env_float("CONTEXT_INPUT_BUDGET_RATIO", 0.70),
            system_budget_ratio=_env_float("CONTEXT_SYSTEM_BUDGET_RATIO", 0.35),
        )

    @property
    def input_budget(self) -> int:
        return int(self.context_window * self.input_budget_ratio)

    @property
    def system_budget(self) -> int:
        return int(self.input_budget * self.system_budget_ratio)

    @property
    def messages_budget(self) -> int:
        return self.input_budget - self.system_budget


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip(), 10)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """兼容旧调用；委托 tokens.truncate_to_tokens。"""
    return truncate_to_tokens(text, max_tokens)


def _format_section(title: str, body: str) -> str:
    """将单块记忆格式化为 Markdown 二级标题段落；空 body 返回空串（不参与组装）。"""
    body = (body or "").strip()
    if not body:
        return ""
    return f"## {title}\n{body}"


def _serialize_task_state(state: dict[str, Any]) -> str:
    """
    从 LangGraph state 提取任务中间态，序列化为 Markdown 列表文本。

    这些内容属于 P1 优先级，注入 System 的「任务中间态」区块，而不是写入
    messages 历史——避免任务元数据占用对话轮次，且每轮 before_model 可重建。
    """
    parts: list[str] = []
    # 标量字段：规划/执行链路中的关键指针
    for key in ("user_goal", "macro_thread_id", "current_todo_id"):
        val = state.get(key)
        if val:
            parts.append(f"- {key}: {val}")
    # 结构化 todo 列表：plan_execute 写入 state["todos"]
    todos = state.get("todos")
    if isinstance(todos, list) and todos:
        parts.append("- todos:")
        for item in todos:
            if isinstance(item, dict):
                tid = item.get("id", "?")
                title = item.get("title", "")
                status = item.get("status", "pending")
                parts.append(f"  - [{status}] {tid}: {title}")
            else:
                parts.append(f"  - {item}")
    return "\n".join(parts).strip()


def build_system_content(
    *,
    sections: MemorySections | None = None,
    task_state: str = "",
    rag_context: str = "",
    budget: ContextBudget | None = None,
) -> str:
    """
    按固定 ## section 标题组装分层 System 内容，并在超预算时分级裁剪。

    优先级（数字越小越重要，越晚被裁）：
      P0 — 角色设定、程序记忆（SOP）：Agent 身份与操作规范，原则上保留
      P1 — 任务中间态：当前目标与 todo 进度
      P2 — 长期记忆（semantic/episodic）、RAG：可压缩的外部知识

    裁剪策略：先尝试完整组装；超 system_budget 时从 P2→P1 逐级减半/删除
    block；若 P0 合计仍超限，对拼接结果做硬截断。
    """
    sections = sections or load_memory_sections()
    budget = budget or ContextBudget.from_env()

    # (title, body, priority)  P0=0, P1=1, P2=2
    raw_sections: list[tuple[str, str, int]] = [
        ("角色设定", sections.role, 0),
        ("程序记忆（procedural）", sections.procedural, 0),
        ("任务中间态", task_state or sections.task_state, 1),
        ("长期记忆（semantic）", sections.semantic, 2),
        ("长期记忆（episodic）", sections.episodic, 2),
        ("业务知识（RAG）", rag_context or sections.rag_hint, 2),
    ]

    # 过滤空 section，得到 (markdown_block, priority) 列表
    blocks: list[tuple[str, int]] = []
    for title, body, prio in raw_sections:
        block = _format_section(title, body)
        if block:
            blocks.append((block, prio))

    if not blocks:
        return f"{CONTEXT_SYSTEM_MARKER}\n(无额外上下文)"

    total = sum(estimate_tokens(b) for b, _ in blocks)
    sys_budget = budget.system_budget  # input_budget * system_budget_ratio
    if total <= sys_budget:
        return CONTEXT_SYSTEM_MARKER + "\n\n" + "\n\n".join(b for b, _ in blocks)

    # 超预算：按优先级从低到高（P2 → P1）迭代裁剪，P0 不参与此循环
    result: list[tuple[str, int]] = list(blocks)  # 复制一份，后续删减/截断只改 result
    for prio in (2, 1):  # 先裁 P2（长期记忆/RAG），再裁 P1（任务态）；P0 角色/SOP 不动
        while sum(estimate_tokens(b) for b, _ in result) > sys_budget:  # 总 token 仍超预算就继续裁
            # 在当前优先级 prio 下，收集所有可裁 block：(列表下标, block 文本)
            candidates = [(i, b) for i, (b, p) in enumerate(result) if p == prio]
            if not candidates:  # 该优先级已无 block（可能都被删光了）
                break  # 退出 while，改试更低优先级或走后面的硬截断
            # 同优先级内先裁最长的 block：一次减 token 最多，循环次数更少
            idx, block = max(candidates, key=lambda x: estimate_tokens(x[1]))
            if estimate_tokens(block) <= 64:  # 已经很短（≤64 token）仍塞不进预算
                result.pop(idx)  # 再截也没意义，整段删掉
                continue  # 回到 while 顶，重新算总 token
            # 否则把该 block 截到「当前长度的一半」，但下限 64 token，避免截成碎片
            new_block = _truncate_to_tokens(block, max(64, estimate_tokens(block) // 2))
            result[idx] = (new_block, prio)  # 原位替换，优先级不变

    # P0（角色 + SOP）仍超预算时，对最终拼接串做硬截断（最后手段）
    joined = "\n\n".join(b for b, _ in result)
    if estimate_tokens(joined) > sys_budget:
        joined = _truncate_to_tokens(joined, sys_budget)
    return CONTEXT_SYSTEM_MARKER + "\n\n" + joined


def _is_injected_system_message(msg: BaseMessage) -> bool:
    """
    判断是否为「可每轮替换」的注入式 System 消息。

    包含两类：
      - [CTX-SYSTEM]：本模块 inject_context_before_model 每轮重建的分层上下文。
      - [会话前文已压缩...]：compactor 写入的压缩摘要（见 _is_compact_summary）。
    二者都带固定前缀，便于从 messages 中识别并去重。

    用于区分「本轮需要可重复注入的系统上下文」与「普通/用户消息」，
    这样在处理历史消息时可以安全去除所有自动生成的 System 层上下文，
    避免它们多轮累加（例如多轮调用 inject_context_before_model 时）。

    举例提醒（PREFIX）：
        CONTEXT_SYSTEM_MARKER: '[CTX-SYSTEM]'
        COMPACT_SUMMARY_PREFIX: '[会话前文已压缩...]'

    只要内容以这些特殊前缀之一开头，即判定为自动注入的 System 消息，予以去重处理。
    """

    # 只关心 type == "system" 的消息，否则直接不是我们要剥离的（如 user/assistant/tool 不处理）
    if getattr(msg, "type", None) != "system":
        return False

    # 兼容两种 content 结构：字符串或 list（LLM 消息格式有时用 list 装分段内容）
    content = getattr(msg, "content", "") or ""
    if isinstance(content, list):
        # 若为 list，则转换为字符串，方便统一处理
        content = str(content)
    text = str(content)

    # 判定逻辑注释：
    #  - 只要内容以 CONTEXT_SYSTEM_MARKER（例：[CTX-SYSTEM]）开头，说明是分层上下文注入
    #  - 或者以 COMPACT_SUMMARY_PREFIX（例：[会话前文已压缩...]）开头，说明是会话摘要（compactor）
    # 这样做的目的是为了统一管理所有“自动注入不可积累”的 system 消息，防止多轮累增冗余
    return (
        text.startswith(CONTEXT_SYSTEM_MARKER)  # 分层上下文标记
        or text.startswith(COMPACT_SUMMARY_PREFIX)  # 会话摘要（自动注入）
    )


def _conversation_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    从 state.messages 中剥离上轮注入的 CTX-SYSTEM，得到「纯对话」序列。

    每轮 before_model 会先去掉旧的 [CTX-SYSTEM]，再基于对话 + 新 System 重组，
    避免 System 层在 checkpointer 里叠乘。压缩摘要（compactor 产物）保留在头部。
    """
    return [m for m in messages if not _is_injected_system_message(m) or _is_compact_summary(m)]


def _is_compact_summary(msg: BaseMessage) -> bool:
    """是否为 compactor 生成的会话压缩摘要（固定在 messages 最前，不参与 CTX 剥离）。"""
    if getattr(msg, "type", None) != "system":
        return False
    content = getattr(msg, "content", "") or ""
    if isinstance(content, list):
        content = str(content)
    return str(content).startswith(COMPACT_SUMMARY_PREFIX)


def trim_messages_to_budget(
    messages: list[BaseMessage],
    *,
    budget: ContextBudget | None = None,
    reserve_tokens: int | None = None,
) -> list[BaseMessage]:
    """
    将对话消息限制在 messages 分区预算内（滚动窗口 + 工具瘦身管线）。

    保留 compactor 压缩摘要于头部；对其余消息执行 apply_working_memory_pruning。
    """
    budget = budget or ContextBudget.from_env()
    reserve = reserve_tokens if reserve_tokens is not None else context_reserve_tokens()
    if not messages:
        return messages

    head: list[BaseMessage] = []
    body = list(messages)
    if body and _is_compact_summary(body[0]):
        head = [body[0]]
        body = body[1:]

    body = apply_working_memory_pruning(
        body,
        messages_budget=budget.messages_budget,
        reserve_tokens=reserve,
    )

    return head + body


# TODO caoyintao 需要阅读代码 2026-06-22-16:00
def apply_context_layers(
    state: dict[str, Any],
    *,
    budget: ContextBudget | None = None,
    rag_context: str = "",
) -> dict[str, list[BaseMessage]]:
    """构建分层上下文并写回 messages（供 before_model 调用）。"""

    # 获取预算,如果没有传入，会从env读取
    budget = budget or ContextBudget.from_env()
    raw_messages: list[BaseMessage] = list(state.get("messages") or [])
    convo = _conversation_messages(raw_messages)
    convo = trim_messages_to_budget(convo, budget=budget)

    task_state = _serialize_task_state(state)
    system_text = build_system_content(
        task_state=task_state, rag_context=rag_context, budget=budget
    )
    system_msg = SystemMessage(content=system_text)
    # add_messages reducer 会追加而非替换；须先 REMOVE_ALL 再写入，避免 CTX / 对话轮次叠乘。
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), system_msg, *convo]}


@before_model(name="enterprise_context_layers")
def inject_context_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    del runtime
    return apply_context_layers(dict(state))


__all__ = [
    "CONTEXT_SYSTEM_MARKER",
    "COMPACT_SUMMARY_PREFIX",
    "ContextBudget",
    "build_system_content",
    "trim_messages_to_budget",
    "apply_context_layers",
    "inject_context_before_model",
]
