"""Unit tests for enterprise context layering."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph_skill_agent.memory.blocks import MemorySections
from langgraph_skill_agent.memory.context import (
    CONTEXT_SYSTEM_MARKER,
    ContextBudget,
    apply_context_layers,
    build_system_content,
    trim_messages_to_budget,
)
from langgraph_skill_agent.memory.session_store import ui_messages_to_lc


def test_context_budget_defaults() -> None:
    budget = ContextBudget()
    assert budget.input_budget == int(128_000 * 0.70)
    assert budget.system_budget == int(budget.input_budget * 0.35)
    assert budget.messages_budget == budget.input_budget - budget.system_budget


def test_build_system_content_has_fixed_sections() -> None:
    sections = MemorySections(
        role="你是助手",
        procedural="使用 rag_search",
        semantic="用户喜欢简洁",
        episodic="上周讨论过 Milvus",
    )
    text = build_system_content(sections=sections, budget=ContextBudget(context_window=10_000))
    assert text.startswith(CONTEXT_SYSTEM_MARKER)
    assert "## 角色设定" in text
    assert "## 程序记忆（procedural）" in text
    assert "## 长期记忆（semantic）" in text
    assert "## 长期记忆（episodic）" in text
    assert "## 业务知识（RAG）" in text


def test_trim_messages_to_budget_keeps_tail() -> None:
    budget = ContextBudget(context_window=500, input_budget_ratio=0.7, system_budget_ratio=0.35)
    msgs = [HumanMessage(content=f"msg-{i} " + "x" * 200) for i in range(20)]
    trimmed = trim_messages_to_budget(msgs, budget=budget, reserve_tokens=0)
    assert len(trimmed) < len(msgs)
    assert trimmed[-1].content.startswith("msg-19")


def test_ui_messages_to_lc_roundtrip_roles() -> None:
    ui = [
        {"role": "user", "content": "你好"},
        {
            "role": "assistant",
            "content": "你好！",
            "tool_results": [{"name": "rag_search", "content": "doc1"}],
        },
    ]
    lc = ui_messages_to_lc(ui)
    assert len(lc) == 2
    assert lc[0].type == "human"
    assert "rag_search" in str(lc[1].content)


def test_apply_context_layers_replaces_injected_system_without_duplicating_convo() -> None:
    human = HumanMessage(content="你好，你有哪些skill", id="h1")
    old_ctx = SystemMessage(content=f"{CONTEXT_SYSTEM_MARKER}\nold", id="ctx-old")
    ai = AIMessage(content="回复", id="a1")
    state = {"messages": [human, old_ctx, ai]}

    out = apply_context_layers(state, budget=ContextBudget(context_window=10_000))
    msgs = out["messages"]
    assert msgs[0].type == "remove"
    ctx_msgs = [m for m in msgs if isinstance(m, SystemMessage)]
    assert len(ctx_msgs) == 1
    assert str(ctx_msgs[0].content).startswith(CONTEXT_SYSTEM_MARKER)
    assert "old" not in str(ctx_msgs[0].content)
    humans = [m for m in msgs if m.type == "human"]
    assert len(humans) == 1
    assert humans[0].content == "你好，你有哪些skill"
