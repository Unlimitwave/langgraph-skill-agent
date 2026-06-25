"""Unit tests for enterprise working-memory pruning."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_skill_agent.memory.context import ContextBudget, trim_messages_to_budget
from langgraph_skill_agent.memory.pruning import (
    PruningConfig,
    apply_rolling_window,
    apply_working_memory_pruning,
    group_conversation_turns,
    slim_tool_messages_in_history,
    slim_tool_output_text,
)
from langgraph_skill_agent.memory.tokens import estimate_tokens


def test_slim_tool_output_text_truncates() -> None:
    huge = "word " * 5000
    slimmed, changed = slim_tool_output_text(huge, max_tokens=50)
    assert changed
    assert estimate_tokens(slimmed) <= 60
    assert "truncated" in slimmed


def test_slim_tool_messages_in_history() -> None:
    msgs = [
        HumanMessage(content="hi"),
        ToolMessage(content="x" * 8000, name="rag_search", tool_call_id="t1"),
    ]
    out = slim_tool_messages_in_history(
        msgs,
        config=PruningConfig(enabled=True, tool_output_max_tokens=100),
    )
    assert estimate_tokens(str(out[1].content)) < estimate_tokens("x" * 8000)


def test_group_conversation_turns_splits_on_human() -> None:
    msgs = [
        HumanMessage(content="q1"),
        AIMessage(content="a1"),
        HumanMessage(content="q2"),
        AIMessage(content="a2"),
    ]
    turns = group_conversation_turns(msgs)
    assert len(turns) == 2
    assert turns[0][0].content == "q1"
    assert turns[1][0].content == "q2"


def test_apply_rolling_window_drops_old_turns() -> None:
    config = PruningConfig(
        enabled=True,
        rolling_max_turns=2,
        rolling_reserve_tokens=0,
        tool_output_max_tokens=0,
        tool_args_max_tokens=0,
    )
    msgs = [
        HumanMessage(content="old"),
        AIMessage(content="old reply"),
        HumanMessage(content="mid"),
        AIMessage(content="mid reply"),
        HumanMessage(content="new"),
        AIMessage(content="new reply"),
    ]
    out = apply_rolling_window(
        msgs,
        messages_budget=100_000,
        config=config,
    )
    assert "old" not in str(out[0].content)
    assert out[-1].content == "new reply"


def test_trim_messages_to_budget_keeps_tail_with_pruning() -> None:
    budget = ContextBudget(context_window=500, input_budget_ratio=0.7, system_budget_ratio=0.35)
    msgs = [HumanMessage(content=f"msg-{i} " + "x" * 200) for i in range(20)]
    trimmed = trim_messages_to_budget(msgs, budget=budget, reserve_tokens=0)
    assert len(trimmed) < len(msgs)
    assert trimmed[-1].content.startswith("msg-19")


def test_apply_working_memory_pruning_slims_old_tool_args() -> None:
    long_body = "y" * 6000
    ai_old = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "write_file",
                "args": {"content": long_body},
                "id": "c1",
                "type": "tool_call",
            }
        ],
    )
    ai_recent = AIMessage(content="recent")
    msgs = [ai_old, HumanMessage(content="h"), ai_recent]
    out = apply_working_memory_pruning(
        msgs,
        messages_budget=200_000,
        config=PruningConfig(
            enabled=True,
            tool_args_max_tokens=80,
            tool_args_keep_recent_messages=1,
            tool_output_max_tokens=0,
            rolling_max_turns=0,
            rolling_reserve_tokens=0,
        ),
    )
    old_calls = out[0].tool_calls
    assert old_calls
    assert "truncated" in str(old_calls[0]["args"]["content"])
    assert out[-1].content == "recent"
