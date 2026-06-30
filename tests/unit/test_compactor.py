"""Unit tests for conversation compaction helpers (no LLM calls)."""

from langchain_core.messages import HumanMessage

from langgraph_skill_agent.memory.compactor import (
    compaction_enabled,
    effective_budget_tokens,
    estimate_messages_tokens,
    estimate_tokens,
    should_compact,
)
from langgraph_skill_agent.memory.context import ContextBudget


def test_estimate_tokens_empty_and_non_empty() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") >= 1


def test_estimate_messages_tokens_sums_messages() -> None:
    msgs = [HumanMessage(content="hello"), HumanMessage(content="world")]
    total = estimate_messages_tokens(msgs)
    assert total == estimate_tokens("### User\nhello") + estimate_tokens("### User\nworld")


def test_compaction_enabled_respects_env(monkeypatch) -> None:
    monkeypatch.setenv("COMPACT_ENABLED", "0")
    assert compaction_enabled() is False
    monkeypatch.setenv("COMPACT_ENABLED", "yes")
    assert compaction_enabled() is True


def test_effective_budget_tokens_defaults(monkeypatch) -> None:
    for name in (
        "CONTEXT_WINDOW",
        "CONTEXT_INPUT_BUDGET_RATIO",
        "CONTEXT_SYSTEM_BUDGET_RATIO",
        "CONTEXT_RESERVE_TOKENS",
        "COMPACT_MAX_CONTEXT_TOKENS",
        "COMPACT_RESERVE_TOKENS",
        "COMPACT_SYSTEM_OVERHEAD_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)
    budget = ContextBudget.from_env()
    max_ctx, reserve, overhead = effective_budget_tokens()
    assert max_ctx == budget.input_budget
    assert reserve == 8_000
    assert overhead == budget.system_budget


def test_should_compact_disabled(monkeypatch) -> None:
    monkeypatch.setenv("COMPACT_ENABLED", "false")
    long_text = "x" * 200_000
    msgs = [HumanMessage(content=long_text)]
    assert should_compact(msgs) is False
