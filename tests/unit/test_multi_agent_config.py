"""Unit tests for multi-agent config flags."""

from langgraph_skill_agent.multi_agent.config import (
    multi_agent_routing_enabled,
    supervisor_max_review_retries,
)


def test_multi_agent_routing_enabled(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_MULTI_AGENT_ROUTING", "1")
    assert multi_agent_routing_enabled() is True
    monkeypatch.setenv("ENABLE_MULTI_AGENT_ROUTING", "0")
    assert multi_agent_routing_enabled() is False


def test_supervisor_max_review_retries_defaults(monkeypatch) -> None:
    monkeypatch.delenv("SUPERVISOR_MAX_REVIEW_RETRIES", raising=False)
    assert supervisor_max_review_retries() == 2
    monkeypatch.setenv("SUPERVISOR_MAX_REVIEW_RETRIES", "5")
    assert supervisor_max_review_retries() == 5
