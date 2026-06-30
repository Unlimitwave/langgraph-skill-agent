"""Unit tests for execution mode routing."""

from langgraph_skill_agent.intent_router import (
    intent_routing_enabled,
    resolve_env_forced_mode,
    resolve_execution_mode,
)


def test_intent_routing_enabled_defaults_true(monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_INTENT_ROUTING", raising=False)
    assert intent_routing_enabled() is True


def test_resolve_execution_mode_skips_router_for_greeting(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_INTENT_ROUTING", "1")
    monkeypatch.setenv("ENABLE_PLAN_ROUTING", "1")
    monkeypatch.setenv("ENABLE_MULTI_AGENT_ROUTING", "1")
    assert resolve_execution_mode("你好") == "direct"


def test_resolve_execution_mode_direct_when_no_advanced_modes(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_INTENT_ROUTING", "1")
    monkeypatch.delenv("ENABLE_PLAN_ROUTING", raising=False)
    monkeypatch.delenv("ENABLE_MULTI_AGENT_ROUTING", raising=False)
    assert resolve_execution_mode("帮我调研并写一份报告还要审查") == "direct"


def test_resolve_env_forced_mode_prefers_supervisor(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_MULTI_AGENT_ROUTING", "1")
    monkeypatch.setenv("ENABLE_PLAN_ROUTING", "1")
    assert resolve_env_forced_mode() == "supervisor"


def test_resolve_env_forced_mode_plan_when_only_plan(monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_MULTI_AGENT_ROUTING", raising=False)
    monkeypatch.setenv("ENABLE_PLAN_ROUTING", "1")
    assert resolve_env_forced_mode() == "plan"


def test_resolve_execution_mode_fixed_supervisor_when_intent_off(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_INTENT_ROUTING", "0")
    monkeypatch.setenv("ENABLE_MULTI_AGENT_ROUTING", "1")
    monkeypatch.setenv("ENABLE_PLAN_ROUTING", "1")
    assert resolve_execution_mode("你好") == "supervisor"
    assert resolve_execution_mode("检索知识库") == "supervisor"


def test_resolve_execution_mode_fixed_plan_when_intent_off(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_INTENT_ROUTING", "0")
    monkeypatch.delenv("ENABLE_MULTI_AGENT_ROUTING", raising=False)
    monkeypatch.setenv("ENABLE_PLAN_ROUTING", "1")
    assert resolve_execution_mode("任意问题") == "plan"
