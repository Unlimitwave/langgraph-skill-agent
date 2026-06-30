"""Unit tests for per-tenant identity and thread namespacing."""

import pytest

from langgraph_skill_agent.utility.tenant import (
    THREAD_NS_SEP,
    AgentContext,
    bare_thread_id,
    build_agent_config,
    build_agent_context,
    build_invoke_kwargs,
    namespaced_thread_id,
    normalize_user_id,
    user_id_from_config,
)


def test_namespaced_thread_id_prefixes_user() -> None:
    tid = namespaced_thread_id("alice", "sess-1")
    assert tid == f"alice{THREAD_NS_SEP}sess-1"
    assert bare_thread_id(tid) == "sess-1"


def test_build_agent_config_carries_user_id() -> None:
    cfg = build_agent_config(thread_id="chat-1", user_id="bob")
    assert cfg["configurable"]["user_id"] == "bob"
    assert cfg["configurable"]["thread_id"].startswith(f"bob{THREAD_NS_SEP}")
    assert cfg["configurable"]["tenant_id"] == "default"


def test_build_agent_context_defaults_tenant() -> None:
    ctx = build_agent_context(user_id="alice")
    assert ctx.user_id == "alice"
    assert ctx.tenant_id == "default"


def test_build_invoke_kwargs_separates_session_and_identity() -> None:
    invoke = build_invoke_kwargs(thread_id="sess-1", user_id="bob", tenant_id="acme")
    assert invoke["context"] == AgentContext(user_id="bob", tenant_id="acme")
    assert invoke["config"]["configurable"]["thread_id"] == f"bob{THREAD_NS_SEP}sess-1"
    assert invoke["config"]["configurable"]["user_id"] == "bob"
    assert invoke["config"]["configurable"]["tenant_id"] == "acme"


def test_user_id_from_config_prefers_explicit_field() -> None:
    cfg = build_agent_config(thread_id="x", user_id="carol")
    assert user_id_from_config(cfg) == "carol"


def test_user_id_from_config_parses_namespaced_thread() -> None:
    cfg = {"configurable": {"thread_id": namespaced_thread_id("dave", "t1")}}
    assert user_id_from_config(cfg) == "dave"


def test_invalid_user_id_rejected() -> None:
    with pytest.raises(ValueError, match="invalid AGENT_USER_ID"):
        normalize_user_id("../evil")
