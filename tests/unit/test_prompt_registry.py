"""Unit tests for versioned prompt registry."""

from langgraph_skill_agent.prompts import (
    clear_prompt_cache,
    get_prompt,
    list_prompt_ids,
    resolve_prompt,
)


def test_list_prompt_ids_includes_core_prompts() -> None:
    ids = list_prompt_ids()
    assert "agent.system" in ids
    assert "intent.router" in ids
    assert "roles.worker" in ids


def test_get_prompt_returns_stable_content() -> None:
    clear_prompt_cache()
    text = get_prompt("agent.system")
    assert "[CTX-SYSTEM]" in text
    meta = resolve_prompt("agent.system")
    assert meta.version == "v1"
    assert meta.stable is True
    assert len(meta.content_hash) == 12


def test_template_prompt_renders() -> None:
    clear_prompt_cache()
    text = get_prompt("intent.router", enabled_modes_block="- direct\n- plan")
    assert "- direct" in text
    assert "- plan" in text


def test_prompt_render_cache(monkeypatch) -> None:
    clear_prompt_cache()
    first = resolve_prompt("plan.planner")
    second = resolve_prompt("plan.planner")
    assert first.content_hash == second.content_hash
    assert first.content is second.content


def test_version_override_via_env(monkeypatch) -> None:
    clear_prompt_cache()
    monkeypatch.setenv("PROMPT_AGENT_SYSTEM_VERSION", "v1")
    meta = resolve_prompt("agent.system")
    assert meta.version == "v1"
