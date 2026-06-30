"""Unit tests for application-layer AgentRuntime."""

from langgraph_skill_agent.utility.agent_policy import resolve_agent_scope
from langgraph_skill_agent.utility.tenant import AgentContext


def test_resolve_scope_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "langgraph_skill_agent.utility.paths.WORKSPACES_ROOT",
        tmp_path / "workspace",
    )
    monkeypatch.setattr(
        "langgraph_skill_agent.utility.paths.SKILLS_DIR",
        tmp_path / "skills",
    )
    (tmp_path / "skills").mkdir()
    ctx = AgentContext(user_id="alice", tenant_id="acme")
    scope = resolve_agent_scope(ctx)
    assert scope.context == ctx
    assert scope.workspace == tmp_path / "workspace" / "alice"
    assert scope.memory_dir == tmp_path / "workspace" / "alice" / "agent_memory"
    assert scope.skill_exec.agent_workspace == scope.workspace
