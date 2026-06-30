"""Unit tests for agent filesystem policy and layered skills."""

from deepagents.middleware.filesystem import FilesystemPermission

from langgraph_skill_agent.utility.agent_policy import (
    SYSTEM_SKILLS_ROUTE,
    USER_SKILLS_SOURCE,
    agent_filesystem_permissions,
    agent_skill_sources,
    build_agent_backend,
)
from langgraph_skill_agent.utility.paths import (
    PROJECT_ROOT,
    SKILLS_DIR,
    WORKSPACES_ROOT,
    get_agent_user_id,
    resolve_agent_memory_dir,
    resolve_agent_workspace,
    resolve_user_skills_dir,
)


def test_per_user_workspace_defaults(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_USER_ID", "default")
    assert WORKSPACES_ROOT == PROJECT_ROOT / "workspace"
    assert get_agent_user_id() == "default"
    assert resolve_agent_workspace() == WORKSPACES_ROOT / "default"
    assert resolve_user_skills_dir() == WORKSPACES_ROOT / "default" / "skills"
    assert resolve_agent_memory_dir() == WORKSPACES_ROOT / "default" / "agent_memory"
    assert SKILLS_DIR == PROJECT_ROOT / "skills"


def test_agent_skill_sources_system_mount_then_user() -> None:
    assert agent_skill_sources() == [SYSTEM_SKILLS_ROUTE, USER_SKILLS_SOURCE]


def test_build_agent_backend_is_composite() -> None:
    backend = build_agent_backend()
    assert type(backend).__name__ == "CompositeBackend"


def test_agent_filesystem_permissions_sandbox() -> None:
    rules = agent_filesystem_permissions()
    assert len(rules) == 3
    assert rules[0] == FilesystemPermission(
        operations=["write"],
        paths=["/system-skills/**"],
        mode="deny",
    )
    assert rules[1] == FilesystemPermission(
        operations=["write"],
        paths=["/**"],
        mode="allow",
    )
    assert rules[2] == FilesystemPermission(
        operations=["read"],
        paths=["/**"],
        mode="allow",
    )
