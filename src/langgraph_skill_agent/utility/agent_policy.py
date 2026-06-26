"""Agent filesystem sandbox (CompositeBackend) and layered skill sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.filesystem import FilesystemPermission
from langgraph.prebuilt.tool_node import ToolRuntime

from langgraph_skill_agent.utility.paths import (
    SKILLS_DIR,
    resolve_agent_memory_dir,
    resolve_agent_workspace,
    resolve_user_skills_dir,
)
from langgraph_skill_agent.utility.tenant import AgentContext


@dataclass(frozen=True)
class ResolvedScope:
    """Per-tenant resource boundaries derived from AgentContext."""

    context: AgentContext
    workspace: Path
    memory_dir: Path
    skill_exec: SkillExecContext


def resolve_agent_scope(ctx: AgentContext) -> ResolvedScope:
    ensure_agent_workspace_dirs(ctx.user_id)
    return ResolvedScope(
        context=ctx,
        workspace=resolve_agent_workspace(ctx.user_id),
        memory_dir=resolve_agent_memory_dir(ctx.user_id),
        skill_exec=build_skill_exec_context(ctx.user_id),
    )


# Virtual mount for platform skills (read-only via permissions).
SYSTEM_SKILLS_ROUTE = "/system-skills/"
# User skills directory relative to the per-user workspace root.
USER_SKILLS_SOURCE = "skills"


@dataclass(frozen=True)
class SkillExecContext:
    """Resolve virtual skill script paths to on-disk locations."""

    agent_workspace: Path
    system_skills_dir: Path

    def resolve_script_path(self, path_arg: str) -> tuple[Path | None, str]:
        if not path_arg or "\x00" in path_arg or ".." in path_arg:
            return None, "invalid path token"

        normalized = path_arg.replace("\\", "/")
        if normalized.startswith(SYSTEM_SKILLS_ROUTE):
            rel = normalized[len(SYSTEM_SKILLS_ROUTE) :].lstrip("/")
            if not rel:
                return None, "empty path under system-skills"
            candidate = (self.system_skills_dir / rel).resolve()
            try:
                candidate.relative_to(self.system_skills_dir.resolve())
            except ValueError:
                return None, f"path escapes system skills root: {path_arg!r}"
            return candidate, ""

        user_prefix = f"{USER_SKILLS_SOURCE}/"
        if normalized.startswith(user_prefix):
            rel = normalized[len(user_prefix) :]
        elif normalized.startswith(f"/{user_prefix}"):
            rel = normalized[len(f"/{user_prefix}") :]
        elif normalized.startswith(f"/{USER_SKILLS_SOURCE}/"):
            rel = normalized[len(f"/{USER_SKILLS_SOURCE}/") :]
        else:
            return None, (f"path must start with system-skills/ or skills/ (got {path_arg!r})")

        if not rel:
            return None, "empty path under skills"
        user_skills = self.agent_workspace / USER_SKILLS_SOURCE
        candidate = (user_skills / rel).resolve()
        try:
            candidate.relative_to(user_skills.resolve())
        except ValueError:
            return None, f"path escapes user skills root: {path_arg!r}"
        return candidate, ""


def ensure_agent_workspace_dirs(user_id: str | None = None) -> Path:
    """Create per-user workspace, skills, and memory directories if missing."""
    ws = resolve_agent_workspace(user_id)
    ws.mkdir(parents=True, exist_ok=True)
    resolve_user_skills_dir(user_id).mkdir(parents=True, exist_ok=True)
    resolve_agent_memory_dir(user_id).mkdir(parents=True, exist_ok=True)
    return ws


def build_skill_exec_context(user_id: str | None = None) -> SkillExecContext:
    return SkillExecContext(
        agent_workspace=resolve_agent_workspace(user_id),
        system_skills_dir=SKILLS_DIR,
    )


def build_agent_backend(user_id: str | None = None) -> BackendProtocol:
    """Workspace sandbox + read-only system skills mount."""
    agent_workspace = resolve_agent_workspace(user_id)
    default = FilesystemBackend(root_dir=str(agent_workspace), virtual_mode=True)
    system_skills = FilesystemBackend(root_dir=str(SKILLS_DIR), virtual_mode=True)
    return CompositeBackend(
        default=default,
        routes={SYSTEM_SKILLS_ROUTE: system_skills},
    )


def backend_for_runtime(runtime: ToolRuntime[AgentContext]) -> BackendProtocol:
    """Resolve per-request filesystem sandbox from Runtime Context."""
    scope = resolve_agent_scope(runtime.context)
    return build_agent_backend(scope.context.user_id)


def agent_skill_sources() -> list[str]:
    """System mount first, user skills second (same name → user overrides)."""
    return [SYSTEM_SKILLS_ROUTE, USER_SKILLS_SOURCE]


def agent_filesystem_permissions() -> list[FilesystemPermission]:
    """Writable sandbox root; system skills mount is read-only."""
    system_glob = f"{SYSTEM_SKILLS_ROUTE.rstrip('/')}/**"
    return [
        FilesystemPermission(
            operations=["write"],
            paths=[system_glob],
            mode="deny",
        ),
        FilesystemPermission(
            operations=["write"],
            paths=["/**"],
            mode="allow",
        ),
        FilesystemPermission(
            operations=["read"],
            paths=["/**"],
            mode="allow",
        ),
    ]
