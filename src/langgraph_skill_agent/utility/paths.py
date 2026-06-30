"""Project root and runtime data directory resolution."""

from __future__ import annotations

import os
import re
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

_USER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def get_project_root() -> Path:
    """Return repository root (parent of ``src/``). Override with ``PROJECT_ROOT`` env."""
    override = os.environ.get("PROJECT_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return PACKAGE_DIR.parent.parent.parent


def _dir_from_env(env_var: str, default: Path) -> Path:
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return default
    path = Path(raw).expanduser()
    root = get_project_root()
    return path if path.is_absolute() else (root / path)


PROJECT_ROOT = get_project_root()
VAR_DIR = PROJECT_ROOT / "var"
AGENT_MEMORY_DIR = _dir_from_env("AGENT_MEMORY_DIR", VAR_DIR / "agent_memory")
RAG_DATA_DIR = _dir_from_env("RAG_DATA_DIR", VAR_DIR / "data")
RAG_STORAGE_DIR = _dir_from_env("RAG_STORAGE_DIR", VAR_DIR / "storage")
CONVERSATION_HISTORY_DIR = _dir_from_env(
    "CONVERSATION_HISTORY_DIR", VAR_DIR / "conversation_history"
)
SKILLS_DIR = _dir_from_env("SKILLS_DIR", PROJECT_ROOT / "skills")
# All per-user sandboxes live under this root (e.g. workspace/default/, workspace/alice/).
WORKSPACES_ROOT = _dir_from_env("WORKSPACE_DIR", PROJECT_ROOT / "workspace")
# Back-compat alias; prefer resolve_agent_workspace() for the active sandbox.
WORKSPACE_DIR = WORKSPACES_ROOT


def get_agent_user_id() -> str:
    """Tenant/user id for workspace isolation. Override with ``AGENT_USER_ID``."""
    raw = os.environ.get("AGENT_USER_ID", "default").strip()
    return raw or "default"


def _validate_user_id(user_id: str) -> str:
    if not _USER_ID_RE.fullmatch(user_id):
        msg = (
            f"invalid AGENT_USER_ID: must match [A-Za-z0-9][A-Za-z0-9._-]{{0,63}}, got {user_id!r}"
        )
        raise ValueError(msg)
    return user_id


def resolve_agent_workspace(user_id: str | None = None) -> Path:
    """Per-user sandbox directory (agent filesystem default root)."""
    uid = _validate_user_id(user_id or get_agent_user_id())
    return WORKSPACES_ROOT / uid


def resolve_user_skills_dir(user_id: str | None = None) -> Path:
    """User-level skills inside the active workspace."""
    return resolve_agent_workspace(user_id) / "skills"


def resolve_agent_memory_dir(user_id: str | None = None) -> Path:
    """Per-user long-term memory blocks (co-located with sandbox)."""
    return resolve_agent_workspace(user_id) / "agent_memory"


def resolve_rag_data_dir(user_id: str | None = None) -> Path:
    """Per-user RAG source documents (co-located with sandbox)."""
    return resolve_agent_workspace(user_id) / "rag_data"


def resolve_rag_storage_dir(user_id: str | None = None) -> Path:
    """Per-user LlamaIndex metadata for RAG (co-located with sandbox)."""
    return resolve_agent_workspace(user_id) / "rag_storage"
