"""Project root and runtime data directory resolution."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


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
