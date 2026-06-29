"""Unit tests for project path resolution."""

import os
from pathlib import Path

import pytest

from langgraph_skill_agent.utility.paths import (
    PROJECT_ROOT,
    get_project_root,
    resolve_agent_workspace,
    resolve_rag_data_dir,
    resolve_rag_storage_dir,
)


def test_project_root_contains_pyproject() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_get_project_root_honors_env_override(monkeypatch, tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    marker.write_text("ok", encoding="utf-8")
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    assert get_project_root() == tmp_path.resolve()


def test_get_project_root_clears_empty_override(monkeypatch) -> None:
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    root = get_project_root()
    assert (root / "pyproject.toml").is_file()
    assert os.environ.get("PROJECT_ROOT", "").strip() == ""


def test_resolve_agent_workspace_explicit_user() -> None:
    ws = resolve_agent_workspace("alice")
    assert ws.name == "alice"
    assert ws.parent == PROJECT_ROOT / "workspace"


def test_resolve_rag_paths_under_user_workspace() -> None:
    data_dir = resolve_rag_data_dir("alice")
    storage_dir = resolve_rag_storage_dir("alice")
    assert data_dir == PROJECT_ROOT / "workspace" / "alice" / "rag_data"
    assert storage_dir == PROJECT_ROOT / "workspace" / "alice" / "rag_storage"


def test_invalid_agent_user_id_rejected() -> None:
    with pytest.raises(ValueError, match="invalid AGENT_USER_ID"):
        resolve_agent_workspace("../evil")
