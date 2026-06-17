"""Unit tests for project path resolution."""

import os
from pathlib import Path

from langgraph_skill_agent.utility.paths import PROJECT_ROOT, get_project_root


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
