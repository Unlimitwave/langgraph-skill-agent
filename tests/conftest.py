"""Pytest defaults: unit tests use in-memory checkpointer (no Postgres required)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _memory_checkpointer_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHECKPOINT_BACKEND", "memory")
    monkeypatch.setenv("HYDRATE_ENABLED", "0")
    monkeypatch.setenv("CHECKPOINT_EXPORT_SNAPSHOT", "0")
    import langgraph_skill_agent.memory.session_store as session_store

    session_store._CHECKPOINTER_SINGLETON = None
    yield
    session_store._CHECKPOINTER_SINGLETON = None
