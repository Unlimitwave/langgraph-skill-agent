"""Per-tenant identity, Runtime Context, and checkpointer thread namespacing."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

THREAD_NS_SEP = "__"


@dataclass(frozen=True)
class AgentContext:
    """Runtime identity injected at invoke/stream time (separate from session thread_id)."""

    user_id: str
    tenant_id: str = "default"


def normalize_user_id(user_id: str | None = None) -> str:
    from langgraph_skill_agent.utility.paths import _validate_user_id, get_agent_user_id

    return _validate_user_id(user_id or get_agent_user_id())


def normalize_tenant_id(tenant_id: str | None = None) -> str:
    raw = (tenant_id or os.environ.get("AGENT_TENANT_ID") or "default").strip()
    return raw or "default"


def namespaced_thread_id(user_id: str, thread_id: str) -> str:
    """Prefix thread_id with user_id so checkpoints cannot collide across tenants."""
    uid = normalize_user_id(user_id)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in thread_id)[:200]
    return f"{uid}{THREAD_NS_SEP}{safe}"


def bare_thread_id(namespaced: str) -> str:
    if THREAD_NS_SEP in namespaced:
        return namespaced.split(THREAD_NS_SEP, 1)[1]
    return namespaced


def build_agent_context(
    *,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> AgentContext:
    return AgentContext(
        user_id=normalize_user_id(user_id),
        tenant_id=normalize_tenant_id(tenant_id),
    )


def build_agent_config(
    *,
    thread_id: str,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> dict:
    """Session config: namespaced thread_id for checkpointer isolation."""
    uid = normalize_user_id(user_id)
    tid = normalize_tenant_id(tenant_id)
    return {
        "configurable": {
            "thread_id": namespaced_thread_id(uid, thread_id),
            "user_id": uid,
            "tenant_id": tid,
        }
    }


def build_invoke_kwargs(
    *,
    thread_id: str,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Session (config) + identity (context) for graph.invoke/stream."""
    ctx = build_agent_context(user_id=user_id, tenant_id=tenant_id)
    return {
        "config": build_agent_config(
            thread_id=thread_id,
            user_id=ctx.user_id,
            tenant_id=ctx.tenant_id,
        ),
        "context": ctx,
    }


def user_id_from_config(config: dict | None) -> str:
    cfg = (config or {}).get("configurable") or {}
    raw_uid = str(cfg.get("user_id") or "").strip()
    if raw_uid:
        return normalize_user_id(raw_uid)
    raw_tid = str(cfg.get("thread_id") or "default")
    if THREAD_NS_SEP in raw_tid:
        return normalize_user_id(raw_tid.split(THREAD_NS_SEP, 1)[0])
    return normalize_user_id(None)


__all__ = [
    "THREAD_NS_SEP",
    "AgentContext",
    "bare_thread_id",
    "build_agent_config",
    "build_agent_context",
    "build_invoke_kwargs",
    "namespaced_thread_id",
    "normalize_tenant_id",
    "normalize_user_id",
    "user_id_from_config",
]
