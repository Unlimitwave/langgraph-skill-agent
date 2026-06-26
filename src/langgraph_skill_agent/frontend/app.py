"""
Streamlit 前端：调用 LangGraph Skill Agent。

会话索引（标题 / thread_id）→ var/session_history/*.json
聊天气泡展示 → 每轮从 checkpointer get_state() 同步（唯一权威）

运行:
  pip install -e ".[ui]"
  langgraph-ui
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from langgraph_skill_agent.memory import (
    get_checkpointer_label,
    maybe_compact_thread,
    persist_thread_snapshot,
    prepare_thread_for_turn,
    sync_ui_messages_from_checkpointer,
)
from langgraph_skill_agent.utility.agent_runtime import get_agent_runtime
from langgraph_skill_agent.utility.hitl import (
    HitlRequest,
    format_hitl_summary,
    get_pending_hitl,
    hitl_to_dict,
)
from langgraph_skill_agent.utility.paths import PROJECT_ROOT, VAR_DIR
from langgraph_skill_agent.utility.streaming import (
    ToolResult,
    format_status_line,
    run_assistant_turn,
)
from langgraph_skill_agent.utility.tenant import normalize_user_id

# Must run before _ensure_sessions() so AGENT_USER_ID from .env is visible on cold start.
load_dotenv(PROJECT_ROOT / ".env")

SESSION_HISTORY_DIR = VAR_DIR / "session_history"


def _render_assistant_block(
    placeholder,
    *,
    status: str | None,
    text: str,
    tool_results: list[ToolResult],
    cursor: bool,
) -> None:
    tail = "▌" if cursor else ""
    with placeholder.container():
        if status and text.strip():
            st.markdown(f"{status}\n\n{text}{tail}")
        elif status:
            st.markdown(f"{status}{tail}")
        elif text.strip():
            st.markdown(f"{text}{tail}")
        elif cursor:
            st.markdown(f"思考中…{tail}")
        elif not tool_results:
            st.markdown("思考中…")

        for tr in tool_results:
            with st.expander(f"🔧 工具 `{tr['name']}`", expanded=False):
                st.code(tr["content"])


def _history_dir() -> Path:
    SESSION_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return SESSION_HISTORY_DIR


def _save_session_index(session_id: str, sess: dict) -> None:
    """仅持久化会话索引（不含 messages；展示内容以 checkpointer 为准）。"""
    path = _history_dir() / f"{session_id}.json"
    messages = sess.get("messages") or []
    payload = {
        "title": sess["title"],
        "thread_id": sess["thread_id"],
        "user_id": sess.get("user_id") or _resolve_ui_agent_user_id(),
        "message_count": len(messages),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_sessions_from_disk() -> tuple[dict, str | None]:
    d = _history_dir()
    files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    sessions: dict = {}
    current_user = _resolve_ui_agent_user_id()
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        owner = str(data.get("user_id") or "default")
        if owner != current_user:
            continue
        sid = str(data.get("thread_id") or path.stem)
        sessions[sid] = {
            "title": data.get("title") or "对话",
            "messages": [],
            "message_count": int(data.get("message_count") or 0),
            "thread_id": sid,
            "user_id": owner,
        }
    default_id = next(iter(sessions)) if sessions else None
    return sessions, default_id


def _resolve_ui_agent_user_id() -> str:
    """Per-browser tenant id for UI; env AGENT_USER_ID overrides (single-tenant / dev)."""
    env_uid = os.environ.get("AGENT_USER_ID", "").strip()
    if env_uid:
        return normalize_user_id(env_uid)
    if "agent_user_id" not in st.session_state:
        st.session_state.agent_user_id = f"ui-{uuid.uuid4().hex[:12]}"
    return normalize_user_id(st.session_state.agent_user_id)


@st.cache_resource
def get_runtime():
    return get_agent_runtime()


def _cached_graph():
    return get_runtime().graph


def _ensure_sessions():
    if "sessions" not in st.session_state:
        loaded, default_id = _load_sessions_from_disk()
        if loaded:
            st.session_state.sessions = loaded
            st.session_state.active_session_id = default_id
        else:
            tid = str(uuid.uuid4())
            st.session_state.sessions = {
                tid: {
                    "title": "新对话",
                    "messages": [],
                    "message_count": 0,
                    "thread_id": tid,
                    "user_id": _resolve_ui_agent_user_id(),
                }
            }
            st.session_state.active_session_id = tid
            _save_session_index(tid, st.session_state.sessions[tid])


def _active_session():
    _ensure_sessions()
    return st.session_state.sessions[st.session_state.active_session_id]


def _invoke_args_for_active():
    active = _active_session()
    return get_runtime().invoke_kwargs(
        thread_id=active["thread_id"],
        user_id=active.get("user_id") or _resolve_ui_agent_user_id(),
    )


def _sync_active_messages_from_checkpointer(*, skip: bool = False) -> None:
    """从 checkpointer 刷新当前会话 UI 气泡。"""
    if skip:
        return
    active = _active_session()
    sync_ui_messages_from_checkpointer(_cached_graph(), _invoke_args_for_active()["config"], active)
    active["message_count"] = len(active.get("messages") or [])


def _new_session():
    tid = str(uuid.uuid4())
    st.session_state.sessions[tid] = {
        "title": "新对话",
        "messages": [],
        "message_count": 0,
        "thread_id": tid,
        "user_id": _resolve_ui_agent_user_id(),
    }
    st.session_state.active_session_id = tid
    _save_session_index(tid, st.session_state.sessions[tid])


def _delete_session(sid: str) -> None:
    path = _history_dir() / f"{sid}.json"
    path.unlink(missing_ok=True)
    st.session_state.sessions.pop(sid, None)
    if not st.session_state.sessions:
        _new_session()
        return
    if st.session_state.active_session_id == sid:
        st.session_state.active_session_id = next(iter(st.session_state.sessions.keys()))


st.set_page_config(page_title="LangGraph Skill Agent", layout="wide")
st.markdown("# LangGraph Skill Agent")
st.caption(
    "DeepSeek + Skills + RAG + MCP（内置 workspace_exec_python / run_skill_script_shell 白名单）"
)

_ensure_sessions()

with st.sidebar:
    st.subheader("会话")
    st.caption(f"用户沙箱：`{_resolve_ui_agent_user_id()}`")
    st.caption(f"项目根：`{PROJECT_ROOT}`")
    st.caption(f"会话索引：`{SESSION_HISTORY_DIR}`")
    st.caption(f"Checkpointer：{get_checkpointer_label()}")
    if st.button("➕ 新建会话", use_container_width=True):
        _new_session()
        st.rerun()

    session_ids = list(st.session_state.sessions.keys())
    if st.session_state.active_session_id not in st.session_state.sessions:
        st.session_state.active_session_id = session_ids[0]

    st.markdown("**历史对话**")
    for sid in session_ids:
        s = st.session_state.sessions[sid]
        n = s.get("message_count", len(s.get("messages") or []))
        label = f"{s['title']}（{n} 条）"
        is_active = sid == st.session_state.active_session_id
        c1, c2 = st.columns([5, 1])
        with c1:
            if st.button(
                label,
                key=f"open_sess_{sid}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if st.session_state.active_session_id != sid:
                    st.session_state.active_session_id = sid
                    st.rerun()
        with c2:
            if st.button("×", key=f"del_sess_{sid}", help="删除此会话"):
                _delete_session(sid)
                st.rerun()

active = _active_session()

_hitl_block = st.session_state.get("hitl_pending")
_hitl_active = isinstance(_hitl_block, dict) and _hitl_block.get("thread_id") == active["thread_id"]
if _hitl_active:
    _cfg = _invoke_args_for_active()["config"]
    if get_pending_hitl(_cached_graph(), _cfg) is None:
        st.session_state.pop("hitl_pending", None)
        st.rerun()

prompt = st.chat_input("输入消息…", disabled=_hitl_active)

prompt_accepted = False
if prompt:
    if _hitl_active:
        st.warning("请先批准或拒绝待处理的工具调用。")
    else:
        if not active.get("messages"):
            active["title"] = (prompt[:28] + "…") if len(prompt) > 28 else prompt
        active["messages"].append({"role": "user", "content": prompt})
        active["message_count"] = len(active["messages"])
        _save_session_index(st.session_state.active_session_id, active)
        prompt_accepted = True

resume_decisions = st.session_state.get("hitl_resume_decisions")
_skip_sync = prompt_accepted or bool(resume_decisions)
_sync_active_messages_from_checkpointer(skip=_skip_sync)
messages = active["messages"]


def _render_message_content(msg: dict) -> None:
    st.markdown(msg["content"])
    for tr in msg.get("tool_results") or []:
        with st.expander(f"🔧 工具 `{tr['name']}`", expanded=False):
            st.code(tr["content"])


def _active_hitl_pending() -> dict[str, Any] | None:
    pending = st.session_state.get("hitl_pending")
    if not isinstance(pending, dict):
        return None
    if pending.get("thread_id") != _active_session()["thread_id"]:
        return None
    return pending


def _clear_hitl_pending() -> None:
    st.session_state.pop("hitl_pending", None)


def _render_hitl_approval(pending: dict[str, Any]) -> None:
    hitl = HitlRequest(
        action_requests=pending.get("hitl", {}).get("action_requests") or [],
        review_configs=pending.get("hitl", {}).get("review_configs") or [],
    )
    st.warning("以下工具调用需要你的审批后才能继续执行。")
    st.markdown(format_hitl_summary(hitl))
    n_actions = len(hitl.action_requests)
    c1, c2 = st.columns(2)
    if c1.button("✅ 批准", key="hitl_approve", use_container_width=True):
        st.session_state.hitl_resume_decisions = [{"type": "approve"} for _ in range(n_actions)]
        st.rerun()
    if c2.button("❌ 拒绝", key="hitl_reject", use_container_width=True):
        st.session_state.hitl_resume_decisions = [
            {
                "type": "reject",
                "message": "用户拒绝了该工具调用，请勿重试除非用户明确要求。",
            }
            for _ in range(n_actions)
        ]
        st.rerun()


def _run_assistant_and_persist(
    *,
    graph,
    invoke: dict,
    user_text: str = "",
    graph_input: Any | None = None,
    text_prefix: str = "",
    tool_results_prefix: list[ToolResult] | None = None,
) -> None:
    active = _active_session()
    had_hitl = _active_hitl_pending() is not None
    cfg = invoke["config"]
    context = invoke["context"]

    prepare_thread_for_turn(
        graph,
        cfg,
        compact_fn=maybe_compact_thread,
    )

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("思考中…")

        def _on_update(
            *,
            status: str | None,
            text: str,
            tool_results: list[ToolResult],
            cursor: bool,
        ) -> None:
            _render_assistant_block(
                placeholder,
                status=status,
                text=text,
                tool_results=tool_results,
                cursor=cursor,
            )

        turn = asyncio.run(
            run_assistant_turn(
                graph,
                user_text=user_text,
                graph_input=graph_input,
                config=cfg,
                context=context,
                on_update=_on_update,
                decide=None,
                text_prefix=text_prefix,
                tool_results_prefix=tool_results_prefix,
            )
        )

        if turn.pending_hitl is not None:
            st.session_state.hitl_pending = {
                "thread_id": active["thread_id"],
                "text": turn.text,
                "tool_results": turn.tool_results,
                "hitl": hitl_to_dict(turn.pending_hitl),
            }
            st.rerun()
            return

        _clear_hitl_pending()
        if not turn.text.strip() and not turn.tool_results:
            status = format_status_line(
                pending_tool_names=[],
                tools_node_running=False,
                agent_node_running=False,
                has_text=False,
            )
            if not status:
                placeholder.markdown("（无回复）")

    sync_ui_messages_from_checkpointer(graph, cfg, active)
    active["message_count"] = len(active["messages"])
    _save_session_index(st.session_state.active_session_id, active)
    persist_thread_snapshot(graph, cfg)
    if had_hitl:
        st.rerun()


for msg in messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_message_content(msg)
        else:
            st.markdown(msg["content"])

resume_decisions = st.session_state.pop("hitl_resume_decisions", None)
hitl_pending = _active_hitl_pending()

if resume_decisions and hitl_pending:
    graph = _cached_graph()
    invoke = _invoke_args_for_active()
    _run_assistant_and_persist(
        graph=graph,
        invoke=invoke,
        graph_input=Command(resume={"decisions": resume_decisions}),
        text_prefix=hitl_pending.get("text") or "",
        tool_results_prefix=hitl_pending.get("tool_results") or [],
    )
    st.stop()
elif prompt_accepted:
    graph = _cached_graph()
    invoke = _invoke_args_for_active()
    _run_assistant_and_persist(
        graph=graph,
        invoke=invoke,
        user_text=prompt,
    )
    st.stop()
elif _active_hitl_pending():
    hitl_pending = _active_hitl_pending()
    with st.chat_message("assistant"):
        pending_ph = st.empty()
        _render_assistant_block(
            pending_ph,
            status="⏸ **等待审批**",
            text=hitl_pending.get("text") or "",
            tool_results=hitl_pending.get("tool_results") or [],
            cursor=False,
        )
        _render_hitl_approval(hitl_pending)
