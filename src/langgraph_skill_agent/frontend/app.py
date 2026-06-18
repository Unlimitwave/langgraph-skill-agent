"""
Streamlit 前端：调用 LangGraph Skill Agent。
会话列表与消息持久化到 var/session_history/*.json

运行:
  pip install -e ".[ui]"
  langgraph-ui
  # 或: streamlit run src/langgraph_skill_agent/frontend/app.py
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st
from langgraph.types import Command

from langgraph_skill_agent.agent_core import build_agent
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

SESSION_HISTORY_DIR = VAR_DIR / "session_history"

# TODO (author:caoyintao): 2026-05-29 待检查整个模块，待测试


# ui渲染，流式刷新 assistant 气泡
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


# 确保目录存在
def _history_dir() -> Path:
    SESSION_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return SESSION_HISTORY_DIR


# 保存会话历史
def _save_session(session_id: str, sess: dict) -> None:
    path = _history_dir() / f"{session_id}.json"
    payload = {
        "title": sess["title"],
        "thread_id": sess["thread_id"],
        "messages": sess["messages"],
        "updated_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 加载会话历史
def _load_sessions_from_disk() -> tuple[dict, str | None]:
    d = _history_dir()
    files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    sessions: dict = {}
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        sid = str(data.get("thread_id") or path.stem)
        sessions[sid] = {
            "title": data.get("title") or "对话",
            "messages": data.get("messages") or [],
            "thread_id": sid,
        }
    default_id = next(iter(sessions)) if sessions else None
    return sessions, default_id


# 缓存 agent 实例
@st.cache_resource
def get_graph():
    return build_agent()


# 初始化 session_state
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
                    "thread_id": tid,
                }
            }
            st.session_state.active_session_id = tid
            _save_session(tid, st.session_state.sessions[tid])


# TODO (author:caoyintao): 2026-06-01 待检查这部分
def _active_session():
    _ensure_sessions()
    return st.session_state.sessions[st.session_state.active_session_id]


def _thread_config_for_active():
    return {"configurable": {"thread_id": _active_session()["thread_id"]}}


# 新建会话
def _new_session():
    tid = str(uuid.uuid4())
    st.session_state.sessions[tid] = {
        "title": "新对话",
        "messages": [],
        "thread_id": tid,
    }
    st.session_state.active_session_id = tid
    _save_session(tid, st.session_state.sessions[tid])


# 删除会话
def _delete_session(sid: str) -> None:
    path = _history_dir() / f"{sid}.json"
    path.unlink(missing_ok=True)
    st.session_state.sessions.pop(sid, None)
    if not st.session_state.sessions:
        tid = str(uuid.uuid4())
        st.session_state.sessions = {
            tid: {
                "title": "新对话",
                "messages": [],
                "thread_id": tid,
            }
        }
        st.session_state.active_session_id = tid
        _save_session(tid, st.session_state.sessions[tid])
        return
    if st.session_state.active_session_id == sid:
        st.session_state.active_session_id = next(iter(st.session_state.sessions.keys()))


# 设置ui页面配置
st.set_page_config(page_title="LangGraph Skill Agent", layout="wide")
st.markdown("# LangGraph Skill Agent")
st.caption(
    "DeepSeek + Skills + RAG + MCP（内置 workspace_exec_python / run_skill_script_shell 白名单）"
)

_ensure_sessions()

# 设置ui sidebar侧边栏
with st.sidebar:
    st.subheader("会话")
    st.caption(f"项目根：`{PROJECT_ROOT}`")
    st.caption(f"持久化：`{SESSION_HISTORY_DIR}`")
    if st.button("➕ 新建会话", use_container_width=True):
        _new_session()
        st.rerun()

    session_ids = list(st.session_state.sessions.keys())
    if st.session_state.active_session_id not in st.session_state.sessions:
        st.session_state.active_session_id = session_ids[0]

    st.markdown("**历史对话**")
    for sid in session_ids:
        s = st.session_state.sessions[sid]
        n = len(s["messages"])
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

# 取当前的active 会话
active = _active_session()
messages = active["messages"]

# 取当前会话的 chat input
_hitl_block = st.session_state.get("hitl_pending")
_hitl_active = isinstance(_hitl_block, dict) and _hitl_block.get("thread_id") == active["thread_id"]
# session 与 checkpointer 不一致时（如审批已完成但未 rerun），自动解除锁定
if _hitl_active:
    _cfg = _thread_config_for_active()
    if get_pending_hitl(get_graph(), _cfg) is None:
        st.session_state.pop("hitl_pending", None)
        st.rerun()
prompt = st.chat_input("输入消息…", disabled=_hitl_active)

prompt_accepted = False
if prompt:
    if _hitl_active:
        st.warning("请先批准或拒绝待处理的工具调用。")
    else:
        if not messages:
            active["title"] = (prompt[:28] + "…") if len(prompt) > 28 else prompt
        messages.append({"role": "user", "content": prompt})
        _save_session(st.session_state.active_session_id, active)
        prompt_accepted = True


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
    cfg: dict,
    user_text: str = "",
    graph_input: Any | None = None,
    text_prefix: str = "",
    tool_results_prefix: list[ToolResult] | None = None,
) -> None:
    active = _active_session()
    messages = active["messages"]
    had_hitl = _active_hitl_pending() is not None

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

    assistant_msg: dict = {"role": "assistant", "content": turn.text}
    if turn.tool_results:
        assistant_msg["tool_results"] = turn.tool_results
    messages.append(assistant_msg)
    _save_session(st.session_state.active_session_id, active)
    # chat_input 在脚本顶部已渲染；审批 resume 完成后需 rerun 才能解除 disabled
    if had_hitl:
        st.rerun()


#
for msg in messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_message_content(msg)
        else:
            st.markdown(msg["content"])

resume_decisions = st.session_state.pop("hitl_resume_decisions", None)
hitl_pending = _active_hitl_pending()

if resume_decisions and hitl_pending:
    graph = get_graph()
    cfg = _thread_config_for_active()
    _run_assistant_and_persist(
        graph=graph,
        cfg=cfg,
        graph_input=Command(resume={"decisions": resume_decisions}),
        text_prefix=hitl_pending.get("text") or "",
        tool_results_prefix=hitl_pending.get("tool_results") or [],
    )
    st.stop()
elif prompt_accepted:
    graph = get_graph()
    cfg = _thread_config_for_active()
    _run_assistant_and_persist(
        graph=graph,
        cfg=cfg,
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
