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

import streamlit as st

from langgraph_skill_agent.agent_core import build_agent
from langgraph_skill_agent.utility.paths import PROJECT_ROOT, VAR_DIR
from langgraph_skill_agent.utility.streaming import format_status_line, stream_assistant_text

SESSION_HISTORY_DIR = VAR_DIR / "session_history"

# TODO (author:caoyintao): 2026-05-29 待检查整个模块，待测试


# ui渲染，流式刷新 assistant 气泡
def _render_assistant_block(
    placeholder,
    *,
    status: str | None,
    text: str,
    cursor: bool,
) -> None:
    tail = "▌" if cursor else ""
    if status and text.strip():
        placeholder.markdown(f"{status}\n\n{text}{tail}")
    elif status:
        placeholder.markdown(f"{status}{tail}")
    elif text.strip():
        placeholder.markdown(f"{text}{tail}")
    elif cursor:
        placeholder.markdown(f"思考中…{tail}")
    else:
        placeholder.markdown("思考中…")


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
st.caption("DeepSeek + Skills + RAG + MCP（内置 workspace_exec / run_skill_script 白名单）")

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
prompt = st.chat_input("输入消息…")

if prompt:
    if not messages:
        # 如果当前会话没有消息，则设置会话标题为 prompt 的前28个字符
        active["title"] = (prompt[:28] + "…") if len(prompt) > 28 else prompt
    messages.append({"role": "user", "content": prompt})
    # 保存用户刚输入的prompt 到会话历史
    _save_session(st.session_state.active_session_id, active)

#
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt:
    # 获取 agent 实例
    graph = get_graph()

    # 获取当前会话的配置
    cfg = _thread_config_for_active()

    # 渲染 assistant 气泡
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("思考中…")

        # 定义一个回调函数，用于更新 assistant 气泡
        def _on_update(*, status: str | None, text: str, cursor: bool) -> None:
            _render_assistant_block(
                placeholder,
                status=status,
                text=text,
                cursor=cursor,
            )

        # 流式输出 assistant 回复
        assistant_text = asyncio.run(
            stream_assistant_text(
                graph,
                user_text=prompt,
                config=cfg,
                on_update=_on_update,
            )
        )
        # 如果 assistant 回复为空，则显示（无回复）
        if not assistant_text.strip():
            status = format_status_line(
                pending_tool_names=[],
                tools_node_running=False,
                agent_node_running=False,
                has_text=False,
            )
            if not status:
                placeholder.markdown("（无回复）")

    # 将 assistant 回复添加到会话历史
    messages.append({"role": "assistant", "content": assistant_text})
    # 保存会话历史
    _save_session(st.session_state.active_session_id, active)
