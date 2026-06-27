"""Streamlit：plan / supervisor 编排任务展示与 HITL 暂停恢复。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import streamlit as st

from langgraph_skill_agent.agent_core import plan_routing_enabled
from langgraph_skill_agent.intent_router import intent_routing_enabled
from langgraph_skill_agent.multi_agent.config import multi_agent_routing_enabled
from langgraph_skill_agent.multi_agent.interactive import (
    drive_supervisor_task,
    hitl_pending_to_dict,
)
from langgraph_skill_agent.multi_agent.roles import role_avatar, role_caption
from langgraph_skill_agent.plan_execute.interactive import drive_plan_task
from langgraph_skill_agent.utility.hitl import HitlRequest, format_hitl_summary
from langgraph_skill_agent.utility.orchestration_callbacks import (
    OrchestrationCallbacks,
    OrchestrationPaused,
)


@dataclass
class UiOrchestrationBuffer:
    status: str = ""
    current_role: str = "plan"
    step_label: str = ""
    step_text: str = ""
    steps: list[dict[str, str]] = field(default_factory=list)
    final: str = ""


def ui_routing_enabled() -> bool:
    return multi_agent_routing_enabled() or plan_routing_enabled()


def _routing_caption() -> str:
    parts = []
    if multi_agent_routing_enabled():
        parts.append("Supervisor")
    if plan_routing_enabled():
        parts.append("Plan")
    if not parts:
        return "编排路由：关闭（仅直连 Deep Agent）"
    mode_label = "Intent 智能路由" if intent_routing_enabled() else "固定模式（Intent 关闭）"
    return f"编排路由：已启用 {' / '.join(parts)} · {mode_label}"


def _clean_step_label(label: str) -> str:
    return re.sub(r"\*\*", "", label).strip()


def _render_agent_bubble(
    role: str,
    label: str,
    text: str,
    *,
    cursor: bool = False,
) -> None:
    with st.chat_message("assistant", avatar=role_avatar(role)):
        st.caption(role_caption(role))
        clean = _clean_step_label(label)
        if clean:
            st.markdown(f"**{clean}**")
        body = (text or "").strip()
        if body:
            st.markdown(f"{body}{'▌' if cursor else ''}")
        elif cursor:
            st.markdown("▌")


def _render_orchestration_bubbles(buf: UiOrchestrationBuffer, *, cursor: bool = False) -> None:
    if buf.status.strip():
        with st.chat_message("assistant", avatar=role_avatar("supervisor")):
            st.caption(role_caption("supervisor"))
            st.markdown(buf.status)
    for step in buf.steps:
        _render_agent_bubble(
            step.get("role") or "plan",
            step.get("label") or "步骤",
            step.get("text") or "",
        )
    if buf.step_label:
        _render_agent_bubble(
            buf.current_role or "plan",
            buf.step_label,
            buf.step_text,
            cursor=cursor,
        )
    if buf.final.strip():
        with st.chat_message("assistant", avatar=role_avatar("supervisor")):
            st.caption(f"{role_caption('supervisor')} · 汇总")
            st.markdown(buf.final)


def render_orchestration_message(msg: dict) -> None:
    """历史消息：按智能体分气泡展示编排结果。"""
    orch = msg.get("orchestration") or {}
    steps = orch.get("steps")
    if isinstance(steps, list) and steps:
        buf = UiOrchestrationBuffer(
            status=str(orch.get("status") or ""),
            steps=[dict(s) for s in steps if isinstance(s, dict)],
            final=str(orch.get("final") or orch.get("final_answer") or ""),
        )
        if not buf.final.strip():
            final_from_content = (msg.get("content") or "").strip()
            if final_from_content and not buf.steps:
                buf.final = final_from_content
        _render_orchestration_bubbles(buf)
        return
    st.markdown(msg.get("content") or "")


def _append_assistant_message(
    active: dict, buf: UiOrchestrationBuffer, *, mode: str, macro_thread_id: str
) -> None:
    final_text = _compose_final_markdown(buf, mode, {"final_answer": buf.final})
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": final_text,
        "orchestration": {
            "mode": mode,
            "macro_thread_id": macro_thread_id,
            "status": buf.status,
            "steps": list(buf.steps),
            "final": buf.final,
        },
    }
    active.setdefault("messages", []).append(msg)
    active["message_count"] = len(active["messages"])


def _build_callbacks(buf: UiOrchestrationBuffer) -> tuple[OrchestrationCallbacks, Any]:
    slot = st.empty()

    def _redraw(*, cursor: bool = False) -> None:
        with slot.container():
            _render_orchestration_bubbles(buf, cursor=cursor)

    def on_status(msg: str) -> None:
        buf.status = msg
        _redraw()

    def on_step_start(_step_id: str, label: str, role: str = "plan") -> None:
        if buf.step_label and (buf.step_text or buf.step_label):
            buf.steps.append(
                {
                    "role": buf.current_role or "plan",
                    "label": buf.step_label,
                    "text": buf.step_text,
                }
            )
        buf.current_role = role
        buf.step_label = label
        buf.step_text = ""
        _redraw(cursor=True)

    def on_token(token: str) -> None:
        buf.step_text += token
        _redraw(cursor=True)

    def on_step_done(_step_id: str, summary: str) -> None:
        if not buf.step_text.strip():
            buf.step_text = summary
        buf.steps.append(
            {
                "role": buf.current_role or "plan",
                "label": buf.step_label,
                "text": buf.step_text,
            }
        )
        buf.step_label = ""
        buf.step_text = ""
        _redraw()

    def on_final(text: str) -> None:
        buf.final = text
        _redraw()

    callbacks = OrchestrationCallbacks(
        on_status=on_status,
        on_step_start=on_step_start,
        on_token=on_token,
        on_step_done=on_step_done,
        on_final=on_final,
    )
    return callbacks, slot


def _paused_from_session(data: dict) -> OrchestrationPaused:
    return OrchestrationPaused(
        mode=str(data.get("mode") or ""),
        macro_thread_id=str(data.get("macro_thread_id") or ""),
        user_goal=str(data.get("user_goal") or ""),
        payload=dict(data.get("payload") or {}),
    )


def run_orchestration_for_ui(
    *,
    mode: str,
    user_goal: str,
    macro_thread_id: str,
    active: dict,
    session_id: str,
    resume: OrchestrationPaused | None = None,
    hitl_decisions: list[dict[str, Any]] | None = None,
) -> None:
    buf = UiOrchestrationBuffer()
    callbacks, slot = _build_callbacks(buf)

    if mode == "supervisor":
        result = drive_supervisor_task(
            user_goal,
            macro_thread_id=macro_thread_id,
            callbacks=callbacks,
            resume=resume,
            hitl_decisions=hitl_decisions,
        )
    else:
        result = drive_plan_task(
            user_goal,
            macro_thread_id=macro_thread_id,
            callbacks=callbacks,
            resume=resume,
            hitl_decisions=hitl_decisions,
        )

    if isinstance(result, OrchestrationPaused):
        pending_view = hitl_pending_to_dict(result)
        if mode == "plan":
            pending_view = _plan_hitl_view(result)
        st.session_state["orch_pending"] = {
            "thread_id": session_id,
            "mode": result.mode,
            "macro_thread_id": result.macro_thread_id,
            "user_goal": result.user_goal,
            "payload": result.payload,
            "view": pending_view,
            "buffer": _buffer_to_dict(buf),
        }
        with slot.container():
            _render_orchestration_bubbles(buf)
            role = pending_view.get("step_role") or buf.current_role or "plan"
            with st.chat_message("assistant", avatar=role_avatar(role)):
                st.caption(f"{role_caption(role)} · 等待审批")
                prefix = pending_view.get("text_prefix") or ""
                if prefix:
                    st.markdown(prefix)
        st.warning("编排任务暂停：以下工具调用需要审批。")
        st.markdown(format_hitl_summary(_hitl_from_view(pending_view)))
        st.rerun()
        return

    if mode == "supervisor" and isinstance(result, dict):
        final = (result.get("final_answer") or "").strip()
        if final and final not in buf.final:
            buf.final = final

    with slot.container():
        _render_orchestration_bubbles(buf)
    _append_assistant_message(
        active,
        buf,
        mode=mode,
        macro_thread_id=macro_thread_id,
    )
    clear_orch_pending()


def _buffer_to_dict(buf: UiOrchestrationBuffer) -> dict[str, Any]:
    return {
        "status": buf.status,
        "current_role": buf.current_role,
        "step_label": buf.step_label,
        "step_text": buf.step_text,
        "steps": list(buf.steps),
        "final": buf.final,
    }


def _buffer_from_dict(data: dict[str, Any]) -> UiOrchestrationBuffer:
    return UiOrchestrationBuffer(
        status=str(data.get("status") or ""),
        current_role=str(data.get("current_role") or "plan"),
        step_label=str(data.get("step_label") or ""),
        step_text=str(data.get("step_text") or ""),
        steps=[dict(s) for s in (data.get("steps") or []) if isinstance(s, dict)],
        final=str(data.get("final") or ""),
    )


def _plan_hitl_view(paused: OrchestrationPaused) -> dict[str, Any]:
    pending = paused.payload.get("step_pending") or {}
    return {
        "mode": paused.mode,
        "macro_thread_id": paused.macro_thread_id,
        "user_goal": paused.user_goal,
        "payload": paused.payload,
        "hitl": pending.get("hitl"),
        "text_prefix": pending.get("text_prefix") or "",
        "tool_results": pending.get("tool_results") or [],
        "step_label": pending.get("step_label") or "Plan",
        "step_role": "plan",
    }


def _hitl_from_view(view: dict[str, Any]) -> HitlRequest:
    hitl = view.get("hitl") or {}
    return HitlRequest(
        action_requests=hitl.get("action_requests") or [],
        review_configs=hitl.get("review_configs") or [],
    )


def format_status_with_partial(buf: UiOrchestrationBuffer, view: dict[str, Any]) -> str:
    """Legacy markdown fallback for callers that still expect plain text."""
    lines: list[str] = []
    if buf.status:
        lines.append(buf.status)
    for step in buf.steps:
        lines.append(f"\n### {step.get('label', '步骤')}")
        if step.get("text"):
            lines.append(step["text"])
    if buf.step_label:
        lines.append(f"\n### {buf.step_label}")
        if buf.step_text:
            lines.append(buf.step_text)
    prefix = view.get("text_prefix") or ""
    if prefix:
        lines.append(prefix)
    if buf.final:
        lines.append("\n---\n")
        lines.append(buf.final)
    return "\n".join(lines).strip()


def _compose_final_markdown(buf: UiOrchestrationBuffer, mode: str, result: Any) -> str:
    lines: list[str] = []
    if buf.status:
        lines.append(buf.status)
    for step in buf.steps:
        lines.append(f"\n### {step.get('label', '步骤')}")
        if step.get("text"):
            lines.append(step["text"])
    if buf.final:
        lines.append("\n---\n")
        lines.append(buf.final)
    body = "\n".join(lines).strip()
    if mode == "supervisor" and isinstance(result, dict):
        final = (result.get("final_answer") or buf.final or "").strip()
        if final and final not in body:
            body = f"{body}\n\n---\n\n{final}".strip() if body else final
    return body or "（编排任务已完成，无文本输出）"


def active_orch_pending(session_id: str) -> dict[str, Any] | None:
    pending = st.session_state.get("orch_pending")
    if not isinstance(pending, dict):
        return None
    if pending.get("thread_id") != session_id:
        return None
    return pending


def clear_orch_pending() -> None:
    st.session_state.pop("orch_pending", None)


def render_orch_pending_bubbles(pending: dict[str, Any]) -> None:
    """编排 HITL 暂停态：恢复已完成的智能体气泡。"""
    buf = _buffer_from_dict(pending.get("buffer") or {})
    view = pending.get("view") or {}
    _render_orchestration_bubbles(buf)
    role = view.get("step_role") or buf.current_role or "plan"
    with st.chat_message("assistant", avatar=role_avatar(role)):
        st.caption(f"{role_caption(role)} · 等待审批")
        prefix = view.get("text_prefix") or buf.step_text or ""
        if prefix:
            st.markdown(prefix)
        elif view.get("step_label"):
            st.markdown(f"**{_clean_step_label(view['step_label'])}**")


def resume_orchestration_after_hitl(
    *,
    active: dict,
    session_id: str,
    decisions: list[dict[str, Any]],
) -> None:
    pending = active_orch_pending(session_id)
    if not pending:
        return
    paused = _paused_from_session(pending)
    run_orchestration_for_ui(
        mode=str(pending.get("mode") or "supervisor"),
        user_goal=str(pending.get("user_goal") or ""),
        macro_thread_id=str(pending.get("macro_thread_id") or ""),
        active=active,
        session_id=session_id,
        resume=paused,
        hitl_decisions=decisions,
    )
    clear_orch_pending()
