"""plan_execute 逐步驱动（支持 Web UI 流式与 HITL 暂停）。"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from langgraph.types import Command

from langgraph_skill_agent.agent_core import build_agent
from langgraph_skill_agent.plan_execute.core import PlanExecuteState, planner_node
from langgraph_skill_agent.utility.agent_runtime import get_agent_runtime
from langgraph_skill_agent.utility.logging_config import configure_logging
from langgraph_skill_agent.utility.orchestration_callbacks import (
    OrchestrationCallbacks,
    OrchestrationPaused,
)
from langgraph_skill_agent.utility.streaming import ToolResult, run_assistant_turn
from langgraph_skill_agent.utility.tenant import AgentContext, normalize_user_id

logger = logging.getLogger(__name__)


def _merge_state(state: PlanExecuteState, patch: dict) -> PlanExecuteState:
    merged = dict(state)
    merged.update(patch)
    return merged  # type: ignore[return-value]


def _build_step_prompt(state: PlanExecuteState) -> tuple[str, str, str] | None:
    tid = state.get("current_todo_id")
    todos = list(state.get("todos") or [])
    macro = (state.get("macro_thread_id") or "macro-default").strip()
    goal = (state.get("user_goal") or "").strip()
    if not tid:
        return None
    current_title = ""
    for t in todos:
        if t["id"] == tid and t["status"] == "pending":
            current_title = t["title"]
            break
    if not current_title:
        return None
    lines = [
        f"【用户总目标】\n{goal}",
        "",
        "【完整任务列表（由规划器生成；请勿擅自跳过未指定的项）】",
    ]
    for t in todos:
        tag = "→ 本回合只做这一项" if t["id"] == tid else ""
        st = "✓" if t["status"] == "done" else "○"
        lines.append(f"  {st} [{t['id']}] {t['title']} {tag}".rstrip())
    lines.extend(
        [
            "",
            f"【本回合要求】只完成步骤 [{tid}]：{current_title}。",
            "未完成项留到后续回合；不要在本回合提前执行后续项。",
            "可使用工具与技能完成本步。",
        ]
    )
    return tid, macro, "\n".join(lines)


def drive_plan_task(
    user_goal: str,
    *,
    macro_thread_id: str | None = None,
    callbacks: OrchestrationCallbacks | None = None,
    resume: OrchestrationPaused | None = None,
    hitl_decisions: list[dict[str, Any]] | None = None,
) -> PlanExecuteState | OrchestrationPaused:
    configure_logging()
    cb = callbacks or OrchestrationCallbacks()
    agent = build_agent()

    if resume:
        macro = resume.macro_thread_id
        state: PlanExecuteState = dict(resume.payload.get("state") or {})  # type: ignore[assignment]
        goal = (resume.user_goal or user_goal or state.get("user_goal") or "").strip()
    else:
        macro = (macro_thread_id or "").strip() or f"macro-{uuid.uuid4().hex[:8]}"
        goal = user_goal.strip()
        state = {
            "user_goal": goal,
            "macro_thread_id": macro,
            "todos": [],
            "current_todo_id": None,
        }
        if cb.on_status:
            cb.on_status("📋 任务规划中…")
        state = _merge_state(state, planner_node(state))

    if resume and resume.payload.get("phase") == "step_hitl" and hitl_decisions is not None:
        pending = resume.payload.get("step_pending") or {}
        ctx_raw = pending.get("context") or {}
        context = AgentContext(
            user_id=str(ctx_raw.get("user_id") or "default"),
            tenant_id=str(ctx_raw.get("tenant_id") or "default"),
        )
        step_tokens: list[str] = []

        def _on_token(t: str, acc: list[str] = step_tokens) -> None:
            acc.append(t)
            if cb.on_token:
                cb.on_token(t)

        def _on_update(
            *,
            status: str | None,
            text: str,
            cursor: bool,
            tool_results: list[ToolResult] | None = None,
            acc: list[str] = step_tokens,
        ) -> None:
            del status, tool_results, cursor
            delta = text[len("".join(acc)) :]
            if delta:
                _on_token(delta)

        turn = asyncio.run(
            run_assistant_turn(
                agent,
                graph_input=Command(resume={"decisions": hitl_decisions}),
                config=pending["config"],
                context=context,
                on_update=_on_update,
                decide=None,
                text_prefix=pending.get("text_prefix") or "",
                tool_results_prefix=pending.get("tool_results") or [],
            )
        )
        if turn.pending_hitl is not None:
            return OrchestrationPaused(
                mode="plan",
                macro_thread_id=macro,
                user_goal=goal,
                payload={
                    **resume.payload,
                    "step_pending": {
                        **pending,
                        "text_prefix": turn.text,
                        "tool_results": turn.tool_results,
                        "hitl": {
                            "action_requests": turn.pending_hitl.action_requests,
                            "review_configs": turn.pending_hitl.review_configs,
                        },
                    },
                },
            )
        tid = pending["todo_id"]
        if cb.on_step_done:
            cb.on_step_done(tid, turn.text.strip())
        state = _merge_state(state, _mark_todo_done(state, tid))

    while state.get("current_todo_id"):
        step = _build_step_prompt(state)
        if step is None:
            nxt = _next_pending_todo(state)
            state = _merge_state(state, {"current_todo_id": nxt})
            if not nxt:
                break
            continue
        tid, macro, prompt = step
        step_title = ""
        for t in state.get("todos") or []:
            if t.get("id") == tid:
                step_title = str(t.get("title") or "")
                break
        if cb.on_step_start:
            cb.on_step_start(tid, f"[{tid}] {step_title}", "plan")

        step_invoke = get_agent_runtime().invoke_kwargs(
            thread_id=f"{macro}:todo:{tid}",
            user_id=normalize_user_id(),
        )
        ctx = step_invoke["context"]
        step_tokens: list[str] = []

        def _on_token(t: str, acc: list[str] = step_tokens) -> None:
            acc.append(t)
            if cb.on_token:
                cb.on_token(t)

        def _on_update(
            *,
            status: str | None,
            text: str,
            cursor: bool,
            tool_results: list[ToolResult] | None = None,
            acc: list[str] = step_tokens,
        ) -> None:
            del status, tool_results, cursor
            delta = text[len("".join(acc)) :]
            if delta:
                _on_token(delta)

        turn = asyncio.run(
            run_assistant_turn(
                agent,
                user_text=prompt,
                config=step_invoke["config"],
                context=ctx,
                on_update=_on_update,
                decide=None,
            )
        )
        if turn.pending_hitl is not None:
            return OrchestrationPaused(
                mode="plan",
                macro_thread_id=macro,
                user_goal=goal,
                payload={
                    "state": state,
                    "phase": "step_hitl",
                    "step_pending": {
                        "todo_id": tid,
                        "config": step_invoke["config"],
                        "context": {"user_id": ctx.user_id, "tenant_id": ctx.tenant_id},
                        "text_prefix": turn.text,
                        "tool_results": turn.tool_results,
                        "hitl": {
                            "action_requests": turn.pending_hitl.action_requests,
                            "review_configs": turn.pending_hitl.review_configs,
                        },
                        "step_label": f"Plan · [{tid}]",
                    },
                },
            )
        if cb.on_step_done:
            cb.on_step_done(tid, turn.text.strip())
        state = _merge_state(state, _mark_todo_done(state, tid))

    if cb.on_final:
        lines = ["**Plan 执行完成**", ""]
        for t in state.get("todos") or []:
            lines.append(f"- [{t['id']}] {t['status']}: {t['title']}")
        cb.on_final("\n".join(lines))
    return state


def _next_pending_todo(state: PlanExecuteState) -> str | None:
    for t in state.get("todos") or []:
        if t.get("status") == "pending":
            return t["id"]
    return None


def _mark_todo_done(state: PlanExecuteState, tid: str) -> dict:
    todos = list(state.get("todos") or [])
    new_todos = []
    for t in todos:
        if t["id"] == tid:
            new_todos.append({"id": t["id"], "title": t["title"], "status": "done"})
        else:
            new_todos.append(dict(t))
    nxt = next((t["id"] for t in new_todos if t["status"] == "pending"), None)
    return {"todos": new_todos, "current_todo_id": nxt}
