"""Supervisor 逐步驱动（支持 Web UI 流式与 HITL 暂停）。"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from langgraph.types import Command

from langgraph_skill_agent.multi_agent.config import supervisor_max_review_retries
from langgraph_skill_agent.multi_agent.handoff import HandoffPayload, TaskRecord
from langgraph_skill_agent.multi_agent.roles import AgentRole
from langgraph_skill_agent.multi_agent.specialists import get_specialist_graph
from langgraph_skill_agent.multi_agent.supervisor import (
    SupervisorState,
    _invoke_specialist,
    _pick_next_runnable_task,
    _prior_artifacts_for_task,
    _requeue_worker_after_failed_review,
    planner_node,
    specialist_node,
    synthesizer_node,
)
from langgraph_skill_agent.utility.logging_config import configure_logging
from langgraph_skill_agent.utility.orchestration_callbacks import (
    OrchestrationCallbacks,
    OrchestrationPaused,
)
from langgraph_skill_agent.utility.streaming import ToolResult, run_assistant_turn
from langgraph_skill_agent.utility.tenant import AgentContext

logger = logging.getLogger(__name__)


def _merge_state(state: SupervisorState, patch: dict) -> SupervisorState:
    merged = dict(state)
    merged.update(patch)
    return merged  # type: ignore[return-value]


def _tasks_from_state(state: SupervisorState) -> list[TaskRecord]:
    return [
        TaskRecord.model_validate(t) if isinstance(t, dict) else t
        for t in (state.get("tasks") or [])
    ]


def drive_supervisor_task(
    user_goal: str,
    *,
    macro_thread_id: str | None = None,
    callbacks: OrchestrationCallbacks | None = None,
    resume: OrchestrationPaused | None = None,
    hitl_decisions: list[dict[str, Any]] | None = None,
) -> SupervisorState | OrchestrationPaused:
    """逐步执行 Supervisor；遇 HITL 返回 OrchestrationPaused 供 UI 跨 rerun 恢复。"""
    configure_logging()
    cb = callbacks or OrchestrationCallbacks()
    macro = (macro_thread_id or "").strip()
    if resume:
        macro = resume.macro_thread_id
        state: SupervisorState = dict(resume.payload.get("state") or {})  # type: ignore[assignment]
        goal = (resume.user_goal or user_goal or state.get("user_goal") or "").strip()
    else:
        if not macro:
            macro = f"supervisor-{uuid.uuid4().hex[:8]}"
        goal = user_goal.strip()
        state = {
            "user_goal": goal,
            "macro_thread_id": macro,
            "tasks": [],
            "current_task_id": None,
            "artifacts": {},
            "final_answer": "",
            "status": "planning",
        }
        if cb.on_status:
            cb.on_status("🧭 Supervisor 规划中…")
        state = _merge_state(state, planner_node(state))

    # HITL resume：完成当前 Specialist 步后再继续 specialist_node 状态机
    if resume and resume.payload.get("phase") == "specialist_hitl" and hitl_decisions is not None:
        pending = resume.payload.get("specialist_pending") or {}
        role = AgentRole(pending["role"])
        payload = HandoffPayload.model_validate(pending["handoff"])
        graph = get_specialist_graph(role)
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
            del status, tool_results
            delta = text[len("".join(acc)) :]
            if delta:
                _on_token(delta)

        turn = asyncio.run(
            run_assistant_turn(
                graph,
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
                mode="supervisor",
                macro_thread_id=macro,
                user_goal=goal,
                payload={
                    **resume.payload,
                    "specialist_pending": {
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
        from langgraph_skill_agent.multi_agent.supervisor import _parse_review_result

        if role is AgentRole.REVIEW:
            result = _parse_review_result(turn.text, payload.task_id)
        else:
            from langgraph_skill_agent.multi_agent.handoff import SpecialistResult

            result = SpecialistResult(
                task_id=payload.task_id,
                role=role,
                status="done",
                summary=turn.text.strip(),
                artifacts={payload.task_id: turn.text.strip()},
            )
        if cb.on_step_done:
            cb.on_step_done(payload.task_id, result.summary)
        # 手动推进：临时写回 specialist 所需 state，再跑 specialist_node 的后半逻辑
        state = _merge_state(
            state,
            _apply_specialist_result_via_node(state, result, payload.task_id),
        )

    while state.get("status") == "executing" and state.get("current_task_id"):
        tid = state["current_task_id"]
        tasks = _tasks_from_state(state)
        current = next((t for t in tasks if t.id == tid), None)
        if current is None or current.status != "pending":
            patch = specialist_node(state)
            state = _merge_state(state, patch)
            continue

        label = f"[{current.id}] {current.title}"
        if cb.on_step_start:
            cb.on_step_start(current.id, label, current.role.value)

        payload = HandoffPayload(
            task_id=current.id,
            role=current.role,
            user_goal=goal,
            objective=current.title,
            prior_artifacts=_prior_artifacts_for_task(current, dict(state.get("artifacts") or {})),
            acceptance="Review 步骤需对照用户总目标与前置 worker 产出进行验收。"
            if current.role is AgentRole.REVIEW
            else "",
            constraints=[
                "只读，禁止写文件" if current.role is AgentRole.RESEARCH else "",
                "只读审查，末尾输出 JSON pass/fail" if current.role is AgentRole.REVIEW else "",
            ],
        )
        payload.constraints = [c for c in payload.constraints if c]

        step_tokens: list[str] = []

        def _on_token(t: str, acc: list[str] = step_tokens) -> None:
            acc.append(t)
            if cb.on_token:
                cb.on_token(t)

        result, hitl, pending_meta = _invoke_specialist(
            role=current.role,
            payload=payload,
            macro_thread_id=macro,
            on_token=_on_token,
            ui_mode=True,
        )
        if hitl is not None:
            return OrchestrationPaused(
                mode="supervisor",
                macro_thread_id=macro,
                user_goal=goal,
                payload={
                    "state": state,
                    "phase": "specialist_hitl",
                    "specialist_pending": pending_meta,
                },
            )
        if result is None:
            break
        if cb.on_step_done:
            cb.on_step_done(current.id, result.summary)
        state = _merge_state(state, _apply_specialist_result_via_node(state, result, current.id))

    if state.get("status") in {"executing", "synthesizing"} and not state.get("current_task_id"):
        if cb.on_status:
            cb.on_status("📝 Supervisor 汇总中…")
        state = _merge_state(state, synthesizer_node(state))
        if cb.on_final and state.get("final_answer"):
            cb.on_final(state["final_answer"])

    return state


def _apply_specialist_result_via_node(state: SupervisorState, result: Any, task_id: str) -> dict:
    """通过既有 specialist_node 逻辑应用单步结果（复用 review 重试等）。"""
    # 先将 result 写入 artifacts，标记 task done，再让 specialist_node 处理 routing
    tasks = _tasks_from_state(state)
    new_artifacts = dict(state.get("artifacts") or {})
    new_artifacts[task_id] = result.summary
    for k, v in (result.artifacts or {}).items():
        new_artifacts[k] = v
    new_tasks = []
    for t in tasks:
        if t.id == task_id:
            new_tasks.append(t.model_copy(update={"status": "done"}))
        else:
            new_tasks.append(t.model_copy())
    if result.role is AgentRole.REVIEW and result.passed is False:
        max_retries = supervisor_max_review_retries()
        worker_id = None
        review_task = next((t for t in new_tasks if t.id == task_id), None)
        if review_task:
            from langgraph_skill_agent.multi_agent.supervisor import _find_worker_for_review

            worker_id = _find_worker_for_review(new_tasks, review_task)
        worker = next((t for t in new_tasks if t.id == worker_id), None) if worker_id else None
        if worker and worker.retry_count < max_retries:
            new_tasks = _requeue_worker_after_failed_review(new_tasks, task_id)
            nxt = _pick_next_runnable_task(new_tasks, new_artifacts)
            return {
                "tasks": [t.model_dump(mode="json") for t in new_tasks],
                "artifacts": new_artifacts,
                "current_task_id": nxt,
                "status": "executing" if nxt else "synthesizing",
            }
    nxt = _pick_next_runnable_task(new_tasks, new_artifacts)
    return {
        "tasks": [t.model_dump(mode="json") for t in new_tasks],
        "artifacts": new_artifacts,
        "current_task_id": nxt,
        "status": "executing" if nxt else "synthesizing",
    }


def hitl_pending_to_dict(paused: OrchestrationPaused) -> dict[str, Any]:
    pending = paused.payload.get("specialist_pending") or {}
    hitl = pending.get("hitl")
    role = str(pending.get("role") or "research")
    return {
        "mode": paused.mode,
        "macro_thread_id": paused.macro_thread_id,
        "user_goal": paused.user_goal,
        "payload": paused.payload,
        "hitl": hitl,
        "text_prefix": pending.get("text_prefix") or "",
        "tool_results": pending.get("tool_results") or [],
        "step_label": pending.get("step_label") or "Specialist",
        "step_role": role,
    }
