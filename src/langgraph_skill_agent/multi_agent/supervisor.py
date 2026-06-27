"""
Supervisor 编排图：Planner → 按角色调度 Specialist → 质量门 → 汇总。

用法（项目根目录，需已配置 .env 中的 DEEPSEEK_API_KEY）：
  langgraph-supervisor "你的复杂目标一句话"
  langgraph-supervisor -t my-macro-1 "目标..."

每步 Specialist 使用独立 checkpointer 线程：{macro_thread_id}:task:{task_id}，
避免多角色工具轨迹混在同一线程。

环境变量（可选）：
  ENABLE_MULTI_AGENT_ROUTING  CLI 自动路由到本流程（见 intent_router）
  SUPERVISOR_MAX_REVIEW_RETRIES  Review 不通过时回 Worker 的最大重试次数（默认 2）
  MACRO_THREAD_ID               未传 -t 时的默认宏任务 id
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from langgraph_skill_agent.agent_core import build_chat_model
from langgraph_skill_agent.multi_agent.config import supervisor_max_review_retries
from langgraph_skill_agent.multi_agent.handoff import (
    HandoffPayload,
    SpecialistResult,
    SupervisorPlanModel,
    TaskRecord,
    build_handoff_prompt,
)
from langgraph_skill_agent.multi_agent.roles import AgentRole, role_display_name
from langgraph_skill_agent.multi_agent.specialists import get_specialist_graph
from langgraph_skill_agent.utility.agent_runtime import get_agent_runtime
from langgraph_skill_agent.utility.hitl import HitlRequest
from langgraph_skill_agent.utility.llm_json import extract_first_json_object, message_content_to_str
from langgraph_skill_agent.utility.logging_config import configure_logging
from langgraph_skill_agent.utility.streaming import (
    ToolResult,
    iter_assistant_text_sync,
    run_assistant_turn,
)
from langgraph_skill_agent.utility.tenant import normalize_user_id

logger = logging.getLogger(__name__)

# Planner 系统提示词
_SUPERVISOR_PLANNER_SYSTEM = (
    "你是企业级任务 Supervisor 的规划器。将用户目标拆成 3～8 个可执行步骤，"
    "并为每步指定 Specialist 角色：\n"
    "- research：检索知识库、阅读文件、产出调研摘要（只读）\n"
    "- worker：写文件、跑技能/脚本、交付产物（可写）\n"
    "- review：只读审查上一步 worker 产出，给出 pass/fail\n\n"
    "规则：\n"
    "1. 典型流水线：research → worker → review；复杂任务可有多轮 worker/review。\n"
    "2. review 步骤必须 depends_on 其审查的 worker 步骤。\n"
    "3. worker 若依赖调研，depends_on 对应 research 步骤。\n"
    "4. 不要写工具调用语法，只写业务目标描述。\n\n"
    "【输出格式】只输出一个 JSON 对象，不要其它说明文字，不要 markdown 代码块。格式：\n"
    '{"tasks": [{"id": "1", "role": "research", "title": "...", "depends_on": []}, '
    '{"id": "2", "role": "worker", "title": "...", "depends_on": ["1"]}, '
    '{"id": "3", "role": "review", "title": "...", "depends_on": ["2"]}]}'
)
# 汇总节点系统提示词
_SYNTHESIZER_SYSTEM = (
    "你是 Supervisor 汇总节点。根据用户目标与各 Specialist 步骤摘要，"
    "生成面向用户的最终答复：简洁、完整、可执行；不要暴露内部 JSON 或 thread id。"
)


class SupervisorState(TypedDict, total=False):
    user_goal: str
    macro_thread_id: str
    tasks: list[TaskRecord]
    current_task_id: str | None
    artifacts: dict[str, str]
    final_answer: str
    status: Literal["planning", "executing", "synthesizing", "done", "failed"]


def _task_records_from_plan(plan: SupervisorPlanModel) -> list[TaskRecord]:
    """
    将 LLM 返回的任务计划（SupervisorPlanModel）转换为 TaskRecord 列表，每个 TaskRecord 表示流水线中的一步。

    主要步骤：
    1. 遍历模型返回的每个 task。
    2. 标准化 id 与标题，若丢失 id 自动生成，保证每步 id 唯一。
    3. 跳过无标题的步骤（保证 task 有说明）。
    4. 去除依赖列表中的空白项，并去重。
    5. 对每步生成 TaskRecord，初始状态为 pending。
    """
    seen: set[str] = set()  # 记录已出现的 step id，保证唯一性
    out: list[TaskRecord] = []  # 最终输出 TaskRecord 列表
    for item in plan.tasks:
        tid = (item.id or "").strip()  # 取出并去除空白的 id
        title = (item.title or "").strip()  # 取出并去除空白的标题
        if not title:
            continue  # 跳过没有标题的 task，防止无意义的步骤进入流程
        if not tid:
            # 若步骤没有 id，则生成唯一 id（如 step-1、step-2）
            tid = f"step-{len(out) + 1}"
        base = tid
        n = 0
        while tid in seen:
            # 保证 id 唯一。如果已经出现，则追加 -1、-2
            n += 1
            tid = f"{base}-{n}"
        seen.add(tid)
        # 处理 depends_on 字段：去除空白、过滤空字符串
        deps = [d.strip() for d in item.depends_on if d and d.strip()]
        # 构建 TaskRecord（初始状态为 pending）
        out.append(
            TaskRecord(
                id=tid,
                role=item.role,
                title=title,
                depends_on=deps,
                status="pending",
            )
        )
    return out


# 调用 Planner 模型，生成任务计划
def _invoke_supervisor_planner(goal: str) -> SupervisorPlanModel | None:
    llm = build_chat_model(streaming=False)
    msg = llm.invoke(
        [SystemMessage(content=_SUPERVISOR_PLANNER_SYSTEM), HumanMessage(content=goal)]
    )
    raw = message_content_to_str(getattr(msg, "content", None))
    data = extract_first_json_object(raw)
    if not data:
        logger.error("Supervisor 规划 JSON 解析失败，原始片段: %r", (raw or "")[:800])
        return None
    try:
        # 验证规划返回的 JSON 是否符合预期
        return SupervisorPlanModel.model_validate(data)
    except ValidationError as e:
        logger.error("Supervisor 规划 JSON 与模型不符: %s data=%r", e, data)
        return None


# Planner 节点：根据用户目标，生成任务计划
def planner_node(state: SupervisorState) -> dict:
    goal = (state.get("user_goal") or "").strip()
    if not goal:
        logger.warning("user_goal 为空，跳过 Supervisor 规划。")
        return {"tasks": [], "current_task_id": None, "status": "failed"}

    # 调用 Planner 模型，生成任务计划
    plan = _invoke_supervisor_planner(goal)
    if plan is None:
        return {"tasks": [], "current_task_id": None, "status": "failed"}

    # 将计划转换为任务记录
    tasks = _task_records_from_plan(plan)
    # 选择第一个可执行任务
    first = _pick_next_runnable_task(tasks, {})
    # 记录日志
    logger.info("Supervisor planner produced %d task(s), first=%s", len(tasks), first)
    return {
        "tasks": [t.model_dump(mode="json") for t in tasks],
        "current_task_id": first,
        "artifacts": state.get("artifacts") or {},
        "status": "executing" if first else "done",
    }


def _dependencies_met(task: TaskRecord, artifacts: dict[str, str]) -> bool:
    if not task.depends_on:
        return True
    # 检查所有依赖是否都已生成
    return all(dep in artifacts for dep in task.depends_on)


def _pick_next_runnable_task(
    tasks: list[TaskRecord],
    artifacts: dict[
        str, str
    ],  # artifacts 表示各个已完成任务产出的结果（每个 key 是任务 ID 或步骤标识，value 是该任务产生的文本/内容）
) -> str | None:
    for task in tasks:
        if task.status != "pending":
            continue
        # 检查依赖是否满足（artifacts 里是否已经包含了该任务需要依赖的其他任务的产出）
        if _dependencies_met(task, artifacts):
            return task.id
    return None


def _prior_artifacts_for_task(task: TaskRecord, artifacts: dict[str, str]) -> dict[str, str]:
    if not task.depends_on:
        return dict(artifacts)
    return {k: artifacts[k] for k in task.depends_on if k in artifacts}


def _parse_review_result(text: str, task_id: str) -> SpecialistResult:
    data = extract_first_json_object(text)
    passed = True
    feedback = ""
    summary = text.strip()
    if data:
        passed = bool(data.get("passed", True))
        fb = data.get("feedback")
        feedback = fb if isinstance(fb, str) else ""
        sm = data.get("summary")
        if isinstance(sm, str) and sm.strip():
            summary = sm.strip()
    return SpecialistResult(
        task_id=task_id,
        role=AgentRole.REVIEW,
        status="done",
        summary=summary,
        passed=passed,
        feedback=feedback or None,
    )


def _invoke_specialist(
    *,
    role: AgentRole,
    payload: HandoffPayload,
    macro_thread_id: str,
    on_token: Any | None = None,
    ui_mode: bool = False,
) -> tuple[SpecialistResult | None, HitlRequest | None, dict[str, Any]]:
    """
    调用 Specialist Agent 执行任务的关键入口。

    参数说明:
        role: 执行任务的 agent 角色，如 WORKER/REVIEW 等。
        payload: HandoffPayload，任务的数据载体，包括目标、依赖等。
        macro_thread_id: 宏观线程 ID，便于追踪分布式的聚合任务。
        on_token: （可选）增量输出 token 时的回调函数（常用于 UI/流式输出）。
        ui_mode: 是否为 UI 交互模式（比如 Web 前端流式体验），若为 True，则支持带 on_update 的增量回调。

    返回值:
        - SpecialistResult 或 None（成功则为产物对象，HITL/人工接管则为 None）
        - HitlRequest 或 None（如果产生 HITL，需人工处理时为 HitlRequest，否则为 None）
        - dict[str, Any]（产出元数据，如各类上下文配置）

    流程梳理:
        1. 按角色组装或获取专属 agent graph。
        2. 用合适的上下文和线程 ID 构造本任务运行时要求。
        3. 组装交接 prompt（任务说明等）。
        4. 区分 UI 流和命令行流，分别做不同处理：
            - UI 流支持 on_update，实现 token 推送，返回 turn/pending_hitl/text 等。
            - 非 UI 流直接流式输出到终端。

        5. 若是 review 任务，则需特殊解析，提取 review 结果/summary/feedback。
        6. 否则正常返回 SpecialistResult。

    """

    # 1. 获取 agent graph 和运行时上下文
    graph = get_specialist_graph(role)
    # 获取运行时上下文
    step_invoke = get_agent_runtime().invoke_kwargs(
        thread_id=f"{macro_thread_id}:task:{payload.task_id}:{role.value}",
        user_id=normalize_user_id(),
    )
    # 构建交接提示词
    prompt = build_handoff_prompt(payload)
    # 构建步骤标签
    step_label = f"[{role_display_name(role)}] 执行 task [{payload.task_id}]"

    # 2. 命令行模式下简单打印阶段提示
    if not ui_mode:
        print(f"\n--- {step_label} ---\n", flush=True)

    # 3. UI 模式：通过 on_update 增量推送 token
    if ui_mode:
        tokens: list[str] = []

        def _on_update(
            *,
            status: str | None,
            text: str,
            cursor: bool,
            tool_results: list[ToolResult] | None = None,
        ) -> None:
            # UI/前端 token 变更推送
            del status, tool_results, cursor
            if on_token is None:
                return
            # 新增 token（增量部分）
            delta = text[len("".join(tokens)) :]
            if delta:
                tokens.append(delta)
                on_token(delta)

        # 4. 执行一轮交互
        turn = asyncio.run(
            run_assistant_turn(
                graph,
                user_text=prompt,
                config=step_invoke["config"],
                context=step_invoke["context"],
                on_update=_on_update,
                decide=None,
            )
        )
        # 5. 判断是否触发 HITL，若触发则组装 meta 返回
        if turn.pending_hitl is not None:
            ctx = step_invoke["context"]
            pending_meta = {
                "config": step_invoke["config"],
                "context": {"user_id": ctx.user_id, "tenant_id": ctx.tenant_id},
                "handoff": payload.model_dump(mode="json"),
                "role": role.value,
                "text_prefix": turn.text,
                "tool_results": turn.tool_results,
                "hitl": {
                    "action_requests": turn.pending_hitl.action_requests,
                    "review_configs": turn.pending_hitl.review_configs,
                },
                "step_label": step_label,
            }
            return None, turn.pending_hitl, pending_meta
        # 若未触发 HITL，正常产出 raw_text
        raw_text = turn.text
    else:
        # 6. 非 UI/命令行流式输出，token 增量 sink 到控制台
        token_sink = on_token or (lambda t: (sys.stdout.write(t), sys.stdout.flush()))
        raw_text = iter_assistant_text_sync(
            graph,
            user_text=prompt,
            config=step_invoke["config"],
            context=step_invoke["context"],
            on_token=token_sink,
        )
        if not ui_mode:
            print(flush=True)

    # 7. 如果是 review 任务，需要专门解析 JSON 检查通过/反馈等结构
    if role is AgentRole.REVIEW:
        return _parse_review_result(raw_text, payload.task_id), None, {}

    # 8. 否则生成标准 SpecialistResult 结构返回
    return (
        SpecialistResult(
            task_id=payload.task_id,
            role=role,
            status="done",
            summary=raw_text.strip(),
            artifacts={payload.task_id: raw_text.strip()},
        ),
        None,
        {},
    )


def _find_worker_for_review(tasks: list[TaskRecord], review_task: TaskRecord) -> str | None:
    for dep in review_task.depends_on:
        for t in tasks:
            if t.id == dep and t.role is AgentRole.WORKER:
                return t.id
    return review_task.depends_on[-1] if review_task.depends_on else None


def _requeue_worker_after_failed_review(
    tasks: list[TaskRecord], review_task_id: str
) -> list[TaskRecord]:
    """
    当 review 任务未通过时，重置其对应的 worker 和 review 任务状态，实现重新执行逻辑。

    步骤说明：
    1. 根据 review_task_id 查找 review 任务对象；
       - 如果查不到，直接返回原 tasks，不做任何操作。
    2. 使用 _find_worker_for_review 查找该 review 依赖的 worker 任务 id；
       - 如果没有找到合适的 worker，也直接返回原 tasks。
    3. 遍历所有 tasks，生成新的任务列表 updated：
       - 对于需要重试的 worker，如果其状态已为 "done"（即已执行完成），
         则重置为 "pending" 并将 retry_count + 1，表示需要重新执行并追踪重试次数。
       - 对于当前 review 任务，同样重置为 "pending"，等待 worker 执行完成后再次 review。
       - 对于其余的任务，保持状态不变，深拷贝一份以避免副作用。
    4. 返回更新后的任务列表，供后续流程使用。
    """
    # 查找指定 review 任务
    review_task = next((t for t in tasks if t.id == review_task_id), None)
    if review_task is None:
        # 未找到 review 任务，原样返回任务列表
        return tasks
    # 查找 review 所对应（依赖）的 worker 任务 id
    worker_id = _find_worker_for_review(tasks, review_task)
    if not worker_id:
        # 没有找到 worker，任务列表不变
        return tasks

    updated: list[TaskRecord] = []
    for t in tasks:
        if t.id == worker_id and t.status == "done":
            # 只对已完成的 worker 任务重置为 pending，并自增 retry_count
            updated.append(
                t.model_copy(update={"status": "pending", "retry_count": t.retry_count + 1})
            )
        elif t.id == review_task_id:
            # 对 review 任务重置为 pending
            updated.append(t.model_copy(update={"status": "pending"}))
        else:
            # 其他任务深拷贝，不做修改
            updated.append(t.model_copy())
    return updated


# 专家节点：执行当前任务
def specialist_node(state: SupervisorState) -> dict:
    tid = state.get("current_task_id")  # 当前要执行的任务 ID
    tasks = [
        TaskRecord.model_validate(t) if isinstance(t, dict) else t
        for t in (state.get("tasks") or [])
    ]
    macro = (state.get("macro_thread_id") or "supervisor-default").strip()  # 宏任务 ID
    goal = (state.get("user_goal") or "").strip()  # 用户目标
    artifacts = dict(state.get("artifacts") or {})  # 已完成任务的产出结果

    if not tid:
        return {"current_task_id": None, "status": "synthesizing"}

    # 获取当前要执行的任务对象
    current = next((t for t in tasks if t.id == tid), None)
    # 如果当前任务不存在，或者该任务不是 pending 状态（可能已完成/失败等），则尝试选择下一个可执行的任务
    if current is None or current.status != "pending":
        nxt = _pick_next_runnable_task(tasks, artifacts)
        # 如果有下一个可执行任务，则返回执行状态；否则说明所有任务已进入新阶段（如汇总）
        return {"current_task_id": nxt, "status": "executing" if nxt else "synthesizing"}

    # 构建交接给 Specialist 节点的 payload，包含任务 id、角色、用户目标、当前子目标，依赖前置工件（如有），
    # 以及验收规则和约束（根据角色区分）。
    payload = HandoffPayload(
        task_id=current.id,  # 当前 Specialist 步骤的唯一 id
        role=current.role,  # Specialist 角色（research / worker / review）
        user_goal=goal,  # 用户总目标，供 Specialist 参考
        objective=current.title,  # 当前 Specialist 本步骤子目标
        prior_artifacts=_prior_artifacts_for_task(current, artifacts),  # 该步骤依赖的前置任务工件
        acceptance=(
            # 对于 review 角色，补充验收规则说明
            "Review 步骤需对照用户总目标与前置 worker 产出进行验收。"
            if current.role is AgentRole.REVIEW
            else ""
        ),
        constraints=[
            # 给 research 角色加“只读”约束
            "只读，禁止写文件" if current.role is AgentRole.RESEARCH else "",
            # 给 review 角色加上输出形式的特殊说明
            "只读审查，末尾输出 JSON pass/fail" if current.role is AgentRole.REVIEW else "",
        ],
    )
    # 过滤掉空字符串的约束内容
    payload.constraints = [c for c in payload.constraints if c]

    try:
        # 调用 Specialist 具体执行任务（交给单步执行的 agent 节点）
        result, hitl, _ = _invoke_specialist(
            role=current.role,
            payload=payload,
            macro_thread_id=macro,
        )
        if hitl is not None or result is None:
            # HITL 出现在非交互驱动场景（如 CLI）应为异常，直接将当前 task 标失败
            logger.error("Specialist HITL 在非 UI 驱动中未预期暂停 task=%s", tid)
            new_tasks = [
                t.model_copy(update={"status": "failed"}) if t.id == tid else t.model_copy()
                for t in tasks
            ]
            return {
                "tasks": [t.model_dump(mode="json") for t in new_tasks],
                "current_task_id": _pick_next_runnable_task(new_tasks, artifacts),
                "status": "failed",
            }
    except Exception as e:
        # Specialist agent 执行报错，记为失败并进入失败流程
        logger.exception("Specialist 执行失败 task=%s role=%s: %s", tid, current.role.value, e)
        new_tasks = [
            t.model_copy(update={"status": "failed"}) if t.id == tid else t.model_copy()
            for t in tasks
        ]
        return {
            "tasks": [t.model_dump(mode="json") for t in new_tasks],
            "current_task_id": _pick_next_runnable_task(new_tasks, artifacts),
            "status": "failed",
        }

    # 整理新工件。当前步骤产出的摘要 summary 记到 artifacts，并合并子字典 artifacts。
    new_artifacts = dict(artifacts)
    new_artifacts[tid] = result.summary  # 当前 task id -> 产出摘要
    for k, v in result.artifacts.items():  # 处理 Specialist agent 附加产物
        new_artifacts[k] = v

    # 重新组装新的任务列表，将当前步骤标记为 done，其他保留原状态
    new_tasks: list[TaskRecord] = []
    for t in tasks:
        if t.id == tid:
            new_tasks.append(t.model_copy(update={"status": "done"}))
        else:
            new_tasks.append(t.model_copy())

    # 如果当前任务为 review 且验证未通过
    if current.role is AgentRole.REVIEW and result.passed is False:
        max_retries = supervisor_max_review_retries()  # 最大 review 重试次数
        worker_id = _find_worker_for_review(new_tasks, current)  # 找到被 review 的 worker 步骤
        worker = next((t for t in new_tasks if t.id == worker_id), None) if worker_id else None
        if worker and worker.retry_count < max_retries:
            # 超过失败次数前，worker 步骤重试且 review 也重新 pending
            logger.info(
                "Review failed for task=%s; re-queue worker=%s (retry %d/%d)",
                tid,
                worker_id,
                worker.retry_count + 1,
                max_retries,
            )
            new_tasks = _requeue_worker_after_failed_review(new_tasks, tid)
            nxt = _pick_next_runnable_task(new_tasks, new_artifacts)
            return {
                "tasks": [t.model_dump(mode="json") for t in new_tasks],
                "artifacts": new_artifacts,
                "current_task_id": nxt,
                "status": "executing" if nxt else "synthesizing",
            }
        # 超过最大 review 重试次数，直接 warning，不再重试
        logger.warning("Review failed and retry budget exhausted for task=%s", tid)

    # 尝试调度下一个可执行的任务。如果没有，则流程进入 synthesizing（汇总）阶段
    nxt = _pick_next_runnable_task(new_tasks, new_artifacts)
    return {
        "tasks": [t.model_dump(mode="json") for t in new_tasks],
        "artifacts": new_artifacts,
        "current_task_id": nxt,
        "status": "executing" if nxt else "synthesizing",
    }


# 汇总节点：根据各 Specialist 步骤摘要，生成最终答复
def synthesizer_node(state: SupervisorState) -> dict:
    goal = (state.get("user_goal") or "").strip()
    artifacts = state.get("artifacts") or {}
    if not goal:
        return {"final_answer": "", "status": "done"}

    lines = [f"用户目标：{goal}", "", "各步骤摘要："]
    for key, val in artifacts.items():
        snippet = (val or "").strip()
        if len(snippet) > 2000:
            snippet = snippet[:2000] + "\n...(truncated)"
        lines.append(f"[{key}]\n{snippet}\n")

    llm = build_chat_model(streaming=False)
    msg = llm.invoke(
        [
            SystemMessage(content=_SYNTHESIZER_SYSTEM),
            HumanMessage(content="\n".join(lines)),
        ]
    )
    answer = message_content_to_str(getattr(msg, "content", None)).strip()
    print("\n--- Supervisor 汇总 ---\n", flush=True)
    print(answer, flush=True)
    return {"final_answer": answer, "status": "done"}


def _route_after_planner(state: SupervisorState) -> Literal["specialist", "synthesizer"]:
    if state.get("current_task_id"):
        return "specialist"
    return "synthesizer"


def _route_after_specialist(state: SupervisorState) -> Literal["specialist", "synthesizer"]:
    status = state.get("status")
    if status == "executing" and state.get("current_task_id"):
        return "specialist"
    return "synthesizer"


def build_supervisor_graph():
    g = StateGraph(SupervisorState)
    # 添加 Planner、Specialist、Synthesizer 节点
    g.add_node("planner", planner_node)
    g.add_node("specialist", specialist_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_edge(START, "planner")
    # 添加 Planner 节点路由，如果当前任务为 executing 状态且有当前任务 ID，则继续执行 Specialist 节点；否则进入汇总节点
    g.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"specialist": "specialist", "synthesizer": "synthesizer"},
    )
    # 添加 Specialist 节点路由，如果当前任务为 executing 状态且有当前任务 ID，则继续执行 Specialist 节点；否则进入汇总节点
    g.add_conditional_edges(
        "specialist",
        _route_after_specialist,
        {"specialist": "specialist", "synthesizer": "synthesizer"},
    )
    g.add_edge("synthesizer", END)
    return g.compile()


def run_supervisor_task(user_goal: str, *, macro_thread_id: str | None = None) -> SupervisorState:
    configure_logging()
    # 获取macro thread id
    macro = (macro_thread_id or os.environ.get("MACRO_THREAD_ID") or "").strip()
    if not macro:
        macro = f"supervisor-{uuid.uuid4().hex[:8]}"

    # 构建 Supervisor 编排图
    graph = build_supervisor_graph()
    final = graph.invoke(
        {
            "user_goal": user_goal.strip(),
            "macro_thread_id": macro,
            "tasks": [],
            "current_task_id": None,
            "artifacts": {},
            "final_answer": "",
            "status": "planning",
        }
    )
    print("\n--- Supervisor 任务状态 ---", flush=True)
    for t in final.get("tasks") or []:
        rec = TaskRecord.model_validate(t) if isinstance(t, dict) else t
        print(f"  [{rec.id}] {rec.role.value}/{rec.status}: {rec.title}", flush=True)
    return final


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Supervisor 多智能体：规划 → Research/Worker/Review → 汇总。"
    )
    p.add_argument("goal", nargs="?", default="", help="一句话描述要完成的事")
    p.add_argument(
        "-t",
        "--thread-id",
        dest="thread_id",
        default=None,
        help="宏任务 id（用于子线程前缀；默认见 MACRO_THREAD_ID 或随机）",
    )
    args = p.parse_args(argv)
    goal = (args.goal or "").strip()
    if not goal and not sys.stdin.isatty():
        goal = sys.stdin.read().strip()
    if not goal:
        p.print_help()
        sys.exit(1)
    run_supervisor_task(goal, macro_thread_id=args.thread_id)


if __name__ == "__main__":
    main()
