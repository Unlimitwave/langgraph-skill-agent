"""Supervisor ↔ Specialist 结构化 Handoff（企业级契约）。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from langgraph_skill_agent.multi_agent.roles import AgentRole


class TaskItem(BaseModel):
    id: str = Field(description="稳定短 id，如 1、research-1")
    role: AgentRole = Field(description="research | worker | review")
    title: str = Field(description="单步可执行描述")
    depends_on: list[str] = Field(default_factory=list, description="前置 task id 列表")


class SupervisorPlanModel(BaseModel):
    tasks: list[TaskItem] = Field(description="3～8 条，顺序与依赖有意义")


class TaskRecord(BaseModel):
    id: str
    role: AgentRole
    title: str
    depends_on: list[str] = Field(default_factory=list)
    status: Literal["pending", "done", "failed", "blocked"] = "pending"
    retry_count: int = 0


class HandoffPayload(BaseModel):
    """Supervisor 下发给 Specialist 的结构化任务包。"""

    task_id: str
    role: AgentRole
    user_goal: str
    objective: str
    prior_artifacts: dict[str, str] = Field(default_factory=dict)
    acceptance: str = ""
    constraints: list[str] = Field(default_factory=list)


class SpecialistResult(BaseModel):
    """Specialist 回传给 Supervisor 的结构化结果。"""

    task_id: str
    role: AgentRole
    status: Literal["done", "failed"] = "done"
    summary: str = ""
    artifacts: dict[str, str] = Field(default_factory=dict)
    passed: bool | None = None
    feedback: str | None = None


def build_handoff_prompt(payload: HandoffPayload) -> str:
    """将 HandoffPayload 渲染为 Specialist 可执行的 prompt。"""
    lines = [
        f"【用户总目标】\n{payload.user_goal}",
        "",
        f"【本步角色】{payload.role.value}",
        f"【本步目标】{payload.objective}",
    ]
    if payload.acceptance:
        lines.extend(["", f"【验收标准】\n{payload.acceptance}"])
    if payload.constraints:
        lines.extend(["", "【约束】", *[f"- {c}" for c in payload.constraints]])
    if payload.prior_artifacts:
        lines.extend(["", "【前置步骤产出（摘要）】"])
        for key, val in payload.prior_artifacts.items():
            snippet = val.strip()
            if len(snippet) > 4000:
                snippet = snippet[:4000] + "\n...(truncated)"
            lines.append(f"--- {key} ---\n{snippet}")
    lines.extend(
        [
            "",
            "【要求】只完成本步目标；完成后给出清晰摘要，便于 Supervisor 汇总。",
        ]
    )
    return "\n".join(lines)
