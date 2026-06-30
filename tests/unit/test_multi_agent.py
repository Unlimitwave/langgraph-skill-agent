"""Unit tests for multi-agent handoff and supervisor helpers."""

from langgraph_skill_agent.multi_agent.handoff import (
    HandoffPayload,
    SupervisorPlanModel,
    TaskItem,
    TaskRecord,
    build_handoff_prompt,
)
from langgraph_skill_agent.multi_agent.roles import AgentRole, permissions_for_role
from langgraph_skill_agent.multi_agent.supervisor import (
    _parse_review_result,
    _pick_next_runnable_task,
    _task_records_from_plan,
)


def test_build_handoff_prompt_includes_goal_and_artifacts() -> None:
    payload = HandoffPayload(
        task_id="2",
        role=AgentRole.WORKER,
        user_goal="写报告",
        objective="根据调研撰写初稿",
        prior_artifacts={"1": "调研摘要内容"},
    )
    text = build_handoff_prompt(payload)
    assert "写报告" in text
    assert "调研摘要内容" in text
    assert "worker" in text


def test_task_records_from_plan_deduplicates_ids() -> None:
    plan = SupervisorPlanModel(
        tasks=[
            TaskItem(id="1", role=AgentRole.RESEARCH, title="调研"),
            TaskItem(id="1", role=AgentRole.WORKER, title="撰写", depends_on=["1"]),
        ]
    )
    records = _task_records_from_plan(plan)
    assert len(records) == 2
    assert records[0].id == "1"
    assert records[1].id == "1-1"


def test_pick_next_runnable_task_respects_dependencies() -> None:
    tasks = [
        TaskRecord(id="1", role=AgentRole.RESEARCH, title="a", status="done"),
        TaskRecord(
            id="2",
            role=AgentRole.WORKER,
            title="b",
            depends_on=["1"],
            status="pending",
        ),
    ]
    assert _pick_next_runnable_task(tasks, {}) is None
    assert _pick_next_runnable_task(tasks, {"1": "done"}) == "2"


def test_parse_review_result_extracts_json() -> None:
    text = '审查完成。\n{"passed": false, "feedback": "缺引用", "summary": "未通过"}'
    result = _parse_review_result(text, "3")
    assert result.passed is False
    assert result.feedback == "缺引用"
    assert result.summary == "未通过"


def test_worker_has_write_permissions_research_read_only() -> None:
    worker = permissions_for_role(AgentRole.WORKER)
    research = permissions_for_role(AgentRole.RESEARCH)
    assert any(p.mode == "allow" and "write" in p.operations for p in worker)
    assert all(not (p.mode == "allow" and "write" in p.operations) for p in research)
