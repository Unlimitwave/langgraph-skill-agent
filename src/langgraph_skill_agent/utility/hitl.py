"""Human-in-the-loop（deepagents interrupt_on）辅助：解析 interrupt、CLI 审批。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TypedDict


# TODO 待阅读 2026-06-22-16:00
class HitlAction(TypedDict, total=False):
    name: str
    args: dict[str, Any]
    description: str


class HitlReviewConfig(TypedDict, total=False):
    action_name: str
    allowed_decisions: list[str]


@dataclass(frozen=True)
class HitlRequest:
    action_requests: list[HitlAction]
    review_configs: list[HitlReviewConfig]


@dataclass
class AssistantTurnResult:
    text: str
    tool_results: list[dict[str, str]]
    pending_hitl: HitlRequest | None = None


def _parse_interrupt_value(value: Any) -> HitlRequest | None:
    if not isinstance(value, dict):
        return None
    actions = value.get("action_requests")
    configs = value.get("review_configs")
    if not isinstance(actions, list) or not actions:
        return None
    return HitlRequest(
        action_requests=actions,
        review_configs=configs if isinstance(configs, list) else [],
    )


def get_pending_hitl(graph: Any, config: dict) -> HitlRequest | None:
    """从 checkpointer 读取当前 thread 的 pending interrupt（LangGraph 标准做法）。"""
    snap = graph.get_state(config)
    interrupts = getattr(snap, "interrupts", ()) or ()
    if not interrupts:
        return None
    return _parse_interrupt_value(getattr(interrupts[0], "value", None))


def allowed_decisions_for(hitl: HitlRequest, action_name: str) -> list[str]:
    for cfg in hitl.review_configs:
        if cfg.get("action_name") == action_name:
            allowed = cfg.get("allowed_decisions")
            if isinstance(allowed, list) and allowed:
                return [str(x) for x in allowed]
    return ["approve", "edit", "reject", "respond"]


def format_hitl_summary(hitl: HitlRequest) -> str:
    lines = ["**待审批工具调用**"]
    for i, action in enumerate(hitl.action_requests, start=1):
        name = action.get("name") or "?"
        args = action.get("args") or {}
        allowed = allowed_decisions_for(hitl, name)
        lines.append(f"{i}. `{name}` — 允许: {', '.join(allowed)}")
        lines.append(f"   参数: `{json.dumps(args, ensure_ascii=False)}`")
        desc = action.get("description")
        if isinstance(desc, str) and desc.strip():
            lines.append(f"   {desc.strip()}")
    return "\n".join(lines)


def hitl_to_dict(hitl: HitlRequest) -> dict[str, Any]:
    return {
        "action_requests": hitl.action_requests,
        "review_configs": hitl.review_configs,
    }


def hitl_from_dict(data: dict[str, Any]) -> HitlRequest:
    return HitlRequest(
        action_requests=data.get("action_requests") or [],
        review_configs=data.get("review_configs") or [],
    )


def prompt_hitl_decisions_cli(hitl: HitlRequest) -> list[dict[str, Any]]:
    """CLI：逐项询问 approve / reject（标准 Command(resume={decisions}) 格式）。"""
    decisions: list[dict[str, Any]] = []
    for action in hitl.action_requests:
        name = action.get("name") or "?"
        args = action.get("args") or {}
        allowed = allowed_decisions_for(hitl, name)
        print(f"\n[需审批] 工具: {name}")
        print(f"  参数: {json.dumps(args, ensure_ascii=False)}")
        print(f"  允许: {', '.join(allowed)}")
        while True:
            choice = input("决策 [approve/reject]: ").strip().lower()
            if choice in {"approve", "a", "y", "yes"}:
                if "approve" in allowed:
                    decisions.append({"type": "approve"})
                    break
            elif choice in {"reject", "r", "n", "no"}:
                if "reject" in allowed:
                    msg = input("拒绝原因（可留空）: ").strip()
                    decision: dict[str, Any] = {"type": "reject"}
                    if msg:
                        decision["message"] = msg
                    decisions.append(decision)
                    break
            print("请输入 approve 或 reject。")
    return decisions
