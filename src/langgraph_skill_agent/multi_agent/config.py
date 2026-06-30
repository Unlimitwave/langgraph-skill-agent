"""多智能体模式运行时开关（与 plan 模式独立）。"""

from __future__ import annotations

import os

from langgraph_skill_agent.utility.logging_config import env_truthy

# CLI / 路由：1 启用 Supervisor 多智能体自动路由
ENABLE_MULTI_AGENT_ROUTING_ENV = "ENABLE_MULTI_AGENT_ROUTING"

# 单个子任务 Review 不通过时，最多回 Worker 重试次数
SUPERVISOR_MAX_REVIEW_RETRIES_ENV = "SUPERVISOR_MAX_REVIEW_RETRIES"


def multi_agent_routing_enabled() -> bool:
    return env_truthy(ENABLE_MULTI_AGENT_ROUTING_ENV)


def supervisor_max_review_retries() -> int:
    raw = (os.environ.get(SUPERVISOR_MAX_REVIEW_RETRIES_ENV) or "2").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 2
