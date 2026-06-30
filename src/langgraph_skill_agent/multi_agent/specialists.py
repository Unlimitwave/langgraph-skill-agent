"""按角色编译 Specialist Deep Agent（单图多租户，进程内按 role 缓存）。"""

from __future__ import annotations

import logging
from typing import Any

from deepagents import create_deep_agent

from langgraph_skill_agent.agent_core import (
    _deepseek_normalize_before_model,
    build_deepseek_chat_model,
    rag_search,
)
from langgraph_skill_agent.memory.context import inject_context_before_model
from langgraph_skill_agent.memory.pruning import slim_tool_output_middleware
from langgraph_skill_agent.memory.session_store import create_checkpointer
from langgraph_skill_agent.multi_agent.roles import (
    ROLE_SYSTEM_PROMPTS,
    AgentRole,
    interrupt_on_for_role,
    permissions_for_role,
    skill_sources_for_role,
)
from langgraph_skill_agent.tool import load_mcp_extra_tools, make_host_skill_tools
from langgraph_skill_agent.utility.agent_policy import backend_for_runtime
from langgraph_skill_agent.utility.tenant import AgentContext

logger = logging.getLogger(__name__)

_SPECIALIST_GRAPHS: dict[AgentRole, Any] = {}


def _tools_for_role(role: AgentRole) -> list[Any]:
    if role is AgentRole.RESEARCH:
        return [rag_search]
    if role is AgentRole.REVIEW:
        return []
    try:
        mcp_tools = load_mcp_extra_tools()
    except Exception as e:
        logger.warning("MCP 工具加载失败（role=%s），已跳过: %s", role.value, e)
        mcp_tools = []
    host_tools = make_host_skill_tools()
    return [*host_tools, rag_search, *mcp_tools]


def build_specialist_graph(role: AgentRole) -> Any:
    """编译指定角色的 Specialist 图（无租户绑定；invoke 时注入 AgentContext）。"""
    model = build_deepseek_chat_model(streaming=True)
    tools = _tools_for_role(role)
    # 根据角色获取系统提示词
    system_prompt = ROLE_SYSTEM_PROMPTS[role]
    return create_deep_agent(
        model=model,
        backend=backend_for_runtime,
        tools=tools,
        skills=skill_sources_for_role(role),
        permissions=permissions_for_role(role),
        checkpointer=create_checkpointer(),
        context_schema=AgentContext,
        middleware=[
            slim_tool_output_middleware,
            inject_context_before_model,
            _deepseek_normalize_before_model,
        ],
        interrupt_on=interrupt_on_for_role(role),
        system_prompt=system_prompt,
    )


def get_specialist_graph(role: AgentRole) -> Any:
    """进程内按 role 懒编译并缓存 Specialist 图。"""
    cached = _SPECIALIST_GRAPHS.get(role)
    if cached is not None:
        return cached
    graph = build_specialist_graph(role)
    _SPECIALIST_GRAPHS[role] = graph
    logger.info("Compiled specialist graph for role=%s", role.value)
    return graph


def clear_specialist_cache() -> None:
    """测试或 force_rebuild 时清空 Specialist 缓存。"""
    _SPECIALIST_GRAPHS.clear()
