"""根据用户输入路由执行模式：直连 / plan / supervisor。"""

from __future__ import annotations

import logging
import os
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from langgraph_skill_agent.agent_core import build_chat_model
from langgraph_skill_agent.multi_agent.config import multi_agent_routing_enabled
from langgraph_skill_agent.prompts import get_prompt
from langgraph_skill_agent.utility.llm_json import extract_first_json_object, message_content_to_str
from langgraph_skill_agent.utility.logging_config import env_truthy

logger = logging.getLogger(__name__)

ExecutionMode = Literal["direct", "plan", "supervisor"]

# 1=每轮 LLM 意图分类（direct/plan/supervisor）；0=按 ENABLE_MULTI_AGENT_ROUTING / ENABLE_PLAN_ROUTING 固定模式
ENABLE_INTENT_ROUTING_ENV = "ENABLE_INTENT_ROUTING"


class RouterModel(BaseModel):
    needs_plan: bool = Field(
        description="为 true 表示应走外层 plan_execute；为 false 表示直接单轮 Deep Agent 对话。"
    )


class ExecutionModeRouterModel(BaseModel):
    mode: ExecutionMode = Field(description="direct | plan | supervisor")


def _plan_routing_enabled() -> bool:
    from langgraph_skill_agent.agent_core import plan_routing_enabled

    return plan_routing_enabled()


def intent_routing_enabled() -> bool:
    """默认开启：与历史行为一致，由 LLM 按用户输入选模式。"""
    raw = os.environ.get(ENABLE_INTENT_ROUTING_ENV)
    if raw is None or not raw.strip():
        return True
    return env_truthy(ENABLE_INTENT_ROUTING_ENV)


def resolve_env_forced_mode() -> ExecutionMode:
    """Intent 关闭时：MULTI_AGENT 优先于 PLAN，均未开则 direct。"""
    if multi_agent_routing_enabled():
        return "supervisor"
    if _plan_routing_enabled():
        return "plan"
    return "direct"


def _quick_heuristic_skip_router(text: str) -> bool | None:
    """返回 True 表示可直接判定为闲聊（不调路由模型）；None 表示交给模型。"""
    t = text.strip()
    if len(t) <= 6 and re.fullmatch(r"[\s\w\u4e00-\u9fff，。！？…~、]{1,20}", t or " "):
        if re.match(
            r"^(你好|您好|hi|hello|在吗|谢谢|多谢|不客气|再见|拜拜|嗯|好|哦|行|ok|OK|好的|收到)[\s!！。?？~…]*$",
            t,
            re.I,
        ):
            return True
    return None


def _normalize_user_text(text: str) -> str:
    return text.replace("\x08", "").strip()


def _enabled_modes_description() -> tuple[list[ExecutionMode], str]:
    modes: list[ExecutionMode] = ["direct"]
    lines = ["- direct（始终可用）"]
    if multi_agent_routing_enabled():
        modes.append("supervisor")
        lines.append("- supervisor")
    if _plan_routing_enabled():
        modes.append("plan")
        lines.append("- plan")
    return modes, "\n".join(lines)


def resolve_execution_mode(user_text: str) -> ExecutionMode:
    """在已启用的 advanced 模式中选 direct / plan / supervisor。"""

    # 获取环境变量，判断是否路由
    if not intent_routing_enabled():
        # 按环境变量，固定走下方已启用的编排模式（MULTI_AGENT 优先于 PLAN）
        return resolve_env_forced_mode()

    user_text = _normalize_user_text(user_text)

    # 快速启发式跳过路由，判断是否闲聊
    if _quick_heuristic_skip_router(user_text) is True:
        return "direct"

    # 获取已启用的编排模式，并构建路由系统提示词
    enabled_modes, modes_block = _enabled_modes_description()
    if enabled_modes == ["direct"]:
        return "direct"

    llm = build_chat_model(streaming=False)
    router_instruction = (
        get_prompt("intent.router", enabled_modes_block=modes_block)
        + "\n\n【输出格式】只输出一个 JSON 对象，不要其它说明文字，不要 markdown 代码块。示例："
        + '{"mode": "direct"}'
    )
    msg = llm.invoke(
        [
            SystemMessage(content=router_instruction),
            HumanMessage(content=user_text),
        ]
    )
    raw = message_content_to_str(getattr(msg, "content", None))
    data = extract_first_json_object(raw)
    if not data:
        logger.warning("路由 JSON 解析失败，保守走直连对话。原始片段: %r", raw[:300])
        return "direct"
    try:
        # 验证路由返回的 JSON 是否符合预期
        mode = ExecutionModeRouterModel.model_validate(data).mode
    except Exception as e:
        logger.warning("路由 JSON 与模式不符: %s data=%r", e, data)
        return "direct"

    if mode not in enabled_modes:
        logger.warning("路由返回未启用的模式 %s，回退 direct", mode)
        return "direct"
    return mode


def user_needs_plan_execute(user_text: str) -> bool:
    """兼容旧接口：仅在 plan 路由开启时等价于 resolve_execution_mode == plan。"""
    if not _plan_routing_enabled():
        return False
    return resolve_execution_mode(user_text) == "plan"


def user_needs_supervisor(user_text: str) -> bool:
    """仅在 multi-agent 路由开启时等价于 resolve_execution_mode == supervisor。"""
    if not multi_agent_routing_enabled():
        return False
    return resolve_execution_mode(user_text) == "supervisor"
