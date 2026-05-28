"""根据用户一句话判断是否适合走「显式 todo + 多步 Deep Agent」。"""
from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent_skills import build_chat_model
from llm_json import extract_first_json_object, message_content_to_str

logger = logging.getLogger(__name__)

_ROUTER_SYSTEM = """你是路由分类器，只判断本轮用户输入是否需要「多步任务规划」。

需要规划（needs_plan 为 true）的典型情况：
- 明确多步骤：调研→总结→写文档、先查再改再测、分阶段交付等
- 依赖多次工具/检索/写文件/跑脚本的综合任务
- 用户要求列计划、分步、todo、里程碑等

不需要规划（needs_plan 为 false）的典型情况：
- 问候、闲聊、感谢、告别
- 单个知识点问答、一句话能答完的翻译/解释
- 极短无实质任务（如「好」「嗯」「继续」且没有新任务描述）

宁可判为 false：不确定时选 false，避免把简单对话做成重型流程。"""


class RouterModel(BaseModel):
    needs_plan: bool = Field(
        description="为 true 表示应走外层 plan_execute；为 false 表示直接单轮 Deep Agent 对话。"
    )


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


def user_needs_plan_execute(user_text: str) -> bool:
    """DeepSeek 等不支持 LangChain 默认 structured parse 时使用普通 JSON 文本解析。"""
    user_text = _normalize_user_text(user_text)
    quick = _quick_heuristic_skip_router(user_text)
    if quick is True:
        return False

    llm = build_chat_model(streaming=False)
    router_instruction = (
        _ROUTER_SYSTEM
        + "\n\n【输出格式】只输出一个 JSON 对象，不要其它说明文字，不要 markdown 代码块。示例："
        + '{"needs_plan": false}'
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
        return False
    try:
        return RouterModel.model_validate(data).needs_plan
    except Exception as e:
        logger.warning("路由 JSON 与模式不符: %s data=%r", e, data)
        return False
