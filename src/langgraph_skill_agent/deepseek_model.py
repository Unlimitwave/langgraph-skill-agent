"""DeepSeek chat model with reasoning_content passback for thinking-mode tool loops."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv

from langgraph_skill_agent.utility.paths import PROJECT_ROOT


class ChatDeepSeekWithReasoningPassback(ChatDeepSeek):
    """Re-inject ``reasoning_content`` on follow-up requests (required by thinking models)."""

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        original_messages = self._convert_input(input_).to_messages()
        payload_messages = payload.get("messages") or []

        if len(original_messages) == len(payload_messages):
            pairs = zip(original_messages, payload_messages, strict=False)
        else:
            pairs = zip(
                (m for m in original_messages if isinstance(m, AIMessage)),
                (m for m in payload_messages if m.get("role") == "assistant"),
                strict=False,
            )

        for orig, out in pairs:
            if not isinstance(orig, AIMessage) or out.get("role") != "assistant":
                continue
            reasoning = orig.additional_kwargs.get("reasoning_content")
            if reasoning is not None:
                out["reasoning_content"] = reasoning
            elif orig.tool_calls:
                out.setdefault("reasoning_content", "")
        return payload


def build_deepseek_chat_model(*, streaming: bool = True) -> ChatDeepSeekWithReasoningPassback:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请在 .env 中设置 DEEPSEEK_API_KEY，或导出该环境变量。")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    return ChatDeepSeekWithReasoningPassback(
        model=model,
        api_key=api_key,
        streaming=streaming,
    )
