"""从 LLM 文本回复中解析 JSON（兼容不支持 response_format / parse 的 API，如 DeepSeek）。"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def message_content_to_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    """从模型输出中取出第一个 JSON 对象并解析为 dict；失败返回 None。"""
    s = text.strip()
    if "```" in s:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.I)
        if m:
            s = m.group(1).strip()
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j <= i:
        return None
    blob = s[i : j + 1]
    try:
        out = json.loads(blob)
    except json.JSONDecodeError:
        logger.debug("JSON decode failed for blob: %s", blob[:200])
        return None
    return out if isinstance(out, dict) else None
