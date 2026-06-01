"""消息 content 归一化为字符串。"""

from __future__ import annotations

from typing import Any


def stringify_message_content(content: Any) -> str:



    """
    如果 content 为 None，返回空字符串，一般情况是模型在调用工具时，content 为 None
    """
    if content is None:
        return ""

    """
            content 中的block 有多种，一种是完整的字符串

            完整的字符串形如：
            content = "Hello, world!"
    """
    if isinstance(content, str):
        return content


    """
            content 中的block 有多种，一种是列表

             (1)list 里每个 block 是 字符串：
             content = ["你好，", "这是", "分段文本"]
             (2)list 里每个 block 是 字典标准多模态格式：
             content = [
                {"type": "text", "text": "图片里有什么？"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/a.png"},
                },
            ]
    """
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)
