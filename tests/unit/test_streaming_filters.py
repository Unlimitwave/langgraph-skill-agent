"""Unit tests for streaming message chunk filters."""

from langchain_core.messages import AIMessageChunk, HumanMessageChunk, SystemMessageChunk

from langgraph_skill_agent.utility.streaming import (
    _is_assistant_message_chunk,
    _is_tool_message_chunk,
)


def test_assistant_chunk_filter() -> None:
    assert _is_assistant_message_chunk(AIMessageChunk(content="hi"))
    assert not _is_assistant_message_chunk(HumanMessageChunk(content="hi"))
    assert not _is_assistant_message_chunk(SystemMessageChunk(content="[CTX-SYSTEM]"))
    assert not _is_tool_message_chunk(AIMessageChunk(content="hi"))
