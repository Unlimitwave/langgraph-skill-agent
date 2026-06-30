"""Unit tests for llm_json helpers."""

from langgraph_skill_agent.utility.llm_json import extract_first_json_object, message_content_to_str


def test_message_content_to_str_handles_none_and_string() -> None:
    assert message_content_to_str(None) == ""
    assert message_content_to_str("hello") == "hello"


def test_message_content_to_str_handles_text_blocks() -> None:
    content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert message_content_to_str(content) == "ab"


def test_extract_first_json_object_from_plain_text() -> None:
    assert extract_first_json_object('{"ok": true, "n": 1}') == {"ok": True, "n": 1}


def test_extract_first_json_object_from_markdown_fence() -> None:
    text = 'Here is the result:\n```json\n{"name": "demo"}\n```\nDone.'
    assert extract_first_json_object(text) == {"name": "demo"}


def test_extract_first_json_object_returns_none_on_invalid() -> None:
    assert extract_first_json_object("no json here") is None
    assert extract_first_json_object("[1, 2, 3]") is None
