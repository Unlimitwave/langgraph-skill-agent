"""Unit tests for message content normalization."""

from langgraph_skill_agent.utility.messages import stringify_message_content


def test_stringify_message_content_none_and_str() -> None:
    assert stringify_message_content(None) == ""
    assert stringify_message_content("hello") == "hello"


def test_stringify_message_content_list_of_strings() -> None:
    assert stringify_message_content(["a", "b"]) == "ab"


def test_stringify_message_content_multimodal_blocks() -> None:
    content = [
        {"type": "text", "text": "see "},
        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
    ]
    assert (
        stringify_message_content(content)
        == "see {'type': 'image_url', 'image_url': {'url': 'https://example.com/x.png'}}"
    )
