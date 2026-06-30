"""Unit tests for session_store (checkpointer authority + UI projection)."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langgraph_skill_agent.memory.context import CONTEXT_SYSTEM_MARKER
from langgraph_skill_agent.memory.session_store import (
    export_snapshot_enabled,
    hydrate_enabled,
    messages_to_ui_display,
    ui_messages_to_lc,
)


def test_hydrate_and_export_disabled_by_default() -> None:
    assert hydrate_enabled() is False
    assert export_snapshot_enabled() is False


def test_messages_to_ui_display_filters_ctx_system_and_merges_tools() -> None:
    msgs = [
        SystemMessage(content=f"{CONTEXT_SYSTEM_MARKER}\nsecret"),
        HumanMessage(content="你好"),
        AIMessage(content=""),
        ToolMessage(content="result", name="rag_search", tool_call_id="t1"),
        AIMessage(content="你好！"),
    ]
    ui = messages_to_ui_display(msgs)
    assert len(ui) == 2
    assert ui[0] == {"role": "user", "content": "你好"}
    assert ui[1]["role"] == "assistant"
    assert ui[1]["content"] == "你好！"
    assert ui[1]["tool_results"][0]["name"] == "rag_search"


def test_ui_messages_to_lc_roundtrip_roles() -> None:
    ui = [
        {"role": "user", "content": "你好"},
        {
            "role": "assistant",
            "content": "你好！",
            "tool_results": [{"name": "rag_search", "content": "doc1"}],
        },
    ]
    lc = ui_messages_to_lc(ui)
    assert len(lc) == 2
    assert lc[0].type == "human"
    assert "rag_search" in str(lc[1].content)
