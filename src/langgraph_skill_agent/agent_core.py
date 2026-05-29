"""
Deep Agent（LangChain deepagents）+ 本地 Skills + RAG。

安装与运行（项目根目录）：
  pip install -e .
  langgraph-agent
  langgraph-ui
  langgraph-plan "目标"
  langgraph-summary
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from dotenv import load_dotenv
from langchain.agents.middleware import before_model
from langchain.agents.middleware.types import AgentState
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime

from langgraph_skill_agent.deepseek_model import (
    ChatDeepSeekWithReasoningPassback,
    build_deepseek_chat_model,
)
from langgraph_skill_agent.memory import load_agent_memory_blocks, maybe_compact_thread, save_conversation_snapshot
from langgraph_skill_agent.rag import _get_rag_retriever
from langgraph_skill_agent.tool import load_mcp_extra_tools, make_host_skill_tools, run_skill_script_in_docker
from langgraph_skill_agent.utility import PROJECT_ROOT, configure_logging, iter_assistant_text_sync

logger = logging.getLogger(__name__)

load_dotenv(PROJECT_ROOT / ".env")


def _content_to_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                parts.append(text if isinstance(text, str) else json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _normalize_messages_for_deepseek(state: dict) -> dict:
    messages = state.get("messages", [])
    normalized: list[BaseMessage] = []
    for msg in messages:
        safe_content = _content_to_str(getattr(msg, "content", ""))
        if hasattr(msg, "model_copy"):
            normalized.append(msg.model_copy(update={"content": safe_content}))
        else:
            normalized.append(msg)
    return {"messages": normalized}


@before_model(name="deepseek_normalize_messages")
async def _deepseek_normalize_before_model(
    state: AgentState, runtime: Runtime
) -> dict[str, Any] | None:
    return _normalize_messages_for_deepseek(dict(state))


@tool
def rag_search(query: str) -> str:
    """知识库检索工具，用于回答用户关于知识库的问题。"""
    import time

    t0 = time.perf_counter()
    results = _get_rag_retriever().retrieve(query)
    lines: list[str] = []
    for i, nws in enumerate(results, start=1):
        node = getattr(nws, "node", None)
        text = (
            (getattr(node, "get_content", None)() if node and hasattr(node, "get_content") else getattr(node, "text", ""))
            or ""
        )
        meta = getattr(node, "metadata", {}) or {}
        source = meta.get("file_path") or meta.get("source") or meta.get("filename") or ""
        page = meta.get("page_label") or meta.get("page") or meta.get("page_number") or ""
        score = nws.score if nws.score is not None else ""
        header = f"[{i}] score={score} source={source} page={page}".strip()
        snippet = text.strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "\n...(truncated)"
        lines.extend([header, snippet, ""])
    out = "\n".join(lines).strip() if lines else "(no results)"
    logger.debug("rag_search %.3fs n=%d", time.perf_counter() - t0, len(results))
    return out


def build_chat_model(*, streaming: bool = True) -> ChatDeepSeekWithReasoningPassback:
    return build_deepseek_chat_model(streaming=streaming)


def _plan_routing_enabled() -> bool:
    return os.environ.get("ENABLE_PLAN_ROUTING", "").strip().lower() in {"1", "true", "yes", "on"}


def build_agent() -> Any:
    # 配置全局的logging配置
    configure_logging()

    # 构建模型
    model = build_deepseek_chat_model(streaming=True)
    backend = FilesystemBackend(root_dir=str(PROJECT_ROOT), virtual_mode=True)

    # 加载MCP工具
    try:
        mcp_tools = load_mcp_extra_tools()
    except Exception as e:
        logger.warning("MCP 工具加载失败，已跳过: %s", e)
        mcp_tools = []

    # 构建host工具
    host_tools = make_host_skill_tools(PROJECT_ROOT)

    # 构建extra工具
    extra_tools = [*host_tools, rag_search, run_skill_script_in_docker, *mcp_tools]

    # 加载记忆块
    memory_block = load_agent_memory_blocks()

    # 构建系统提示
    system_prompt = f"""You are a helpful assistant, you can use the tools to help the user.
When a skill asks to run a Python script under skills/:
- Prefer run_skill_script_in_docker with path relative to skills/ (e.g. test-calc-script/run_calc.py).
- Or workspace_exec with program python/python3 and argv_tail like ["skills/test-calc-script/run_calc.py"].
For whitelisted skill shell scripts, use run_skill_script with a registered script_id (e.g. test-calc.run);
never use workspace_exec with bash.

The following files were loaded at session start and are authoritative for persona, user preferences, and long-term facts:
{memory_block}
"""

    # 构建agent
    return create_deep_agent(
        model=model,
        backend=backend,
        tools=extra_tools,
        skills=["skills"],
        checkpointer=MemorySaver(),
        middleware=[_deepseek_normalize_before_model],
        interrupt_on={
            "write_file": True,
            "read_file": False,
            "edit_file": True,
            "run_skill_script_in_docker": False,
            "workspace_exec": False,
            "run_skill_script": False,
        },
        system_prompt=system_prompt,
    )


def main() -> None:
    # 构建agent
    agent = build_agent()

    # 设置线程ID
    thread_id = os.environ.get("THREAD_ID", "demo-thread-1")
    config = {"configurable": {"thread_id": thread_id}}
    print("持续对话（quit / exit / q 退出）\n")

    # 持续对话
    while True:
        try:
            user_text = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit", "q"}:
            print("再见。")
            break
        
        # 判断是否需要规划，读取.env中的ENABLE_PLAN_ROUTING配置
        if _plan_routing_enabled():
            from langgraph_skill_agent.intent_router import user_needs_plan_execute
            from langgraph_skill_agent.plan_execute import run_macro_task

            if user_needs_plan_execute(user_text):
                print("\n[路由] 判定为复杂任务 → 显式规划 + 分步执行\n", flush=True)
                run_macro_task(user_text, macro_thread_id=f"{thread_id}-plan")
                print()
                continue

        def _compact_trace(msg: str) -> None:
            if os.environ.get("RAG_TRACE", "").strip() in {"1", "true", "yes", "on"}:
                logger.info("[COMPACT] %s", msg)
        # 根据上下文长度。如果上下文过长，压缩对话上下文
        maybe_compact_thread(agent, config, on_trace=_compact_trace)
        sys.stdout.write("助手: ")
        sys.stdout.flush()
        iter_assistant_text_sync(
            agent,
            user_text=user_text,
            config=config,
            on_token=lambda t: (sys.stdout.write(t), sys.stdout.flush()),
        )
        print()

    try:
        p = save_conversation_snapshot(agent, config)
        logger.info("对话已保存: %s", p)
    except Exception as e:
        logger.error("保存对话失败: %s", e)


if __name__ == "__main__":
    main()
