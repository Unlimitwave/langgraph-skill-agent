"""
Deep Agent（LangChain deepagents）+ 本地 Skills + RAG。

安装与运行（项目根目录）：
  pip install -e .
  langgraph-agent
  langgraph-ui
  langgraph-plan "目标"
  langgraph-supervisor "目标"
  langgraph-summary
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.agents.middleware import before_model
from langchain.agents.middleware.types import AgentState
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.runtime import Runtime

from langgraph_skill_agent.deepseek_model import (
    ChatDeepSeekWithReasoningPassback,
    build_deepseek_chat_model,
)
from langgraph_skill_agent.memory import (
    maybe_compact_thread,
    persist_thread_snapshot,
    prepare_thread_for_turn,
    save_conversation_snapshot,
)
from langgraph_skill_agent.memory.context import inject_context_before_model
from langgraph_skill_agent.memory.pruning import slim_tool_output_middleware
from langgraph_skill_agent.memory.session_store import create_checkpointer
from langgraph_skill_agent.rag import _get_rag_retriever
from langgraph_skill_agent.tool import load_mcp_extra_tools, make_host_skill_tools
from langgraph_skill_agent.utility import PROJECT_ROOT, configure_logging, iter_assistant_text_sync
from langgraph_skill_agent.utility.agent_policy import (
    agent_filesystem_permissions,
    agent_skill_sources,
    backend_for_runtime,
)
from langgraph_skill_agent.utility.agent_runtime import get_agent_runtime
from langgraph_skill_agent.utility.tenant import AgentContext

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
                parts.append(
                    text if isinstance(text, str) else json.dumps(item, ensure_ascii=False)
                )
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
def _deepseek_normalize_before_model(
    state: AgentState, runtime: Runtime[AgentContext]
) -> dict[str, Any] | None:
    del runtime
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
            getattr(node, "get_content", None)()
            if node and hasattr(node, "get_content")
            else getattr(node, "text", "")
        ) or ""
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


def plan_routing_enabled() -> bool:
    return os.environ.get("ENABLE_PLAN_ROUTING", "").strip().lower() in {"1", "true", "yes", "on"}


def build_agent_graph() -> Any:
    """编译单图（无租户绑定）；身份在 invoke 时通过 AgentContext 注入。"""
    configure_logging()
    model = build_deepseek_chat_model(streaming=True)

    try:
        mcp_tools = load_mcp_extra_tools()
    except Exception as e:
        logger.warning("MCP 工具加载失败，已跳过: %s", e)
        mcp_tools = []

    host_tools = make_host_skill_tools()
    extra_tools = [*host_tools, rag_search, *mcp_tools]

    system_prompt = (
        "You are a helpful assistant. Follow the layered [CTX-SYSTEM] context in messages "
        "for persona, memory, task state, and procedures."
    )

    return create_deep_agent(
        model=model,
        backend=backend_for_runtime,
        tools=extra_tools,
        skills=agent_skill_sources(),
        permissions=agent_filesystem_permissions(),
        checkpointer=create_checkpointer(),
        context_schema=AgentContext,
        middleware=[
            slim_tool_output_middleware,
            inject_context_before_model,
            _deepseek_normalize_before_model,
        ],
        interrupt_on={
            "write_file": True,
            "read_file": False,
            "edit_file": True,
            "workspace_exec_python": False,
            "run_skill_script_shell": False,
        },
        system_prompt=system_prompt,
    )


def build_agent(*, force_rebuild: bool = False) -> Any:
    """返回 compiled graph（单图多租户）。推荐用 get_agent_runtime().graph。"""
    # 通俗写法，便于理解：先获取 runtime，再取它的 graph 属性
    runtime = get_agent_runtime(force_rebuild=force_rebuild)
    agent_graph = runtime.graph
    return agent_graph


def main() -> None:
    # 构建runtime,其中runtime会构建基础的graph
    runtime = get_agent_runtime()
    agent = runtime.graph
    thread_id = os.environ.get("THREAD_ID", "demo-thread-1")
    invoke = runtime.invoke_kwargs(thread_id=thread_id)

    """
    默认invoke_kwargs返回的是:
    {
    "context": AgentContext(user_id="default", tenant_id="default"),  # 除非设了环境变量
    "config": {
        "configurable": {
        "thread_id": "default__demo-thread-1",  # user_id + "__" + thread_id
        "user_id": "default",
        "tenant_id": "default",
        }
    }
    }
    """

    config = invoke["config"]
    context = invoke["context"]
    print("持续对话（quit / exit / q 退出）\n")

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

        from langgraph_skill_agent.intent_router import resolve_execution_mode
        from langgraph_skill_agent.multi_agent.config import multi_agent_routing_enabled

        if multi_agent_routing_enabled() or plan_routing_enabled():
            # 根据用户输入，路由到对应的执行模式
            mode = resolve_execution_mode(user_text)
            if mode == "supervisor":
                from langgraph_skill_agent.multi_agent.supervisor import run_supervisor_task

                print(
                    "\n[路由] 判定为多智能体任务 → Supervisor（Research/Worker/Review）\n",
                    flush=True,
                )
                run_supervisor_task(user_text, macro_thread_id=f"{thread_id}-supervisor")
                print()
                continue
            if mode == "plan":
                from langgraph_skill_agent.plan_execute import run_macro_task

                print("\n[路由] 判定为复杂任务 → 显式规划 + 分步执行\n", flush=True)
                run_macro_task(user_text, macro_thread_id=f"{thread_id}-plan")
                print()
                continue

        def _compact_trace(msg: str) -> None:
            if os.environ.get("RAG_TRACE", "").strip() in {"1", "true", "yes", "on"}:
                logger.info("[COMPACT] %s", msg)

        prepare_thread_for_turn(
            agent,
            config,
            compact_fn=maybe_compact_thread,
            compact_kwargs={"on_trace": _compact_trace},
        )
        sys.stdout.write("助手: ")
        sys.stdout.flush()

        iter_assistant_text_sync(
            agent,
            user_text=user_text,
            config=config,
            context=context,
            on_token=lambda t: (sys.stdout.write(t), sys.stdout.flush()),
        )
        print()
        try:
            persist_thread_snapshot(agent, config)
        except Exception as e:
            logger.warning("导出 thread 快照失败: %s", e)

    try:
        p = save_conversation_snapshot(agent, config)
        logger.info("对话已保存: %s", p)
    except Exception as e:
        logger.error("保存对话失败: %s", e)


if __name__ == "__main__":
    main()
