"""
Deep Agent（LangChain deepagents，底层为 LangGraph）+ 本地 Skills + RAG（LlamaIndex + Milvus 混合检索）。

在项目根目录创建 .env（勿提交到 Git）：
  DEEPSEEK_API_KEY=sk-...
  # 可选：DEEPSEEK_MODEL=deepseek-chat

  # 远端 embedding（OpenAI-compatible，路径含 /v1）
  EMBED_BASE_URL=http://<host>:<port>/v1
  EMBED_API_KEY=dummy
  EMBED_MODEL=bge-m3
  EMBED_DIM=1024

  # Milvus 2.4+ standalone（稠密 + 内置 BM25 + Milvus RRFRanker）
  MILVUS_URI=http://<host>:19530
  MILVUS_TOKEN=
  MILVUS_COLLECTION=rag_llamaindex
  MILVUS_METRIC=IP
  MILVUS_RRF_K=60

  # 可选：知识库与索引目录（默认项目下 data/、storage/）
  RAG_DATA_DIR=./data
  RAG_STORAGE_DIR=./storage
  RAG_TOP_K=8
  # 更新 PDF 解析或知识库后，设 1 可删除本地索引并 drop Milvus collection 后全量重建
  RAG_FORCE_REBUILD=0

  # Skills 脚本在 Docker 中执行（工具 run_skill_script_in_docker）
  # SKILL_DOCKER_IMAGE=python:3.12-slim
  # SKILL_DOCKER_TIMEOUT=120

  # FastMCP 工具（mcp_tools.py）
  # MCP_TOOLS=0 关闭；MCP_CLIENT_URL=http://127.0.0.1:8000/mcp 连接远端；默认进程内置服务
  # MCP_TOOL_NAME_PREFIX=mcp_

  # 日志：DEBUG / INFO / WARNING / ERROR（默认 INFO），输出到 stderr
  # LOG_LEVEL=INFO
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path


from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from utility import stream_assistant_reply, _normalize_skill_sources, save_conversation_snapshot, _load_agent_memory_blocks
from rag import _get_rag_retriever
from skill_docker_runner import run_skill_script_in_docker
from compactor import maybe_compact_thread
from mcp_tools import load_mcp_extra_tools
from logging_config import configure_logging

PROJECT_ROOT = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

SKILL_SOURCES = ["./skills"]

load_dotenv(PROJECT_ROOT / ".env")



def _trace(msg: str) -> None:
    if os.environ.get("RAG_TRACE", "").strip() in {"1", "true", "yes", "on"}:
        logger.info("[RAG_TRACE] %s", msg)



def build_chat_model(*, streaming: bool = True) -> ChatOpenAI:
    """与 Deep Agent 相同环境变量（DEEPSEEK_*）的 Chat 模型；外层规划等可设 streaming=False。"""
    base = _build_deepseek_chat_model()
    if base.streaming == streaming:
        return base
    return base.model_copy(update={"streaming": streaming})

def _build_deepseek_chat_model() -> ChatOpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        msg = "请在 .env 中设置 DEEPSEEK_API_KEY，或导出该环境变量。"
        raise ValueError(msg)
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://api.deepseek.com",
        streaming=True,
    )


@tool
def rag_search(query: str) -> str:
    """知识库检索工具，用于回答用户关于知识库的问题。"""
    t0 = time.perf_counter()
    retriever = _get_rag_retriever()
    t1 = time.perf_counter()
    results = retriever.retrieve(query)
    t2 = time.perf_counter()

    lines: list[str] = []
    for i, nws in enumerate(results, start=1):
        node = getattr(nws, "node", None)
        text = (getattr(node, "get_content", None)() if node and hasattr(node, "get_content") else getattr(node, "text", "")) or ""
        meta = getattr(node, "metadata", {}) or {}
        source = meta.get("file_path") or meta.get("source") or meta.get("filename") or ""
        page = meta.get("page_label") or meta.get("page") or meta.get("page_number") or ""
        score = nws.score if nws.score is not None else ""

        header = f"[{i}] score={score} source={source} page={page}".strip()
        snippet = text.strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "\n...(truncated)"
        lines.append(header)
        lines.append(snippet)
        lines.append("")

    out = "\n".join(lines).strip() if lines else "(no results)"
    t3 = time.perf_counter()
    _trace(
        "rag_search breakdown: "
        f"get_retriever={t1 - t0:.3f}s retrieve={t2 - t1:.3f}s format={t3 - t2:.3f}s total={t3 - t0:.3f}s "
        f"(n_results={len(results)})"
    )
    return out


def build_agent():
    """使用 FilesystemBackend，从项目根目录下的 skills/ 加载 Agent Skills。"""
    configure_logging()
    model = _build_deepseek_chat_model()
    checkpointer = MemorySaver()

    backend = FilesystemBackend(root_dir=str(PROJECT_ROOT), virtual_mode=True)

    try:
        mcp_tools = load_mcp_extra_tools()
    except Exception as e:
        logger.warning("MCP 工具加载失败，已跳过: %s", e)
        mcp_tools = []
    extra_tools = [rag_search, run_skill_script_in_docker, *mcp_tools]

    memory_block = _load_agent_memory_blocks(PROJECT_ROOT)
    full_system_prompt = f"""You are a helpful assistant, you can use the tools to help the user. 
    When a skill asks to run a Python script under skills/, prefer the tool run_skill_script_in_docker with the path relative to skills (e.g. test-calc-script/run_calc.py) so execution stays isolated in Docker.
   
   
   The following files were loaded at session start and are authoritative,for persona, user preferences, and long-term facts:
    {memory_block}
    """
    logger.debug("System prompt:\n%s", full_system_prompt)

    agent = create_deep_agent(
        model=model,
        backend=backend,
        tools=extra_tools,
        skills=["./skills"],
        checkpointer=checkpointer,
        interrupt_on={
        "write_file": True,  # Default: approve, edit, reject
        "read_file": False,  # No interrupts needed
        "edit_file": True,  # Default: approve, edit, reject
        "run_skill_script_in_docker": False,
        },
        system_prompt=(
        full_system_prompt
        ),
    )
    return agent



def main() -> None:
    agent = build_agent()
    thread_id = os.environ.get("THREAD_ID", "demo-thread-1")
    config = {"configurable": {"thread_id": thread_id}}
    print("持续对话（...）\n")
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
        from intent_router import user_needs_plan_execute
        from plan_execute import run_macro_task
        if user_needs_plan_execute(user_text):
            print("\n[路由] 判定为复杂任务 → 显式规划 + 分步执行\n", flush=True)
            macro_id = f"{thread_id}-plan"
            run_macro_task(user_text, macro_thread_id=macro_id)
            print()
            continue
        def _compact_trace(msg: str) -> None:
            if os.environ.get("RAG_TRACE", "").strip() in {"1", "true", "yes", "on"}:
                logger.info("[COMPACT] %s", msg)
        maybe_compact_thread(agent, config, on_trace=_compact_trace)
        stream_assistant_reply(agent, user_text, config)
        print()
    try:
        p = save_conversation_snapshot(agent, config)
        logger.info("对话已保存: %s", p)
    except Exception as e:
        logger.error("保存对话失败: %s", e)


if __name__ == "__main__":
    main()
