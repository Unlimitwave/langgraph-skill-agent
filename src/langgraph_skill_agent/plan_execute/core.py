"""
显式 todo 外层 LangGraph：planner（结构化计划）→ executor（逐步调用现有 Deep Agent）。

用法（项目根目录，需已配置 .env 中的 DEEPSEEK_API_KEY）：
  langgraph-plan "你的复杂目标一句话"
  langgraph-plan -t my-macro-1 "目标..."
  # 或：python -m langgraph_skill_agent.plan_execute "目标..."

每步 Deep Agent 使用独立 checkpointer 线程：{macro_thread_id}:todo:{todo_id}，
避免多步工具记录混在同一线程。

环境变量（可选）：
  MACRO_THREAD_ID   未传 -t 时的默认宏任务 id（否则为随机 macro-xxxxxxxx）
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from langgraph_skill_agent.agent_core import build_agent, build_chat_model
from langgraph_skill_agent.prompts import get_prompt
from langgraph_skill_agent.utility.agent_runtime import get_agent_runtime
from langgraph_skill_agent.utility.llm_json import extract_first_json_object, message_content_to_str
from langgraph_skill_agent.utility.logging_config import configure_logging
from langgraph_skill_agent.utility.streaming import stream_assistant_reply
from langgraph_skill_agent.utility.tenant import normalize_user_id

logger = logging.getLogger(__name__)


class TodoItemModel(BaseModel):
    id: str = Field(description="稳定短 id，如 1、2、step-a")
    title: str = Field(description="单步可执行描述")


class PlanModel(BaseModel):
    todos: list[TodoItemModel] = Field(description="3～8 条，顺序有意义")


class TodoItem(TypedDict):
    id: str
    title: str
    status: Literal["pending", "done"]


class PlanExecuteState(TypedDict, total=False):
    user_goal: str
    macro_thread_id: str
    todos: list[TodoItem]
    current_todo_id: str | None


def _invoke_planner_llm(goal: str) -> PlanModel | None:
    """普通 chat 输出 JSON，避免 DeepSeek 不支持 LangChain structured parse。"""
    llm = build_chat_model(streaming=False)
    msg = llm.invoke(
        [SystemMessage(content=get_prompt("plan.planner")), HumanMessage(content=goal)]
    )
    raw = message_content_to_str(getattr(msg, "content", None))
    data = extract_first_json_object(raw)
    if not data:
        logger.error("规划器 JSON 解析失败，原始片段: %r", (raw or "")[:800])
        return None
    try:
        return PlanModel.model_validate(data)
    except Exception as e:
        logger.error("规划 JSON 与 PlanModel 不符: %s data=%r", e, data)
        return None


def planner_node(state: PlanExecuteState) -> dict:
    goal = (state.get("user_goal") or "").strip()
    if not goal:
        logger.warning("user_goal 为空，跳过规划。")
        return {"todos": [], "current_todo_id": None}
    plan = _invoke_planner_llm(goal)
    if plan is None:
        return {"todos": [], "current_todo_id": None}
    seen: set[str] = set()
    todos: list[TodoItem] = []
    for t in plan.todos:
        tid = (t.id or "").strip()
        title = (t.title or "").strip()
        if not title:
            continue
        if not tid:
            tid = f"step-{len(todos) + 1}"
        base = tid
        n = 0
        while tid in seen:
            n += 1
            tid = f"{base}-{n}"
        seen.add(tid)
        todos.append({"id": tid, "title": title, "status": "pending"})
    first = todos[0]["id"] if todos else None
    logger.info("Planner produced %d todo(s), first=%s", len(todos), first)
    return {"todos": todos, "current_todo_id": first}


def _route_after_planner(state: PlanExecuteState) -> Literal["executor", "done"]:
    if state.get("todos") and state.get("current_todo_id"):
        return "executor"
    return "done"


def _route_after_executor(state: PlanExecuteState) -> Literal["executor", "done"]:
    if state.get("current_todo_id"):
        return "executor"
    return "done"


def build_executor_node(agent):
    def executor_node(state: PlanExecuteState) -> dict:
        tid = state.get("current_todo_id")
        todos = list(state.get("todos") or [])
        macro = (state.get("macro_thread_id") or "macro-default").strip()
        goal = (state.get("user_goal") or "").strip()

        if not tid:
            return {"current_todo_id": None}

        current_title = ""
        for t in todos:
            if t["id"] == tid and t["status"] == "pending":
                current_title = t["title"]
                break

        if not current_title:
            nxt = next((t["id"] for t in todos if t["status"] == "pending"), None)
            return {"current_todo_id": nxt}

        lines = [
            f"【用户总目标】\n{goal}",
            "",
            "【完整任务列表（由规划器生成；请勿擅自跳过未指定的项）】",
        ]
        for t in todos:
            tag = "→ 本回合只做这一项" if t["id"] == tid else ""
            st = "✓" if t["status"] == "done" else "○"
            lines.append(f"  {st} [{t['id']}] {t['title']} {tag}".rstrip())
        lines.extend(
            [
                "",
                f"【本回合要求】只完成步骤 [{tid}]：{current_title}。",
                "未完成项留到后续回合；不要在本回合提前执行后续项。",
                "可使用工具与技能完成本步。",
            ]
        )
        prompt = "\n".join(lines)
        step_invoke = get_agent_runtime().invoke_kwargs(
            thread_id=f"{macro}:todo:{tid}",
            user_id=normalize_user_id(),
        )
        print(f"\n--- 执行 todo [{tid}] ---\n", flush=True)
        stream_assistant_reply(
            agent,
            prompt,
            step_invoke["config"],
            context=step_invoke["context"],
        )

        new_todos: list[TodoItem] = []
        for t in todos:
            if t["id"] == tid:
                new_todos.append({"id": t["id"], "title": t["title"], "status": "done"})
            else:
                new_todos.append(dict(t))

        nxt = next((t["id"] for t in new_todos if t["status"] == "pending"), None)
        logger.info("Finished todo %s; next=%s", tid, nxt)
        return {"todos": new_todos, "current_todo_id": nxt}

    return executor_node


def build_plan_execute_graph(agent):
    g = StateGraph(PlanExecuteState)
    g.add_node("planner", planner_node)
    g.add_node("executor", build_executor_node(agent))
    g.add_edge(START, "planner")
    g.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"executor": "executor", "done": END},
    )
    g.add_conditional_edges(
        "executor",
        _route_after_executor,
        {"executor": "executor", "done": END},
    )
    return g.compile()


def run_macro_task(user_goal: str, *, macro_thread_id: str | None = None) -> PlanExecuteState:
    configure_logging()
    agent = build_agent()
    macro = (macro_thread_id or os.environ.get("MACRO_THREAD_ID") or "").strip()
    if not macro:
        macro = f"macro-{uuid.uuid4().hex[:8]}"
    graph = build_plan_execute_graph(agent)
    final = graph.invoke(
        {
            "user_goal": user_goal.strip(),
            "macro_thread_id": macro,
            "todos": [],
            "current_todo_id": None,
        }
    )
    print("\n--- 全部 todo 状态 ---", flush=True)
    for t in final.get("todos") or ():
        print(f"  [{t['id']}] {t['status']}: {t['title']}", flush=True)
    return final


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="显式 todo：先结构化规划，再逐步调用 Deep Agent。")
    p.add_argument("goal", nargs="?", default="", help="一句话描述要完成的事")
    p.add_argument(
        "-t",
        "--thread-id",
        dest="thread_id",
        default=None,
        help="宏任务 id（用于子线程前缀；默认见 MACRO_THREAD_ID 或随机）",
    )
    args = p.parse_args(argv)
    goal = (args.goal or "").strip()
    if not goal and not sys.stdin.isatty():
        goal = sys.stdin.read().strip()
    if not goal:
        p.print_help()
        sys.exit(1)
    run_macro_task(goal, macro_thread_id=args.thread_id)


if __name__ == "__main__":
    main()
