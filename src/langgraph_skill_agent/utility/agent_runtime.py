"""
应用层 AgentRuntime — 单图多租户的显式调用入口。

与 LangGraph 框架内建的 ``langgraph.runtime.Runtime`` 是两回事：

+---------------------------+------------------------------------------+
| 框架 Runtime              | 本模块 AgentRuntime                       |
+---------------------------+------------------------------------------+
| LangGraph 在节点执行时    | 你的业务代码在 invoke 前调用              |
| 自动构造并注入 middleware | ``get_agent_runtime()`` 拿到单例图        |
| / 工具                    | ``invoke_kwargs()`` 拼 config + context |
|                           | ``resolve()`` → agent_policy 解析沙箱   |
+---------------------------+------------------------------------------+

一次用户消息的调用链（从上到下读即可）::

  CLI / UI / plan_execute
       │
       ▼
  runtime.invoke_kwargs(thread_id, user_id)
       │  → config.configurable.thread_id  （会话，checkpointer 键）
       │  → context=AgentContext             （身份，middleware/tools 读）
       ▼
  runtime.graph.stream({"messages": [...]}, config=..., context=...)
       │
       ├─► before_model: inject_context_before_model
       │        runtime.context → resolve_agent_scope() → 读 agent_memory/*.md
       │
       ├─► agent 节点: LLM 推理，可能产生 tool_calls
       │
       ├─► tools 节点: ToolRuntime.context → resolve_agent_scope() → 沙箱
       │
       └─► checkpointer: 按 thread_id 持久化 messages state

你**不需要**自己实现框架 Runtime；只要理解上面这条链，并在入口用本类即可。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph_skill_agent.utility.agent_policy import ResolvedScope, resolve_agent_scope
from langgraph_skill_agent.utility.tenant import build_invoke_kwargs


@dataclass
class AgentRuntime:
    """单图多租户运行时：编译一次，每次调用注入 identity context。"""

    graph: Any

    resolve = staticmethod(resolve_agent_scope)

    def invoke_kwargs(
        self,
        *,
        thread_id: str,
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        return build_invoke_kwargs(
            thread_id=thread_id,
            user_id=user_id,
            tenant_id=tenant_id,
        )


_RUNTIME_SINGLETON: AgentRuntime | None = None


def get_agent_runtime(*, force_rebuild: bool = False) -> AgentRuntime:
    """返回进程内单例 AgentRuntime（懒编译图）。"""
    global _RUNTIME_SINGLETON
    if _RUNTIME_SINGLETON is not None and not force_rebuild:
        return _RUNTIME_SINGLETON
    from langgraph_skill_agent.agent_core import build_agent_graph

    _RUNTIME_SINGLETON = AgentRuntime(graph=build_agent_graph())
    return _RUNTIME_SINGLETON


__all__ = [
    "AgentRuntime",
    "ResolvedScope",
    "get_agent_runtime",
]
