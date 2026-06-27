"""编排任务（plan / supervisor）的可选回调，CLI 与 Web UI 共用。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OrchestrationCallbacks:
    """编排过程事件；均为可选。"""

    on_status: Callable[[str], None] | None = None
    on_step_start: Callable[..., None] | None = None
    on_token: Callable[[str], None] | None = None
    on_step_done: Callable[[str, str], None] | None = None
    on_final: Callable[[str], None] | None = None


@dataclass
class AgentTurnCallbacks:
    """单次 Deep Agent / Specialist 调用的流式更新。"""

    on_token: Callable[[str], None] | None = None
    on_tool_results: Callable[[list[dict[str, str]]], None] | None = None


@dataclass
class OrchestrationPaused:
    """编排因 HITL 暂停；由 UI 保存并在审批后 resume。"""

    mode: str
    macro_thread_id: str
    user_goal: str
    payload: dict[str, Any] = field(default_factory=dict)
