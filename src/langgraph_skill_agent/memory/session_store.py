"""
会话持久化：Postgres / Sqlite checkpointer 为唯一权威；UI 仅展示。

生产默认：
  CHECKPOINT_BACKEND=postgres + POSTGRES_URI
  HYDRATE_ENABLED=0（灾备恢复见 restore_thread_from_snapshot）
  CHECKPOINT_EXPORT_SNAPSHOT=0（可选 JSON 导出供 langgraph-summary / 审计）

本地开发可设 CHECKPOINT_BACKEND=sqlite|memory。
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    message_to_dict,
    messages_from_dict,
)
from langgraph.checkpoint.memory import MemorySaver

from langgraph_skill_agent.memory.context import (
    _is_injected_system_message,
)
from langgraph_skill_agent.utility.messages import stringify_message_content
from langgraph_skill_agent.utility.paths import CONVERSATION_HISTORY_DIR, VAR_DIR

logger = logging.getLogger(__name__)

SESSION_HISTORY_DIR = VAR_DIR / "session_history"
CHECKPOINT_DB_PATH = VAR_DIR / "checkpoints.sqlite"

_CHECKPOINTER_SINGLETON: Any | None = None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def hydrate_enabled() -> bool:
    """灾备 / 迁移：仅在显式开启时从 JSON 灌入空 thread（非热路径）。"""
    return _env_bool("HYDRATE_ENABLED", False)


def export_snapshot_enabled() -> bool:
    """是否每轮额外导出 conversation_history JSON（默认关）。"""
    return _env_bool("CHECKPOINT_EXPORT_SNAPSHOT", False)


def checkpoint_backend() -> str:
    return os.environ.get("CHECKPOINT_BACKEND", "postgres").strip().lower()


def _postgres_uri() -> str | None:
    for key in ("POSTGRES_URI", "DATABASE_URL", "CHECKPOINT_POSTGRES_URI"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw
    return None


def _redact_uri(uri: str) -> str:
    if "@" not in uri:
        return uri
    scheme, rest = uri.split("://", 1) if "://" in uri else ("", uri)
    if "@" in rest:
        creds, host = rest.rsplit("@", 1)
        user = creds.split(":", 1)[0] if creds else ""
        return f"{scheme}://{user}:***@{host}" if scheme else f"{user}:***@{host}"
    return uri


def get_checkpointer_label() -> str:
    backend = checkpoint_backend()
    if backend == "postgres":
        uri = _postgres_uri()
        return f"PostgresSaver ({_redact_uri(uri)})" if uri else "PostgresSaver (未配置 URI)"
    if backend == "sqlite":
        return f"SqliteSaver ({CHECKPOINT_DB_PATH})"
    return "MemorySaver（仅进程内，非生产）"


def create_checkpointer():
    """创建（并缓存）checkpointer：postgres → sqlite → memory。"""
    global _CHECKPOINTER_SINGLETON
    if _CHECKPOINTER_SINGLETON is not None:
        return _CHECKPOINTER_SINGLETON

    backend = checkpoint_backend()

    if backend == "postgres":
        uri = _postgres_uri()
        if not uri:
            raise RuntimeError(
                "CHECKPOINT_BACKEND=postgres 但未设置 POSTGRES_URI / DATABASE_URL。"
                "生产环境请配置 PostgreSQL 连接串。"
            )
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as e:
            raise RuntimeError(
                "缺少 langgraph-checkpoint-postgres 或 psycopg。请安装: "
                "pip install langgraph-checkpoint-postgres 'psycopg[binary,pool]'"
            ) from e

        pool = ConnectionPool(
            conninfo=uri,
            min_size=1,
            max_size=int(os.environ.get("POSTGRES_POOL_MAX", "10")),
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        )
        saver = PostgresSaver(pool)
        saver.setup()
        _CHECKPOINTER_SINGLETON = saver
        logger.info("使用 PostgresSaver: %s", _redact_uri(uri))
        return saver

    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
            saver = SqliteSaver(conn)
            _CHECKPOINTER_SINGLETON = saver
            logger.info("使用 SqliteSaver: %s", CHECKPOINT_DB_PATH)
            return saver
        except ImportError:
            logger.warning("未安装 langgraph-checkpoint-sqlite，回退 MemorySaver")

    _CHECKPOINTER_SINGLETON = MemorySaver()
    logger.warning("使用 MemorySaver（进程重启丢失 checkpoint，仅适合本地调试）")
    return _CHECKPOINTER_SINGLETON


def _safe_thread_id(thread_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in thread_id)[:200]


def snapshot_path(thread_id: str, *, hist_dir: Path | None = None) -> Path:
    hist_dir = hist_dir or CONVERSATION_HISTORY_DIR
    return hist_dir / f"{_safe_thread_id(thread_id)}.json"


def session_ui_path(thread_id: str) -> Path:
    return SESSION_HISTORY_DIR / f"{_safe_thread_id(thread_id)}.json"


def get_thread_messages(compiled: Any, config: dict) -> list[BaseMessage]:
    snap = compiled.get_state(config)
    return list((snap.values or {}).get("messages") or [])


def thread_is_empty(compiled: Any, config: dict) -> bool:
    return len(get_thread_messages(compiled, config)) == 0


def messages_to_ui_display(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """Checkpointer messages → UI 聊天气泡（过滤注入 System，合并 Tool 到 assistant）。"""
    visible = [m for m in messages if not _is_injected_system_message(m)]
    ui: list[dict[str, Any]] = []
    i = 0
    while i < len(visible):
        msg = visible[i]
        if msg.type == "human":
            ui.append(
                {
                    "role": "user",
                    "content": stringify_message_content(getattr(msg, "content", "")),
                }
            )
            i += 1
            continue
        if msg.type == "ai":
            tool_results: list[dict[str, str]] = []
            j = i + 1
            while j < len(visible) and visible[j].type == "tool":
                tool_msg = visible[j]
                tool_results.append(
                    {
                        "name": str(getattr(tool_msg, "name", None) or "tool"),
                        "content": stringify_message_content(getattr(tool_msg, "content", "")),
                    }
                )
                j += 1
            content = stringify_message_content(getattr(msg, "content", ""))
            k = j
            while k < len(visible) and visible[k].type == "ai":
                next_content = stringify_message_content(getattr(visible[k], "content", ""))
                if next_content.strip():
                    content = next_content
                k += 1
            if content.strip() or tool_results:
                entry: dict[str, Any] = {"role": "assistant", "content": content}
                if tool_results:
                    entry["tool_results"] = tool_results
                ui.append(entry)
            i = k
            continue
        i += 1
    return ui


def sync_ui_messages_from_checkpointer(
    compiled: Any,
    config: dict,
    session: dict[str, Any],
) -> list[dict[str, Any]]:
    """用 checkpointer 权威 state 刷新 session['messages']（仅 UI 展示）。"""
    ui_msgs = messages_to_ui_display(get_thread_messages(compiled, config))
    session["messages"] = ui_msgs
    if ui_msgs and session.get("title") in (None, "", "新对话"):
        first_user = next((m["content"] for m in ui_msgs if m.get("role") == "user"), "")
        if first_user:
            session["title"] = (first_user[:28] + "…") if len(first_user) > 28 else first_user
    return ui_msgs


def ui_messages_to_lc(messages: list[dict[str, Any]]) -> list[BaseMessage]:
    """UI / JSON 快照 → LangChain messages（灾备导入，工具链有损）。"""
    out: list[BaseMessage] = []
    for msg in messages:
        role = msg.get("role")
        content = str(msg.get("content") or "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            tool_results = msg.get("tool_results") or []
            if tool_results:
                lines = [content] if content.strip() else []
                for tr in tool_results:
                    name = tr.get("name", "tool")
                    body = str(tr.get("content") or "")
                    lines.append(f"[工具 {name} 返回]\n{body}")
                content = "\n\n".join(lines).strip()
            out.append(AIMessage(content=content or "(无回复)"))
    return out


def load_thread_snapshot(thread_id: str) -> list[BaseMessage] | None:
    """灾备：加载 conversation_history 完整快照；否则尝试 UI JSON。"""
    path = snapshot_path(thread_id)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            raw = data.get("messages") or []
            if raw:
                return list(messages_from_dict(raw))
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
            logger.warning("读取对话快照失败 %s: %s", path, e)

    ui_path = session_ui_path(thread_id)
    if ui_path.is_file():
        try:
            data = json.loads(ui_path.read_text(encoding="utf-8"))
            ui_msgs = data.get("messages") or []
            if ui_msgs:
                return ui_messages_to_lc(ui_msgs)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("读取 UI 会话失败 %s: %s", ui_path, e)
    return None


def restore_thread_from_snapshot(
    compiled: Any,
    config: dict,
    *,
    thread_id: str | None = None,
    ui_messages: list[dict[str, Any]] | None = None,
) -> bool:
    """
    灾备 / 迁移：将 JSON 快照写入空 thread 的 checkpointer。
    仅在 checkpoint 为空时执行；生产热路径不应调用。
    """
    if not thread_is_empty(compiled, config):
        logger.warning("restore_thread_from_snapshot: thread 非空，跳过")
        return False

    tid = thread_id or str((config.get("configurable") or {}).get("thread_id") or "default")
    messages = load_thread_snapshot(tid)
    if not messages and ui_messages:
        messages = ui_messages_to_lc(ui_messages)
    if not messages:
        return False

    compiled.update_state(config, {"messages": messages})
    logger.info("灾备恢复：已将 %d 条消息写入 thread %s", len(messages), tid)
    return True


def hydrate_thread_if_needed(
    compiled: Any,
    config: dict,
    *,
    ui_messages: list[dict[str, Any]] | None = None,
) -> bool:
    """兼容旧名：仅在 HYDRATE_ENABLED=1 时委托 restore_thread_from_snapshot。"""
    if not hydrate_enabled():
        return False
    return restore_thread_from_snapshot(compiled, config, ui_messages=ui_messages)


def persist_thread_snapshot(
    compiled: Any,
    config: dict,
    *,
    hist_dir: Path | None = None,
) -> Path | None:
    """可选：导出 conversation_history JSON（CHECKPOINT_EXPORT_SNAPSHOT=1）。"""
    if not export_snapshot_enabled():
        return None

    hist_dir = hist_dir or CONVERSATION_HISTORY_DIR
    hist_dir.mkdir(parents=True, exist_ok=True)
    thread_id = str((config.get("configurable") or {}).get("thread_id") or "default")
    messages = get_thread_messages(compiled, config)
    if not messages:
        return None

    path = snapshot_path(thread_id, hist_dir=hist_dir)
    payload = {
        "thread_id": thread_id,
        "messages": [message_to_dict(m) for m in messages],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.debug("已导出 checkpoint 快照: %s", path)
    return path


def prepare_thread_for_turn(
    compiled: Any,
    config: dict,
    *,
    ui_messages: list[dict[str, Any]] | None = None,
    compact_fn: Any | None = None,
    compact_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    每轮模型调用前：可选灾备 hydrate → compact。
    ui_messages 仅 HYDRATE_ENABLED=1 时用于空 thread 恢复（向后兼容）。
    """
    if hydrate_enabled():
        hydrate_thread_if_needed(compiled, config, ui_messages=ui_messages)
    if compact_fn is not None:
        compact_fn(compiled, config, **(compact_kwargs or {}))


__all__ = [
    "CHECKPOINT_DB_PATH",
    "SESSION_HISTORY_DIR",
    "checkpoint_backend",
    "create_checkpointer",
    "export_snapshot_enabled",
    "get_checkpointer_label",
    "get_thread_messages",
    "hydrate_enabled",
    "hydrate_thread_if_needed",
    "load_thread_snapshot",
    "messages_to_ui_display",
    "persist_thread_snapshot",
    "prepare_thread_for_turn",
    "restore_thread_from_snapshot",
    "sync_ui_messages_from_checkpointer",
    "thread_is_empty",
    "ui_messages_to_lc",
]
