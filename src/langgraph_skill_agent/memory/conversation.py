"""对话快照持久化。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import message_to_dict

from langgraph_skill_agent.utility.paths import CONVERSATION_HISTORY_DIR


def save_conversation_snapshot(
    agent,
    config: dict,
    *,
    hist_dir: Path | None = None,
) -> Path:
    hist_dir = hist_dir or CONVERSATION_HISTORY_DIR
    hist_dir.mkdir(parents=True, exist_ok=True)
    thread_id = str((config.get("configurable") or {}).get("thread_id") or "default")
    safe_tid = "".join(c if c.isalnum() or c in "-_" else "_" for c in thread_id)[:200]
    snap = agent.get_state(config)
    values = getattr(snap, "values", None) or {}
    messages = values.get("messages") or []
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = hist_dir / f"{safe_tid}_{ts}.json"
    payload = {
        "thread_id": thread_id,
        "saved_at_utc": ts,
        "messages": [message_to_dict(m) for m in messages],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
