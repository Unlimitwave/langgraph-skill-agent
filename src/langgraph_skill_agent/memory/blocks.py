"""加载 agent_memory 目录下的 Markdown 记忆块。"""

from __future__ import annotations

from pathlib import Path

from langgraph_skill_agent.utility.paths import AGENT_MEMORY_DIR


def load_agent_memory_blocks(mem_dir: Path | None = None) -> str:
    mem_dir = mem_dir or AGENT_MEMORY_DIR
    parts: list[str] = []
    for name, title in [
        ("soul.md", "## Agent soul (persona)"),
        ("user.md", "## User profile"),
        ("Memory.md", "## Long-term memory"),
    ]:
        p = mem_dir / name
        if p.is_file():
            text = p.read_text(encoding="utf-8").strip()
            if text:
                parts.append(f"{title}\n{text}")
    return "\n\n".join(parts).strip()
