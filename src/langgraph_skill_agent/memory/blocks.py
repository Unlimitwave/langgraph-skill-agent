"""加载 agent_memory 目录下的 Markdown 记忆块（分层 section）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from langgraph_skill_agent.utility.paths import AGENT_MEMORY_DIR

_SECTION_SPLIT = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)

_DEFAULT_PROCEDURAL = """\
When a skill asks to run a Python script under skills/:
- Use workspace_exec_python with program python/python3 and argv_tail like ["skills/test-calc-script/run_calc.py"].
For whitelisted skill shell scripts, use run_skill_script_shell with a registered script_id (e.g. test-calc.run);
never use workspace_exec_python with bash.
Use rag_search when the user asks about knowledge-base content.
"""


@dataclass
class MemorySections:
    """分层记忆块，对应 System 各 section。"""

    role: str = ""
    procedural: str = ""
    semantic: str = ""
    episodic: str = ""
    task_state: str = ""
    rag_hint: str = field(
        default="按需调用 rag_search 工具检索业务知识库；本节为占位，检索结果在工具返回中。"
    )


def _read_md(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _split_md_sections(text: str) -> dict[str, str]:
    if not text.strip():
        return {}
    parts = _SECTION_SPLIT.split(text)
    if len(parts) == 1:
        return {"_body": text.strip()}
    sections: dict[str, str] = {}
    # parts[0] 可能是前言
    if parts[0].strip():
        sections["_preamble"] = parts[0].strip()
    for i in range(1, len(parts), 2):
        title = parts[i].strip().lower()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body:
            sections[title] = body
    return sections


def _pick_section(sections: dict[str, str], *keys: str) -> str:
    for key in keys:
        k = key.lower()
        if k in sections:
            return sections[k]
    return ""


def load_memory_sections(mem_dir: Path | None = None) -> MemorySections:
    mem_dir = mem_dir or AGENT_MEMORY_DIR
    out = MemorySections()

    soul_text = _read_md(mem_dir / "soul.md")
    soul_parts = _split_md_sections(soul_text)
    out.role = _pick_section(
        soul_parts, "角色设定", "agent soul", "persona", "soul"
    ) or soul_parts.get("_body", soul_parts.get("_preamble", ""))
    proc_from_soul = _pick_section(
        soul_parts, "程序记忆", "程序记忆（procedural）", "procedural", "sop"
    )
    procedural_file = _read_md(mem_dir / "procedural.md")
    out.procedural = procedural_file or proc_from_soul or _DEFAULT_PROCEDURAL

    user_text = _read_md(mem_dir / "user.md")
    user_parts = _split_md_sections(user_text)
    out.semantic = _pick_section(
        user_parts, "长期记忆（semantic）", "semantic", "user profile", "user"
    ) or user_parts.get("_body", user_parts.get("_preamble", ""))

    memory_text = _read_md(mem_dir / "Memory.md")
    mem_parts = _split_md_sections(memory_text)
    sem = _pick_section(mem_parts, "长期记忆（semantic）", "semantic")
    epi = _pick_section(mem_parts, "长期记忆（episodic）", "episodic", "long-term memory", "memory")
    if sem:
        out.semantic = (out.semantic + "\n\n" + sem).strip() if out.semantic else sem
    if epi:
        out.episodic = epi
    elif memory_text and not sem:
        out.episodic = mem_parts.get("_body", mem_parts.get("_preamble", memory_text))

    task_text = _read_md(mem_dir / "task_state.md")
    if task_text:
        out.task_state = task_text

    return out


def load_agent_memory_blocks(mem_dir: Path | None = None) -> str:
    """兼容旧接口：扁平拼接记忆块。"""
    s = load_memory_sections(mem_dir)
    parts: list[str] = []
    for title, body in [
        ("Agent soul (persona)", s.role),
        ("User profile", s.semantic),
        ("Long-term memory", s.episodic),
    ]:
        if body.strip():
            parts.append(f"## {title}\n{body}")
    return "\n\n".join(parts).strip()
