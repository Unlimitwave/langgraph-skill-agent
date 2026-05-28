"""
手动运行：根据对话快照适当更新 agent_memory/soul.md、user.md、Memory.md。

用法：
  python summary.py                          # 使用 conversation_history/ 下最新的快照
  python summary.py path/to/snapshot.json    # 指定快照
  python summary.py --dry-run                # 只打印建议内容，不写文件
  python summary.py --no-backup              # 不写 .bak 备份

依赖：与 agent_skills 相同（.env 中 DEEPSEEK_API_KEY；可选 DEEPSEEK_MODEL）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from logging_config import configure_logging
from langchain_core.messages import BaseMessage, messages_from_dict
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
MEM_DIR = PROJECT_ROOT / "agent_memory"
HIST_DIR = PROJECT_ROOT / "conversation_history"

SOUL_PATH = MEM_DIR / "soul.md"
USER_PATH = MEM_DIR / "user.md"
MEMORY_PATH = MEM_DIR / "Memory.md"

logger = logging.getLogger(__name__)


class UpdatedMemoryFiles(BaseModel):
    """模型输出的三个 Markdown 文件的完整替换内容（不是 diff）。"""

    soul_md: str = Field(
        description="agent 人格、语气、价值观、行为边界等；无则保持原意压缩重写。"
    )
    user_md: str = Field(
        description="用户称呼、偏好、长期目标、禁忌等；对话中未提及则保留原文。"
    )
    memory_md: str = Field(
        description="跨对话需记住的事实、约定、项目状态；用条目列表，避免重复。"
    )


def _build_chat_model(*, streaming: bool = False) -> ChatOpenAI:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请在 .env 中设置 DEEPSEEK_API_KEY。")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://api.deepseek.com",
        streaming=streaming,
        temperature=0.3,
    )


def _stringify_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _message_role_label(msg: BaseMessage) -> str:
    t = getattr(msg, "type", None) or ""
    if t in ("human", "user"):
        return "User"
    if t in ("ai", "assistant"):
        return "Assistant"
    if t == "system":
        return "System"
    if t == "tool":
        return "Tool"
    return t or "Unknown"


def messages_json_to_transcript(
    raw_messages: list[dict[str, Any]],
    *,
    max_chars: int | None = None,
    include_tools: bool = False,
) -> str:
    """将 save_conversation_snapshot 保存的 messages 列表转成可读剧本。"""
    try:
        msgs = messages_from_dict(raw_messages)
    except Exception as e:
        raise ValueError(f"无法解析消息列表: {e}") from e

    lines: list[str] = []
    for m in msgs:
        role = _message_role_label(m)
        if role == "Tool" and not include_tools:
            continue
        text = _stringify_message_content(getattr(m, "content", None)).strip()
        if not text:
            continue
        if role == "Tool" and include_tools:
            name = getattr(m, "name", None) or "tool"
            text = f"[{name}] {text}"
        lines.append(f"### {role}\n{text}")

    out = "\n\n".join(lines)
    if max_chars is not None and len(out) > max_chars:
        head = max_chars // 4
        tail = max_chars - head - 80
        out = (
            out[:head]
            + f"\n\n...(中间已省略，共截断至约 {max_chars} 字符)...\n\n"
            + out[-tail:]
        )
    return out


def _find_latest_snapshot() -> Path:
    if not HIST_DIR.is_dir():
        raise FileNotFoundError(f"目录不存在: {HIST_DIR}")
    json_files = sorted(
        HIST_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not json_files:
        raise FileNotFoundError(f"{HIST_DIR} 下没有 .json 快照，请先运行对话并保存。")
    return json_files[0]


def _read_text(path: Path) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


def _backup_file(path: Path) -> Path | None:
    if not path.is_file():
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bak = path.with_suffix(path.suffix + f".bak.{ts}")
    bak.write_bytes(path.read_bytes())
    return bak


def run_summary(
    snapshot_path: Path,
    *,
    dry_run: bool,
    write_backup: bool,
    max_transcript_chars: int | None,
) -> None:
    configure_logging()
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    raw_messages = payload.get("messages") or []
    if not raw_messages:
        raise ValueError("快照中没有 messages。")

    thread_id = payload.get("thread_id", "")
    transcript = messages_json_to_transcript(
        raw_messages,
        max_chars=max_transcript_chars,
        include_tools=False,
    )

    current_soul = _read_text(SOUL_PATH)
    current_user = _read_text(USER_PATH)
    current_memory = _read_text(MEMORY_PATH)

    summarizer_system = """你是一个「记忆整理」助手。输入包含：
1) 当前三个记忆文件的完整内容（可能为空）
2) 一段对话转录

请根据对话**适度**更新三个文件的内容：
- soul.md：仅当对话体现了应对人设/语气/原则的调整时才改；不要堆砌闲聊。
- user.md：用户偏好、称呼、习惯、长期目标、明确禁忌。
- Memory.md：值得跨会话记住的事实、约定、任务进度；条目化，去重合并旧内容。

原则：保守修改，不要删除仍有效的旧信息；新增内容要有依据；不要编造对话里没出现的「事实」。
输出必须通过结构化字段给出三个文件的**完整最终 Markdown 正文**（整文件替换，不是 patch）。"""

    user_block = f"""## 元数据
- snapshot: {snapshot_path.name}
- thread_id: {thread_id}

## 当前 soul.md
```markdown
{current_soul or "(空)"}
```

## 当前 user.md
```markdown
{current_user or "(空)"}
```

## 当前 Memory.md
```markdown
{current_memory or "(空)"}
```

## 对话转录（User / Assistant）
{transcript}
"""

    model = _build_chat_model(streaming=False)
    # DeepSeek 等网关不支持 OpenAI 的 json_schema response_format；用工具调用解析结构化输出。
    structured = model.with_structured_output(
        UpdatedMemoryFiles, method="function_calling"
    )
    result: UpdatedMemoryFiles = structured.invoke(
        [
            {"role": "system", "content": summarizer_system},
            {"role": "user", "content": user_block},
        ]
    )

    soul_preview = (result.soul_md[:800] + "…") if len(result.soul_md) > 800 else result.soul_md
    user_preview = (result.user_md[:800] + "…") if len(result.user_md) > 800 else result.user_md
    mem_preview = (
        (result.memory_md[:800] + "…") if len(result.memory_md) > 800 else result.memory_md
    )
    logger.info("--- 拟写入 soul.md（前 800 字预览）---\n%s", soul_preview)
    logger.info("--- 拟写入 user.md（前 800 字预览）---\n%s", user_preview)
    logger.info("--- 拟写入 Memory.md（前 800 字预览）---\n%s", mem_preview)

    if dry_run:
        logger.info("dry-run：未写入文件")
        return

    MEM_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for path, content in [
        (SOUL_PATH, result.soul_md),
        (USER_PATH, result.user_md),
        (MEMORY_PATH, result.memory_md),
    ]:
        if write_backup:
            bak = _backup_file(path)
            if bak:
                written.append(str(bak))
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        written.append(str(path))

    logger.info("已写入:\n%s", "\n".join(f"  {p}" for p in written))


def _parse_max_chars(s: str | None) -> int | None:
    if not s:
        raw = os.environ.get("SUMMARY_MAX_TRANSCRIPT_CHARS", "48000") or "48000"
        return int(raw)
    v = s.strip().lower()
    if v in {"", "0", "none", "full"}:
        return None
    return int(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="根据对话快照更新 agent_memory/*.md")
    parser.add_argument(
        "snapshot",
        nargs="?",
        default=None,
        help="conversation_history 下的快照 json；省略则用最新文件",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只调用模型并预览，不写文件",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="写入前不备份 .bak.*",
    )
    parser.add_argument(
        "--max-chars",
        default=None,
        help="转录最大字符数；默认环境变量 SUMMARY_MAX_TRANSCRIPT_CHARS 或 48000；0 表示不截断",
    )
    args = parser.parse_args()

    snap = Path(args.snapshot).resolve() if args.snapshot else _find_latest_snapshot()
    if not snap.is_file():
        raise SystemExit(f"文件不存在: {snap}")

    max_chars = _parse_max_chars(args.max_chars)
    run_summary(
        snap,
        dry_run=args.dry_run,
        write_backup=not args.no_backup,
        max_transcript_chars=max_chars,
    )


if __name__ == "__main__":
    main()
