"""企业级提示词注册表：版本化 manifest + 进程内渲染缓存。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

_MANIFEST_NAME = "manifest.json"
_DEFAULT_VERSION_ENV = "PROMPT_DEFAULT_VERSION"


@dataclass(frozen=True)
class PromptMeta:
    """已解析提示词元数据（便于日志/观测）。"""

    prompt_id: str
    version: str
    description: str
    stable: bool
    content_hash: str
    content: str


def _env_version_key(prompt_id: str) -> str:
    return f"PROMPT_{prompt_id.upper().replace('.', '_')}_VERSION"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@lru_cache(maxsize=1)
def _load_manifest() -> dict[str, Any]:
    raw = resources.files(__package__).joinpath(_MANIFEST_NAME).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict) or "prompts" not in data:
        raise ValueError(f"Invalid prompt manifest: {_MANIFEST_NAME}")
    return data


def _resolve_version(prompt_id: str, version: str | None) -> str:
    if version:
        return version
    per_prompt = os.environ.get(_env_version_key(prompt_id))
    if per_prompt and per_prompt.strip():
        return per_prompt.strip()
    default = os.environ.get(_DEFAULT_VERSION_ENV)
    if default and default.strip():
        return default.strip()
    manifest = _load_manifest()
    manifest_default = manifest.get("default_version")
    if isinstance(manifest_default, str) and manifest_default.strip():
        return manifest_default.strip()
    versions = _prompt_versions(prompt_id)
    if not versions:
        raise KeyError(f"Unknown prompt id: {prompt_id!r}")
    return sorted(versions)[-1]


def _prompt_versions(prompt_id: str) -> dict[str, Any]:
    manifest = _load_manifest()
    prompts = manifest.get("prompts") or {}
    entry = prompts.get(prompt_id)
    if not isinstance(entry, dict):
        return {}
    return {k: v for k, v in entry.items() if isinstance(v, dict)}


def _raw_prompt_entry(prompt_id: str, version: str) -> dict[str, Any]:
    versions = _prompt_versions(prompt_id)
    entry = versions.get(version)
    if not isinstance(entry, dict):
        known = ", ".join(sorted(versions)) or "(none)"
        raise KeyError(f"Prompt {prompt_id!r} has no version {version!r}; known: {known}")
    content = entry.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Prompt {prompt_id!r}@{version} has empty content")
    return entry


@lru_cache(maxsize=256)
def _render_prompt(
    prompt_id: str,
    version: str,
    fmt_items: tuple[tuple[str, str], ...],
) -> str:
    entry = _raw_prompt_entry(prompt_id, version)
    content = entry["content"]
    if fmt_items:
        content = content.format(**dict(fmt_items))
    return content


def resolve_prompt(
    prompt_id: str,
    *,
    version: str | None = None,
    **fmt: Any,
) -> PromptMeta:
    """解析提示词并返回元数据（含 content_hash，便于审计与缓存命中观测）。"""
    resolved_version = _resolve_version(prompt_id, version)
    fmt_items = tuple(sorted((k, str(v)) for k, v in fmt.items()))
    content = _render_prompt(prompt_id, resolved_version, fmt_items)
    entry = _raw_prompt_entry(prompt_id, resolved_version)
    return PromptMeta(
        prompt_id=prompt_id,
        version=resolved_version,
        description=str(entry.get("description") or ""),
        stable=bool(entry.get("stable", True)),
        content_hash=_content_hash(content),
        content=content,
    )


def get_prompt(
    prompt_id: str,
    *,
    version: str | None = None,
    **fmt: Any,
) -> str:
    """加载版本化提示词；模板变量通过关键字参数传入。"""
    return resolve_prompt(prompt_id, version=version, **fmt).content


def list_prompt_ids() -> list[str]:
    manifest = _load_manifest()
    prompts = manifest.get("prompts") or {}
    return sorted(prompts.keys())


def clear_prompt_cache() -> None:
    """测试或热重载 manifest 后清空进程内缓存。"""
    _load_manifest.cache_clear()
    _render_prompt.cache_clear()


__all__ = [
    "PromptMeta",
    "clear_prompt_cache",
    "get_prompt",
    "list_prompt_ids",
    "resolve_prompt",
]
