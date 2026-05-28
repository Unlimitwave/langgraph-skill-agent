"""
在只读挂载的 skills/ 目录下，用一次性 Docker 容器执行白名单内的 Python 脚本。

环境变量（可选）：
  SKILL_DOCKER_IMAGE   默认 python:3.12-slim
  SKILL_DOCKER_TIMEOUT 秒，默认 120
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parent
SKILLS_ROOT = PROJECT_ROOT / "skills"

_DEFAULT_IMAGE = "python:3.12-slim"
_DEFAULT_TIMEOUT = 120
_MAX_OUTPUT_BYTES = 256_000


def _docker_binary() -> str | None:
    return shutil.which("docker")


def _normalize_script_path(skill_relative: str) -> Path:
    """将用户/模型输入解析为 skills/ 下的真实文件路径，禁止穿越。"""
    raw = skill_relative.strip().replace("\\", "/").lstrip("/")
    if raw.startswith("skills/"):
        raw = raw[len("skills/") :]
    if not raw or ".." in Path(raw).parts:
        raise ValueError("非法路径：仅允许 skills 目录下的相对路径，且不能包含 ..")
    candidate = (SKILLS_ROOT / raw).resolve()
    skills_resolved = SKILLS_ROOT.resolve()
    try:
        candidate.relative_to(skills_resolved)
    except ValueError as e:
        raise ValueError("路径必须位于 skills/ 目录内") from e
    if not candidate.is_file():
        raise ValueError(f"文件不存在或不是普通文件: {candidate.relative_to(PROJECT_ROOT)}")
    if candidate.suffix.lower() != ".py":
        raise ValueError("仅允许执行 .py 脚本")
    return candidate


def _container_script_path(host_file: Path) -> str:
    rel_under_skills = host_file.relative_to(SKILLS_ROOT.resolve())
    return f"/workspace/skills/{rel_under_skills.as_posix()}"


@tool
def run_skill_script_in_docker(skill_script_path: str) -> str:
    """在隔离的 Docker 容器中执行 skills 目录下的 Python 脚本（只读挂载 skills）。

    skill_script_path 示例：test-calc-script/run_calc.py 或 skills/test-calc-script/run_calc.py
    容器内无默认外网（--network none）。标准输出与标准错误合并返回。
    """
    t0 = time.perf_counter()
    docker_exe = _docker_binary()
    if not docker_exe:
        return "错误：未找到 docker 可执行文件，请安装 Docker 并确保 PATH 可用。"

    try:
        host_script = _normalize_script_path(skill_script_path)
    except ValueError as e:
        return f"错误：{e}"

    image = os.environ.get("SKILL_DOCKER_IMAGE", _DEFAULT_IMAGE).strip() or _DEFAULT_IMAGE
    try:
        timeout_s = int(os.environ.get("SKILL_DOCKER_TIMEOUT", str(_DEFAULT_TIMEOUT)))
    except ValueError:
        timeout_s = _DEFAULT_TIMEOUT
    timeout_s = max(5, min(timeout_s, 3600))

    inner_py = _container_script_path(host_script)
    skills_mount = f"{SKILLS_ROOT.resolve().as_posix()}:/workspace/skills:ro"

    cmd = [
        docker_exe,
        "run",
        "--rm",
        "--network",
        "none",
        "-v",
        skills_mount,
        "-w",
        "/workspace",
        image,
        "python",
        inner_py,
    ]

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout_s,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except subprocess.TimeoutExpired:
        return f"错误：Docker 执行超过 {timeout_s} 秒已终止。"
    except FileNotFoundError:
        return "错误：docker 命令启动失败（FileNotFoundError）。"

    out_parts: list[str] = []
    if completed.stdout:
        out_parts.append(completed.stdout)
    if completed.stderr:
        out_parts.append(completed.stderr)
    combined = "\n".join(out_parts).rstrip("\n")
    if len(combined.encode("utf-8", errors="replace")) > _MAX_OUTPUT_BYTES:
        combined = combined[:_MAX_OUTPUT_BYTES] + "\n...(truncated: output too large)"

    elapsed = time.perf_counter() - t0
    header = (
        f"[docker] image={image} script={host_script.relative_to(PROJECT_ROOT)} "
        f"exit={completed.returncode} elapsed={elapsed:.2f}s\n"
    )
    body = combined if combined else "(no stdout/stderr)"
    if completed.returncode != 0:
        return header + body + f"\n\n(进程退出码 {completed.returncode})"
    return header + body
