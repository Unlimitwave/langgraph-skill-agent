"""Skill 脚本执行工具：本机 argv、白名单 shell、Docker 隔离。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Literal

from langchain_core.tools import tool

from langgraph_skill_agent.utility.paths import PROJECT_ROOT, SKILLS_DIR

_MAX_ARGV = 48
_MAX_ARG_LEN = 8000
_TIMEOUT_S = 120
_MAX_SCRIPT_ARGS = 8
_BASH_PATH = "/bin/bash"
_ALLOWED_PROGRAM = frozenset({"python", "python3"})

SkillScriptId = Literal["test-calc.run"]

_SKILL_SCRIPT_REGISTRY: dict[SkillScriptId, str] = {
    "test-calc.run": "skills/test-calc-script/run_calc.sh",
}
_SKILL_SCRIPT_TIMEOUT_S: dict[SkillScriptId, int] = {
    "test-calc.run": 120,
}

_DEFAULT_IMAGE = "python:3.12-slim"
_DEFAULT_DOCKER_TIMEOUT = 120
_MAX_OUTPUT_BYTES = 256_000


def _looks_like_filesystem_path(arg: str) -> bool:
    if arg.startswith("-"):
        return False
    return "/" in arg or "\\" in arg or arg.endswith(".py")


def _must_stay_under_repo(repo: Path, path_arg: str) -> tuple[bool, str]:
    if not _looks_like_filesystem_path(path_arg):
        return True, ""
    if "\x00" in path_arg or ".." in path_arg:
        return False, "invalid path token"
    candidate = (repo / path_arg).resolve()
    try:
        candidate.relative_to(repo.resolve())
    except ValueError:
        return False, f"path outside workspace: {path_arg!r}"
    return True, ""


def _format_proc_output(proc: subprocess.CompletedProcess[str]) -> str:
    parts: list[str] = []
    if proc.stdout:
        parts.append(proc.stdout.rstrip())
    if proc.stderr:
        parts.append("[stderr]\n" + proc.stderr.rstrip())
    parts.append(f"[exit_code={proc.returncode}]")
    return "\n".join(parts).strip()


def _validate_script_args(script_id: SkillScriptId, script_args: list[str] | None) -> str | None:
    if script_args is None:
        return None
    if not isinstance(script_args, list) or len(script_args) > _MAX_SCRIPT_ARGS:
        return f"error: script_args must be a list of length <= {_MAX_SCRIPT_ARGS}"
    for a in script_args:
        if not isinstance(a, str):
            return "error: every script_args element must be a string"
        if len(a) > _MAX_ARG_LEN or "\x00" in a or ".." in a:
            return "error: invalid script_args token"
    if script_id == "test-calc.run" and script_args:
        return "error: test-calc.run does not accept script_args"
    return None


def _resolve_registered_script(repo_root: Path, script_id: SkillScriptId) -> tuple[Path | None, str]:
    rel = _SKILL_SCRIPT_REGISTRY.get(script_id)
    if rel is None:
        return None, f"error: unknown script_id {script_id!r}"
    ok, err = _must_stay_under_repo(repo_root, rel)
    if not ok:
        return None, f"error: {err}"
    path = (repo_root / rel).resolve()
    if not path.is_file():
        return None, f"error: script not found: {rel}"
    return path, ""


def _normalize_script_path(skill_relative: str) -> Path:
    raw = skill_relative.strip().replace("\\", "/").lstrip("/")
    if raw.startswith("skills/"):
        raw = raw[len("skills/") :]
    if not raw or ".." in Path(raw).parts:
        raise ValueError("非法路径：仅允许 skills 目录下的相对路径，且不能包含 ..")
    candidate = (SKILLS_DIR / raw).resolve()
    try:
        candidate.relative_to(SKILLS_DIR.resolve())
    except ValueError as e:
        raise ValueError("路径必须位于 skills/ 目录内") from e
    if not candidate.is_file():
        raise ValueError(f"文件不存在或不是普通文件: {candidate.relative_to(PROJECT_ROOT)}")
    if candidate.suffix.lower() != ".py":
        raise ValueError("仅允许执行 .py 脚本")
    return candidate


def make_host_skill_tools(repo_root: Path) -> list:
    """本机执行：workspace_exec（Python argv）与 run_skill_script（白名单 shell）。"""
    repo_root = repo_root.resolve()

    @tool
    def workspace_exec(
        program: Annotated[str, "Must be `python` or `python3`."],
        argv_tail: Annotated[
            list[str],
            'Argv after interpreter, e.g. ["skills/test-calc-script/run_calc.py"].',
        ],
    ) -> str:
        """Run Python from repo root without a shell. For shell scripts use run_skill_script."""
        prog = program.strip().lower()
        if prog not in _ALLOWED_PROGRAM:
            return f"error: program must be one of {sorted(_ALLOWED_PROGRAM)}, got {program!r}"
        if not isinstance(argv_tail, list) or len(argv_tail) > _MAX_ARGV:
            return f"error: argv_tail must be a list of length <= {_MAX_ARGV}"

        i = 0
        while i < len(argv_tail):
            a = argv_tail[i]
            if not isinstance(a, str):
                return "error: every argv element must be a string"
            if len(a) > _MAX_ARG_LEN:
                return "error: argument too long"
            if a == "-c" and i + 1 < len(argv_tail):
                if "\x00" in argv_tail[i + 1] or len(argv_tail[i + 1]) > _MAX_ARG_LEN:
                    return "error: invalid -c payload"
                i += 2
                continue
            ok, err = _must_stay_under_repo(repo_root, a)
            if not ok:
                return f"error: {err}"
            i += 1

        try:
            proc = subprocess.run(
                [sys.executable, *argv_tail],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return f"error: timeout after {_TIMEOUT_S}s"
        except OSError as e:
            return f"error: failed to spawn process: {e}"
        return _format_proc_output(proc)

    @tool
    def run_skill_script(
        script_id: Annotated[SkillScriptId, "Registered skill shell script id."],
        script_args: Annotated[list[str] | None, "Optional argv for the shell script."] = None,
    ) -> str:
        """Run a whitelisted bash script from repo root. Do NOT use workspace_exec with bash."""
        arg_err = _validate_script_args(script_id, script_args)
        if arg_err:
            return arg_err
        path, err = _resolve_registered_script(repo_root, script_id)
        if err:
            return err
        assert path is not None
        timeout = _SKILL_SCRIPT_TIMEOUT_S.get(script_id, _TIMEOUT_S)
        try:
            proc = subprocess.run(
                [_BASH_PATH, str(path), *(script_args or [])],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"error: timeout after {timeout}s (script_id={script_id})"
        except OSError as e:
            return f"error: failed to spawn process: {e}"
        return _format_proc_output(proc)

    return [workspace_exec, run_skill_script]


@tool
def run_skill_script_in_docker(skill_script_path: str) -> str:
    """在 Docker 容器中执行 skills/ 下的 Python 脚本（只读挂载，无网络）。"""
    t0 = time.perf_counter()
    docker_exe = shutil.which("docker")
    if not docker_exe:
        return "错误：未找到 docker 可执行文件，请安装 Docker 并确保 PATH 可用。"

    try:
        host_script = _normalize_script_path(skill_script_path)
    except ValueError as e:
        return f"错误：{e}"

    image = os.environ.get("SKILL_DOCKER_IMAGE", _DEFAULT_IMAGE).strip() or _DEFAULT_IMAGE
    try:
        timeout_s = int(os.environ.get("SKILL_DOCKER_TIMEOUT", str(_DEFAULT_DOCKER_TIMEOUT)))
    except ValueError:
        timeout_s = _DEFAULT_DOCKER_TIMEOUT
    timeout_s = max(5, min(timeout_s, 3600))

    rel = host_script.relative_to(SKILLS_DIR.resolve())
    inner_py = f"/workspace/skills/{rel.as_posix()}"
    skills_mount = f"{SKILLS_DIR.resolve().as_posix()}:/workspace/skills:ro"

    try:
        completed = subprocess.run(
            [
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
            ],
            capture_output=True,
            timeout=timeout_s,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except subprocess.TimeoutExpired:
        return f"错误：Docker 执行超过 {timeout_s} 秒已终止。"
    except FileNotFoundError:
        return "错误：docker 命令启动失败（FileNotFoundError）。"

    out_parts = [completed.stdout, completed.stderr]
    combined = "\n".join(p for p in out_parts if p).rstrip("\n")
    if len(combined.encode("utf-8", errors="replace")) > _MAX_OUTPUT_BYTES:
        combined = combined[:_MAX_OUTPUT_BYTES] + "\n...(truncated: output too large)"

    header = (
        f"[docker] image={image} script={host_script.relative_to(PROJECT_ROOT)} "
        f"exit={completed.returncode} elapsed={time.perf_counter() - t0:.2f}s\n"
    )
    body = combined if combined else "(no stdout/stderr)"
    if completed.returncode != 0:
        return header + body + f"\n\n(进程退出码 {completed.returncode})"
    return header + body
