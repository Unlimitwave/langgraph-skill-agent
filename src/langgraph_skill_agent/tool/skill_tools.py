"""Skill 脚本执行工具：本机 argv、白名单 shell。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated, Literal

from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime

from langgraph_skill_agent.utility.agent_policy import SkillExecContext, resolve_agent_scope
from langgraph_skill_agent.utility.tenant import AgentContext

_MAX_ARGV = 48
_MAX_ARG_LEN = 8000
_TIMEOUT_S = 120
_MAX_SCRIPT_ARGS = 8
_BASH_PATH = "/bin/bash"
_ALLOWED_PROGRAM = frozenset({"python", "python3"})

SkillScriptId = Literal["test-calc.run"]

# Paths relative to platform SKILLS_DIR on disk.
_SKILL_SCRIPT_REGISTRY: dict[SkillScriptId, str] = {
    "test-calc.run": "test-calc-script/run_calc.sh",
}
_SKILL_SCRIPT_TIMEOUT_S: dict[SkillScriptId, int] = {
    "test-calc.run": 120,
}


def _looks_like_filesystem_path(arg: str) -> bool:
    if arg.startswith("-"):
        return False
    return "/" in arg or "\\" in arg or arg.endswith(".py")


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


def _resolve_registered_script(
    ctx: SkillExecContext, script_id: SkillScriptId
) -> tuple[Path | None, str]:
    rel = _SKILL_SCRIPT_REGISTRY.get(script_id)
    if rel is None:
        return None, f"error: unknown script_id {script_id!r}"
    path = (ctx.system_skills_dir / rel).resolve()
    try:
        path.relative_to(ctx.system_skills_dir.resolve())
    except ValueError:
        return None, f"error: script escapes system skills dir: {rel!r}"
    if not path.is_file():
        return None, f"error: script not found: {rel}"
    return path, ""


def make_host_skill_tools() -> list:
    """本机执行：workspace_exec_python（Python argv）与 run_skill_script_shell（白名单 shell）."""

    @tool
    def workspace_exec_python(
        program: Annotated[str, "Must be `python` or `python3`."],
        argv_tail: Annotated[
            list[str],
            'Argv after interpreter, e.g. ["/system-skills/.../run.py"] or ["skills/.../run.py"].',
        ],
        runtime: ToolRuntime[AgentContext],
    ) -> str:
        """Run Python without a shell. Paths must be under /system-skills/ or skills/."""
        scope = resolve_agent_scope(runtime.context)
        ctx = scope.skill_exec
        workspace = scope.workspace.resolve()
        prog = program.strip().lower()
        if prog not in _ALLOWED_PROGRAM:
            return f"error: program must be one of {sorted(_ALLOWED_PROGRAM)}, got {program!r}"
        if not isinstance(argv_tail, list) or len(argv_tail) > _MAX_ARGV:
            return f"error: argv_tail must be a list of length <= {_MAX_ARGV}"

        resolved_argv: list[str] = []
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
                resolved_argv.extend(["-c", argv_tail[i + 1]])
                i += 2
                continue
            if _looks_like_filesystem_path(a):
                resolved, err = ctx.resolve_script_path(a)
                if err:
                    return f"error: {err}"
                assert resolved is not None
                resolved_argv.append(str(resolved))
            else:
                resolved_argv.append(a)
            i += 1

        try:
            proc = subprocess.run(
                [sys.executable, *resolved_argv],
                cwd=str(workspace),
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
    def run_skill_script_shell(
        script_id: Annotated[SkillScriptId, "Registered skill shell script id."],
        runtime: ToolRuntime[AgentContext],
        script_args: Annotated[list[str] | None, "Optional argv for the shell script."] = None,
    ) -> str:
        """Run a whitelisted bash script from platform skills. Do NOT use workspace_exec_python with bash."""
        scope = resolve_agent_scope(runtime.context)
        ctx = scope.skill_exec
        workspace = scope.workspace.resolve()
        arg_err = _validate_script_args(script_id, script_args)
        if arg_err:
            return arg_err
        path, err = _resolve_registered_script(ctx, script_id)
        if err:
            return err
        assert path is not None
        timeout = _SKILL_SCRIPT_TIMEOUT_S.get(script_id, _TIMEOUT_S)
        try:
            proc = subprocess.run(
                [_BASH_PATH, str(path), *(script_args or [])],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"error: timeout after {timeout}s (script_id={script_id})"
        except OSError as e:
            return f"error: failed to spawn process: {e}"
        return _format_proc_output(proc)

    return [workspace_exec_python, run_skill_script_shell]
