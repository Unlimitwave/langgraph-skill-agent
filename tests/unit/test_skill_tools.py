"""Unit tests for skill script path guards."""

from pathlib import Path

from langgraph_skill_agent.utility.agent_policy import SkillExecContext


def _ctx(tmp_path: Path) -> SkillExecContext:
    ws = tmp_path / "workspace" / "alice"
    system = tmp_path / "platform-skills"
    (ws / "skills" / "mine").mkdir(parents=True)
    (system / "demo").mkdir(parents=True)
    return SkillExecContext(agent_workspace=ws, system_skills_dir=system)


def test_resolve_system_skill_script(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    script = ctx.system_skills_dir / "demo" / "run.py"
    script.write_text("print(1)", encoding="utf-8")

    resolved, err = ctx.resolve_script_path("/system-skills/demo/run.py")
    assert err == ""
    assert resolved == script.resolve()


def test_resolve_user_skill_script(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    script = ctx.agent_workspace / "skills" / "mine" / "run.py"
    script.write_text("print(1)", encoding="utf-8")

    resolved, err = ctx.resolve_script_path("skills/mine/run.py")
    assert err == ""
    assert resolved == script.resolve()


def test_resolve_rejects_src_paths(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _, err = ctx.resolve_script_path("src/agent_core.py")
    assert err
    _, err = ctx.resolve_script_path("/var/agent_memory/soul.md")
    assert err
