"""Specialist 角色定义：prompt、工具白名单、文件系统权限。"""

from __future__ import annotations

from enum import StrEnum

from deepagents.middleware.filesystem import FilesystemPermission

from langgraph_skill_agent.utility.agent_policy import (
    SYSTEM_SKILLS_ROUTE,
    agent_filesystem_permissions,
    agent_skill_sources,
)


class AgentRole(StrEnum):
    """Supervisor 可调度的 Specialist 角色。"""

    RESEARCH = "research"
    WORKER = "worker"
    REVIEW = "review"


# 角色系统提示词
ROLE_SYSTEM_PROMPTS: dict[AgentRole, str] = {
    AgentRole.RESEARCH: (
        "You are a Research Specialist. Your job is to gather facts from the knowledge base "
        "and readable workspace files only.\n"
        "- Use rag_search and read_file; never write or edit files.\n"
        "- Cite sources (file paths / RAG hits) in your summary.\n"
        "- End with a concise structured brief the Worker can act on."
    ),
    AgentRole.WORKER: (
        "You are a Worker Specialist. Execute the assigned step using tools and skills.\n"
        "- Follow [CTX-SYSTEM] persona and memory when present.\n"
        "- Produce concrete artifacts (files, scripts output) in the user workspace.\n"
        "- Stay within the single step objective; do not replan the whole project."
    ),
    AgentRole.REVIEW: (
        "You are a Review Specialist (read-only). Audit the step output against acceptance criteria.\n"
        "- Use read_file only; never write or edit.\n"
        "- Compare against prior Research briefs and Worker artifacts when provided.\n"
        "【输出格式】回复末尾必须包含且仅包含一个 JSON 对象（不要 markdown 代码块）：\n"
        '{"passed": true|false, "feedback": "具体改进建议或确认说明", "summary": "审查结论一句话"}'
    ),
}


def read_only_filesystem_permissions() -> list[FilesystemPermission]:
    """Research / Review：全局只读，禁止任何写操作。"""
    return [
        FilesystemPermission(
            operations=["write"],
            paths=["/**"],
            mode="deny",
        ),
        FilesystemPermission(
            operations=["read"],
            paths=["/**"],
            mode="allow",
        ),
    ]


def permissions_for_role(role: AgentRole) -> list[FilesystemPermission]:
    if role is AgentRole.WORKER:
        return agent_filesystem_permissions()
    return read_only_filesystem_permissions()


def skill_sources_for_role(role: AgentRole) -> list[str]:
    if role is AgentRole.WORKER:
        return agent_skill_sources()
    return [SYSTEM_SKILLS_ROUTE]


def interrupt_on_for_role(role: AgentRole) -> dict[str, bool]:
    """Worker 写操作需 HITL；只读角色一律拦截写类工具。"""
    if role is AgentRole.WORKER:
        return {
            "write_file": True,
            "read_file": False,
            "edit_file": True,
            "workspace_exec_python": False,
            "run_skill_script_shell": False,
        }
    return {
        "write_file": True,
        "read_file": False,
        "edit_file": True,
        "workspace_exec_python": True,
        "run_skill_script_shell": True,
    }


def role_display_name(role: AgentRole) -> str:
    return {
        AgentRole.RESEARCH: "Research",
        AgentRole.WORKER: "Worker",
        AgentRole.REVIEW: "Review",
    }[role]


def role_avatar(role: AgentRole | str) -> str:
    """Streamlit chat_message avatar（机器人 / 角色图标）。"""
    key = role.value if isinstance(role, AgentRole) else str(role).lower()
    return {
        "supervisor": "🧭",
        "research": "🤖",
        "worker": "🦾",
        "review": "🔎",
        "plan": "📋",
    }.get(key, "🤖")


def role_caption(role: AgentRole | str) -> str:
    key = role.value if isinstance(role, AgentRole) else str(role).lower()
    if key == "supervisor":
        return "Supervisor"
    if key == "plan":
        return "Plan"
    try:
        return role_display_name(AgentRole(key))
    except ValueError:
        return key.capitalize()
