"""LangGraph Deep Agent with local Skills and RAG."""

from langgraph_skill_agent.utility.paths import (
    AGENT_MEMORY_DIR,
    CONVERSATION_HISTORY_DIR,
    PROJECT_ROOT,
    RAG_DATA_DIR,
    RAG_STORAGE_DIR,
    SKILLS_DIR,
    VAR_DIR,
    get_project_root,
)

__all__ = [
    "AGENT_MEMORY_DIR",
    "CONVERSATION_HISTORY_DIR",
    "PROJECT_ROOT",
    "RAG_DATA_DIR",
    "RAG_STORAGE_DIR",
    "SKILLS_DIR",
    "VAR_DIR",
    "get_project_root",
]
__version__ = "0.1.0"
