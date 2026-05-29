"""Streamlit Web UI for LangGraph Skill Agent."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_APP_PATH = Path(__file__).resolve().parent / "app.py"


def main() -> None:
    """Launch the Streamlit app (requires optional ``ui`` extras)."""
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(_APP_PATH), *sys.argv[1:]],
        check=True,
    )
