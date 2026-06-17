# syntax=docker/dockerfile:1
# langgraph-skill-agent — Streamlit UI runtime image (Phase 1 标准制品)
#
# Build:  make docker-build
# Run:    make docker-up  (see docker-compose.yml)

# --- builder: resolve dependencies with uv (frozen lockfile) ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
COPY src ./src

RUN uv sync --frozen --no-dev --extra ui --no-editable

# --- runtime: minimal image, non-root user ---
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

RUN groupadd -r app && useradd -r -g app -d /app app \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PROJECT_ROOT=/app \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app pyproject.toml uv.lock ./
COPY --chown=app:app src ./src
COPY --chown=app:app skills ./skills

RUN mkdir -p /app/var && chown -R app:app /app

USER app

EXPOSE 8501

# Streamlit built-in health endpoint (no custom FastAPI /health in this project)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -sf http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/langgraph_skill_agent/frontend/app.py", \
    "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
