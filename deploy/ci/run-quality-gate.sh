#!/usr/bin/env bash
# Gitee Go / 本地 CI 质量门禁入口（与 Makefile `make ci` 对齐）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v make >/dev/null 2>&1; then
  echo "error: make is required (install build-essential / make)" >&2
  exit 1
fi

make ci
