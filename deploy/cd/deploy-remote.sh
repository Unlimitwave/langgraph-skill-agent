#!/usr/bin/env bash
# 远程 CD 部署入口：拉 Registry 不可变镜像 → compose up → 健康检查
# 由 GitHub Actions Deploy workflow 或运维手动调用。
#
# 前置（服务器）:
#   - 已安装 Docker + Compose plugin
#   - 仓库部署目录含 docker-compose.yml / docker-compose.prod.yml / deploy/
#   - .env 已配置（POSTGRES_PASSWORD、API Key 等，不进镜像）
#   - private GHCR: docker login ghcr.io
#
# 用法:
#   export IMAGE=ghcr.io/<owner>/langgraph-skill-agent
#   export TAG=<git-sha>
#   ./deploy/cd/deploy-remote.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

: "${IMAGE:?set IMAGE e.g. ghcr.io/owner/langgraph-skill-agent}"
: "${TAG:?set TAG e.g. git commit SHA}"

APP_PORT="${APP_PORT:-8501}"
HEALTH_URL="http://127.0.0.1:${APP_PORT}/_stcore/health"

echo "==> Deploy ${IMAGE}:${TAG}"

docker compose -f docker-compose.yml -f docker-compose.prod.yml pull app
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

if [ -x deploy/postgres/ensure-databases.sh ]; then
  chmod +x deploy/postgres/ensure-databases.sh deploy/postgres/init-databases.sh 2>/dev/null || true
  ./deploy/postgres/ensure-databases.sh
fi

echo "==> Waiting for health: ${HEALTH_URL}"
for _ in $(seq 1 30); do
  if curl -sf "$HEALTH_URL" >/dev/null; then
    echo "OK: health check passed"
    docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
    exit 0
  fi
  sleep 2
done

echo "error: health check failed after 60s" >&2
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs --tail=80 app || true
exit 1
