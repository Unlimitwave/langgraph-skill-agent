#!/usr/bin/env bash
# 幂等创建 Docker / 裸机隔离库（已有 pg_data 卷时 make postgres-init-dbs 调用）。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then
	set -a
	# shellcheck disable=SC1091
	source .env
	set +a
fi

USER="${POSTGRES_USER:-langgraph}"
DOCKER_DB="${POSTGRES_DB_DOCKER:-langgraph_docker}"
LOCAL_DB="${POSTGRES_DB_LOCAL:-langgraph_local}"
SERVICE="${POSTGRES_SERVICE:-postgres}"

compose=(docker compose -f docker-compose.yml -f docker-compose.prod.yml)

if ! "${compose[@]}" ps --status running --services 2>/dev/null | grep -qx "$SERVICE"; then
	echo "Postgres 未运行，正在启动…" >&2
	"${compose[@]}" up -d "$SERVICE"
fi

_create_db() {
	local db="$1"
	local exists
	exists="$("${compose[@]}" exec -T "$SERVICE" psql -U "$USER" -d postgres -tAc \
		"SELECT 1 FROM pg_database WHERE datname = '${db}'" | tr -d '[:space:]')"
	if [ "$exists" = "1" ]; then
		echo "  skip: ${db} (exists)"
		return 0
	fi
	"${compose[@]}" exec -T "$SERVICE" psql -U "$USER" -d postgres -v ON_ERROR_STOP=1 \
		-c "CREATE DATABASE \"${db}\""
	"${compose[@]}" exec -T "$SERVICE" psql -U "$USER" -d postgres -v ON_ERROR_STOP=1 \
		-c "GRANT ALL PRIVILEGES ON DATABASE \"${db}\" TO \"${USER}\""
	echo "  created: ${db}"
}

echo "Ensuring Postgres databases (user=${USER})…"
_create_db "$DOCKER_DB"
_create_db "$LOCAL_DB"
echo "OK: docker=${DOCKER_DB}, local=${LOCAL_DB}"
