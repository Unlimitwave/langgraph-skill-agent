#!/bin/bash
# 首次初始化 pg_data 卷时执行：POSTGRES_DB 已由镜像创建，此处补建裸机开发库。
set -euo pipefail

LOCAL_DB="${POSTGRES_DB_LOCAL:-langgraph_local}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE ${LOCAL_DB};
    GRANT ALL PRIVILEGES ON DATABASE ${LOCAL_DB} TO ${POSTGRES_USER};
EOSQL
