# langgraph-skill-agent 常用命令（跨层编排：质量 / 测试 / 制品）
# 用法: make help
#
# 铁律: Makefile、pre-commit、CI 共用 pyproject.toml 中的 ruff / pytest 配置。
# 前置: uv (https://docs.astral.sh/uv/)，Python 3.12（见 .python-version）

.PHONY: help install lint format test test-integration run-ui run-agent \
	python-check pre-commit-install pre-commit check ci clean \
	build docker-build docker-up docker-up-milvus docker-down docker-logs \
	docker-stack-up docker-prod-up postgres-init-dbs

COMPOSE_FILES = -f docker-compose.yml
COMPOSE_MILVUS = $(COMPOSE_FILES) -f docker-compose.milvus.yml

# --- 镜像标签（本地默认 local；CI/CD 请传 TAG=$(git rev-parse --short HEAD)）---
IMAGE ?= langgraph-skill-agent
TAG   ?= local

# ---------------------------------------------------------------------------
# 环境与依赖
# ---------------------------------------------------------------------------

help: ## 显示所有命令
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

install: ## 安装开发依赖（uv sync）
	uv sync --all-groups --extra ui

python-check: ## 校验 Python 3.12.x
	@uv run python -c "import sys; v=sys.version_info; assert v.major==3 and v.minor==12, f'expected Python 3.12.x, got {sys.version}'; print(f'OK: Python {sys.version.split()[0]}')"

pre-commit-install: install ## 安装 git pre-commit hook
	uv run pre-commit install

pre-commit: ## 手动跑 pre-commit（全文件）
	uv run pre-commit run --all-files

# ---------------------------------------------------------------------------
# 第二层：代码质量
# ---------------------------------------------------------------------------

lint: ## Ruff 静态检查
	uv run ruff check src tests

format: ## Ruff 格式化 + 可安全 autofix
	uv run ruff format src tests
	uv run ruff check --fix src tests

# ---------------------------------------------------------------------------
# 第三层：自动化测试
# ---------------------------------------------------------------------------

test: ## 单元测试（跳过 integration）
	uv run pytest -q -m "not integration"

test-integration: ## 集成测试（需 Milvus 等，见 docker-compose）
	uv run pytest -q -m integration

# ---------------------------------------------------------------------------
# 第四层：构建与制品
# ---------------------------------------------------------------------------

build: ## Python wheel 打包（dist/）
	uv build

docker-build: ## 构建 Docker 镜像（TAG 默认 local）
	docker build -t $(IMAGE):$(TAG) .

docker-up: ## 启动 app 容器（Milvus 用 .env 远程地址）
	docker compose $(COMPOSE_FILES) up -d --build

docker-up-milvus: ## 启动 app + 本地 Milvus 栈（无远程 Milvus 时用）
	docker compose $(COMPOSE_MILVUS) up -d --build

docker-down: ## 停止 compose 容器（含可选 Milvus 栈）
	docker compose $(COMPOSE_MILVUS) down 2>/dev/null || true
	docker compose $(COMPOSE_FILES) down

docker-logs: ## 跟踪 app 容器日志
	docker compose logs -f app

docker-stack-up: ## 本地全栈：app + Postgres（docker/local 分库，PG 映射 127.0.0.1:5432）
	docker compose $(COMPOSE_FILES) -f docker-compose.prod.yml up -d --build
	@$(MAKE) postgres-init-dbs
	@docker compose $(COMPOSE_FILES) -f docker-compose.prod.yml up -d app

postgres-init-dbs: ## 幂等创建 langgraph_docker / langgraph_local 隔离库
	@chmod +x deploy/postgres/ensure-databases.sh deploy/postgres/init-databases.sh
	@./deploy/postgres/ensure-databases.sh

docker-prod-up: ## 生产/测试服：仅拉 Registry 镜像（需 IMAGE + TAG）
	docker compose $(COMPOSE_FILES) -f docker-compose.prod.yml up -d
	@$(MAKE) postgres-init-dbs

# ---------------------------------------------------------------------------
# 运行应用（裸机 / 虚拟环境）
# ---------------------------------------------------------------------------

run-ui: ## 启动 Streamlit UI（http://localhost:8501）
	uv run langgraph-ui

run-agent: ## 启动 CLI 交互 Agent
	uv run langgraph-agent

# ---------------------------------------------------------------------------
# 跨层组合
# ---------------------------------------------------------------------------

check: lint test ## MR 前一键自检（lint + 单测）

ci: ## CI 门禁（frozen lockfile + lint + 单测 + wheel；GitHub Actions 调用）
	uv sync --frozen --all-groups --extra ui
	@$(MAKE) python-check check build

clean: ## 清理缓存与构建产物
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info htmlcov
