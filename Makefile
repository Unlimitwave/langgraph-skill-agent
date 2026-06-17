# langgraph-skill-agent 常用命令
# 用法: make <target>   例如 make install / make test
#
# 前置: 需安装 uv (https://docs.astral.sh/uv/)
# Python 版本由 .python-version 锁定为 3.12

.PHONY: install lint format test test-integration run-ui run-agent python-check pre-commit-install pre-commit

# ---------------------------------------------------------------------------
# 环境与依赖
# ---------------------------------------------------------------------------

# 安装/同步依赖（含 dev 工具与 Streamlit UI 可选包）
install:
	uv sync --all-groups --extra ui

# 校验当前虚拟环境是否为 Python 3.12.x（pre-commit 也会调用）
python-check:
	@uv run python -c "import sys; v=sys.version_info; assert v.major==3 and v.minor==12, f'expected Python 3.12.x, got {sys.version}'; print(f'OK: Python {sys.version.split()[0]}')"

# 安装 git pre-commit hook（clone 后首次建议执行一次）
pre-commit-install: install
	uv run pre-commit install

# 手动对所有文件跑 pre-commit（不提交也可用来自查）
pre-commit:
	uv run pre-commit run --all-files

# ---------------------------------------------------------------------------
# 代码质量
# ---------------------------------------------------------------------------

# 静态检查（ruff lint，不修改文件）
lint:
	uv run ruff check src tests

# 自动格式化并修复可安全处理的 lint 问题
format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

# 单元测试（默认 CI / pre-commit 使用，跳过需 Milvus 等的 integration）
test:
	uv run pytest -q -m "not integration"

# 集成测试（需 Milvus 等外部服务在线，并配置 .env）
test-integration:
	uv run pytest -q -m integration

# ---------------------------------------------------------------------------
# 运行应用
# ---------------------------------------------------------------------------

# 启动 Streamlit Web UI（http://localhost:8501）
run-ui:
	uv run langgraph-ui

# 启动 CLI 交互式 Agent
run-agent:
	uv run langgraph-agent
