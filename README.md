# LangGraph Skill Agent

基于 [LangGraph](https://github.com/langchain-ai/langgraph) 与 [Deep Agents](https://github.com/langchain-ai/deepagents) 构建的智能体项目，集成本地 **Skills**、**RAG 知识库检索**（本机文档 + LlamaIndex 元数据 + 远程 Milvus / Embedding）、**MCP 工具**、**CLI 对话记忆压缩**与可选的 **多步任务规划** 能力。默认使用 DeepSeek 作为对话模型。

## 功能概览

| 能力 | 说明 |
|------|------|
| Deep Agent | 基于 `create_deep_agent`，支持文件读写、Skills 调用、工具编排 |
| 本地 Skills | `skills/` 目录下的 `SKILL.md` 定义可复用技能，Agent 按需加载 |
| RAG 检索 | 本机 `var/data` 建索引；向量与 BM25 存远程 Milvus；混合检索（向量 + RRF） |
| MCP 工具 | 可选接入 FastMCP 外部工具（`MCP_TOOLS=1`） |
| Skill 脚本 | 支持本机执行 `skills/` 下的 Python / Shell 脚本（`workspace_exec_python` / `run_skill_script_shell`） |
| 对话记忆 | 长期记忆块（`soul.md` / `user.md` / `Memory.md`）；CLI 支持上下文压缩与退出快照 |
| 任务规划 | 复杂任务可走 `plan_execute` 外层图：规划 → 分步执行（CLI 可自动路由） |
| Web UI | Streamlit 前端，流式输出；会话持久化到 `var/session_history/` |

## 项目结构

```
langgrpah-skills/
├── src/langgraph_skill_agent/     # 主包
│   ├── agent_core.py              # Agent 构建与 CLI 入口（langgraph-agent）
│   ├── plan_execute.py            # 多步规划执行（langgraph-plan）
│   ├── intent_router.py           # 判断是否需要走规划流程
│   ├── deepseek_model.py          # DeepSeek 模型封装
│   ├── frontend/                  # Streamlit Web UI（langgraph-ui）
│   ├── memory/                    # 对话快照、压缩、摘要 CLI
│   │   ├── compactor.py           # 上下文 token 压缩
│   │   ├── conversation.py        # 对话持久化
│   │   ├── summary.py             # 记忆摘要更新（langgraph-summary）
│   │   └── blocks.py              # 加载 agent 记忆块
│   ├── rag/
│   │   └── retriever.py           # RAG 索引构建与检索
│   ├── tool/
│   │   ├── skill_tools.py         # Skill 脚本执行工具
│   │   └── mcp_tools.py           # MCP 工具加载
│   └── utility/                   # 路径、日志、流式输出、JSON 解析等
├── skills/                        # 本地 Skills 定义
│   ├── demo-greeting/SKILL.md
│   └── test-calc-script/          # 示例脚本 Skill
├── tests/
│   ├── unit/                      # 单元测试
│   └── integration/               # 集成测试（需 Milvus 等外部服务）
├── var/                           # 运行时数据（git 忽略；各子目录按需创建）
│   ├── agent_memory/              # 长期记忆 Markdown
│   ├── conversation_history/      # CLI 退出快照（供 langgraph-summary）
│   ├── session_history/           # Web UI 会话 JSON
│   ├── data/                      # RAG 原始文档（建索引前需手动创建并放入文件）
│   └── storage/                   # LlamaIndex 本地元数据缓存（向量在 Milvus）
├── .env.example                   # 环境变量模板
├── Dockerfile                     # Streamlit 运行时镜像（标准制品）
├── docker-compose.yml             # 默认：仅 app（连 .env 远程 Milvus）
├── docker-compose.milvus.yml      # 可选 overlay：本地 Milvus 栈
├── docker-compose.prod.yml        # 生产 overlay：仅拉 Registry 镜像
├── Makefile                       # 跨层命令编排（lint / test / build / docker）
└── pyproject.toml                 # 项目配置与 CLI 入口
```

## 环境要求

- **Python 3.12**（由 `.python-version` 锁定）
- **[uv](https://docs.astral.sh/uv/)** 包管理器（推荐）
- **DeepSeek API Key**（必需）
- **Milvus** + **Embedding 服务**（RAG 需要；通常为**独立远程服务**，见下方「RAG 部署」）
- **Docker**（可选：打包 Streamlit 镜像；本地 Milvus 栈见 `docker-compose.milvus.yml`）

## 快速开始

### 1. 安装依赖

```bash
# 克隆仓库后，在项目根目录执行
make install
```

等价于 `uv sync --all-groups --extra ui`，会安装主依赖、开发工具与 Streamlit UI。

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少填写 DEEPSEEK_API_KEY
```

主要配置项：

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥（必需） |
| `DEEPSEEK_MODEL` | 模型名称，默认 `deepseek-chat` |
| `EMBED_BASE_URL` | OpenAI 兼容 Embedding 服务地址（可与 Agent 分机部署） |
| `EMBED_MODEL` / `EMBED_DIM` | Embedding 模型与维度 |
| `MILVUS_URI` | Milvus 连接地址（本地或远程，如 `http://host:19530`） |
| `MILVUS_TOKEN` | Milvus 鉴权 token（远程实例常用） |
| `MILVUS_COLLECTION` | 向量集合名称 |
| `RAG_FORCE_REBUILD` | `1` 强制重建索引（删本地 storage 并 drop Milvus collection） |
| `RAG_TRACE` | `1` 输出 RAG/MCP 调试日志，`0` 关闭 |
| `AGENT_TOOL_TRACE` | `1` 输出 Agent 工具调用调试日志，`0` 关闭 |
| `MCP_TOOLS` | `1` 启用 MCP 工具，`0` 关闭 |
| `ENABLE_PLAN_ROUTING` | `1` 启用 **CLI** 自动路由到规划流程 |
| `COMPACT_ENABLED` | `1` 启用 **CLI** 上下文压缩（默认开启） |
| `THREAD_ID` | CLI 会话线程 ID |

完整选项见 [`.env.example`](.env.example)。

### 3. RAG 部署（可选）

RAG 由三部分组成，**彼此分离、可分机部署**：

| 组件 | 存放位置 | 说明 |
|------|----------|------|
| 原始文档 | 本机 `var/data/`（或 `RAG_DATA_DIR`） | 建索引时的 PDF / 文档来源 |
| 向量与稀疏索引 | **远程 Milvus**（`MILVUS_URI`） | 稠密向量 + BM25 数据在 collection 内 |
| LlamaIndex 元数据 | 本机 `var/storage/`（或 `RAG_STORAGE_DIR`） | docstore / index_store 本地缓存 |

**典型用法（Milvus 在远程服务器）**——与本仓库 `docker-compose` 里的 Milvus 栈无关：

```bash
# .env 示例
MILVUS_URI=http://your-milvus-host:19530
MILVUS_TOKEN=your-token-if-needed
MILVUS_COLLECTION=rag_llamaindex
EMBED_BASE_URL=http://your-embed-host:8080/v1
```

在本机准备文档并启动 Agent（`make run-ui` 或 `make run-agent`）即可；首次调用 `rag_search` 时会读取 `var/data/`、向远程 Embedding 服务请求向量，并写入远程 Milvus。

```bash
mkdir -p var/data
# 将 PDF / 文档放入 var/data/
make run-ui   # 或 make run-agent
```

- 索引**不会**自动创建 `var/data/`，目录不存在时会报错。
- 已有本地 `var/storage/` 且 Milvus 中 collection 完好时，会直接加载；换模型 / 改 schema / 索引损坏时，可临时设 `RAG_FORCE_REBUILD=1` 重建。
- 远程 Milvus 需保证本机网络可达（防火墙、TLS、token 等按你的集群配置）。

### 4. 启动应用

**Web UI（推荐）：**

```bash
make run-ui
# 浏览器访问 http://localhost:8501
```

**CLI 交互式 Agent：**

```bash
make run-agent
# 输入 quit / exit / q 退出
```

**多步任务规划：**

```bash
uv run langgraph-plan "调研某主题并写一份总结文档"
uv run langgraph-plan -t my-task-1 "你的复杂目标"
```

**更新长期记忆摘要**（读取 CLI 退出快照，见下方「CLI 与 Web UI」）：

```bash
uv run langgraph-summary              # var/conversation_history/ 下最新快照
uv run langgraph-summary --dry-run      # 预览，不写文件
```

### CLI 与 Web UI 差异

| 能力 | CLI（`make run-agent`） | Web UI（`make run-ui`） |
|------|---------------------------|-------------------------|
| 上下文压缩 `compactor` | ✅ 每轮对话前自动压缩 | ❌ 未接入 |
| 规划自动路由 `ENABLE_PLAN_ROUTING` | ✅ 可开启 | ❌ 未接入；复杂任务请用 `langgraph-plan` |
| 会话持久化 | 退出时写入 `var/conversation_history/` | 实时写入 `var/session_history/*.json` |
| `langgraph-summary` | ✅ 使用 CLI 快照 | ❌ UI 会话需手动转换格式后才可用 |

长期记忆块（`var/agent_memory/*.md`）两种入口共用；摘要从 CLI 快照更新是推荐工作流。

## 可用 CLI 命令

安装后可通过 `uv run` 或直接调用以下入口：

| 命令 | 说明 |
|------|------|
| `langgraph-agent` | CLI 持续对话 |
| `langgraph-ui` | 启动 Streamlit Web UI |
| `langgraph-plan` | 多步规划 + 分步执行 |
| `langgraph-summary` | 从对话快照更新记忆文件 |

## Docker 部署（Phase 1）

**默认用法（远程 Milvus）**——compose 只启动 Streamlit app，Milvus / Embedding 地址完全由 `.env` 决定：

```bash
cp .env.example .env
# .env: MILVUS_URI=http://your-remote-host:19530
make docker-up                # 仅 build + 启动 app
# 浏览器 http://localhost:8501
make docker-logs
make docker-down
```

**可选：本地 Milvus 栈**（无远程实例、全本机联调时）：

```bash
make docker-up-milvus         # app + etcd + minio + milvus
# 此时 MILVUS_URI 会被 overlay 设为 http://milvus:19530
```

**Docker 下 RAG / 外部服务注意：**

| 项 | 说明 |
|----|------|
| `MILVUS_URI` | 默认从 `.env` 读取远程地址；仅 `make docker-up-milvus` 时覆盖为 `http://milvus:19530` |
| `EMBED_BASE_URL` | 容器内 `127.0.0.1` 是容器自身；Embedding 在宿主机时用 `host.docker.internal`（Mac/Windows）或宿主机 IP |
| RAG 文档 | app 使用 volume `app_var`（`/app/var`），不会自动挂载本机 `./var/data/`；需 bind mount 或把文档放进 volume |

| 命令 | 作用 |
|------|------|
| `make docker-up` | 仅 app（远程 Milvus） |
| `make docker-up-milvus` | app + 本地 Milvus 栈 |
| `make docker-build` | 仅构建镜像 `langgraph-skill-agent:local` |
| `make build` | Python wheel 到 `dist/` |
| `make docker-prod-up` | 生产：拉 Registry 镜像（需 `IMAGE` + `TAG`） |

**制品标签**：生产部署请用 Git commit SHA，不要用 `latest`：

```bash
export IMAGE=registry.example.com/langgraph-skill-agent
export TAG=$(git rev-parse --short HEAD)
make docker-build TAG=$TAG
# push 后于服务器：
make docker-prod-up IMAGE=$IMAGE TAG=$TAG
```

**与参考标准的取舍**（本项目）：

| 参考项 | 是否采用 | 说明 |
|--------|----------|------|
| Makefile 跨层编排 | ✅ | `lint` / `test` / `build` / `docker-*` / `check` |
| 多阶段 Dockerfile + 非 root | ✅ | uv 构建，Streamlit 8501 |
| docker-compose 多环境 | ✅ | 默认 app only + 可选 `docker-compose.milvus.yml` + 生产 overlay |
| `/health` FastAPI 探针 | ❌ | 使用 Streamlit 内置 `/_stcore/health` |
| Alembic `make migrate` | ❌ | 无关系型 DB 迁移 |
| mypy `make typecheck` | ⏸ | Phase 0 未引入，后续可加 |
| Compose 内 PostgreSQL | ❌ | 向量库为 Milvus；默认连远程，本地栈见 `docker-compose.milvus.yml` |

本地 Milvus 栈数据在 `deploy/volumes/`（gitignore）。应用运行时数据在 Docker volume `app_var`（`/app/var`）。**远程 Milvus 数据由你的集群自行持久化**，与本仓库 volume 无关。

## 开发

### 常用 Make 命令

```bash
make help               # 列出全部命令
make install            # 安装依赖
make check              # MR 前：lint + 单测
make python-check       # 校验 Python 3.12
make pre-commit-install # 安装 git pre-commit hook
make lint               # ruff 静态检查
make format             # ruff 格式化
make test               # 单元测试（跳过 integration）
make test-integration   # 集成测试（需 Milvus 在线）
make build              # Python wheel 打包
make docker-build       # 构建 Docker 镜像
make docker-up          # Docker 启动 app（远程 Milvus）
make docker-up-milvus   # Docker 启动 app + 本地 Milvus（可选）
make pre-commit         # 手动跑全部 pre-commit 检查
```

### 编写 Skill

在 `skills/<skill-name>/` 下创建 `SKILL.md`，参考 [`skills/demo-greeting/SKILL.md`](skills/demo-greeting/SKILL.md)：

```markdown
---
name: my-skill
description: 何时使用该技能的简短说明
---

# 技能标题

## Instructions
1. 步骤一
2. 步骤二
```

如需执行脚本，可在 `skill_tools.py` 中注册 shell 脚本 ID，或使用 `workspace_exec_python` 运行 `skills/` 下的 Python 脚本。

### 测试

```bash
# 单元测试
make test

# 集成测试（需配置 .env 且 Milvus 可用）
make test-integration
```

## 架构简述

```
用户输入
   │
   ├─ CLI + ENABLE_PLAN_ROUTING=1 ──→ intent_router ──→ plan_execute（规划 → 逐步 Deep Agent）
   │
   └─ 直接对话 ──→ Deep Agent
                    ├─ Skills（skills/SKILL.md）
                    ├─ rag_search → 本机 var/data + var/storage + 远程 Milvus / Embedding
                    ├─ MCP 工具
                    ├─ Skill 脚本（本机 workspace_exec_python / run_skill_script_shell）
                    └─ 文件系统 Backend（读写项目内文件）
```

- **CLI**：每轮前 `compactor` 压缩上下文；退出时快照 → `var/conversation_history/` → 可跑 `langgraph-summary`。
- **Web UI**：无 compactor / 规划路由；会话 → `var/session_history/`。

## License

见项目仓库说明。
