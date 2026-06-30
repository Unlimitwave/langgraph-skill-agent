# LangGraph Skill Agent

基于 [LangGraph](https://github.com/langchain-ai/langgraph) 与 [Deep Agents](https://github.com/langchain-ai/deepagents) 构建的智能体项目，集成本地 **Skills**、**RAG 知识库检索**（本机文档 + LlamaIndex 元数据 + 远程 Milvus / Embedding）、**MCP 工具**、**CLI 对话记忆压缩**与可选的 **多步任务规划** 能力。默认使用 DeepSeek 作为对话模型。

## 功能概览

| 能力 | 说明 |
|------|------|
| Deep Agent | 基于 `create_deep_agent`，支持文件读写、Skills 调用、工具编排 |
| 分层 Skills | 系统级挂载 `/system-skills/` + 用户级 `skills/`（同名时用户覆盖系统） |
| 沙箱工作区 | `CompositeBackend`：默认根为 `workspace/{AGENT_USER_ID}/`；系统 skills 只读挂载 |
| RAG 检索 | 每用户 `workspace/{user_id}/rag_data/` 建索引；向量存共享 Milvus collection（`user_id`+`tenant_id` filter）；混合检索（向量 + BM25 + RRF） |
| MCP 工具 | 可选接入 FastMCP 外部工具（`MCP_TOOLS=1`） |
| Skill 脚本 | 虚拟路径 `/system-skills/...` 或 `skills/...`；Shell 走平台白名单 |
| 对话记忆 | 长期记忆块（`soul.md` / `user.md` / `Memory.md`）；CLI 支持上下文压缩与退出快照 |
| 任务规划 | 复杂任务可走 `plan_execute` 外层图：规划 → 分步执行（CLI 可自动路由） |
| Supervisor 多智能体 | Research / Worker / Review 分角色协作 + 质量门 + 汇总（CLI 可自动路由） |
| Web UI | Streamlit 前端，流式输出；对话状态存 Postgres checkpointer，侧边栏索引在 `var/session_history/` |

## 项目结构

```
langgrpah-skills/
├── src/langgraph_skill_agent/     # 主包
│   ├── agent_core.py              # Agent 构建与 CLI 入口（langgraph-agent）
│   ├── plan_execute.py            # 多步规划执行（langgraph-plan）
│   ├── multi_agent/               # Supervisor 多智能体（langgraph-supervisor）
│   │   ├── supervisor.py          # 编排图：规划 → Specialist → 汇总
│   │   ├── specialists.py         # 按角色编译 Deep Agent
│   │   ├── roles.py               # 角色 prompt / 工具 / 权限
│   │   └── handoff.py             # 结构化 Handoff 契约
│   ├── intent_router.py           # 判断是否需要走规划流程
│   ├── deepseek_model.py          # DeepSeek 模型封装
│   ├── frontend/                  # Streamlit Web UI（langgraph-ui）
│   ├── memory/                    # 对话快照、压缩、摘要 CLI
│   │   ├── compactor.py           # 上下文 token 压缩
│   │   ├── conversation.py        # 对话持久化
│   │   ├── summary.py             # 记忆摘要更新（langgraph-summary）
│   │   └── blocks.py              # 加载 agent 记忆块
│   ├── rag/
│   │   ├── retriever.py           # RAG 索引构建与检索（多租户 filter）
│   │   └── rag_readme.md          # RAG 模块详细说明
│   ├── tool/
│   │   ├── skill_tools.py         # Skill 脚本执行工具
│   │   └── mcp_tools.py           # MCP 工具加载
│   └── utility/                   # 路径、日志、流式输出、JSON 解析等
├── skills/                        # 系统级 Skills 磁盘目录（挂载为 /system-skills/，只读）
│   ├── demo-greeting/SKILL.md
│   └── test-calc-script/          # 示例脚本 Skill
├── workspace/                     # 多用户沙箱根目录
│   └── default/                   # AGENT_USER_ID=default 时可写工作区
│       ├── skills/                # 用户级 Skills（虚拟路径 skills/）
│       ├── rag_data/              # 该用户 RAG 源文档
│       ├── rag_storage/           # 该用户 LlamaIndex 元数据
│       └── …                      # 用户任务产物
├── tests/
│   ├── unit/                      # 单元测试
│   └── integration/               # 集成测试（需 Milvus 等外部服务）
├── var/                           # 运行时数据（git 忽略；各子目录按需创建）
│   ├── agent_memory/              # 长期记忆 Markdown
│   ├── conversation_history/      # CLI 退出快照（供 langgraph-summary）
│   ├── session_history/           # Web UI 会话 JSON
│   ├── data/                      # legacy：default 用户 RAG 源文档回退路径
│   └── storage/                   # legacy：default 用户 LlamaIndex 元数据回退（向量在 Milvus）
├── deploy/
│   └── postgres/                  # Postgres 分库初始化脚本
├── .env.example                   # 环境变量模板
├── Dockerfile                     # Streamlit 运行时镜像（标准制品）
├── docker-compose.yml             # 默认：仅 app（连 .env 远程 Milvus）
├── docker-compose.milvus.yml      # 可选 overlay：本地 Milvus 栈
├── docker-compose.prod.yml        # 生产 / 本地全栈 overlay：Postgres + 分库 checkpointer
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
| `ENABLE_MULTI_AGENT_ROUTING` | `1` 启用 **CLI** 自动路由到 Supervisor 多智能体 |
| `SUPERVISOR_MAX_REVIEW_RETRIES` | Review 不通过时 Worker 最大重试次数（默认 `2`） |
| `COMPACT_ENABLED` | `1` 启用 **CLI** 跨轮 LLM 摘要压缩（默认开启） |
| `CONTEXT_WINDOW` | 模型上下文窗口 token 数（默认 `128000`）；compactor / 裁剪瀑布共用 |
| `CONTEXT_RESERVE_TOKENS` | 预留给本轮回复 + 工具输出（默认 `8000`） |
| `THREAD_ID` | CLI 会话线程 ID |
| `CHECKPOINT_BACKEND` | checkpointer 类型：`postgres`（默认）/ `sqlite` / `memory` |
| `POSTGRES_URI` | 裸机 `run-ui` / `run-agent` 的 Postgres 连接串（库名见下方「Checkpointer」） |
| `POSTGRES_DB_DOCKER` | Docker app 使用的 database，默认 `langgraph_docker` |
| `POSTGRES_DB_LOCAL` | 裸机 UI 使用的 database，默认 `langgraph_local` |
| `POSTGRES_PASSWORD` | Postgres 用户密码（`docker-stack-up` 必需） |

完整选项见 [`.env.example`](.env.example)。

### 3. RAG 部署（可选）

RAG 由三部分组成，**彼此分离、可分机部署**；身份与 Agent 沙箱一致，走 `AgentContext` 的 `user_id` / `tenant_id`（详见 [`src/langgraph_skill_agent/rag/rag_readme.md`](src/langgraph_skill_agent/rag/rag_readme.md)）。

| 组件 | 存放位置 | 说明 |
|------|----------|------|
| 原始文档 | `workspace/{user_id}/rag_data/` | 每用户独立；`default` 可回退 `var/data/` |
| 向量与稀疏索引 | **远程 Milvus**（`MILVUS_URI`） | 共享 collection；每条向量带 `user_id` / `tenant_id` 标量，检索时 filter |
| LlamaIndex 元数据 | `workspace/{user_id}/rag_storage/` | 每用户 docstore；`default` 可回退 `var/storage/` |

**典型用法（Milvus 在远程服务器）**——与本仓库 `docker-compose` 里的 Milvus 栈无关：

```bash
# .env 示例
MILVUS_URI=http://your-milvus-host:19530
MILVUS_TOKEN=your-token-if-needed
MILVUS_COLLECTION=rag_llamaindex
EMBED_BASE_URL=http://your-embed-host:8080/v1
```

在本机准备文档并启动 Agent（`make run-ui` 或 `make run-agent`）即可；首次对该用户调用 `rag_search` 时会读取其 `rag_data/`、向远程 Embedding 请求向量，并写入远程 Milvus（带租户标量）。

**单租户 / CLI（`default` 用户，兼容旧路径）：**

```bash
mkdir -p var/data
# 将 PDF / 文档放入 var/data/
make run-ui   # 或 make run-agent
```

**多用户 Web UI（每浏览器 `ui-<id>`）：**

```bash
mkdir -p workspace/ui-abc123/rag_data
# 将文档放入该目录
make run-ui
```

- 用户无 `rag_data` 文件时，`rag_search` 返回 `(no results)`，不中断对话。
- 已有该用户 `rag_storage/` 且 Milvus collection schema 一致时会直接加载；换模型 / 改 schema / **从无租户标量升级** 时，设 `RAG_FORCE_REBUILD=1` 重建。
- 远程 Milvus 需保证本机网络可达（防火墙、TLS、token 等按你的集群配置）。

### 4. 启动应用

Web UI 与 CLI 默认使用 **Postgres checkpointer** 持久化 LangGraph 对话状态。本地开发常见两种跑法：**裸机改代码** 或 **Docker 全栈**；二者共用 **一个 Postgres 容器、两个 database**，数据互不干扰。

#### Checkpointer 与 Postgres 分库

| 概念 | 说明 |
|------|------|
| Postgres **实例** | 一个 `postgres:16-alpine` 容器，监听 `127.0.0.1:5432`（供裸机连接） |
| **database** | 同一实例内的逻辑库；Docker UI 与裸机 UI 各用一库 |
| `langgraph_docker` | Docker 内 Streamlit app 的 checkpoint（compose 自动配置） |
| `langgraph_local` | 裸机 `make run-ui` / `make run-agent` 的 checkpoint（`.env` 中 `POSTGRES_URI`） |
| `var/session_history/` | 仅存侧边栏标题与 `thread_id`；**聊天气泡内容以 checkpointer 为准** |

连接串示例（库名在最后一段）：

```text
postgresql://langgraph:密码@localhost:5432/langgraph_local
                                              ↑ database 名
```

无 Postgres 时可改 `.env`：`CHECKPOINT_BACKEND=sqlite` 或 `memory`（见 `.env.example`）。

#### 方式 A：裸机 UI / CLI（改代码即时生效，推荐日常开发）

**1. 先起 Postgres（可后台常驻）：**

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d postgres
```

**初次搭环境或旧 pg 卷可能缺库时**（只需一次，或不确定时跑一下），再执行：

```bash
make postgres-init-dbs    # 幂等：库已存在则 skip，不会重复创建
```

> **不必每次启动都跑。** 全新 `pg_data` 卷首次 `up postgres` 时，`init-databases.sh` 会自动建 `langgraph_local`（`langgraph_docker` 由 `POSTGRES_DB` 创建）。日常开发只要 Postgres 在跑、数据卷未删，直接 `make run-ui` 即可；容器 stop/start 或重启电脑后，库仍在 `pg_data` 里。

**2. 启动 UI 或 CLI：**

```bash
make run-ui               # http://localhost:8501
# 或
make run-agent
```

`.env` 需包含（示例见 `.env.example`）：

```bash
CHECKPOINT_BACKEND=postgres
POSTGRES_URI=postgresql://langgraph:你的密码@localhost:5432/langgraph_local
```

若 Docker app 已占用 8501，可停掉 app 或换端口：

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml stop app
make run-ui
# 或：uv run langgraph-ui -- --server.port 8502
```

#### 方式 B：Docker 全栈 UI（接近部署环境）

```bash
make docker-stack-up
# 自动：Postgres + 建库 + 构建并启动 app
# 浏览器 http://localhost:8501
```

Docker app 使用 compose 内配置的 `@postgres:5432/langgraph_docker`，**无需**改 `.env` 里的 `POSTGRES_URI`。

| 命令 | 作用 |
|------|------|
| `make docker-stack-up` | Postgres + app（分库隔离，推荐本地全栈） |
| `make postgres-init-dbs` | 幂等补建两库（**仅初次/旧卷/不确定时**；日常不必重复） |
| `make docker-logs` | 查看 app 日志 |
| `make docker-down` | 停止 compose 服务 |

**方式 A 与 B 可同时运行**（不同 database，checkpoint 不冲突）；但 **8501 端口只能给一个 UI**，另一个需停 app 或改用 8502。

**多步任务规划**（与 UI 入口无关）：

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
| 规划自动路由 `ENABLE_PLAN_ROUTING` | ✅ 可开启 | ✅ 可开启 |
| 多智能体路由 `ENABLE_MULTI_AGENT_ROUTING` | ✅ 可开启 | ✅ 可开启 |
| 会话持久化 | 退出时写入 `var/conversation_history/` | checkpointer（Postgres 等）+ 侧边栏索引 `var/session_history/*.json` |
| `langgraph-summary` | ✅ 使用 CLI 快照 | ❌ UI 会话需手动转换格式后才可用 |

长期记忆块（`var/agent_memory/*.md`）两种入口共用；摘要从 CLI 快照更新是推荐工作流。

## 可用 CLI 命令

安装后可通过 `uv run` 或直接调用以下入口：

| 命令 | 说明 |
|------|------|
| `langgraph-agent` | CLI 持续对话 |
| `langgraph-ui` | 启动 Streamlit Web UI |
| `langgraph-plan` | 多步规划 + 分步执行 |
| `langgraph-supervisor` | Supervisor 多智能体（Research/Worker/Review） |
| `langgraph-summary` | 从对话快照更新记忆文件 |

## Docker 部署

### 仅 app（远程 Milvus，无 Postgres checkpointer）

compose 只启动 Streamlit app；checkpointer 仍读 `.env`（若 `CHECKPOINT_BACKEND=postgres` 需自行提供可达的 Postgres，或改为 `sqlite`）：

```bash
cp .env.example .env
# .env: MILVUS_URI=http://your-remote-host:19530
make docker-up                # 仅 build + 启动 app
# 浏览器 http://localhost:8501
make docker-logs
make docker-down
```

### 本地全栈：app + Postgres（推荐，checkpointer 分库）

```bash
cp .env.example .env
# 填写 DEEPSEEK_API_KEY、POSTGRES_PASSWORD 等
make docker-stack-up          # Postgres + 建库 + app；PG 映射 127.0.0.1:5432
# 浏览器 http://localhost:8501  （Docker UI → langgraph_docker）
make docker-logs
make docker-down
```

裸机调试时可只起 Postgres，与 Docker UI 共用实例、不同库：

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d postgres
# 仅初次或旧 pg 卷缺库时：make postgres-init-dbs
make run-ui                   # 裸机 UI → langgraph_local
```

### 可选：本地 Milvus 栈

无远程 Milvus、全本机联调时：

```bash
make docker-up-milvus         # app + etcd + minio + milvus
# 此时 MILVUS_URI 会被 overlay 设为 http://milvus:19530
```

**Docker 下 RAG / 外部服务注意：**

| 项 | 说明 |
|----|------|
| `MILVUS_URI` | 默认从 `.env` 读取远程地址；仅 `make docker-up-milvus` 时覆盖为 `http://milvus:19530` |
| `EMBED_BASE_URL` | 容器内 `127.0.0.1` 是容器自身；Embedding 在宿主机时用 `host.docker.internal`（Mac/Windows）或宿主机 IP |
| RAG 文档 | 每用户 `workspace/{user_id}/rag_data/`（与沙箱同根）；Docker 需 bind mount `./workspace` 或 `./var`；legacy `default` 可仍用 volume 内 `/app/var/data/` |

| 命令 | 作用 |
|------|------|
| `make docker-up` | 仅 app（远程 Milvus；无 compose 内 Postgres） |
| `make docker-stack-up` | app + Postgres（分库 checkpointer，本地全栈推荐） |
| `make postgres-init-dbs` | 幂等补建两库（**仅初次/旧卷/不确定时**；`docker-stack-up` 会自动调用） |
| `make docker-up-milvus` | app + 本地 Milvus 栈 |
| `make docker-build` | 仅构建镜像 `langgraph-skill-agent:local` |
| `make build` | Python wheel 到 `dist/` |
| `make docker-prod-up` | 生产：拉 Registry 镜像 + Postgres（需 `IMAGE` + `TAG`） |

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
| docker-compose 多环境 | ✅ | 默认 app only + 可选 Milvus + `docker-compose.prod.yml`（Postgres） |
| `/health` FastAPI 探针 | ❌ | 使用 Streamlit 内置 `/_stcore/health` |
| LangGraph Postgres checkpointer | ✅ | 默认 `postgres`；Docker / 裸机分库（`langgraph_docker` / `langgraph_local`） |
| Alembic `make migrate` | ❌ | checkpoint 表由 `PostgresSaver.setup()` 自动迁移 |
| mypy `make typecheck` | ⏸ | Phase 0 未引入，后续可加 |

本地 Milvus 栈数据在 `deploy/volumes/`（gitignore）。Postgres 数据在 Docker volume `pg_data`；应用运行时数据在 `app_var`（`/app/var`）。**远程 Milvus 数据由你的集群自行持久化**，与本仓库 volume 无关。

## 开发

### 常用 Make 命令

```bash
make help               # 列出全部命令
make install            # 安装依赖
make check              # MR 前：lint + 单测
make ci                 # CI 门禁（与 GitHub Actions 一致）
make python-check       # 校验 Python 3.12
make pre-commit-install # 安装 git pre-commit hook
make lint               # ruff 静态检查
make format             # ruff 格式化
make test               # 单元测试（跳过 integration）
make test-integration   # 集成测试（需 Milvus 在线）
make build              # Python wheel 打包
make docker-build       # 构建 Docker 镜像
make docker-up          # Docker 启动 app（远程 Milvus）
make docker-stack-up    # Docker 启动 app + Postgres（checkpointer 分库）
make postgres-init-dbs  # 幂等补建两库（仅初次/旧卷/不确定时）
make docker-up-milvus   # Docker 启动 app + 本地 Milvus（可选）
make run-ui             # 裸机 Streamlit（需 Postgres 容器 + langgraph_local）
make run-agent          # 裸机 CLI
make pre-commit         # 手动跑全部 pre-commit 检查
```

### CI（GitHub Actions）

流水线配置在 [`.github/workflows/ci.yml`](.github/workflows/ci.yml)，入口脚本 [`deploy/ci/run-quality-gate.sh`](deploy/ci/run-quality-gate.sh)。开通与分支保护见 [`deploy/ci/README.md`](deploy/ci/README.md)。

### 编写 Skill

**系统级**（平台维护）：在仓库 `skills/<skill-name>/` 下创建 `SKILL.md`（Agent 内虚拟路径 `/system-skills/<skill-name>/`）。

**用户级**：在 `workspace/<AGENT_USER_ID>/skills/<skill-name>/` 下创建 `SKILL.md`（Agent 内虚拟路径 `skills/<skill-name>/`）。同名时 **用户级优先**。

参考 [`skills/demo-greeting/SKILL.md`](skills/demo-greeting/SKILL.md)：

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

如需执行脚本：系统 Skill 用 `["/system-skills/<skill>/script.py"]`；用户 Skill 用 `["skills/<skill>/script.py"]`。Shell 需在 `skill_tools.py` 注册白名单。

### 安全边界（CompositeBackend 沙箱）

| 虚拟路径 | 磁盘位置 | 读 | 写 |
|----------|----------|----|----|
| `/`（工作区根） | `workspace/{AGENT_USER_ID}/` | ✅ | ✅（HITL 审批） |
| `skills/` | `workspace/{user}/skills/` | ✅ | ✅ |
| `/system-skills/` | 仓库 `skills/`（只读挂载） | ✅ | ❌ |
| `src/`、`var/`、`.env` | — | ❌ 不可见 | ❌ |

多用户：默认 Web UI 为每个浏览器分配 `ui-<id>` 沙箱（**勿设 `AGENT_USER_ID`**）。CLI 或单租户部署可设 `AGENT_USER_ID=alice` → `workspace/alice/`。Checkpointer 线程自动加 `{user_id}__` 前缀，避免跨用户串话；记忆在 `workspace/{user}/agent_memory/`；RAG 文档在 `workspace/{user}/rag_data/`，Milvus 检索带 `user_id` + `tenant_id` filter，不会跨用户命中向量。

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
   ├─ CLI + 路由开启 ──→ intent_router ──→ supervisor（Research/Worker/Review → 汇总）
   │                                    └─→ plan_execute（规划 → 逐步 Deep Agent）
   │
   └─ 直接对话 ──→ Deep Agent
                    ├─ Skills（/system-skills/ → skills/，后者覆盖同名）
                    ├─ rag_search → workspace/{user}/rag_data + Milvus（metadata filter，不经 FS 工具）
                    ├─ MCP 工具
                    ├─ Skill 脚本（/system-skills/ 或 skills/）
                    └─ CompositeBackend（沙箱 workspace/{user}/ + 只读 /system-skills/）
```

- **CLI 路由**：`ENABLE_MULTI_AGENT_ROUTING=1` 与 `ENABLE_PLAN_ROUTING=1` 可同时开启；`intent_router` 在已启用模式中选 `supervisor` / `plan` / `direct`。
- **Web UI 路由**：与 CLI 共用相同环境变量；复杂任务在聊天框输入后自动分流，编排过程流式展示，Worker 写操作支持 HITL 审批。
- **CLI**：每轮前 `compactor` 压缩上下文；退出时快照 → `var/conversation_history/` → 可跑 `langgraph-summary`。
- **Web UI**：无 compactor；对话 checkpoint → Postgres（或 sqlite/memory）；侧边栏索引 → `var/session_history/`。

## License

见项目仓库说明。
