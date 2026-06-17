# LangGraph Skill Agent

基于 [LangGraph](https://github.com/langchain-ai/langgraph) 与 [Deep Agents](https://github.com/langchain-ai/deepagents) 构建的智能体项目，集成本地 **Skills**、**RAG 知识库检索**（LlamaIndex + Milvus）、**MCP 工具**、**对话记忆压缩**与可选的 **多步任务规划** 能力。默认使用 DeepSeek 作为对话模型。

## 功能概览

| 能力 | 说明 |
|------|------|
| Deep Agent | 基于 `create_deep_agent`，支持文件读写、Skills 调用、工具编排 |
| 本地 Skills | `skills/` 目录下的 `SKILL.md` 定义可复用技能，Agent 按需加载 |
| RAG 检索 | LlamaIndex 索引 + Milvus 向量库，混合检索（向量 + BM25 RRF） |
| MCP 工具 | 可选接入 FastMCP 外部工具（`MCP_TOOLS=1`） |
| Skill 脚本 | 支持本机 / Docker 隔离执行 `skills/` 下的 Python / Shell 脚本 |
| 对话记忆 | 长期记忆块（`soul.md` / `user.md` / `Memory.md`）+ 上下文压缩 |
| 任务规划 | 复杂任务可走 `plan_execute` 外层图：规划 → 分步执行 |
| Web UI | Streamlit 前端，支持流式输出与会话历史持久化 |

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
├── var/                           # 运行时数据（git 忽略，首次运行自动创建）
│   ├── agent_memory/              # 长期记忆 Markdown
│   ├── conversation_history/      # 对话快照
│   ├── session_history/           # Web UI 会话历史
│   ├── data/                      # RAG 原始文档
│   └── storage/                   # LlamaIndex 本地缓存
├── .env.example                   # 环境变量模板
├── Makefile                       # 常用开发命令
└── pyproject.toml                 # 项目配置与 CLI 入口
```

## 环境要求

- **Python 3.12**（由 `.python-version` 锁定）
- **[uv](https://docs.astral.sh/uv/)** 包管理器（推荐）
- **DeepSeek API Key**（必需）
- **Milvus** + **Embedding 服务**（RAG 功能需要，见下方配置）
- **Docker**（可选，用于 Skill 脚本沙箱执行）

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
| `EMBED_BASE_URL` | OpenAI 兼容 Embedding 服务地址 |
| `EMBED_MODEL` / `EMBED_DIM` | Embedding 模型与维度 |
| `MILVUS_URI` | Milvus 连接地址 |
| `MILVUS_COLLECTION` | 向量集合名称 |
| `MCP_TOOLS` | 设为 `1` 启用 MCP 工具 |
| `ENABLE_PLAN_ROUTING` | 设为 `1` 启用 CLI 自动路由到规划流程 |
| `THREAD_ID` | CLI 会话线程 ID |

完整选项见 [`.env.example`](.env.example)。

### 3. 准备 RAG 数据（可选）

将待索引的 PDF / 文档放入 `var/data/`，首次调用 `rag_search` 工具时会自动构建索引并写入 Milvus。

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

**更新长期记忆摘要：**

```bash
uv run langgraph-summary              # 使用最新对话快照
uv run langgraph-summary --dry-run      # 预览，不写文件
```

## 可用 CLI 命令

安装后可通过 `uv run` 或直接调用以下入口：

| 命令 | 说明 |
|------|------|
| `langgraph-agent` | CLI 持续对话 |
| `langgraph-ui` | 启动 Streamlit Web UI |
| `langgraph-plan` | 多步规划 + 分步执行 |
| `langgraph-summary` | 从对话快照更新记忆文件 |

## 开发

### 常用 Make 命令

```bash
make install            # 安装依赖
make python-check       # 校验 Python 3.12
make pre-commit-install # 安装 git pre-commit hook
make lint               # ruff 静态检查
make format             # ruff 格式化
make test               # 单元测试（跳过 integration）
make test-integration   # 集成测试（需 Milvus 在线）
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

如需执行脚本，可在 `skill_tools.py` 中注册脚本 ID，或使用 `run_skill_script_in_docker` 在 Docker 中运行 `skills/` 下的 Python 脚本。

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
   ├─ [ENABLE_PLAN_ROUTING=1] ──→ intent_router ──→ plan_execute（规划 → 逐步 Deep Agent）
   │
   └─ 直接对话 ──→ Deep Agent
                    ├─ Skills（skills/SKILL.md）
                    ├─ rag_search（LlamaIndex + Milvus）
                    ├─ MCP 工具
                    ├─ Skill 脚本（本机 / Docker）
                    └─ 文件系统 Backend（读写项目内文件）
```

对话过程中，`compactor` 会在上下文接近 token 上限时自动压缩历史消息；退出 CLI 时会将会话快照保存到 `var/conversation_history/`。

## License

见项目仓库说明。
