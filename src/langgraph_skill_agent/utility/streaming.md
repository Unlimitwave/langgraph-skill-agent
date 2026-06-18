# 流式输出说明（读 `streaming.py` 前先看）

本文件说明 `stream_assistant_text` / `run_assistant_turn` 消费的 LangGraph v2 流式事件，以及与本项目 `redraw` / `buf` / `tool_results` / HITL 的对应关系。

- 流式实现：同目录 [`streaming.py`](./streaming.py)
- 人工审批（interrupt 解析、CLI 提示）：[`hitl.py`](./hitl.py)
- Streamlit 审批 UI：[`frontend/app.py`](../frontend/app.py)

---

## 1. 两层 `type`（最容易混）

| 层级 | 字段位置 | 取值示例 | 含义 |
|------|----------|----------|------|
| **外层** | `chunk["type"]` | `"messages"` / `"tasks"` | LangGraph **流事件种类**（`stream_mode` 订阅项） |
| **内层** | `chunk["data"][0].type` | `"ai"` / `"tool"` 等 | LangChain **消息种类**（仅在 `messages` 事件里） |

```text
chunk = {
  "type": "messages",          ← 外层：消息流事件
  "data": (AIMessageChunk / ToolMessage, metadata),
}

chunk = {
  "type": "tasks",             ← 外层：节点调度事件
  "data": {"name": "agent" | "tools", ...},
}
```

本项目 `astream` **只订阅** `messages` 与 `tasks`；其它外层 type 在代码里直接 `continue` 跳过。

---

## 2. `messages` 与 `tasks` 各管什么

```text
                    graph.astream(stream_mode=["messages", "tasks"])
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
    chunk.type == "messages"        chunk.type == "tasks"
    （消息内容流）                    （节点生命周期，需 checkpointer）
           │                               │
    ┌──────┼──────┐                 ┌─────┴─────┐
    ▼      ▼      ▼                 ▼           ▼
  正文   tool call  ToolMessage   agent 节点   tools 节点
  token  (在 AI 消息里) (tool 消息)  开始/结束    开始/结束
```

| 你想知道的事 | 看哪种 chunk / API |
|-------------|-------------------|
| 助手说了哪个字 | `messages` + `AIMessageChunk.content` |
| 模型要调什么工具 | `messages` + `tool_calls` / `tool_call_chunks` |
| 工具返回了什么 | `messages` + `ToolMessage`（内层 `type=="tool"`） |
| 模型节点是否在跑 | `tasks` + `name in ("agent", "model")` 的开始/结束 |
| 工具节点是否在跑 | `tasks` + `name == "tools"` 的开始/结束 |
| **是否有待审批 interrupt** | **流结束后** `graph.get_state(config).interrupts`（见 [`hitl.get_pending_hitl`](./hitl.py)） |

**要点**：

- `tasks` 不携带 tool call 参数或工具正文的流式 token；具体内容仍在 `messages` 里。
- `tasks` 结束时的 `data.interrupts` 非空只表示「该节点发生了 interrupt」，本项目**仅打 `AGENT_TOOL_TRACE` 日志**，**不**用它驱动审批 UI；审批以 `get_state().interrupts` 为准。

---

## 3. tool call 与 ToolMessage（都在外层 `messages` 里）

| 阶段 | 内层消息 | 典型形态 | `streaming.py` 如何处理 |
|------|----------|----------|-------------------------|
| **tool call**（模型决定调工具） | `AIMessageChunk` | `content` 常为空；有 `tool_calls` / `tool_call_chunks` | `tool_names_from_message_chunk` → `pending_tool_names` |
| **ToolMessage**（工具执行完毕） | `ToolMessage` | `type=="tool"`，有 `name`、`content` | 进 `tool_results`，**不进** `buf`（CLI 正文不含工具输出） |

- **tool call** = 模型说「我要调用 `rag_search`」→ 还在 **AI 消息**里。
- **ToolMessage** = 工具说「`rag_search` 返回了 xxx」→ 单独的 **tool 消息**。

---

## 4. v2 公共外壳

不论哪种 mode，每个 chunk 都是：

```python
{
    "type": "messages" | "tasks" | ...,
    "ns": (),                    # 子图时非空，如 ("agent:abc123",)
    "data": <因 type 而异>,
}
```

- **`messages`**：`data` = `(message_chunk, metadata)` 二元组 → 代码：`message_chunk, _meta = chunk["data"]`
- **`tasks`**：`data` = `dict`，**开始**（有 `input` + `triggers`）与 **结束**（有 `result`）互斥

### 4.1 其它 `stream_mode`（本项目未订阅）

LangGraph v2 还支持 `values`、`updates`、`checkpoints`、`debug`、`custom` 等。当前实现**不消费**它们，但了解有助于读官方文档：

| `stream_mode` | 外层 `type` | 与 HITL 的关系 |
|---------------|-------------|----------------|
| `values` | `"values"` | 每步全量 state；part 上可有 `interrupts` |
| `updates` | `"updates"` | 增量更新；可能有 `__interrupt__` 键 |
| `checkpoints` | `"checkpoints"` | 类似 `get_state()` 的快照事件 |
| `tasks` | `"tasks"` | 节点结束 `data.interrupts` 可非空（见 §7 T5） |
| `messages` | `"messages"` | 与 HITL 无直接关系 |
| `custom` / `debug` | `"custom"` / `"debug"` | 调试用 |

本项目 HITL 检测路径：**`astream` 一段结束 → `get_pending_hitl(graph, config)` → 有则暂停 → `Command(resume={"decisions": [...]})` 再开下一段流**。不依赖订阅 `values` / `updates`。

---

## 5. 一次完整调工具的时间线

用户：「帮我搜一下 LangGraph 文档」

```text
T1  tasks   agent 开始          → agent_depth++  → 可能显示「⏳ 模型推理中」
M2  messages AIMessageChunk     → tool_call_chunks 出现 rag_search → pending_tool_names
M3  messages AIMessageChunk     → 完整 tool_calls
T2  tasks   agent 结束          → agent_depth--
T3  tasks   tools 开始          → tools_depth++    → 「🔄 工具正在运行」
M4  messages ToolMessage        → tool_results 增加
T4  tasks   tools 结束          → tools_depth--，清空 pending_tool_names
T5  tasks   agent 开始          → 模型读工具结果后继续生成
M5  messages AIMessageChunk     → buf 累积最终回答正文
T?  tasks   agent 结束
```

`format_status_line` 优先级：**pending 工具名** > **tools 节点在跑** > **agent 在跑且无正文**。

### 5.1 带 HITL（`interrupt_on`）的写文件时间线

用户：「在 workspace 写入 test.txt」（`write_file` 在 `agent_core.interrupt_on` 里为 `True`）

```text
M1  messages AIMessageChunk     → 可能先有正文 token（「好的，我来写…」）
M2  messages AIMessageChunk     → tool_calls 出现 write_file
T1  tasks   tools 开始
T2  tasks   tools 结束          → data.interrupts 非空（仅 trace 日志）
                                → astream 循环结束
                                → get_pending_hitl() 返回 action_requests
── 暂停：CLI 询问 approve/reject；Streamlit 显示审批按钮 ──
    resume：graph_input = Command(resume={"decisions": [...]})
M3  messages …                  → 下一段 astream（text_prefix 接上段正文）
M4  messages ToolMessage        → write_file 执行结果（若已 approve）
…   可能再次 interrupt（如连续 write_file）→ 重复审批循环
```

多段流由 `run_assistant_turn` 用 `text_prefix` / `tool_results_prefix` 拼接；**同一 `thread_id`** 下 resume。

---

## 6. `chunk.type == "messages"` 具体例子

### M1：模型吐正文 token

```python
{
    "type": "messages",
    "ns": (),
    "data": (
        {"type": "AIMessageChunk", "content": "好", "tool_calls": [], "tool_call_chunks": []},
        {"langgraph_node": "agent", "langgraph_step": 3},
    ),
}
```

→ `buf.append("好")` → `redraw()`

### M2：仅有 tool call，尚无正文

```python
{
    "type": "messages",
    "ns": (),
    "data": (
        {
            "type": "AIMessageChunk",
            "content": "",
            "tool_call_chunks": [
                {"name": "rag_search", "args": '{"query": "LangGraph', "index": 0, ...}
            ],
        },
        {"langgraph_node": "agent"},
    ),
}
```

→ `piece` 为空仍 `redraw()`；`pending_tool_names = ["rag_search"]`

### M3：tool call 流式补全 / 完整 tool_calls

```python
{
    "type": "messages",
    "ns": (),
    "data": (
        {
            "type": "AIMessageChunk",
            "content": "",
            "tool_calls": [{"name": "rag_search", "args": {"query": "LangGraph 文档"}, "id": "call_xyz"}],
            "response_metadata": {"finish_reason": "tool_calls"},
        },
        {"langgraph_node": "agent"},
    ),
}
```

### M4：工具返回（ToolMessage）

```python
{
    "type": "messages",
    "ns": (),
    "data": (
        {
            "type": "tool",
            "name": "rag_search",
            "content": "【检索结果】LangGraph 是...",
            "tool_call_id": "call_xyz",
        },
        {"langgraph_node": "tools"},
    ),
}
```

→ `tool_results.append(...)`，不进 `buf`

### M5：根据工具结果继续回答

```python
{
    "type": "messages",
    "ns": (),
    "data": (
        {"type": "AIMessageChunk", "content": "根据检索，LangGraph 是", ...},
        {"langgraph_node": "agent"},
    ),
}
```

---

## 7. `chunk.type == "tasks"` 具体例子

### T1：agent 节点开始

```python
{
    "type": "tasks",
    "ns": (),
    "data": {
        "id": "task-1111-...",
        "name": "agent",
        "input": {"messages": [{"type": "human", "content": "帮我搜一下 LangGraph 文档"}]},
        "triggers": ["messages"],
    },
}
```

代码判断：`"triggers" in data and "input" in data` → 节点**开始** → `agent_depth += 1`

### T2：agent 节点结束

```python
{
    "type": "tasks",
    "ns": (),
    "data": {
        "id": "task-1111-...",
        "name": "agent",
        "result": {"messages": [{"type": "ai", "tool_calls": [...]}]},
        "error": None,
        "interrupts": [],
    },
}
```

代码判断：`"result" in data` → 节点**结束** → `agent_depth -= 1`
正文仍以 `messages` 流的 token 为准，不靠 `tasks.result` 拼 UI。

### T3：tools 节点开始

```python
{
    "type": "tasks",
    "ns": (),
    "data": {
        "id": "task-2222-...",
        "name": "tools",
        "input": {"messages": [...]},
        "triggers": ["messages"],
    },
}
```

→ `tools_depth += 1`

### T4：tools 节点结束（无 interrupt）

```python
{
    "type": "tasks",
    "ns": (),
    "data": {
        "id": "task-2222-...",
        "name": "tools",
        "result": {"messages": [{"type": "tool", "name": "rag_search", "content": "..."}]},
        "error": None,
        "interrupts": [],
    },
}
```

→ `tools_depth -= 1`；归零时 `pending_tool_names = []`

### T5：tools 节点结束（HITL interrupt，非空 `interrupts`）

`write_file` / `edit_file` 等命中 `interrupt_on` 时，工具**尚未执行**，图在 checkpointer 中暂停：

```python
{
    "type": "tasks",
    "ns": (),
    "data": {
        "id": "task-3333-...",
        "name": "tools",
        "result": {...},
        "error": None,
        "interrupts": [{
            "value": {
                "action_requests": [{
                    "name": "write_file",
                    "args": {"file_path": "/workspace/test.txt", "content": "hello"},
                    "description": "Tool execution requires approval\n\nTool: write_file\n...",
                }],
                "review_configs": [{
                    "action_name": "write_file",
                    "allowed_decisions": ["approve", "edit", "reject", "respond"],
                }],
            },
            "id": "6409921951981aa384434ee2c37305dc",
        }],
    },
}
```

`streaming.py` 对非空 `interrupts` 仅 `_tool_trace`；**审批逻辑在 `run_assistant_turn` 流结束后调用 `get_pending_hitl`**（与 `interrupts[0].value` 结构一致）。

---

## 8. 与 `streaming.py` 核心逻辑的对应

### 状态变量

| 变量 | 作用 |
|------|------|
| `buf` | 助手正文累积（不含 ToolMessage）；可通过 `text_prefix` 跨 HITL 段拼接 |
| `tool_results` | 工具返回，供 Streamlit expander；可通过 `tool_results_prefix` 跨段累积 |
| `pending_tool_names` | 从 AI 消息解析出的待执行工具名 |
| `agent_depth` / `tools_depth` | 由 `tasks` 事件维护的节点运行深度 |

### `graph_input`（首段 vs resume 段）

| 场景 | 传入 `stream_assistant_text` 的输入 |
|------|-------------------------------------|
| 用户新消息 | `user_text` → 内部构造 `{"messages": [HumanMessage(...)]}` |
| HITL resume | `graph_input=Command(resume={"decisions": [...]})`，`user_text=""` |

由 `run_assistant_turn` 循环：流式一段 → `get_pending_hitl` → 有则 `decide`（CLI）或返回 `pending_hitl`（UI）→ `Command(resume=...)` 下一段。

### `redraw()` 调用时机

1. 流开始前初始化
2. `tasks`：agent/tools 节点开始或结束
3. `messages`：工具名变化、正文 token、工具结果；或 `content` 为空但状态已变
4. 循环结束：单独 `on_update(..., cursor=False)` 去掉流式光标

### 消费方

| 入口 | 用法 |
|------|------|
| `stream_assistant_text` | async 核心；消费 `messages`/`tasks` chunk；`on_update` 推快照 |
| `run_assistant_turn` | 多段流 + HITL 循环；包装 `stream_assistant_text` + `get_pending_hitl` + `Command(resume=...)` |
| `iter_assistant_text_sync` | CLI：`run_assistant_turn` + `decide=prompt_hitl_decisions_cli` |
| `stream_assistant_reply` | CLI 一行封装（写 stdout） |
| [`hitl.get_pending_hitl`](./hitl.py) | 读 `graph.get_state(config).interrupts`，解析 `action_requests` / `review_configs` |
| [`frontend/app.py`](../frontend/app.py) | `decide=None`；`pending_hitl` 时 `session_state` + 审批按钮；resume 用 `Command` |

---

## 9. HITL 与流式的职责划分

```text
  interrupt_on（agent 配置）
           │
           ▼
  astream(messages, tasks)  ──►  UI：token / 状态行 / tool_results
           │
           ▼（一段结束）
  get_state().interrupts?  ──►  有：暂停，等待人工 decisions
           │
           ▼
  Command(resume={decisions})  ──►  下一段 astream（非新的 HumanMessage）
```

- **流式**：负责「看得见」的助手输出与工具结果展示。
- **HITL**：负责「能不能执行」敏感工具；与流式 chunk type **无新增订阅关系**。
- **权威状态**：checkpointer 里的 `interrupts`；Streamlit 的 `hitl_pending` 仅为 UI 镜像，需与 `get_pending_hitl` 同步（见 `app.py` 顶部清理逻辑）。

---

## 10. 延伸阅读

- LangGraph Streaming：`stream_mode`、`version="v2"`、`StreamPart` / `TasksStreamPart` / `MessagesStreamPart`
- [deepagents Human-in-the-loop](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop)：`interrupt_on` + `Command(resume={"decisions": [...]})`
- 本项目 agent 由 `create_deep_agent` 构建，图中节点名通常为 `agent`、`tools`（及可能的 `model`）
