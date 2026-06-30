# 流式输出说明（v3 event streaming）

本文件说明 LangGraph **`stream_events(version="v3")` / `astream_events(version="v3")`** 的事件模型，以及与当前项目 v2 实现（[`streaming.py`](./streaming.py) + [`streaming.md`](./streaming.md)）的对应关系。

- 当前项目实现（v2）：[`streaming.py`](./streaming.py) 使用 `graph.astream(..., stream_mode=["messages", "tasks"], version="v2")`
- v2 事件说明：[`streaming.md`](./streaming.md)
- 人工审批：[`hitl.py`](./hitl.py)

> **状态**：v3 API 在 LangGraph 中标记为 **experimental / beta**，可能变更。需要 `langgraph` 较新版本（本项目 `pyproject.toml` 要求 `>=1.1.0`；v3 在 1.1.x 已可用）。

---

## 0. v2 与 v3 一句话对比

| | v2（当前项目） | v3 |
|---|----------------|-----|
| 入口 | `graph.astream(..., stream_mode=[...], version="v2")` | `graph.stream_events(..., version="v3")` |
| 返回值 | `AsyncIterator[StreamPart]`，每个 chunk 是 `{type, ns, data}` | `GraphRunStream` / `AsyncGraphRunStream`，**由调用方迭代驱动** pump |
| 消费方式 | 分支 `chunk["type"] == "messages" \| "tasks"` | 迭代 **typed projection**（`run.messages`、`run.values`…）或原始 `ProtocolEvent` |
| 模型输出 | `AIMessageChunk` 元组 `(chunk, metadata)` | **content-block 协议**（`message-start` → `content-block-delta` → `message-finish`） |
| 节点生命周期 | 直接订阅 `tasks` chunk | 默认 **`tasks` 被折叠进 `lifecycle`**；原始 tasks 需注册 `TasksTransformer` |
| HITL | 流结束后 `graph.get_state(config).interrupts` | 可直接读 `stream.interrupted` / `stream.interrupts`（仍建议 checkpointer + thread_id） |

---

## 1. 三层结构（最容易混）

v3 比 v2 多一层 **typed projection**，但底层仍由 Pregel 的 v2 `StreamPart` 转换而来。

```text
graph.astream(..., version="v2")          ← v3 内部仍走这条链路
        │
        ▼ convert_to_protocol_event
ProtocolEvent（原始协议层）                 ← for event in stream / transformer.process
  method = "messages" | "tasks" | "values" | ...
        │
        ▼ StreamTransformer 管道
Typed Projection（应用消费层）              ← stream.messages / stream.lifecycle / ...
```

| 层级 | 访问方式 | 含义 |
|------|----------|------|
| **投影层** | `stream.messages`、`stream.values`、`stream.lifecycle`… | 框架内置 transformer 产出的 **类型化迭代器** |
| **协议层** | `for event in stream:` | 统一外壳 `ProtocolEvent` |
| **messages 内层** | `event["params"]["data"]` 里 `payload["event"]` | content-block 生命周期（仅 `method=="messages"` 且 payload 为 dict 时） |

### 1.1 ProtocolEvent 公共外壳

不论哪种 channel，每个原始事件都是：

```python
{
    "type": "event",
    "seq": 42,                          # 单 run 内单调递增，用于排序
    "method": "messages",               # channel 名，对应 v2 的 chunk["type"]
    "params": {
        "namespace": [],                # 子图路径，如 ["tools:91ac"]；根图为 []
        "timestamp": 1770000000000,     # 墙钟毫秒，可能回拨，勿单独依赖排序
        "data": <因 method 而异>,
        "interrupts": (...),            # 仅 values 等事件可能有
    },
}
```

v2 → v3 的转换规则（源码 `langgraph.stream._convert.convert_to_protocol_event`）：

```python
# v2 StreamPart
{"type": "messages", "ns": (), "data": (payload, metadata)}

# 等价 ProtocolEvent
{"type": "event", "method": "messages", "params": {
    "namespace": list(ns),
    "timestamp": <now_ms>,
    "data": (payload, metadata),   # 结构与 v2 相同
}}
```

### 1.2 默认内置 transformer

`stream_events(version="v3")` 默认注册（源码 `Pregel._pregel_stream_v3`）：

| Transformer | Projection | `required_stream_modes` |
|-------------|------------|-------------------------|
| `ValuesTransformer` | `stream.values` | `values` |
| `MessagesTransformer` | `stream.messages` | `messages` |
| `LifecycleTransformer` | `stream.lifecycle` | `tasks` |
| `SubgraphTransformer` | `stream.subgraphs` | `tasks` |

因此默认底层订阅 **`values` + `messages` + `tasks`** 三种 mode；**不**包含 `tools` / `updates` / `custom` / `checkpoints` / `debug`（需额外 transformer）。

**注意**：`LifecycleTransformer` 会把 `tasks` 事件 **从主协议 log 中 suppress**（`process` 返回 `False`）。直接 `for event in stream` **看不到** `method=="tasks"`，应改用 `stream.lifecycle` 或额外注册 `TasksTransformer` 得到 `stream.tasks`。

---

## 2. 各 projection / channel 各管什么

```text
              graph.stream_events(input, version="v3")
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    stream.messages   stream.values   stream.lifecycle
    （模型输出）        （全量 state）    （子图/子 agent 生命周期）
           │               │               │
    .text / .reasoning   stream.output    started / completed /
    .tool_calls          最终 state       interrupted / failed
           │
           ▼
    原始 method=="messages" 还可看到 ToolMessage（投影层会忽略）
```

| 你想知道的事 | v3 推荐 API |
|-------------|------------|
| 助手说了哪个字 | `async for message in stream.messages:` → `async for token in message.text` |
| 模型推理过程（thinking） | `message.reasoning`（v2 里常为空 content，v3 独立通道） |
| 模型要调什么工具 | `message.tool_calls` 或 raw `content-block-delta` 里 `tool_call` / `tool_call_chunk` |
| 工具返回了什么 | **不在** `stream.messages`；见 §3.2 |
| 根图 agent/tools 节点是否在跑 | 默认 `lifecycle` **只跟踪子图**；根节点用 `stream.tasks`（需 `TasksTransformer`）或从 `message.node` / metadata 推断 |
| 是否有待审批 interrupt | `stream.interrupted` / `stream.interrupts`，或流结束后 `get_pending_hitl`（与 v2 相同） |

---

## 3. tool call 与工具结果（v3 职责拆分）

v3 把 **模型侧 tool call** 与 **工具执行结果** 拆到不同 surface：

| 阶段 | v2（当前） | v3 |
|------|----------|-----|
| 模型决定调工具 | `messages` + `AIMessageChunk.tool_calls` / `tool_call_chunks` | `messages` channel 的 **content-block**（`type: tool_call` / `tool_call_chunk`）→ `message.tool_calls` |
| 工具执行中 | `tasks` + `name=="tools"` 开始/结束 | 可选 `tools` channel（`tool-started`…，**默认未订阅**）；或 `TasksTransformer` → `stream.tasks` |
| 工具返回 | `messages` + `ToolMessage` | **`run.messages` 故意忽略**（`role=="tool"` 的 message-start 被 skip）；ToolMessage 仍可能出现在 **raw** `method=="messages"` 事件里 |

### 3.1 迁移 [`streaming.py`](./streaming.py) 时的映射

| 变量 | v2 来源 | v3 等价思路 |
|------|---------|------------|
| `buf` | `messages` 里非 tool 的 `AIMessageChunk.content` | `message.text` 迭代 |
| `pending_tool_names` | `tool_calls` / `tool_call_chunks` | `message.tool_calls` 或 block-delta |
| `tool_results` | `messages` 里 `ToolMessage` | raw `messages` 事件中的 `ToolMessage` 对象，或 `stream.values` / `UpdatesTransformer` |
| `agent_depth` / `tools_depth` | `tasks` 事件 | 注册 `TasksTransformer` 读 `stream.tasks`，或自定义 transformer |

---

## 4. 推荐消费方式（投影层）

### 4.1 最简：只关心助手正文

```python
stream = await graph.astream_events(
    {"messages": [HumanMessage(content=user_text)]},
    config=config,
    version="v3",
)

async for message in stream.messages:
    async for token in message.text:
        buf.append(token)          # 等价 streaming.py 的 buf
        redraw()

final_state = await stream.output  # 或 stream.output（sync 版 drive 完后读取）
```

每个 `message` 是 **一次 LLM 调用** 的 `ChatModelStream` / `AsyncChatModelStream`（draft / refine 等多轮调用会自动分成多个 message 对象，**无需** v2 里按 `langgraph_node` 手动 reset accumulator）。

### 4.2 并发消费多路投影（async）

```python
stream = await graph.astream_events(input, config=config, version="v3")

async def consume_text():
    async for message in stream.messages:
        async for token in message.text:
            ...

async def consume_state():
    async for snapshot in stream.values:
        ...

await asyncio.gather(consume_text(), consume_state())
```

### 4.3 严格到达顺序：interleave

```python
stream = graph.stream_events(input, config=config, version="v3")

for name, item in stream.interleave("values", "messages", "lifecycle"):
    if name == "messages":
        ...
    elif name == "values":
        ...
```

### 4.4 HITL resume

```python
from langgraph.types import Command

stream = await graph.astream_events(input, config=config, version="v3")
async for message in stream.messages:
    ...

if stream.interrupted:
    pending = stream.interrupts          # 或仍用 get_pending_hitl(graph, config)

stream = await graph.astream_events(
    Command(resume={"decisions": decisions}),
    config=config,
    version="v3",
)
final = await stream.output
```

---

## 5. 一次完整调工具的时间线（v3 视角）

用户：「帮我搜一下 LangGraph 文档」

```text
V1  values            → state 快照更新
M1  messages          → message-start (role=ai)
M2  messages          → content-block-start (type=tool_call_chunk, name=rag_search)
M3  messages          → content-block-delta (args 流式片段)
M4  messages          → content-block-finish (完整 tool_call)
M5  messages          → message-finish
    stream.messages   → 本次 LLM 调用的 ChatModelStream 关闭

T1  tasks (内部)      → LifecycleTransformer 消费；主 log 不可见
    stream.tasks*     → 若注册 TasksTransformer：agent/tools 开始/结束
M6  messages (raw)    → ToolMessage(name=rag_search, content=...)  ← 投影层不进 run.messages

M7  messages          → 新一轮 message-start
M8  messages          → content-block-delta (type=text-delta, text="根据检索…")
M9  messages          → message-finish (+ usage_metadata 可选)

* 默认未注册 TasksTransformer；见 §1.2
```

### 5.1 带 HITL 的写文件时间线

```text
M1–M4  messages       → 可能先有 text-delta，再有 write_file tool_call block
T1     tasks 结束     → data.interrupts 非空（Lifecycle → interrupted）
                       → stream.interrupted == True
── 暂停：CLI / Streamlit 审批 ──
    resume：Command(resume={"decisions": [...]})
M5+    messages       → 下一段 stream_events
M6     messages(raw)  → ToolMessage(write_file 结果)（若 approve）
```

---

## 6. `method == "messages"` — content-block 协议示例

内层 `params["data"]` 仍是 **`(payload, metadata)` 二元组**（与 v2 相同）。启用 v3 后，`payload` 多为 **协议 dict**（带 `"event"` 键），而非 `AIMessageChunk`。

`metadata` 常见字段：`langgraph_node`、`langgraph_step`、`run_id`（`MessagesTransformer` 用 `run_id` 关联 `ChatModelStream`）。

### 6.1 消息生命周期（`data.event` 取值）

| `data.event` | 含义 |
|--------------|------|
| `message-start` | 一条 LLM 输出开始；含 `role`（`ai` / `tool`） |
| `content-block-start` | 一个 content block 开始（text / reasoning / tool_call…） |
| `content-block-delta` | block 增量 |
| `content-block-finish` | block 结束，带完整 content |
| `message-finish` | 整条消息结束；可含 `usage` |
| `message-error` | 不可恢复的模型调用失败 |

### M1：正文 token（text-delta）

```python
{
    "type": "event",
    "seq": 10,
    "method": "messages",
    "params": {
        "namespace": [],
        "timestamp": 1770000000010,
        "data": (
            {
                "event": "content-block-delta",
                "index": 0,
                "delta": {"type": "text-delta", "text": "好"},
            },
            {"langgraph_node": "agent", "langgraph_step": 3, "run_id": "run-abc"},
        ),
    },
}
```

→ 投影层：`message.text` 产出 `"好"`
→ 等价 v2：`AIMessageChunk(content="好")`

### M2：推理 token（reasoning-delta，DeepSeek 等）

```python
{
    "type": "event",
    "method": "messages",
    "params": {
        "namespace": [],
        "timestamp": 1770000000011,
        "data": (
            {
                "event": "content-block-delta",
                "index": 0,
                "delta": {"type": "reasoning-delta", "reasoning": "先分析用户意图…"},
            },
            {"langgraph_node": "agent", "run_id": "run-abc"},
        ),
    },
}
```

→ 投影层：`message.reasoning`（**v2 里这些内容通常在 content 之外，容易漏掉**）

### M3：tool call 流式开始（tool_call_chunk）

```python
{
    "type": "event",
    "method": "messages",
    "params": {
        "namespace": [],
        "data": (
            {
                "event": "content-block-start",
                "index": 0,
                "content": {
                    "type": "tool_call_chunk",
                    "id": "call_xyz",
                    "name": "rag_search",
                    "args": "",
                },
            },
            {"langgraph_node": "agent", "run_id": "run-abc"},
        ),
    },
}
```

### M4：tool call 参数流式增量

```python
{
    "type": "event",
    "method": "messages",
    "params": {
        "namespace": [],
        "data": (
            {
                "event": "content-block-delta",
                "index": 0,
                "delta": {
                    "type": "block-delta",
                    "fields": {
                        "type": "tool_call_chunk",
                        "id": "call_xyz",
                        "name": "rag_search",
                        "args": '{"query": "LangGraph',
                        "index": 0,
                    },
                },
            },
            {"langgraph_node": "agent", "run_id": "run-abc"},
        ),
    },
}
```

→ 等价 v2：`tool_call_chunks` 里 `args` 字符串片段

### M5：完整 tool_call（content-block-finish）

```python
{
    "type": "event",
    "method": "messages",
    "params": {
        "namespace": [],
        "data": (
            {
                "event": "content-block-finish",
                "index": 0,
                "content": {
                    "type": "tool_call",
                    "id": "call_xyz",
                    "name": "rag_search",
                    "args": {"query": "LangGraph 文档"},
                },
            },
            {"langgraph_node": "agent", "run_id": "run-abc"},
        ),
    },
}
```

→ 等价 v2：`tool_calls` 完整列表

### M6：message-finish（含 usage 可选）

```python
{
    "type": "event",
    "method": "messages",
    "params": {
        "namespace": [],
        "data": (
            {"event": "message-finish", "usage": {"output_tokens": 42, "input_tokens": 100}},
            {"langgraph_node": "agent", "run_id": "run-abc"},
        ),
    },
}
```

### M7：工具返回（ToolMessage — 仅 raw 层）

`run.messages` **不会**为 ToolMessage 创建 `ChatModelStream`；但底层 `messages` mode 仍可能推送：

```python
{
    "type": "event",
    "method": "messages",
    "params": {
        "namespace": [],
        "data": (
            ToolMessage(
                content="【检索结果】LangGraph 是...",
                name="rag_search",
                tool_call_id="call_xyz",
            ),
            {"langgraph_node": "tools", "run_id": "run-tool-1"},
        ),
    },
}
```

→ 等价 v2 §M4；迁移时若要 `tool_results`，需监听 **raw 事件** 或 `values`/`updates`

### M8：legacy AIMessageChunk（v3 投影层忽略）

若 payload 是 `AIMessageChunk` 元组（旧 `on_llm_new_token` 路径），`MessagesTransformer` **不会**写入 `run.messages`。v3 内部通过 `CONFIG_KEY_STREAM_MESSAGES_V2: True` 尽量走 content-block 路径；若仍收到 chunk 元组，只能走 raw 协议层自行解析。

---

## 7. `method == "tasks"` 与 `stream.lifecycle`

v2 里直接可见的 tasks chunk，在 v3 默认配置下：

- **主协议 log**：被 `LifecycleTransformer` suppress
- **`stream.lifecycle`**：仅 **子图 / 命名 subagent**（`namespace` 深度 > 0）
- **`stream.tasks`**：需显式注册 `TasksTransformer`

### 7.1 原始 tasks（注册 TasksTransformer 后）

与 v2 结构相同，只是包在 `ProtocolEvent` 里：

#### T1：agent 节点开始

```python
{
    "type": "event",
    "method": "tasks",
    "params": {
        "namespace": [],
        "timestamp": 1770000000020,
        "data": {
            "id": "task-1111-...",
            "name": "agent",
            "input": {"messages": [{"type": "human", "content": "帮我搜一下 LangGraph 文档"}]},
            "triggers": ["messages"],
        },
    },
}
```

#### T2：tools 节点结束（HITL interrupt）

```python
{
    "type": "event",
    "method": "tasks",
    "params": {
        "namespace": [],
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
    },
}
```

→ 等价 v2 §T5；v3 还可映射为 `stream.lifecycle` 的 `interrupted` 或 `stream.interrupts`

### 7.2 lifecycle 投影（子图）

```python
{
    "type": "event",
    "method": "lifecycle",
    "params": {
        "namespace": [],
        "data": {
            "event": "started",           # started | completed | failed | interrupted | drained
            "namespace": ["researcher:6f4d"],
            "graph_name": "researcher",
            "trigger_call_id": "91ac",
            "cause": {"type": "toolCall", "tool_call_id": "call_xyz"},
        },
    },
}
```

---

## 8. 其它 channel 示例（opt-in）

默认 v3 **不**订阅这些 mode；注册对应 transformer 或自定义 transformer 声明 `required_stream_modes` 后才会出现。

### 8.1 `tools` channel

| `data.event` | 含义 |
|--------------|------|
| `tool-started` | 工具开始；含 `tool_call_id`、`tool_name`、`input` |
| `tool-output-delta` | 工具输出流式片段 |
| `tool-finished` | 工具完成；含 `output` |
| `tool-error` | 工具失败 |

```python
{
    "type": "event",
    "method": "tools",
    "params": {
        "namespace": [],
        "data": {
            "event": "tool-started",
            "tool_call_id": "call_xyz",
            "tool_name": "rag_search",
            "input": {"query": "LangGraph 文档"},
        },
    },
}
```

### 8.2 `values` channel

```python
{
    "type": "event",
    "method": "values",
    "params": {
        "namespace": [],
        "data": {"messages": [...], ...},   # 完整 state 快照
        "interrupts": (),                   # 非空表示 interrupt
    },
}
```

→ `stream.output` / `stream.interrupted` / `stream.interrupts` 由 `ValuesTransformer` 跟踪

### 8.3 `updates` channel（需 `UpdatesTransformer`）

```python
{
    "type": "event",
    "method": "updates",
    "params": {
        "namespace": [],
        "data": {"agent": {"messages": [AIMessage(...)]}},
    },
}
```

### 8.4 `custom` channel（节点内 `get_stream_writer()`）

```python
{
    "type": "event",
    "method": "custom",
    "params": {
        "namespace": [],
        "data": {"kind": "progress", "message": "retrieving context"},
    },
}
```

---

## 9. 与 [`streaming.py`](./streaming.py) 概念对照

| streaming.py 概念 | v2 实现 | v3 等价 |
|-------------------|---------|---------|
| 入口 | `graph.astream(..., stream_mode=["messages","tasks"], version="v2")` | `await graph.astream_events(..., version="v3")` |
| 正文 `buf` | 累积 `AIMessageChunk.content` | `async for m in stream.messages: async for t in m.text: buf+=t` |
| `pending_tool_names` | 解析 `tool_calls` / `tool_call_chunks` | `m.tool_calls` 或 raw content-block |
| `tool_results` | `ToolMessage` in `messages` | raw `method=="messages"` 中的 `ToolMessage`，或 state snapshot |
| `agent_depth` / `tools_depth` | `tasks` 开始/结束 | `TasksTransformer` → `stream.tasks`；或自定义 |
| `redraw()` 触发 | messages + tasks 事件 | 每个 text token / tool_call 更新 / tasks 或 lifecycle 变化 |
| HITL 检测 | 段结束后 `get_pending_hitl` | `stream.interrupted` **或** 仍用 `get_pending_hitl`（权威 checkpointer） |
| resume | `Command(resume=...)` + 下一段 `astream` | 相同，但入口改为 `astream_events` |

### 9.1 v3 不允许的参数

以下 kwargs 会 **直接 TypeError**（v3 自行管理）：

- `stream_mode=...`
- `subgraphs=...`

stream mode 由所有 transformer 的 `required_stream_modes` **并集** 决定。

---

## 10. HITL 与流式的职责划分（v3）

```text
  interrupt_on（agent 配置）
           │
           ▼
  astream_events(version="v3")  ──►  stream.messages：token / tool_calls
           │                         stream.interrupted / .interrupts
           ▼（一段结束）
  get_pending_hitl()?  ──►  可与 stream.interrupts 交叉验证；UI 仍以 checkpointer 为准
           │
           ▼
  Command(resume={decisions})  ──►  下一段 astream_events
```

- **投影层**：负责「看得见」的模型输出（正文、推理、tool call 参数流）。
- **ToolMessage / 工具节点状态**：默认不在 `run.messages`；按 §3.2 选 raw / tasks / values。
- **HITL**：`stream.interrupts` 便于流式 UI 同步；[`hitl.get_pending_hitl`](./hitl.py) 仍是 checkpointer 权威读法。

---

## 11. 何时用 raw 事件 vs 投影

| 需求 | 推荐 |
|------|------|
| 只要助手正文 | `stream.messages` → `.text` |
| 要 reasoning + text + tool_call **严格交错顺序** | `for event in stream` 过滤 `method=="messages"` 的 content-block-delta |
| 要 tool 执行结果 | raw messages 里的 `ToolMessage`，或 `stream.values` |
| 要根图 agent/tools 节点调度 | 加 `TasksTransformer` → `stream.tasks` |
| 要子 agent / 子图 | `stream.subgraphs` 或 `stream.lifecycle` |
| 自定义业务指标 | 编写 `StreamTransformer` + `StreamChannel` → `stream.extensions` |

---

## 12. 延伸阅读

- [LangGraph Event streaming（v3）](https://docs.langchain.com/oss/python/langgraph/event-streaming)
- [LangGraph Streaming（v2 stream_mode）](https://docs.langchain.com/oss/python/langgraph/streaming) — v3 底层仍依赖此链路
- [LangGraph v3 Event Streaming（架构 walkthrough）](https://vadim.blog/langgraph-v3-event-streaming-typed-projections)
- 本项目 v2 消费说明：[`streaming.md`](./streaming.md)
- [deepagents Human-in-the-loop](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop)
