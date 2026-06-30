# LangGraph v2 流式输出教学文档

本文档分两部分：

1. **基础篇**：LangGraph `stream` / `astream` 的 v2 流式 API；按 `values` → `updates` → `messages` → `tasks` / `custom` 顺序讲解各 `stream_mode`。
2. **项目篇**：本仓库 `[streaming.py](./streaming.py)` 如何消费 `messages` + `tasks`，以及与 `redraw` / `buf` / `tool_results` / HITL 的对应关系。

相关文件：


| 文件                                      | 说明                                 |
| --------------------------------------- | ---------------------------------- |
| `[streaming.py](./streaming.py)`        | 流式实现（CLI / Streamlit 共用）           |
| `[hitl.py](./hitl.py)`                  | interrupt 解析、CLI 审批                |
| `[frontend/app.py](../frontend/app.py)` | Streamlit 审批 UI                    |
| `[streaming_v3.md](./streaming_v3.md)`  | v3 event streaming（实验性 API，本项目未采用） |


---

## 目录

- [Part A · LangGraph v2 流式基础](#part-a--langgraph-v2-流式基础)
  - [A.1 为什么需要流式](#a1-为什么需要流式)
  - [A.2 最小用法：`stream` / `astream](#a2-最小用法stream--astream)`
  - [A.3 `version="v2"` 与 `StreamPart` 外壳](#a3-versionv2-与-streampart-外壳)
  - [A.4 `stream_mode` 一览](#a4-stream_mode-一览)
  - [A.5 选课指南：UI 要什么，就订什么 mode](#a5-选课指南ui-要什么就订什么-mode)
  - [A.6 `values`：完整 state 快照](#a6-values完整-state-快照)
  - [A.7 `updates`：按节点的 state 增量](#a7-updates按节点的-state-增量)
  - [A.8 `messages`：内容流（正文、tool call、工具返回）](#a8-messages内容流正文tool-call工具返回)
  - [A.9 `tasks` / `custom` 与其它 mode](#a9-tasks--custom-与其它-mode)
- [Part B · 本项目流式实现](#part-b--本项目流式实现)
  - [B.1 本项目订阅了哪些 mode](#b1-本项目订阅了哪些-mode)
  - [B.2 `messages` 与 `tasks` 各管什么](#b2-messages-与-tasks-各管什么)
  - [B.3 tool call 与 ToolMessage](#b3-tool-call-与-toolmessage)
  - [B.4 一次完整调工具的时间线](#b4-一次完整调工具的时间线)
  - [B.5 带 HITL 的写文件时间线](#b5-带-hitl-的写文件时间线)
    - [B.5.1 chunk 级时间线（单段 `astream` 内）](#b51-chunk-级时间线单段-astream-内)
    - [B.5.2 CLI 端到端时间线（入口 `iter_assistant_text_sync`）](#b52-cli-端到端时间线入口-iter_assistant_text_sync)
  - [B.6 `messages` chunk 具体例子](#b6-messages-chunk-具体例子)
  - [B.7 `tasks` chunk 具体例子](#b7-tasks-chunk-具体例子)
  - [B.8 与 `streaming.py` 核心逻辑的对应](#b8-与-streamingpy-核心逻辑的对应)
  - [B.9 HITL 与流式的职责划分](#b9-hitl-与流式的职责划分)
- [延伸阅读](#延伸阅读)

---

# Part A · LangGraph v2 流式基础

## A.1 为什么需要流式

LangGraph 的「流式」不只是把 LLM 的 token 打到终端，而是**把图执行过程实时暴露出来**：

- 节点每步改了哪些 state（`values` 全量快照 / `updates` 增量）
- 模型正在吐哪个字、tool call、工具返回（`messages`）
- 哪个节点开始/结束、是否 interrupt（`tasks`）
- 节点里手动推送的进度（`custom`）

对聊天 UI 来说，通常同时关心 **内容流**（`messages`，用户看得见）和 **节点生命周期**（`tasks`，显示「推理中」「工具运行中」）。这正是本项目选择 `messages` + `tasks` 的原因（见 [Part B](#part-b--本项目流式实现)）。

---

## A.2 最小用法：`stream` / `astream`

图编译后调用：

```python
# 同步
for chunk in graph.stream(inputs, stream_mode="updates", version="v2"):
    ...

# 异步（本项目 streaming.py 使用）
async for chunk in graph.astream(inputs, stream_mode=["messages", "tasks"], version="v2"):
    ...
```


| 参数             | 含义                                                                |
| -------------- | ----------------------------------------------------------------- |
| `inputs`       | 图输入，如 `{"messages": [HumanMessage(...)]}` 或 `Command(resume=...)` |
| `stream_mode`  | 字符串或列表，决定订阅哪些事件                                                   |
| `version="v2"` | **强烈建议写上**；统一 chunk 结构（见下一节）                                      |
| `config`       | 含 `thread_id` 等；配合 checkpointer 才能用 `tasks` / HITL                |


---

## A.3 `version="v2"` 与 `StreamPart` 外壳

LangGraph ≥ 1.1 引入 v2 格式。不论订阅几种 `stream_mode`、是否有子图，**每个 chunk 形状一致**：

```python
{
    "type": "values" | "updates" | "messages" | "custom" | "checkpoints" | "tasks" | "debug",
    "ns": (),           # 子图命名空间；根图为空元组，子图如 ("agent:abc123",)
    "data": ...,        # 实际载荷，因 type 而异
}
```

v1（默认）格式会随 mode 数量、是否开子图而变化（单 mode 直接 yield `data`，多 mode yield `(mode, data)` 元组等），阅读和类型收窄都更麻烦。**新项目请固定 `version="v2"`**。

类型可从 `langgraph.types` 导入：`StreamPart`、`ValuesStreamPart`、`UpdatesStreamPart`、`MessagesStreamPart` 等；对 `chunk["type"]` 分支后，编辑器可自动收窄 `chunk["data"]` 类型。

---

## A.4 `stream_mode` 一览


| `stream_mode`     | 外层 `chunk["type"]` | `data` 大致形态                     | 典型用途                               |
| ----------------- | ------------------ | ------------------------------- | ---------------------------------- |
| `**values**`      | `"values"`         | 完整 state 快照                     | 每步后看全量状态、调试、检测 interrupt           |
| `**updates**`     | `"updates"`        | `{节点名: 该节点返回的增量}`               | 只看谁改了什么，payload 更小                 |
| `**messages**`    | `"messages"`       | `(message_chunk, metadata)` 二元组 | LLM token、tool_call 片段、ToolMessage |
| `**custom**`      | `"custom"`         | 任意 dict                         | 节点内 `get_stream_writer()` 推送进度     |
| `**checkpoints**` | `"checkpoints"`    | 同 `get_state()` 格式              | 观察 checkpoint 写入（需 checkpointer）   |
| `**tasks**`       | `"tasks"`          | 节点开始/结束的 dict                   | 节点生命周期、错误、interrupt 痕迹             |
| `**debug**`       | `"debug"`          | checkpoints + tasks + 元数据       | 全量调试                               |


> **需 checkpointer 的 mode**：`tasks`、`checkpoints`、`debug`（以及依赖持久化状态的 HITL）。本项目在 `agent_core.py` 使用 `MemorySaver`。

下面按 `**values` → `updates` → `messages` → `tasks` / `custom`** 顺序逐节展开；聊天 Agent 实际最常用的是后两者（见 [A.5](#a5-选课指南ui-要什么就订什么-mode) 与 [Part B](#part-b--本项目流式实现)）。

---

## A.5 选课指南：UI 要什么，就订什么 mode

做流式 UI 时，先问「用户要看到什么」，再选 `stream_mode`：


| 你想展示…                 | 订阅                      | 原因                         |
| --------------------- | ----------------------- | -------------------------- |
| 助手逐字输出、tool call、工具返回 | `messages`              | 唯一带 **token 粒度** 的 mode    |
| 「模型推理中」「工具运行中」        | `tasks`                 | 节点 **开始 / 结束** 事件，不携带正文    |
| 每步后的完整 state 快照       | `values`                | 全量 state，适合快照 UI、调试        |
| 谁改了 state 的哪几个字段      | `updates`               | 按节点名的增量，payload 更小         |
| 节点内自定义进度              | `custom`                | `get_stream_writer()` 手动推送 |
| checkpoint / 全量调试     | `checkpoints` / `debug` | 开发排错向                      |


**推荐阅读顺序**（与下文章节编号一致）：

1. [A.6 `values](#a6-values完整-state-快照)` → [A.7 `updates](#a7-updates按节点的-state-增量)` —— 理解 state 流的两种粒度。
2. [A.8 `messages](#a8-messages内容流正文tool-call工具返回)` → [A.9 `tasks` / `custom](#a9-tasks--custom-与其它-mode)` —— 聊天 Agent 标配，本项目即 `messages` + `tasks`（见 [Part B](#part-b--本项目流式实现)）。

```text
典型聊天 UI
    │
    ├─ 正文区 ────────── messages（AIMessageChunk.content）
    ├─ 状态行 ────────── tasks（agent / tools 节点开始结束）
    ├─ 工具名 / 参数 ─── messages（tool_calls / tool_call_chunks）
    ├─ 工具结果 ──────── messages（ToolMessage）
    └─ 侧边栏 state 树 ─ values 或 updates（一般不用来拼正文）
```

---

## A.6 `values`：完整 state 快照

### 6.1 它解决什么问题

`values` 在每个 super-step 结束后 emit **合并后的完整 state**。适合：

- 侧边栏「当前世界长什么样」的快照 UI
- 调试：直接看每步后全量 state
- 检测 interrupt：v2 的 `ValuesStreamPart` 上可带 `interrupts` 字段

**聊天正文不要用 `values` 拼字** —— token 粒度在 `messages` 里更细（见 [A.8](#a8-messages内容流正文tool-call工具返回)）。

### 6.2 示例图

下面两节（A.6 / A.7）共用同一小图（无 LLM，纯 state 变更）：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    topic: str
    joke: str

def refine_topic_node(state: State):
    return {"topic": state["topic"] + " and cats"}

def generate_joke_node(state: State):
    return {"joke": f"This is a joke about {state['topic']}"}

graph = (
    StateGraph(State)
    .add_node(refine_topic_node)
    .add_node(generate_joke_node)
    .add_edge(START, "refine_topic_node")
    .add_edge("refine_topic_node", "generate_joke_node")
    .add_edge("generate_joke_node", END)
    .compile()
)
```

### 6.3 订阅与输出

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="values",
    version="v2",
):
    if chunk["type"] == "values":
       print(chunk)
```

```text
{'type': 'values', 'ns': (), 'data': {'topic': 'ice cream'}, 'interrupts': ()}
{'type': 'values', 'ns': (), 'data': {'topic': 'ice cream and cats'}, 'interrupts': ()}
{'type': 'values', 'ns': (), 'data': {'topic': 'ice cream and cats', 'joke': 'This is a joke about ice cream and cats'}, 'interrupts': ()}
```

特点：

- **payload 大**：每步都是全量 state；对话图里 `messages` 列表很长时尤其明显。
- `**data` 结构**：与 State schema 一致，可直接当「当前 state」渲染。
- **含初始输入**：第一条往往是图输入合并后的快照（上例先 emit 仅含 `topic` 的状态）。

与 `updates` 的对比见 [A.7](#a7-updates按节点的-state-增量)。

---

## A.7 `updates`：按节点的 state 增量

### 7.1 它解决什么问题

`updates` 在每个节点执行完后，只 emit **该节点 return 的字典**，`data` 键为节点名。适合：

- 日志：关注「谁写了什么」
- 增量 UI：只 patch 变更字段，payload 更小

同样**不用于拼聊天正文**。

### 7.2 订阅与输出

沿用 [A.6](#a6-values完整-state-快照) 的示例图：

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="updates",
    version="v2",
):
    if chunk["type"] == "updates":
        print(chunk)
```

```text
{'type': 'updates', 'ns': (), 'data': {'refine_topic_node': {'topic': 'ice cream and cats'}}}
{'type': 'updates', 'ns': (), 'data': {'generate_joke_node': {'joke': 'This is a joke about ice cream and cats'}}}
```

特点：

- **带节点名**：`data` 是 `{node_name: partial_state}`；同一步多个节点会 **分多条** chunk 发出。
- **不含未改字段**：上例第一步不会告诉你 `joke` 是什么（那时还没写入）。
- **与 HITL**：`data` 里可能出现 `__interrupt__` 键。

### 7.3 与 `values` 对比


| 维度              | `values`                 | `updates`                     |
| --------------- | ------------------------ | ----------------------------- |
| 每次 emit         | 合并后完整 state              | 单节点增量                         |
| `data` 结构       | 与 State schema 一致        | `{节点名: partial_state}`        |
| 长 `messages` 列表 | 完整数组，payload 大           | 通常只有新增片段                      |
| 适用场景            | 快照 UI、调试                 | 日志、增量 UI、「谁写了什么」              |
| 与 HITL          | part 上可有 `interrupts` 字段 | `data` 里可能有 `__interrupt__` 键 |


---

## A.8 `messages`：内容流（正文、tool call、工具返回）

### 8.1 它解决什么问题

`messages` 把图里所有 **LangChain 消息级输出** 实时推出来：模型 token、tool call 片段、工具执行结果。聊天 UI 的「打字机效果」几乎都靠它。

### 8.2 最小示例

```python
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START

project_root = Path.cwd()
if not (project_root / ".env").exists():
    project_root = project_root.parent
load_dotenv(project_root / ".env")


@dataclass
class MyState:
    topic: str
    joke: str = ""


model = init_chat_model(
    model=os.environ["DEEPSEEK_MODEL"],
    model_provider="deepseek",
    api_key=os.environ["DEEPSEEK_API_KEY"],
)

def call_model(state: MyState):
    # 即使用 .invoke()，LangGraph 仍会把模型输出拆成 AIMessageChunk 流出
    model_response = model.invoke(
        [{"role": "user", "content": f"Generate a joke about {state.topic}"}]
    )
    return {"joke": model_response.content}

graph = (
    StateGraph(MyState)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .compile()
)

for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        message_chunk, metadata = chunk["data"]
        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)
```

终端会逐字打印笑话正文。若打开 debug 打印完整 chunk，每条大致如下（字段已省略）：

```text
{'type': 'messages', 'data': (AIMessageChunk(content='Why'), {'langgraph_node': 'call_model', ...})}
{'type': 'messages', 'data': (AIMessageChunk(content=' did'), {'langgraph_node': 'call_model', ...})}
...
```

### 8.3 chunk 结构：先分清两层 `type`

读 `messages` 事件时有 **两层「类型」**，初学者极易混淆：


| 层级     | 字段位置                        | 取值示例                                 | 含义                                        |
| ------ | --------------------------- | ------------------------------------ | ----------------------------------------- |
| **外层** | `chunk["type"]`             | `"messages"` / `"tasks"` …           | LangGraph **流事件种类**（= 你订阅的 `stream_mode`） |
| **内层** | `chunk["data"][0].type` 或类名 | `"ai"` / `"tool"` / `AIMessageChunk` | LangChain **消息种类**（仅外层为 `messages` 时存在）   |


记忆口诀：**外层 type = stream_mode 名；内层 type = 消息角色（ai / tool / human）**。

每条 `messages` chunk 的 `data` **永远是二元组** `(message_chunk, metadata)`，不是裸字符串：

```python
{
    "type": "messages",       # 外层
    "ns": (),
    "data": (
        AIMessageChunk(content="你", ...),   # 内层消息对象
        {"langgraph_node": "agent", "langgraph_step": 2, ...},  # metadata
    ),
}
```

`metadata` 常用字段：`langgraph_node`（产出该片段的节点名）、`langgraph_step`、`tags`（模型 tag，可过滤多模型图）。

### 8.4 三类内容，都在同一条 `messages` 流里

`message_chunk` 是 LangChain 消息对象，不是 LangGraph 概念。Agent 图里常见三种：


| 阶段            | 内层消息                          | 识别方式                                              | `content` 典型值 |
| ------------- | ----------------------------- | ------------------------------------------------- | ------------- |
| **正文 token**  | `AIMessageChunk`              | 有 `content` 字符串                                   | `"你"`、`"好"` … |
| **tool call** | `AIMessageChunk`              | 有 `tool_calls` / `tool_call_chunks`，`content` 常为空 | `""`          |
| **工具返回**      | `ToolMessage`（`type=="tool"`） | 有 `name`、`tool_call_id`                           | 工具返回字符串       |


**正文示例** —— 直接拼进 UI：

```python
AIMessageChunk(content="好", tool_calls=[], tool_call_chunks=[])
```

**tool call 示例** —— 模型决定调工具，参数常流式补全：

```python
AIMessageChunk(
    content="",
    tool_call_chunks=[{"name": "search_web", "args": '{"query":', "index": 0, ...}],
)
# 随后可能补全为完整 tool_calls（finish_reason: tool_calls）
```

**ToolMessage 示例** —— 工具跑完后的返回（没有单独的 `tools` stream_mode）：

```python
ToolMessage(type="tool", name="search_web", content="检索结果...", tool_call_id="call_xyz")
```

因此：`**messages` = 模型侧输出 + 工具侧返回**，都挂在同一条外层事件流上；区分靠内层 `type` 与字段（`tool_calls` vs `name` / `tool_call_id`）。

### 8.5 实用过滤

```python
message_chunk, metadata = chunk["data"]

# 只显示某节点的正文
if metadata.get("langgraph_node") == "agent" and message_chunk.content:
    buf.append(message_chunk.content)

# 按模型 tag 过滤（多 LLM 图）
if metadata.get("tags") == ["joke"]:
    ...
```

模型打 `nostream` tag 可让该次调用 **不出现在 `messages` 流**（仍正常写入 state）。

---

## A.9 `tasks` / `custom` 与其它 mode

### 9.1 `tasks`：节点生命周期

#### 它解决什么问题

`messages` 告诉你 **「说了什么」**，但不告诉你 **「谁在跑」**。用户常见需求：

- 模型还没吐字时，显示「⏳ 推理中…」
- 工具执行时，显示「🔄 工具运行中」

这些靠 `tasks`：每个节点 **开始** 和 **结束** 各 emit 一条事件。

```text
  messages                          tasks
  ────────                          ─────
  "你" "好" "，" …                  agent 开始 ──► agent 结束
  tool_call: rag_search             tools 开始 ──► tools 结束
  ToolMessage: 检索结果…            agent 开始 ──► agent 结束
                                    （不携带 token 正文）
```

#### 前置条件：checkpointer

`tasks` **必须** 配合 checkpointer（如 `MemorySaver`）和带 `thread_id` 的 `config`，否则不会 emit。HITL 同样依赖持久化状态。

#### 开始事件 vs 结束事件

同一节点用 `data["id"]` 关联开始与结束两条 chunk：


| 时机     | `data` 里有什么                               | `data` 里没有什么     |
| ------ | ----------------------------------------- | ---------------- |
| **开始** | `id`、`name`、`input`、`triggers`            | `result`、`error` |
| **结束** | `id`、`name`、`result`、`error`、`interrupts` | `input`          |


- `name`：节点名，Agent 图里常见 `"agent"` / `"model"` / `"tools"`。
- `result`：该节点 return 的 state 增量（与 [A.7 `updates](#a7-updates按节点的-state-增量)` 里单节点 payload 同类，但挂在生命周期事件上）。
- `interrupts`：非空表示该节点发生了 interrupt（HITL）；**审批 UI 仍以流结束后 `get_state().interrupts` 为准**（见 [B.9](#b9-hitl-与流式的职责划分)）。

#### 示例

沿用 [A.8](#a8-messages内容流正文tool-call工具返回) 的 `call_model` 图：

```python
from langgraph.checkpoint.memory import MemorySaver

graph = (
    StateGraph(MyState)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .compile(checkpointer=MemorySaver())
)

for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="tasks",
    version="v2",
    config={"configurable": {"thread_id": "demo-1"}},
):
    if chunk["type"] == "tasks":
        print(chunk)
```

输出（两条，同一 `id`）：

```text
# 开始：有 input，无 result
{'type': 'tasks', 'data': {'id': '2131…', 'name': 'call_model',
  'input': MyState(topic='ice cream', joke=''), 'triggers': ('branch:to:call_model',)}}

# 结束：有 result，无 input
{'type': 'tasks', 'data': {'id': '2131…', 'name': 'call_model',
  'error': None, 'result': {'joke': 'Why did the ice cream…'}, 'interrupts': []}}
```

消费模式：维护 `agent_depth` / `tools_depth` 计数器，开始 `++`、结束 `--`，深度 > 0 即显示对应状态行。完整时间线见 [B.4](#b4-一次完整调工具的时间线)。

### 9.2 `custom`：节点内自定义进度

节点内通过 `get_stream_writer()` 手动推送任意 dict：

```python
from langgraph.config import get_stream_writer

def my_node(state):
    writer = get_stream_writer()
    writer({"status": "thinking..."})
    return {"answer": "done"}
```

流里 `chunk["type"] == "custom"`，`data` 即 `writer(...)` 传入的对象。适合进度条、非 LLM 的中间状态。

### 9.3 多 mode 组合订阅

v2 下多 mode **不会** 变成 `(mode, data)` 元组；一律是 `StreamPart`，用 `part["type"]` 分支：

```python
for part in graph.astream(
    inputs,
    stream_mode=["messages", "tasks"],   # 本项目实际组合
    version="v2",
    config=config,
):
    if part["type"] == "messages":
        msg, meta = part["data"]
        if msg.content:
            print(msg.content, end="")
    elif part["type"] == "tasks":
        data = part["data"]
        if "result" in data:          # 结束事件
            print(f"节点 {data['name']} 完成")
        else:                         # 开始事件
            print(f"节点 {data['name']} 开始")
```

需要状态面板时可再加上 `values` / `updates` / `custom`，逻辑相同，按 `part["type"]` 分支即可。

### 9.4 其它 mode（按需了解）


| mode          | 要点                                                        |
| ------------- | --------------------------------------------------------- |
| `checkpoints` | 每次写入 checkpoint 时 emit，格式对齐 `get_state()`（需 checkpointer） |
| `debug`       | `checkpoints` + `tasks` + 额外调试字段的超集                       |


---

# Part B · 本项目流式实现

读完 Part A 后，本节说明本仓库 **实际代码** 如何选择和消费流式事件。

## B.1 本项目订阅了哪些 mode

`[streaming.py](./streaming.py)` 中：

```python
async for chunk in graph.astream(
    payload,
    config=config,
    stream_mode=["messages", "tasks"],
    version="v2",
):
```


| 订阅         | 用途                                                       |
| ---------- | -------------------------------------------------------- |
| `messages` | 累积助手正文 `buf`、工具结果 `tool_results`、解析 `pending_tool_names` |
| `tasks`    | 维护 `agent_depth` / `tools_depth`，驱动状态行「推理中 / 工具运行中」      |


**未订阅** `values` / `updates` / `custom` / `checkpoints` / `debug`；其它外层 `type` 在代码里直接 `continue`。

与 Part A 中 mode 的关系（含 HITL）：


| `stream_mode`      | 外层 `type`              | 本项目  | 与 HITL 的关系                             |
| ------------------ | ---------------------- | ---- | -------------------------------------- |
| `messages`         | `"messages"`           | ✅ 消费 | 无直接关系                                  |
| `tasks`            | `"tasks"`              | ✅ 消费 | 节点结束 `data.interrupts` 可非空（仅 trace 日志） |
| `values`           | `"values"`             | ❌    | part 上可有 `interrupts`                  |
| `updates`          | `"updates"`            | ❌    | 可能有 `__interrupt__` 键                  |
| `checkpoints`      | `"checkpoints"`        | ❌    | 类似 `get_state()` 快照                    |
| `custom` / `debug` | `"custom"` / `"debug"` | ❌    | 调试用                                    |


本项目 HITL 路径：`**astream` 一段结束 → `get_pending_hitl(graph, config)` → 有则暂停 → `Command(resume={"decisions": [...]})` 再开下一段流**。不依赖订阅 `values` / `updates`。

---

## B.2 `messages` 与 `tasks` 各管什么

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


| 你想知道的事               | 看哪种 chunk / API                                                                       |
| -------------------- | ------------------------------------------------------------------------------------- |
| 助手说了哪个字              | `messages` + `AIMessageChunk.content`                                                 |
| 模型要调什么工具             | `messages` + `tool_calls` / `tool_call_chunks`                                        |
| 工具返回了什么              | `messages` + `ToolMessage`（内层 `type=="tool"`）                                         |
| 模型节点是否在跑             | `tasks` + `name in ("agent", "model")` 的开始/结束                                         |
| 工具节点是否在跑             | `tasks` + `name == "tools"` 的开始/结束                                                    |
| **是否有待审批 interrupt** | **流结束后** `graph.get_state(config).interrupts`（见 `[hitl.get_pending_hitl](./hitl.py)`） |


要点：

- `tasks` **不**携带 tool call 参数或工具正文的流式 token；具体内容仍在 `messages` 里。
- `tasks` 结束时的 `data.interrupts` 非空只表示「该节点发生了 interrupt」，本项目**仅打 `AGENT_TOOL_TRACE` 日志**，**不**用它驱动审批 UI；审批以 `get_state().interrupts` 为准。

---

## B.3 tool call 与 ToolMessage

都在外层 `messages` 里，阶段不同：


| 阶段                      | 内层消息             | 典型形态                                              | `streaming.py` 如何处理                                    |
| ----------------------- | ---------------- | ------------------------------------------------- | ------------------------------------------------------ |
| **tool call**（模型决定调工具）  | `AIMessageChunk` | `content` 常为空；有 `tool_calls` / `tool_call_chunks` | `tool_names_from_message_chunk` → `pending_tool_names` |
| **ToolMessage**（工具执行完毕） | `ToolMessage`    | `type=="tool"`，有 `name`、`content`                 | 进 `tool_results`，**不进** `buf`（CLI 正文不含工具输出）            |


- **tool call** = 模型说「我要调用 `rag_search`」→ 还在 **AI 消息**里。
- **ToolMessage** = 工具说「`rag_search` 返回了 xxx」→ 单独的 **tool 消息**。

---

## B.4 一次完整调工具的时间线

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

---

## B.5 带 HITL 的写文件时间线

用户：「在 workspace 写入 test.txt」（`write_file` 在 `agent_core.interrupt_on` 里为 `True`）

HITL 涉及两层视角，建议对照阅读：

| 视角 | 关注函数 | 时间线粒度 |
| ---- | -------- | ---------- |
| chunk 级 | `stream_assistant_text` | 单段 `astream` 内的 `messages` / `tasks` 事件 |
| 轮次级 | `run_assistant_turn` | 多段流 + 审批 + `Command(resume=...)` 的完整循环 |

多段流由 `run_assistant_turn` 用 `text_prefix` / `tool_results_prefix` 拼接；**同一 `thread_id`** 下 resume。

### B.5.1 chunk 级时间线（单段 `astream` 内）

第一段 `astream`（用户消息进入，尚未 resume）：

```text
M1  messages AIMessageChunk     → 可能先有正文 token（「好的，我来写…」）
M2  messages AIMessageChunk     → tool_calls 出现 write_file
T1  tasks   tools 开始          → status「正在调用工具 write_file」
T2  tasks   tools 结束          → data.interrupts 非空（仅 AGENT_TOOL_TRACE 日志）
                                → astream 循环结束
                                → get_pending_hitl() 返回 action_requests
```

`tasks` 结束时的 `data.interrupts` **不驱动审批 UI**；审批以流结束后 `get_state().interrupts` 为准（见 [B.9](#b9-hitl-与流式的职责划分)）。

### B.5.2 CLI 端到端时间线（入口 `iter_assistant_text_sync`）

以下从 **函数调用栈** 描述同一轮对话（CLI 路径：`decide=prompt_hitl_decisions_cli`）：

```text
T0  iter_assistant_text_sync(user_text="...", decide=prompt_hitl_decisions_cli)
      └─ run_assistant_turn 进入 while 循环

T1  stream_assistant_text 开始 astream(HumanMessage)
      ├─ messages: 模型流式输出（若有正文则 on_token 增量打印）
      ├─ messages: AIMessageChunk 出现 tool_call write_file
      ├─ tasks: tools 节点 start → status「正在调用工具 write_file」
      └─ tasks: tools 节点 end，interrupts 非空（仅 trace，工具尚未真正执行）

T2  astream 结束
      └─ get_pending_hitl() → HitlRequest(action_requests=[write_file...])

T3  prompt_hitl_decisions_cli(hitl)
      终端打印工具名、参数，等待输入 approve/reject

T4  decisions = [{"type": "approve"}]
      payload = Command(resume={"decisions": decisions})
      user_text = ""（避免 resume 时再注入 HumanMessage）
      回到 while 循环，再次 stream_assistant_text(resume payload)

T5  第二段 astream：图根据决策真正执行 write_file
      ├─ messages ToolMessage → write_file 执行结果，进入 tool_results
      ├─ tasks: agent 继续 → 模型读工具结果后生成最终回复
      └─ messages: 正文 token 继续累积到 buf（text_prefix 接上段）

T6  astream 结束 → get_pending_hitl() 为 None
      └─ 返回 AssistantTurnResult；iter_assistant_text_sync 返回 turn.text
```

若用户 **reject**，T4 改为 `[{"type": "reject", "message": "..."}]`；resume 后工具不执行，模型通常会改口回复。若连续多个敏感工具，T2–T4 可能重复（每段流结束后各审一次）。

**Streamlit 差异**（同一 `run_assistant_turn`，`decide=None`）：T3 不阻塞终端，而是在 T2 后直接返回 `pending_hitl`，由 `frontend/app.py` 存 `session_state.hitl_pending` 并显示审批按钮；用户点按钮后下次 rerun 用 `Command(resume=...)` + 保存的 `text_prefix` / `tool_results_prefix` 续跑（见 [B.8 消费方](#消费方) 表）。

---

## B.6 `messages` chunk 具体例子

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

## B.7 `tasks` chunk 具体例子

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

## B.8 与 `streaming.py` 核心逻辑的对应

### 状态变量


| 变量                            | 作用                                                       |
| ----------------------------- | -------------------------------------------------------- |
| `buf`                         | 助手正文累积（不含 ToolMessage）；可通过 `text_prefix` 跨 HITL 段拼接      |
| `tool_results`                | 工具返回，供 Streamlit expander；可通过 `tool_results_prefix` 跨段累积 |
| `pending_tool_names`          | 从 AI 消息解析出的待执行工具名                                        |
| `agent_depth` / `tools_depth` | 由 `tasks` 事件维护的节点运行深度                                    |


### `graph_input`（首段 vs resume 段）


| 场景          | 传入 `stream_assistant_text` 的输入                                    |
| ----------- | ----------------------------------------------------------------- |
| 用户新消息       | `user_text` → 内部构造 `{"messages": [HumanMessage(...)]}`            |
| HITL resume | `graph_input=Command(resume={"decisions": [...]})`，`user_text=""` |


由 `run_assistant_turn` 循环：流式一段 → `get_pending_hitl` → 有则 `decide`（CLI）或返回 `pending_hitl`（UI）→ `Command(resume=...)` 下一段。

### `redraw()` 调用时机

1. 流开始前初始化
2. `tasks`：agent/tools 节点开始或结束
3. `messages`：工具名变化、正文 token、工具结果；或 `content` 为空但状态已变
4. 循环结束：单独 `on_update(..., cursor=False)` 去掉流式光标

### 消费方


| 入口                                      | 用法                                                                                    |
| --------------------------------------- | ------------------------------------------------------------------------------------- |
| `stream_assistant_text`                 | async 核心；消费 `messages`/`tasks` chunk；`on_update` 推快照                                  |
| `run_assistant_turn`                    | 多段流 + HITL 循环；包装 `stream_assistant_text` + `get_pending_hitl` + `Command(resume=...)` |
| `iter_assistant_text_sync`              | CLI：`run_assistant_turn` + `decide=prompt_hitl_decisions_cli`                         |
| `stream_assistant_reply`                | CLI 一行封装（写 stdout）                                                                    |
| `[hitl.get_pending_hitl](./hitl.py)`    | 读 `graph.get_state(config).interrupts`，解析 `action_requests` / `review_configs`        |
| `[frontend/app.py](../frontend/app.py)` | `decide=None`；`pending_hitl` 时 `session_state` + 审批按钮；resume 用 `Command`              |


---

## B.9 HITL 与流式的职责划分

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

## 延伸阅读

- [LangGraph Streaming（官方）](https://docs.langchain.com/oss/python/langgraph/streaming)：`stream_mode`、`version="v2"`、`StreamPart` / `TasksStreamPart` / `MessagesStreamPart`
- [deepagents Human-in-the-loop](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop)：`interrupt_on` + `Command(resume={"decisions": [...]})`
- 本项目 agent 由 `create_deep_agent` 构建，图中节点名通常为 `agent`、`tools`（及可能的 `model`）
- 若关注 v3 实验性 API，见同目录 `[streaming_v3.md](./streaming_v3.md)`
