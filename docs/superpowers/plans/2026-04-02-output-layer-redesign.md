# 用户交互输出体系重整 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重新分配事件级别，实现三层输出模式（PROGRESS/DETAIL/TRACE），补全 ThinkingDelta 发射，消除重复输出。

**Architecture:** 只调整事件类型的默认级别和 LLM 流式解析逻辑，不改 EventLevel 枚举、EventBus 过滤机制和 CLIInterface 渲染。所有用户可见内容走 EventBus 统一路径。

**Tech Stack:** Python 3.13, asyncio, OpenAI SDK (DeepSeek 兼容接口)

---

### Task 1: 事件级别重新分配

**Files:**
- Modify: `src/events/types.py`
- Modify: `tests/events/test_bus.py`

- [ ] **Step 1: 更新 test_level_filter 测试以反映新的级别分配**

`tests/events/test_bus.py` 中的 `test_level_filter` 当前断言 `NodeStarted`（PROGRESS）通过、`ToolCalled`（DETAIL）被过滤、`TokenDelta`（PROGRESS，刚改的）被过滤。改为：`TokenDelta`（PROGRESS）通过、`ToolCalled`（DETAIL）被过滤、`NodeStarted`（TRACE）被过滤。

```python
@pytest.mark.asyncio
async def test_level_filter():
    """高于 bus level 的事件不被广播。"""
    bus = EventBus(level=EventLevel.PROGRESS)
    received = []

    async def consumer():
        async for event in bus.subscribe():
            received.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    # PROGRESS 事件应该通过
    await bus.emit(TokenDelta(timestamp=time.time(), source="l", delta="x"))
    # DETAIL 事件应该被过滤
    await bus.emit(ToolCalled(timestamp=time.time(), source="a", tool_name="t", args={}))
    # TRACE 事件应该被过滤
    await bus.emit(NodeStarted(timestamp=time.time(), source="n", node_type="agent"))
    await asyncio.sleep(0)

    bus.close()
    await task

    assert len(received) == 1
    assert received[0].type == "token_delta"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/events/test_bus.py::test_level_filter -v`
Expected: FAIL — `NodeStarted` 仍是 PROGRESS 会通过过滤，`TokenDelta` 也通过，收到 2 个事件而非 1 个。

- [ ] **Step 3: 修改 src/events/types.py 中的事件级别**

```python
# --- PROGRESS 级别 ---

@dataclass
class TokenDelta(Event):
    """流式 token — 默认可见。"""
    delta: str = ""
    level: EventLevel = field(default=EventLevel.PROGRESS, init=False)
    type: Literal["token_delta"] = field(default="token_delta", init=False)


@dataclass
class ThinkingDelta(Event):
    """思考过程 — 默认可见。"""
    content: str = ""
    level: EventLevel = field(default=EventLevel.PROGRESS, init=False)
    type: Literal["thinking_delta"] = field(default="thinking_delta", init=False)


@dataclass
class ErrorOccurred(Event):
    error: str = ""
    level: EventLevel = field(default=EventLevel.PROGRESS, init=False)
    type: Literal["error"] = field(default="error", init=False)


# --- DETAIL 级别 ---
# AgentStarted, AgentEnded, ToolCalled, ToolResult, Handoff — 保持不变


# --- TRACE 级别 ---

@dataclass
class GraphStarted(Event):
    level: EventLevel = field(default=EventLevel.TRACE, init=False)
    type: Literal["graph_started"] = field(default="graph_started", init=False)


@dataclass
class GraphEnded(Event):
    output: Any = None
    level: EventLevel = field(default=EventLevel.TRACE, init=False)
    type: Literal["graph_ended"] = field(default="graph_ended", init=False)


@dataclass
class NodeStarted(Event):
    node_type: str = ""
    level: EventLevel = field(default=EventLevel.TRACE, init=False)
    type: Literal["node_started"] = field(default="node_started", init=False)


@dataclass
class NodeEnded(Event):
    output_summary: str = ""
    level: EventLevel = field(default=EventLevel.TRACE, init=False)
    type: Literal["node_ended"] = field(default="node_ended", init=False)
```

注意整个文件的排列顺序也要调整：PROGRESS 块放 `TokenDelta`、`ThinkingDelta`、`ErrorOccurred`；DETAIL 块不变；TRACE 块放 `GraphStarted/Ended`、`NodeStarted/Ended`。

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/events/test_bus.py -v`
Expected: ALL PASS

- [ ] **Step 5: 更新 test_multiple_subscribers 中使用的事件**

`test_multiple_subscribers` 当前在 PROGRESS 级别 bus 上发射 `NodeStarted`。由于 `NodeStarted` 现在是 TRACE，需要改为发射一个 PROGRESS 事件（如 `ErrorOccurred`）：

```python
@pytest.mark.asyncio
async def test_multiple_subscribers():
    """多个订阅者各自独立收到事件。"""
    bus = EventBus(level=EventLevel.PROGRESS)
    received_a = []
    received_b = []

    async def consumer_a():
        async for event in bus.subscribe():
            received_a.append(event)

    async def consumer_b():
        async for event in bus.subscribe():
            received_b.append(event)

    task_a = asyncio.create_task(consumer_a())
    task_b = asyncio.create_task(consumer_b())
    await asyncio.sleep(0)

    await bus.emit(ErrorOccurred(timestamp=time.time(), source="n", error="test"))
    await asyncio.sleep(0)

    bus.close()
    await task_a
    await task_b

    assert len(received_a) == 1
    assert len(received_b) == 1
```

- [ ] **Step 6: 运行全部事件测试确认通过**

Run: `uv run pytest tests/events/ -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add src/events/types.py tests/events/test_bus.py
git commit -m "refactor: 重新分配事件级别 — TokenDelta/ThinkingDelta→PROGRESS, Graph/Node→TRACE"
```

---

### Task 2: 补充 ThinkingDelta 发射

**Files:**
- Modify: `src/llm/openai.py`
- Modify: `tests/llm/test_openai.py`

- [ ] **Step 1: 写失败测试 — ThinkingDelta 在流式中被发射**

在 `tests/llm/test_openai.py` 的 `TestEventBusEmission` 类中添加：

```python
from src.events.types import TokenDelta, ThinkingDelta

@pytest.mark.asyncio
async def test_thinking_delta_emitted_for_reasoning_content(self):
    """DeepSeek reasoning_content 应触发 ThinkingDelta 事件。"""
    bus = MagicMock(spec=EventBus)
    bus.emit = AsyncMock()
    with patch("src.llm.openai.AsyncOpenAI"):
        p = OpenAIProvider(
            api_key="k", base_url="u", model="m", event_bus=bus
        )

    chunks = [
        _make_chunk(content=None, reasoning_content="让我想想"),
        _make_chunk(content="答案是42"),
        _make_chunk(finish_reason="stop"),
    ]
    mock_create = AsyncMock(return_value=_async_iter(chunks))
    p._client.chat.completions.create = mock_create

    await p.chat(messages=[])

    emitted = [call.args[0] for call in bus.emit.call_args_list]
    thinking_events = [e for e in emitted if isinstance(e, ThinkingDelta)]
    token_events = [e for e in emitted if isinstance(e, TokenDelta)]

    assert len(thinking_events) == 1
    assert thinking_events[0].content == "让我想想"
    assert len(token_events) == 1
    assert token_events[0].delta == "答案是42"
```

同时更新 `_make_chunk` helper 支持 `reasoning_content` 参数：

```python
def _make_chunk(content=None, tool_calls=None, finish_reason=None, reasoning_content=None):
    """Build a minimal fake OpenAI stream chunk."""
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls or []
    delta.reasoning_content = reasoning_content

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason

    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk
```

- [ ] **Step 2: 添加测试 — silent 模式不发射 ThinkingDelta**

```python
@pytest.mark.asyncio
async def test_thinking_delta_not_emitted_when_silent(self):
    bus = MagicMock(spec=EventBus)
    bus.emit = AsyncMock()
    with patch("src.llm.openai.AsyncOpenAI"):
        p = OpenAIProvider(
            api_key="k", base_url="u", model="m", event_bus=bus
        )

    chunks = [
        _make_chunk(reasoning_content="thinking..."),
        _make_chunk(content="answer"),
        _make_chunk(finish_reason="stop"),
    ]
    mock_create = AsyncMock(return_value=_async_iter(chunks))
    p._client.chat.completions.create = mock_create

    await p.chat(messages=[], silent=True)

    bus.emit.assert_not_called()
```

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest tests/llm/test_openai.py::TestEventBusEmission::test_thinking_delta_emitted_for_reasoning_content -v`
Expected: FAIL — `_parse_stream` 不处理 `reasoning_content`

- [ ] **Step 4: 在 openai.py 中实现 ThinkingDelta 发射**

在 `src/llm/openai.py` 中：

1. 添加 import：
```python
from src.events.types import TokenDelta, ThinkingDelta
```

2. 在 `_parse_stream` 的 `async for chunk in stream:` 循环中，`if delta.content:` 块之前添加：
```python
            # 思考内容（DeepSeek reasoning_content）
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning and not silent and self._bus:
                await self._bus.emit(ThinkingDelta(
                    timestamp=time.time(),
                    source=self.model,
                    content=reasoning,
                ))
```

- [ ] **Step 5: 运行测试确认通过**

Run: `uv run pytest tests/llm/test_openai.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/llm/openai.py tests/llm/test_openai.py
git commit -m "feat: emit ThinkingDelta for DeepSeek reasoning_content in streaming"
```

---

### Task 3: 更新 config.yaml 注释

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: 更新 events.level 注释**

```yaml
events:
    level: progress   # progress(流式回答+思考) | detail(+工具/Agent) | trace(+图/节点调试)
```

- [ ] **Step 2: Commit**

```bash
git add config.yaml
git commit -m "docs: 更新 events.level 注释，说明三层输出模式含义"
```

---

### Task 4: 更新设计文档中的级别表（一致性）

**Files:**
- Modify: `docs/superpowers/specs/2026-04-02-event-bus-design.md`

- [ ] **Step 1: 检查 event-bus-design.md 中的级别表**

读取 `docs/superpowers/specs/2026-04-02-event-bus-design.md`，找到事件级别表格并更新为新的分配，使文档与代码一致。将 `TokenDelta`/`ThinkingDelta` 标为 PROGRESS，`Graph*/Node*` 标为 TRACE。

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-04-02-event-bus-design.md
git commit -m "docs: 同步 event-bus 设计文档中的事件级别表"
```
