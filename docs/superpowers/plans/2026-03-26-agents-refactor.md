# Agents 模块全面重写 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用图引擎 + Agent Runner 混合架构替代现有 FSM 驱动的多智能体系统，支持并行执行、条件路由、类型安全状态、生命周期钩子和可观测性。

**Architecture:** 两层架构 — 外层 GraphEngine 负责节点间的拓扑编排（顺序/并行/条件路由），内层 AgentRunner 负责单个 agent 的工具调用循环。RunContext[StateT, DepsT] 提供类型安全的共享状态和依赖注入。

**Tech Stack:** Python 3.13, Pydantic v2 (BaseModel, Generic), asyncio (gather for parallel), OpenAI API (via existing `call_model`)

**Design Spec:** `docs/superpowers/specs/2026-03-26-agents-refactor-design.md`

---

## File Structure

```
src/agents/                    # 全部新建（替换现有文件）
├── __init__.py                # 公共 API 导出
├── agent.py                   # Agent dataclass + AgentResult + HandoffRequest
├── registry.py                # AgentRegistry 注册表（重写）
├── context.py                 # RunContext[StateT, DepsT] + TraceEvent + DictState + EmptyDeps
├── guardrails.py              # Guardrail + GuardrailResult
├── hooks.py                   # AgentHooks + GraphHooks
├── runner.py                  # AgentRunner — 单 agent 执行引擎
├── graph/
│   ├── __init__.py            # graph 子包导出
│   ├── types.py               # GraphNode protocol + AgentNode + FunctionNode + Edge + NodeResult + CompiledGraph + ParallelGroup
│   ├── builder.py             # GraphBuilder 声明式构建 + compile 验证
│   └── engine.py              # GraphEngine 异步图执行器

tests/agents/                  # 全部新建
├── __init__.py
├── test_agent.py              # Agent / AgentResult / HandoffRequest 测试
├── test_registry.py           # AgentRegistry 测试
├── test_context.py            # RunContext 泛型 + TraceEvent 测试
├── test_guardrails.py         # Guardrail 测试
├── test_hooks.py              # Hooks 触发测试
├── test_runner.py             # AgentRunner 测试（mock call_model）
├── graph/
│   ├── __init__.py
│   ├── test_types.py          # 节点类型测试
│   ├── test_builder.py        # GraphBuilder 编译 + 验证测试
│   └── test_engine.py         # GraphEngine 执行测试
```

---

### Task 1: Agent 数据模型

**Files:**
- Create: `src/agents/agent.py`
- Create: `tests/agents/__init__.py`
- Create: `tests/agents/test_agent.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/__init__.py` (empty) and `tests/agents/test_agent.py`:

```python
"""Agent / AgentResult / HandoffRequest 数据模型测试。"""
import pytest
from pydantic import BaseModel


class DummyOutput(BaseModel):
    score: float


def test_agent_basic_creation():
    from src.agents.agent import Agent

    agent = Agent(
        name="test",
        description="A test agent",
        instructions="You are a test agent.",
    )
    assert agent.name == "test"
    assert agent.description == "A test agent"
    assert agent.instructions == "You are a test agent."
    assert agent.tools == []
    assert agent.handoffs == []
    assert agent.output_model is None
    assert agent.input_guardrails == []
    assert agent.output_guardrails == []
    assert agent.hooks is None


def test_agent_with_all_fields():
    from src.agents.agent import Agent

    agent = Agent(
        name="weather",
        description="Weather agent",
        instructions="Check weather.",
        tools=["get_weather"],
        handoffs=["calendar_agent"],
        output_model=DummyOutput,
    )
    assert agent.tools == ["get_weather"]
    assert agent.handoffs == ["calendar_agent"]
    assert agent.output_model is DummyOutput


def test_agent_dynamic_instructions():
    from src.agents.agent import Agent

    def make_instructions(ctx):
        return f"Handle: {ctx.input}"

    agent = Agent(
        name="dynamic",
        description="Dynamic agent",
        instructions=make_instructions,
    )
    assert callable(agent.instructions)


def test_agent_result():
    from src.agents.agent import AgentResult

    result = AgentResult(text="hello")
    assert result.text == "hello"
    assert result.data == {}
    assert result.handoff is None


def test_agent_result_with_handoff():
    from src.agents.agent import AgentResult, HandoffRequest

    handoff = HandoffRequest(target="calendar", task="book meeting")
    result = AgentResult(text="transferring", handoff=handoff)
    assert result.handoff.target == "calendar"
    assert result.handoff.task == "book meeting"


def test_handoff_request():
    from src.agents.agent import HandoffRequest

    req = HandoffRequest(target="email_agent", task="send report")
    assert req.target == "email_agent"
    assert req.task == "send report"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/agents/test_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agents.agent'`

- [ ] **Step 3: Write the implementation**

Create `src/agents/agent.py`:

```python
"""Agent 数据模型 — 声明式定义一个 agent 是什么、能做什么。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel


@dataclass
class HandoffRequest:
    """Agent 请求将任务交接到另一个 agent。"""

    target: str
    task: str


@dataclass
class AgentResult:
    """单个 agent 的执行结果。"""

    text: str
    data: dict = field(default_factory=dict)
    handoff: Optional[HandoffRequest] = None


@dataclass
class Agent:
    """Agent 定义。

    Attributes:
        name: 唯一标识。
        description: 一句话描述，用于 handoff 工具生成。
        instructions: 系统提示，支持字符串或 Callable[[RunContext], str] 动态生成。
        tools: 允许使用的工具名列表。
        handoffs: 可 handoff 到的 agent 名列表。
        output_model: 结构化输出的 Pydantic 模型。
        input_guardrails: 输入护栏列表。
        output_guardrails: 输出护栏列表。
        hooks: 生命周期钩子。
    """

    name: str
    description: str
    instructions: str | Callable[..., str]
    tools: list[str] = field(default_factory=list)
    handoffs: list[str] = field(default_factory=list)
    output_model: Optional[Type[BaseModel]] = None
    input_guardrails: list = field(default_factory=list)
    output_guardrails: list = field(default_factory=list)
    hooks: Optional[Any] = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/agents/test_agent.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/agents/agent.py tests/agents/__init__.py tests/agents/test_agent.py
git commit -m "feat(agents): add Agent, AgentResult, HandoffRequest data models"
```

---

### Task 2: RunContext 泛型上下文 + TraceEvent

**Files:**
- Create: `src/agents/context.py`
- Create: `tests/agents/test_context.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_context.py`:

```python
"""RunContext 泛型 + TraceEvent 测试。"""
import time
import pytest
from pydantic import BaseModel, ConfigDict


class MyState(BaseModel):
    counter: int = 0
    result: str = ""


class MyDeps(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    api_key: str = "test-key"


def test_run_context_typed_state():
    from src.agents.context import RunContext

    ctx = RunContext(input="hello", state=MyState(), deps=MyDeps())
    assert ctx.state.counter == 0
    ctx.state.counter = 5
    assert ctx.state.counter == 5


def test_run_context_typed_deps():
    from src.agents.context import RunContext

    ctx = RunContext(input="hi", state=MyState(), deps=MyDeps(api_key="secret"))
    assert ctx.deps.api_key == "secret"


def test_run_context_defaults():
    from src.agents.context import RunContext, DictState, EmptyDeps

    ctx = RunContext(input="test", state=DictState(), deps=EmptyDeps())
    assert ctx.current_agent == ""
    assert ctx.depth == 0
    assert ctx.trace == []


def test_run_context_dict_state_allows_extra():
    from src.agents.context import RunContext, DictState, EmptyDeps

    ctx = RunContext(input="test", state=DictState(), deps=EmptyDeps())
    ctx.state.weather = {"temp": 25}
    assert ctx.state.weather == {"temp": 25}


def test_trace_event():
    from src.agents.context import TraceEvent

    event = TraceEvent(node="weather", event="start", timestamp=time.time())
    assert event.node == "weather"
    assert event.event == "start"
    assert event.data == {}


def test_trace_event_with_data():
    from src.agents.context import TraceEvent

    event = TraceEvent(
        node="weather",
        event="tool_call",
        timestamp=1234567890.0,
        data={"tool": "get_weather", "args": {"city": "Beijing"}},
    )
    assert event.data["tool"] == "get_weather"


def test_run_context_add_trace():
    from src.agents.context import RunContext, DictState, EmptyDeps, TraceEvent

    ctx = RunContext(input="test", state=DictState(), deps=EmptyDeps())
    ctx.trace.append(TraceEvent(node="a", event="start", timestamp=1.0))
    ctx.trace.append(TraceEvent(node="a", event="end", timestamp=2.0))
    assert len(ctx.trace) == 2
    assert ctx.trace[0].event == "start"
    assert ctx.trace[1].event == "end"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/agents/test_context.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/agents/context.py`:

```python
"""RunContext — 贯穿图执行的泛型共享上下文。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict


StateT = TypeVar("StateT", bound=BaseModel)
DepsT = TypeVar("DepsT", bound=BaseModel)


class DictState(BaseModel):
    """默认的宽松状态，允许任意 key-value。"""

    model_config = ConfigDict(extra="allow")


class EmptyDeps(BaseModel):
    """无外部依赖时使用。"""

    pass


@dataclass
class TraceEvent:
    """一次执行事件的记录。"""

    node: str
    event: str  # "start" | "end" | "tool_call" | "handoff" | "error"
    timestamp: float
    data: dict = field(default_factory=dict)


@dataclass
class RunContext(Generic[StateT, DepsT]):
    """贯穿整个图执行的共享上下文。

    两个泛型参数：
    - StateT: 共享状态结构，节点间传递数据的唯一通道。
    - DepsT: 外部依赖，由使用者定义，框架只负责传递。
    """

    input: str
    state: StateT
    deps: DepsT
    trace: list[TraceEvent] = field(default_factory=list)
    current_agent: str = ""
    depth: int = 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/agents/test_context.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/agents/context.py tests/agents/test_context.py
git commit -m "feat(agents): add RunContext[StateT, DepsT] with TraceEvent"
```

---

### Task 3: Guardrails

**Files:**
- Create: `src/agents/guardrails.py`
- Create: `tests/agents/test_guardrails.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_guardrails.py`:

```python
"""Guardrail + GuardrailResult 测试。"""
import pytest
from pydantic import BaseModel


@pytest.fixture
def make_context():
    from src.agents.context import RunContext, DictState, EmptyDeps

    def _make(input_text="test"):
        return RunContext(input=input_text, state=DictState(), deps=EmptyDeps())

    return _make


@pytest.mark.asyncio
async def test_guardrail_passes(make_context):
    from src.agents.guardrails import Guardrail, GuardrailResult

    async def always_pass(ctx, text):
        return GuardrailResult(passed=True)

    guard = Guardrail(name="pass_guard", check=always_pass)
    result = await guard.check(make_context(), "hello")
    assert result.passed is True


@pytest.mark.asyncio
async def test_guardrail_blocks(make_context):
    from src.agents.guardrails import Guardrail, GuardrailResult

    async def block_bad(ctx, text):
        if "bad" in text:
            return GuardrailResult(passed=False, message="Contains bad content", action="block")
        return GuardrailResult(passed=True)

    guard = Guardrail(name="bad_guard", check=block_bad)
    result = await guard.check(make_context(), "this is bad")
    assert result.passed is False
    assert result.action == "block"
    assert "bad" in result.message


@pytest.mark.asyncio
async def test_guardrail_warn(make_context):
    from src.agents.guardrails import Guardrail, GuardrailResult

    async def warn_check(ctx, text):
        return GuardrailResult(passed=False, message="warning", action="warn")

    guard = Guardrail(name="warn_guard", check=warn_check)
    result = await guard.check(make_context(), "something")
    assert result.passed is False
    assert result.action == "warn"


@pytest.mark.asyncio
async def test_run_guardrails_all_pass(make_context):
    from src.agents.guardrails import Guardrail, GuardrailResult, run_guardrails

    async def pass_check(ctx, text):
        return GuardrailResult(passed=True)

    guards = [
        Guardrail(name="g1", check=pass_check),
        Guardrail(name="g2", check=pass_check),
    ]
    result = await run_guardrails(guards, make_context(), "hello")
    assert result is None  # None means all passed


@pytest.mark.asyncio
async def test_run_guardrails_first_block_stops(make_context):
    from src.agents.guardrails import Guardrail, GuardrailResult, run_guardrails

    async def block_check(ctx, text):
        return GuardrailResult(passed=False, message="blocked", action="block")

    async def pass_check(ctx, text):
        return GuardrailResult(passed=True)

    guards = [
        Guardrail(name="blocker", check=block_check),
        Guardrail(name="passer", check=pass_check),
    ]
    result = await run_guardrails(guards, make_context(), "hello")
    assert result is not None
    assert result.passed is False
    assert result.action == "block"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/agents/test_guardrails.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/agents/guardrails.py`:

```python
"""Guardrail — Agent 输入/输出护栏。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional


@dataclass
class GuardrailResult:
    """护栏检查结果。"""

    passed: bool
    message: str = ""
    action: str = "block"  # "block" | "warn" | "rewrite"


@dataclass
class Guardrail:
    """输入/输出护栏。"""

    name: str
    check: Callable[..., Awaitable[GuardrailResult]]


async def run_guardrails(
    guardrails: list[Guardrail],
    context: Any,
    text: str,
) -> Optional[GuardrailResult]:
    """依次执行护栏列表，遇到 block 立即返回，全部通过返回 None。"""
    for guard in guardrails:
        result = await guard.check(context, text)
        if not result.passed and result.action == "block":
            return result
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/agents/test_guardrails.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/agents/guardrails.py tests/agents/test_guardrails.py
git commit -m "feat(agents): add Guardrail with block/warn/rewrite actions"
```

---

### Task 4: Lifecycle Hooks

**Files:**
- Create: `src/agents/hooks.py`
- Create: `tests/agents/test_hooks.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_hooks.py`:

```python
"""AgentHooks + GraphHooks 测试。"""
import pytest


@pytest.mark.asyncio
async def test_agent_hooks_on_start():
    from src.agents.hooks import AgentHooks
    from src.agents.agent import Agent
    from src.agents.context import RunContext, DictState, EmptyDeps

    calls = []

    async def on_start(agent, ctx):
        calls.append(("start", agent.name))

    hooks = AgentHooks(on_start=on_start)
    agent = Agent(name="test", description="test", instructions="test")
    ctx = RunContext(input="hi", state=DictState(), deps=EmptyDeps())

    await hooks.on_start(agent, ctx)
    assert calls == [("start", "test")]


@pytest.mark.asyncio
async def test_agent_hooks_on_end():
    from src.agents.hooks import AgentHooks
    from src.agents.agent import Agent, AgentResult
    from src.agents.context import RunContext, DictState, EmptyDeps

    calls = []

    async def on_end(agent, ctx, result):
        calls.append(("end", result.text))

    hooks = AgentHooks(on_end=on_end)
    agent = Agent(name="test", description="test", instructions="test")
    ctx = RunContext(input="hi", state=DictState(), deps=EmptyDeps())
    result = AgentResult(text="done")

    await hooks.on_end(agent, ctx, result)
    assert calls == [("end", "done")]


@pytest.mark.asyncio
async def test_agent_hooks_none_is_noop():
    from src.agents.hooks import AgentHooks
    from src.agents.agent import Agent
    from src.agents.context import RunContext, DictState, EmptyDeps

    hooks = AgentHooks()  # all None
    agent = Agent(name="test", description="test", instructions="test")
    ctx = RunContext(input="hi", state=DictState(), deps=EmptyDeps())

    # Should not raise
    await hooks.on_start(agent, ctx)


@pytest.mark.asyncio
async def test_graph_hooks_on_node_start():
    from src.agents.hooks import GraphHooks
    from src.agents.context import RunContext, DictState, EmptyDeps

    calls = []

    async def on_node_start(node_name, ctx):
        calls.append(node_name)

    hooks = GraphHooks(on_node_start=on_node_start)
    ctx = RunContext(input="hi", state=DictState(), deps=EmptyDeps())

    await hooks.on_node_start("weather", ctx)
    assert calls == ["weather"]


@pytest.mark.asyncio
async def test_graph_hooks_none_is_noop():
    from src.agents.hooks import GraphHooks
    from src.agents.context import RunContext, DictState, EmptyDeps

    hooks = GraphHooks()
    ctx = RunContext(input="hi", state=DictState(), deps=EmptyDeps())

    # Should not raise
    await hooks.on_graph_start(ctx)
    await hooks.on_node_start("x", ctx)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/agents/test_hooks.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/agents/hooks.py`:

```python
"""Lifecycle Hooks — Agent 级和图级的生命周期钩子。"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional


class AgentHooks:
    """Agent 级钩子。

    所有钩子均为可选，未设置时调用为 no-op。
    """

    def __init__(
        self,
        on_start: Optional[Callable[..., Awaitable[None]]] = None,
        on_end: Optional[Callable[..., Awaitable[None]]] = None,
        on_tool_call: Optional[Callable[..., Awaitable[None]]] = None,
        on_handoff: Optional[Callable[..., Awaitable[None]]] = None,
        on_error: Optional[Callable[..., Awaitable[None]]] = None,
    ):
        self._on_start = on_start
        self._on_end = on_end
        self._on_tool_call = on_tool_call
        self._on_handoff = on_handoff
        self._on_error = on_error

    async def on_start(self, agent: Any, context: Any) -> None:
        if self._on_start:
            await self._on_start(agent, context)

    async def on_end(self, agent: Any, context: Any, result: Any) -> None:
        if self._on_end:
            await self._on_end(agent, context, result)

    async def on_tool_call(self, agent: Any, context: Any, tool_name: str, args: dict) -> None:
        if self._on_tool_call:
            await self._on_tool_call(agent, context, tool_name, args)

    async def on_handoff(self, agent: Any, context: Any, handoff: Any) -> None:
        if self._on_handoff:
            await self._on_handoff(agent, context, handoff)

    async def on_error(self, agent: Any, context: Any, error: Exception) -> None:
        if self._on_error:
            await self._on_error(agent, context, error)


class GraphHooks:
    """图级钩子。

    所有钩子均为可选，未设置时调用为 no-op。
    """

    def __init__(
        self,
        on_graph_start: Optional[Callable[..., Awaitable[None]]] = None,
        on_graph_end: Optional[Callable[..., Awaitable[None]]] = None,
        on_node_start: Optional[Callable[..., Awaitable[None]]] = None,
        on_node_end: Optional[Callable[..., Awaitable[None]]] = None,
    ):
        self._on_graph_start = on_graph_start
        self._on_graph_end = on_graph_end
        self._on_node_start = on_node_start
        self._on_node_end = on_node_end

    async def on_graph_start(self, context: Any) -> None:
        if self._on_graph_start:
            await self._on_graph_start(context)

    async def on_graph_end(self, context: Any, result: Any) -> None:
        if self._on_graph_end:
            await self._on_graph_end(context, result)

    async def on_node_start(self, node_name: str, context: Any) -> None:
        if self._on_node_start:
            await self._on_node_start(node_name, context)

    async def on_node_end(self, node_name: str, context: Any, result: Any) -> None:
        if self._on_node_end:
            await self._on_node_end(node_name, context, result)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/agents/test_hooks.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/agents/hooks.py tests/agents/test_hooks.py
git commit -m "feat(agents): add AgentHooks and GraphHooks lifecycle hooks"
```

---

### Task 5: AgentRegistry（重写）

**Files:**
- Create: `src/agents/registry.py` (覆盖旧文件)
- Create: `tests/agents/test_registry.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_registry.py`:

```python
"""AgentRegistry 测试。"""
import pytest
from src.agents.agent import Agent


@pytest.fixture
def registry():
    from src.agents.registry import AgentRegistry

    return AgentRegistry()


@pytest.fixture
def weather_agent():
    return Agent(
        name="weather_agent",
        description="查询天气",
        instructions="你是天气查询专家。",
        tools=["get_weather"],
    )


@pytest.fixture
def calendar_agent():
    return Agent(
        name="calendar_agent",
        description="管理日历",
        instructions="你是日历管理专家。",
        tools=["create_event"],
    )


def test_register_and_get(registry, weather_agent):
    registry.register(weather_agent)
    assert registry.get("weather_agent") is weather_agent


def test_get_nonexistent_returns_none(registry):
    assert registry.get("nonexistent") is None


def test_all_agents(registry, weather_agent, calendar_agent):
    registry.register(weather_agent)
    registry.register(calendar_agent)
    agents = registry.all_agents()
    assert len(agents) == 2
    names = {a.name for a in agents}
    assert names == {"weather_agent", "calendar_agent"}


def test_register_overwrite(registry, weather_agent):
    registry.register(weather_agent)
    updated = Agent(
        name="weather_agent",
        description="Updated",
        instructions="Updated instructions.",
    )
    registry.register(updated)
    assert registry.get("weather_agent").description == "Updated"


def test_has(registry, weather_agent):
    assert not registry.has("weather_agent")
    registry.register(weather_agent)
    assert registry.has("weather_agent")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/agents/test_registry.py -v`
Expected: FAIL (old registry doesn't have `has` method, imports mismatch)

- [ ] **Step 3: Write the implementation**

Overwrite `src/agents/registry.py`:

```python
"""AgentRegistry — Agent 注册表。"""

from __future__ import annotations

from typing import Optional

from src.agents.agent import Agent


class AgentRegistry:
    """管理所有已注册的 Agent。"""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        """注册一个 Agent（同名覆盖）。"""
        self._agents[agent.name] = agent

    def get(self, name: str) -> Optional[Agent]:
        """根据名称获取 Agent，不存在返回 None。"""
        return self._agents.get(name)

    def has(self, name: str) -> bool:
        """检查 Agent 是否已注册。"""
        return name in self._agents

    def all_agents(self) -> list[Agent]:
        """返回所有已注册的 Agent 列表。"""
        return list(self._agents.values())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/agents/test_registry.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/agents/registry.py tests/agents/test_registry.py
git commit -m "feat(agents): rewrite AgentRegistry for new Agent model"
```

---

### Task 6: Graph Types — 节点、边、编译结果

**Files:**
- Create: `src/agents/graph/__init__.py`
- Create: `src/agents/graph/types.py`
- Create: `tests/agents/graph/__init__.py`
- Create: `tests/agents/graph/test_types.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/graph/__init__.py` (empty) and `tests/agents/graph/test_types.py`:

```python
"""图类型测试 — GraphNode, AgentNode, FunctionNode, Edge, NodeResult。"""
import pytest
from src.agents.agent import Agent, AgentResult
from src.agents.context import RunContext, DictState, EmptyDeps


@pytest.fixture
def context():
    return RunContext(input="test", state=DictState(), deps=EmptyDeps())


@pytest.mark.asyncio
async def test_function_node_execute(context):
    from src.agents.graph.types import FunctionNode, NodeResult

    async def greet(ctx):
        return NodeResult(output=f"Hello {ctx.input}")

    node = FunctionNode(name="greeter", fn=greet)
    assert node.name == "greeter"
    result = await node.execute(context)
    assert result.output == "Hello test"
    assert result.next is None
    assert result.handoff is None


@pytest.mark.asyncio
async def test_function_node_with_next(context):
    from src.agents.graph.types import FunctionNode, NodeResult

    async def router_fn(ctx):
        return NodeResult(output="routed", next="target_node")

    node = FunctionNode(name="router", fn=router_fn)
    result = await node.execute(context)
    assert result.next == "target_node"


def test_node_result_defaults():
    from src.agents.graph.types import NodeResult

    result = NodeResult(output="data")
    assert result.output == "data"
    assert result.next is None
    assert result.handoff is None


def test_edge_unconditional():
    from src.agents.graph.types import Edge

    edge = Edge(source="a", target="b")
    assert edge.source == "a"
    assert edge.target == "b"
    assert edge.condition is None


def test_edge_conditional():
    from src.agents.graph.types import Edge

    edge = Edge(source="a", target="b", condition=lambda ctx: ctx.state.counter > 0)
    assert edge.condition is not None


def test_parallel_group():
    from src.agents.graph.types import ParallelGroup

    pg = ParallelGroup(nodes=["a", "b"], then="c")
    assert pg.nodes == ["a", "b"]
    assert pg.then == "c"


def test_compiled_graph():
    from src.agents.graph.types import CompiledGraph, FunctionNode, NodeResult, Edge

    async def noop(ctx):
        return NodeResult(output=None)

    node = FunctionNode(name="a", fn=noop)
    graph = CompiledGraph(
        nodes={"a": node},
        edges=[],
        entry="a",
        parallel_groups=[],
    )
    assert graph.entry == "a"
    assert "a" in graph.nodes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/agents/graph/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/agents/graph/__init__.py`:

```python
"""Graph 子包 — 图引擎相关类型和实现。"""
```

Create `src/agents/graph/types.py`:

```python
"""图类型定义 — 节点、边、执行结果、编译后的图。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Protocol, runtime_checkable

from src.agents.agent import HandoffRequest


@dataclass
class NodeResult:
    """节点执行结果。"""

    output: Any
    next: Optional[str | list[str]] = None
    handoff: Optional[HandoffRequest] = None


@runtime_checkable
class GraphNode(Protocol):
    """图节点协议。"""

    name: str

    async def execute(self, context: Any) -> NodeResult: ...


class AgentNode:
    """包装一个 Agent，内部用 AgentRunner 驱动。"""

    def __init__(self, agent: Any, runner: Any = None):
        self.name: str = agent.name
        self.agent = agent
        self.runner = runner

    async def execute(self, context: Any) -> NodeResult:
        if self.runner is None:
            raise RuntimeError(f"AgentNode '{self.name}' has no runner assigned")
        result = await self.runner.run(self.agent, context)
        return NodeResult(
            output={"text": result.text, "data": result.data},
            handoff=result.handoff,
        )


class FunctionNode:
    """包装一个普通 async 函数。"""

    def __init__(self, name: str, fn: Callable[..., Awaitable[NodeResult]]):
        self.name = name
        self.fn = fn

    async def execute(self, context: Any) -> NodeResult:
        return await self.fn(context)


@dataclass
class Edge:
    """节点间的连接。"""

    source: str
    target: str
    condition: Optional[Callable[..., bool]] = None


@dataclass
class ParallelGroup:
    """一组需要并行执行的节点。"""

    nodes: list[str]
    then: str


@dataclass
class CompiledGraph:
    """编译后的图，不可变，可复用。"""

    nodes: dict[str, GraphNode]
    edges: list[Edge]
    entry: str
    parallel_groups: list[ParallelGroup] = field(default_factory=list)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/agents/graph/test_types.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/agents/graph/__init__.py src/agents/graph/types.py tests/agents/graph/__init__.py tests/agents/graph/test_types.py
git commit -m "feat(agents): add graph types — nodes, edges, CompiledGraph"
```

---

### Task 7: GraphBuilder — 声明式图构建 + 编译验证

**Files:**
- Create: `src/agents/graph/builder.py`
- Create: `tests/agents/graph/test_builder.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/graph/test_builder.py`:

```python
"""GraphBuilder 编译 + 验证测试。"""
import pytest
from src.agents.agent import Agent
from src.agents.graph.types import NodeResult


@pytest.fixture
def simple_agent():
    return Agent(name="agent_a", description="Agent A", instructions="Do A.")


@pytest.fixture
def another_agent():
    return Agent(name="agent_b", description="Agent B", instructions="Do B.")


async def dummy_fn(ctx):
    return NodeResult(output="done")


def test_builder_add_agent_and_compile(simple_agent):
    from src.agents.graph.builder import GraphBuilder

    graph = GraphBuilder()
    graph.add_agent("agent_a", simple_agent)
    graph.set_entry("agent_a")
    compiled = graph.compile()
    assert compiled.entry == "agent_a"
    assert "agent_a" in compiled.nodes


def test_builder_add_function_and_compile():
    from src.agents.graph.builder import GraphBuilder

    graph = GraphBuilder()
    graph.add_function("fn_a", dummy_fn)
    graph.set_entry("fn_a")
    compiled = graph.compile()
    assert "fn_a" in compiled.nodes


def test_builder_add_edge(simple_agent, another_agent):
    from src.agents.graph.builder import GraphBuilder

    graph = GraphBuilder()
    graph.add_agent("agent_a", simple_agent)
    graph.add_agent("agent_b", another_agent)
    graph.set_entry("agent_a")
    graph.add_edge("agent_a", "agent_b")
    compiled = graph.compile()
    assert len(compiled.edges) == 1
    assert compiled.edges[0].source == "agent_a"
    assert compiled.edges[0].target == "agent_b"


def test_builder_add_conditional_edge(simple_agent, another_agent):
    from src.agents.graph.builder import GraphBuilder

    graph = GraphBuilder()
    graph.add_agent("agent_a", simple_agent)
    graph.add_agent("agent_b", another_agent)
    graph.set_entry("agent_a")
    graph.add_edge("agent_a", "agent_b", condition=lambda ctx: True)
    compiled = graph.compile()
    assert compiled.edges[0].condition is not None


def test_builder_add_parallel(simple_agent, another_agent):
    from src.agents.graph.builder import GraphBuilder

    graph = GraphBuilder()
    graph.add_agent("agent_a", simple_agent)
    graph.add_agent("agent_b", another_agent)
    graph.add_function("merge", dummy_fn)
    graph.set_entry("agent_a")
    graph.add_parallel(["agent_a", "agent_b"], then="merge")
    compiled = graph.compile()
    assert len(compiled.parallel_groups) == 1
    assert compiled.parallel_groups[0].nodes == ["agent_a", "agent_b"]
    assert compiled.parallel_groups[0].then == "merge"


def test_builder_chain_api(simple_agent):
    from src.agents.graph.builder import GraphBuilder

    graph = GraphBuilder()
    result = graph.add_agent("agent_a", simple_agent)
    assert result is graph  # returns self for chaining


def test_compile_fails_without_entry(simple_agent):
    from src.agents.graph.builder import GraphBuilder

    graph = GraphBuilder()
    graph.add_agent("agent_a", simple_agent)
    with pytest.raises(ValueError, match="entry"):
        graph.compile()


def test_compile_fails_with_unknown_entry():
    from src.agents.graph.builder import GraphBuilder

    graph = GraphBuilder()
    graph.set_entry("nonexistent")
    with pytest.raises(ValueError, match="nonexistent"):
        graph.compile()


def test_compile_fails_with_unknown_edge_target(simple_agent):
    from src.agents.graph.builder import GraphBuilder

    graph = GraphBuilder()
    graph.add_agent("agent_a", simple_agent)
    graph.set_entry("agent_a")
    graph.add_edge("agent_a", "nonexistent")
    with pytest.raises(ValueError, match="nonexistent"):
        graph.compile()


def test_compile_fails_with_unknown_parallel_node(simple_agent):
    from src.agents.graph.builder import GraphBuilder

    graph = GraphBuilder()
    graph.add_agent("agent_a", simple_agent)
    graph.add_function("merge", dummy_fn)
    graph.set_entry("agent_a")
    graph.add_parallel(["agent_a", "nonexistent"], then="merge")
    with pytest.raises(ValueError, match="nonexistent"):
        graph.compile()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/agents/graph/test_builder.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/agents/graph/builder.py`:

```python
"""GraphBuilder — 声明式图构建器。"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from src.agents.agent import Agent
from src.agents.graph.types import (
    AgentNode,
    CompiledGraph,
    Edge,
    FunctionNode,
    GraphNode,
    NodeResult,
    ParallelGroup,
)


class GraphBuilder:
    """声明式图构建器，支持链式调用。"""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[Edge] = []
        self._parallel_groups: list[ParallelGroup] = []
        self._entry: Optional[str] = None

    def add_agent(self, name: str, agent: Agent) -> GraphBuilder:
        """添加一个 agent 节点。"""
        self._nodes[name] = AgentNode(agent=agent)
        return self

    def add_function(self, name: str, fn: Callable[..., Awaitable[NodeResult]]) -> GraphBuilder:
        """添加一个函数节点。"""
        self._nodes[name] = FunctionNode(name=name, fn=fn)
        return self

    def set_entry(self, name: str) -> GraphBuilder:
        """设置入口节点。"""
        self._entry = name
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        condition: Optional[Callable[..., bool]] = None,
    ) -> GraphBuilder:
        """添加一条边（可选条件）。"""
        self._edges.append(Edge(source=source, target=target, condition=condition))
        return self

    def add_parallel(self, nodes: list[str], then: str) -> GraphBuilder:
        """声明一组并行执行的节点，完成后汇聚到 then 节点。"""
        self._parallel_groups.append(ParallelGroup(nodes=list(nodes), then=then))
        return self

    def compile(self) -> CompiledGraph:
        """编译图：验证合法性后返回 CompiledGraph。"""
        # 验证入口
        if self._entry is None:
            raise ValueError("Graph has no entry node. Call set_entry() before compile().")
        if self._entry not in self._nodes:
            raise ValueError(f"Entry node '{self._entry}' not found in registered nodes.")

        # 验证边
        for edge in self._edges:
            if edge.source not in self._nodes:
                raise ValueError(f"Edge source '{edge.source}' not found in registered nodes.")
            if edge.target not in self._nodes:
                raise ValueError(f"Edge target '{edge.target}' not found in registered nodes.")

        # 验证并行组
        for pg in self._parallel_groups:
            for node_name in pg.nodes:
                if node_name not in self._nodes:
                    raise ValueError(
                        f"Parallel group node '{node_name}' not found in registered nodes."
                    )
            if pg.then not in self._nodes:
                raise ValueError(
                    f"Parallel group 'then' node '{pg.then}' not found in registered nodes."
                )

        return CompiledGraph(
            nodes=dict(self._nodes),
            edges=list(self._edges),
            entry=self._entry,
            parallel_groups=list(self._parallel_groups),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/agents/graph/test_builder.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/agents/graph/builder.py tests/agents/graph/test_builder.py
git commit -m "feat(agents): add GraphBuilder with compile-time validation"
```

---

### Task 8: AgentRunner — 单 Agent 执行引擎

**Files:**
- Create: `src/agents/runner.py`
- Create: `tests/agents/test_runner.py`

这是核心模块，需要 mock `call_model` 和 `ToolRouter`。

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_runner.py`:

```python
"""AgentRunner 测试 — mock call_model 和 ToolRouter。"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel, ConfigDict

from src.agents.agent import Agent, AgentResult, HandoffRequest
from src.agents.context import RunContext, DictState, EmptyDeps
from src.agents.registry import AgentRegistry


class RunnerDeps(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tool_router: object


@pytest.fixture
def mock_router():
    router = AsyncMock()
    router.route = AsyncMock(return_value="tool result")
    router.get_all_schemas = MagicMock(return_value=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ])
    return router


@pytest.fixture
def registry():
    reg = AgentRegistry()
    reg.register(Agent(
        name="calendar_agent",
        description="管理日历",
        instructions="日历专家。",
    ))
    return reg


@pytest.fixture
def simple_agent():
    return Agent(
        name="test_agent",
        description="Test",
        instructions="You are a test agent.",
        tools=["get_weather"],
    )


@pytest.fixture
def handoff_agent(registry):
    return Agent(
        name="orchestrator",
        description="Orchestrator",
        instructions="You orchestrate.",
        handoffs=["calendar_agent"],
    )


@pytest.mark.asyncio
async def test_runner_simple_response(simple_agent, mock_router):
    from src.agents.runner import AgentRunner

    ctx = RunContext(
        input="hello",
        state=DictState(),
        deps=RunnerDeps(tool_router=mock_router),
    )

    with patch("src.agents.runner.call_model", new_callable=AsyncMock) as mock_call:
        # LLM returns text, no tool calls
        mock_call.return_value = ("Hello back!", {}, None)

        runner = AgentRunner(registry=AgentRegistry())
        result = await runner.run(simple_agent, ctx)

    assert result.text == "Hello back!"
    assert result.handoff is None


@pytest.mark.asyncio
async def test_runner_tool_call_loop(simple_agent, mock_router):
    from src.agents.runner import AgentRunner

    ctx = RunContext(
        input="weather in Beijing",
        state=DictState(),
        deps=RunnerDeps(tool_router=mock_router),
    )

    with patch("src.agents.runner.call_model", new_callable=AsyncMock) as mock_call:
        # Round 1: LLM calls get_weather tool
        mock_call.side_effect = [
            (
                "",
                {0: {"id": "call_1", "name": "get_weather", "arguments": '{"city": "Beijing"}'}},
                "tool_calls",
            ),
            # Round 2: LLM gives final answer
            ("Beijing is sunny, 25°C.", {}, None),
        ]

        runner = AgentRunner(registry=AgentRegistry())
        result = await runner.run(simple_agent, ctx)

    assert "25" in result.text
    mock_router.route.assert_called_once_with("get_weather", {"city": "Beijing"})


@pytest.mark.asyncio
async def test_runner_handoff_detection(handoff_agent, mock_router, registry):
    from src.agents.runner import AgentRunner

    ctx = RunContext(
        input="book a meeting",
        state=DictState(),
        deps=RunnerDeps(tool_router=mock_router),
    )

    with patch("src.agents.runner.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = (
            "",
            {0: {
                "id": "call_1",
                "name": "transfer_to_calendar_agent",
                "arguments": json.dumps({"task": "Book meeting tomorrow"}),
            }},
            "tool_calls",
        )

        runner = AgentRunner(registry=registry)
        result = await runner.run(handoff_agent, ctx)

    assert result.handoff is not None
    assert result.handoff.target == "calendar_agent"
    assert result.handoff.task == "Book meeting tomorrow"


@pytest.mark.asyncio
async def test_runner_max_rounds(simple_agent, mock_router):
    from src.agents.runner import AgentRunner

    ctx = RunContext(
        input="loop",
        state=DictState(),
        deps=RunnerDeps(tool_router=mock_router),
    )

    with patch("src.agents.runner.call_model", new_callable=AsyncMock) as mock_call:
        # Always returns tool calls — should stop at max_tool_rounds
        mock_call.return_value = (
            "",
            {0: {"id": "call_1", "name": "get_weather", "arguments": "{}"}},
            "tool_calls",
        )

        runner = AgentRunner(registry=AgentRegistry(), max_tool_rounds=2)
        result = await runner.run(simple_agent, ctx)

    # Should have called call_model exactly max_tool_rounds times
    assert mock_call.call_count == 2
    assert result.text != ""  # Should have some fallback text


@pytest.mark.asyncio
async def test_runner_dynamic_instructions(mock_router):
    from src.agents.runner import AgentRunner

    def make_instructions(ctx):
        return f"Handle input: {ctx.input}"

    agent = Agent(
        name="dynamic",
        description="Dynamic",
        instructions=make_instructions,
    )
    ctx = RunContext(
        input="test input",
        state=DictState(),
        deps=RunnerDeps(tool_router=mock_router),
    )

    with patch("src.agents.runner.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("OK", {}, None)

        runner = AgentRunner(registry=AgentRegistry())
        await runner.run(agent, ctx)

    # Check the system message contains dynamic content
    messages = mock_call.call_args[0][0]
    assert "Handle input: test input" in messages[0]["content"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/agents/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/agents/runner.py`:

```python
"""AgentRunner — 驱动单个 Agent 完成任务的工具调用循环。"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.agents.agent import Agent, AgentResult, HandoffRequest
from src.agents.context import RunContext, TraceEvent
from src.agents.guardrails import run_guardrails
from src.agents.registry import AgentRegistry
from src.core.async_api import call_model
from src.core.structured_output import build_output_schema, parse_output

logger = logging.getLogger(__name__)

HANDOFF_PREFIX = "transfer_to_"


class AgentRunner:
    """驱动单个 Agent 完成任务的循环。

    职责边界：只负责一个 agent 的工具调用循环和 handoff 检测。
    不关心图的拓扑 — 图引擎决定收到 HandoffRequest 后怎么做。
    """

    def __init__(
        self,
        registry: AgentRegistry,
        max_tool_rounds: int = 10,
        max_result_length: int = 4000,
    ):
        self.registry = registry
        self.max_tool_rounds = max_tool_rounds
        self.max_result_length = max_result_length

    async def run(self, agent: Agent, context: RunContext) -> AgentResult:
        """执行 agent，返回 AgentResult。"""
        hooks = agent.hooks

        # 1. hooks.on_start
        if hooks:
            await hooks.on_start(agent, context)

        # 2. input guardrails
        block = await run_guardrails(agent.input_guardrails, context, context.input)
        if block:
            return AgentResult(text=block.message)

        # 3. 构建 instructions
        if callable(agent.instructions):
            system_prompt = agent.instructions(context)
        else:
            system_prompt = agent.instructions

        # 4. 构建 messages
        task = context.input
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        # 5. 构建工具列表
        tools = self._build_tools(agent, context)
        handoff_tools = self._build_handoff_tools(agent)

        all_tools = tools + handoff_tools
        if not all_tools:
            all_tools = None

        # 6. 工具调用循环
        final_text = ""
        for round_idx in range(self.max_tool_rounds):
            content, tool_calls, _ = await call_model(
                messages,
                tools=all_tools,
                silent=True,
            )

            if not tool_calls:
                final_text = content
                break

            # 检查是否有 handoff 调用
            for tc in tool_calls.values():
                tc_name = tc.get("name", "")
                if tc_name.startswith(HANDOFF_PREFIX):
                    target_name = tc_name[len(HANDOFF_PREFIX):]
                    try:
                        args = json.loads(tc["arguments"])
                    except json.JSONDecodeError:
                        args = {}
                    handoff = HandoffRequest(
                        target=target_name,
                        task=args.get("task", context.input),
                    )
                    context.trace.append(TraceEvent(
                        node=agent.name,
                        event="handoff",
                        timestamp=time.time(),
                        data={"target": target_name, "task": handoff.task},
                    ))
                    if hooks:
                        await hooks.on_handoff(agent, context, handoff)
                    return AgentResult(text=content or "", handoff=handoff)

            # 普通工具调用 — 执行并追加消息
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content if content else None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls.values()
                ],
            }
            messages.append(assistant_msg)

            for tc in tool_calls.values():
                tool_name = tc["name"]
                try:
                    args = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    args = {}

                context.trace.append(TraceEvent(
                    node=agent.name,
                    event="tool_call",
                    timestamp=time.time(),
                    data={"tool": tool_name, "args": args},
                ))
                if hooks:
                    await hooks.on_tool_call(agent, context, tool_name, args)

                # 执行工具
                tool_router = getattr(context.deps, "tool_router", None)
                if tool_router:
                    result_text = await tool_router.route(tool_name, args)
                else:
                    result_text = f"Error: no tool_router in deps"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": str(result_text),
                })
        else:
            # 超过 max_tool_rounds — 强制取最终文本
            content, _, _ = await call_model(messages, silent=True)
            final_text = content

        # 截断
        if len(final_text) > self.max_result_length:
            final_text = final_text[: self.max_result_length] + "…(已截断)"

        # 7. output guardrails
        block = await run_guardrails(agent.output_guardrails, context, final_text)
        if block:
            final_text = block.message

        # 8. 结构化输出
        structured_data: dict = {}
        if agent.output_model is not None:
            output_schema = build_output_schema(
                "agent_output",
                f"将结果整理为 {agent.output_model.__name__} 结构",
                agent.output_model,
            )
            _, struct_calls, _ = await call_model(
                messages + [{"role": "user", "content": "请将结果整理为结构化数据。"}],
                tools=[output_schema],
                silent=True,
            )
            parsed = parse_output(struct_calls, "agent_output", agent.output_model)
            if parsed is not None:
                structured_data = parsed.model_dump()

        result = AgentResult(text=final_text, data=structured_data)

        # 9. hooks.on_end
        if hooks:
            await hooks.on_end(agent, context, result)

        return result

    def _build_tools(self, agent: Agent, context: RunContext) -> list[dict]:
        """从 deps.tool_router 过滤 agent 允许的工具。"""
        tool_router = getattr(context.deps, "tool_router", None)
        if not tool_router or not agent.tools:
            return []
        all_schemas = tool_router.get_all_schemas()
        return [s for s in all_schemas if s["function"]["name"] in agent.tools]

    def _build_handoff_tools(self, agent: Agent) -> list[dict]:
        """为 agent.handoffs 中的每个目标生成 transfer_to_<name> 工具。"""
        tools = []
        for target_name in agent.handoffs:
            target = self.registry.get(target_name)
            description = target.description if target else target_name
            tools.append({
                "type": "function",
                "function": {
                    "name": f"{HANDOFF_PREFIX}{target_name}",
                    "description": f"将任务交接给 {target_name}: {description}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "交接给目标 agent 的任务描述",
                            }
                        },
                        "required": ["task"],
                    },
                },
            })
        return tools
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/agents/test_runner.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/agents/runner.py tests/agents/test_runner.py
git commit -m "feat(agents): add AgentRunner with tool loop, handoff detection, guardrails"
```

---

### Task 9: GraphEngine — 图执行引擎

**Files:**
- Create: `src/agents/graph/engine.py`
- Create: `tests/agents/graph/test_engine.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/graph/test_engine.py`:

```python
"""GraphEngine 执行测试 — 顺序、并行、条件、handoff。"""
import pytest
from unittest.mock import AsyncMock, patch
from pydantic import BaseModel, ConfigDict

from src.agents.agent import Agent, AgentResult, HandoffRequest
from src.agents.context import RunContext, DictState, EmptyDeps, TraceEvent
from src.agents.graph.types import NodeResult, FunctionNode
from src.agents.graph.builder import GraphBuilder
from src.agents.registry import AgentRegistry
from src.agents.hooks import GraphHooks


@pytest.fixture
def registry():
    return AgentRegistry()


@pytest.fixture
def engine(registry):
    from src.agents.graph.engine import GraphEngine

    return GraphEngine(registry=registry)


# --- Sequential execution ---

@pytest.mark.asyncio
async def test_single_function_node(engine):
    async def greet(ctx):
        return NodeResult(output=f"Hello {ctx.input}")

    graph = GraphBuilder()
    graph.add_function("greet", greet)
    graph.set_entry("greet")
    compiled = graph.compile()

    ctx = RunContext(input="World", state=DictState(), deps=EmptyDeps())
    result = await engine.run(compiled, ctx)
    assert result.output == "Hello World"


@pytest.mark.asyncio
async def test_sequential_function_nodes(engine):
    async def step1(ctx):
        return NodeResult(output="step1_done")

    async def step2(ctx):
        return NodeResult(output=f"step2 got {ctx.state.step1}")

    graph = GraphBuilder()
    graph.add_function("step1", step1)
    graph.add_function("step2", step2)
    graph.set_entry("step1")
    graph.add_edge("step1", "step2")
    compiled = graph.compile()

    ctx = RunContext(input="test", state=DictState(), deps=EmptyDeps())
    result = await engine.run(compiled, ctx)
    assert result.output == "step2 got step1_done"


# --- Conditional routing ---

@pytest.mark.asyncio
async def test_conditional_edge(engine):
    async def check(ctx):
        return NodeResult(output="checked")

    async def path_a(ctx):
        return NodeResult(output="took path A")

    async def path_b(ctx):
        return NodeResult(output="took path B")

    graph = GraphBuilder()
    graph.add_function("check", check)
    graph.add_function("path_a", path_a)
    graph.add_function("path_b", path_b)
    graph.set_entry("check")
    graph.add_edge("check", "path_a", condition=lambda ctx: ctx.state.check == "checked")
    graph.add_edge("check", "path_b", condition=lambda ctx: ctx.state.check != "checked")
    compiled = graph.compile()

    ctx = RunContext(input="test", state=DictState(), deps=EmptyDeps())
    result = await engine.run(compiled, ctx)
    assert result.output == "took path A"


# --- Parallel execution ---

@pytest.mark.asyncio
async def test_parallel_execution(engine):
    async def task_a(ctx):
        return NodeResult(output="result_a")

    async def task_b(ctx):
        return NodeResult(output="result_b")

    async def merge(ctx):
        a = ctx.state.task_a
        b = ctx.state.task_b
        return NodeResult(output=f"merged: {a} + {b}")

    graph = GraphBuilder()
    graph.add_function("task_a", task_a)
    graph.add_function("task_b", task_b)
    graph.add_function("merge", merge)
    graph.set_entry("task_a")  # entry is part of parallel group
    graph.add_parallel(["task_a", "task_b"], then="merge")
    compiled = graph.compile()

    ctx = RunContext(input="test", state=DictState(), deps=EmptyDeps())
    result = await engine.run(compiled, ctx)
    assert "result_a" in result.output
    assert "result_b" in result.output


# --- Handoff ---

@pytest.mark.asyncio
async def test_handoff_to_graph_node(registry):
    from src.agents.graph.engine import GraphEngine

    agent_a = Agent(name="agent_a", description="A", instructions="Do A.", handoffs=["agent_b"])
    agent_b = Agent(name="agent_b", description="B", instructions="Do B.")
    registry.register(agent_a)
    registry.register(agent_b)

    graph = GraphBuilder()
    graph.add_agent("agent_a", agent_a)
    graph.add_agent("agent_b", agent_b)
    graph.set_entry("agent_a")
    compiled = graph.compile()

    engine = GraphEngine(registry=registry)
    ctx = RunContext(input="test", state=DictState(), deps=EmptyDeps())

    with patch("src.agents.runner.call_model", new_callable=AsyncMock) as mock_call:
        import json

        mock_call.side_effect = [
            # agent_a returns handoff
            (
                "",
                {0: {
                    "id": "c1",
                    "name": "transfer_to_agent_b",
                    "arguments": json.dumps({"task": "do B stuff"}),
                }},
                "tool_calls",
            ),
            # agent_b returns final answer
            ("B is done.", {}, None),
        ]

        result = await engine.run(compiled, ctx)

    assert result.output["text"] == "B is done."


@pytest.mark.asyncio
async def test_handoff_to_dynamic_agent(registry):
    from src.agents.graph.engine import GraphEngine

    agent_a = Agent(name="agent_a", description="A", instructions="Do A.", handoffs=["dynamic_b"])
    dynamic_b = Agent(name="dynamic_b", description="Dynamic B", instructions="Do dynamic B.")
    registry.register(agent_a)
    registry.register(dynamic_b)

    # Only agent_a is in the graph; dynamic_b is only in registry
    graph = GraphBuilder()
    graph.add_agent("agent_a", agent_a)
    graph.set_entry("agent_a")
    compiled = graph.compile()

    engine = GraphEngine(registry=registry)
    ctx = RunContext(input="test", state=DictState(), deps=EmptyDeps())

    with patch("src.agents.runner.call_model", new_callable=AsyncMock) as mock_call:
        import json

        mock_call.side_effect = [
            (
                "",
                {0: {
                    "id": "c1",
                    "name": "transfer_to_dynamic_b",
                    "arguments": json.dumps({"task": "dynamic task"}),
                }},
                "tool_calls",
            ),
            ("Dynamic B done.", {}, None),
        ]

        result = await engine.run(compiled, ctx)

    assert result.output["text"] == "Dynamic B done."


# --- Tracing ---

@pytest.mark.asyncio
async def test_trace_events_recorded(engine):
    async def step(ctx):
        return NodeResult(output="done")

    graph = GraphBuilder()
    graph.add_function("step", step)
    graph.set_entry("step")
    compiled = graph.compile()

    ctx = RunContext(input="test", state=DictState(), deps=EmptyDeps())
    await engine.run(compiled, ctx)

    start_events = [e for e in ctx.trace if e.event == "start"]
    end_events = [e for e in ctx.trace if e.event == "end"]
    assert len(start_events) >= 1
    assert len(end_events) >= 1
    assert start_events[0].node == "step"


# --- Graph hooks ---

@pytest.mark.asyncio
async def test_graph_hooks_called(registry):
    from src.agents.graph.engine import GraphEngine

    calls = []

    async def on_graph_start(ctx):
        calls.append("graph_start")

    async def on_graph_end(ctx, result):
        calls.append("graph_end")

    async def on_node_start(name, ctx):
        calls.append(f"node_start:{name}")

    async def on_node_end(name, ctx, result):
        calls.append(f"node_end:{name}")

    hooks = GraphHooks(
        on_graph_start=on_graph_start,
        on_graph_end=on_graph_end,
        on_node_start=on_node_start,
        on_node_end=on_node_end,
    )
    engine = GraphEngine(registry=registry, hooks=hooks)

    async def step(ctx):
        return NodeResult(output="done")

    graph = GraphBuilder()
    graph.add_function("step", step)
    graph.set_entry("step")
    compiled = graph.compile()

    ctx = RunContext(input="test", state=DictState(), deps=EmptyDeps())
    await engine.run(compiled, ctx)

    assert "graph_start" in calls
    assert "graph_end" in calls
    assert "node_start:step" in calls
    assert "node_end:step" in calls
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/agents/graph/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/agents/graph/engine.py`:

```python
"""GraphEngine — 异步图执行器。"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from src.agents.agent import Agent
from src.agents.context import RunContext, TraceEvent
from src.agents.graph.types import (
    AgentNode,
    CompiledGraph,
    FunctionNode,
    GraphNode,
    NodeResult,
    ParallelGroup,
)
from src.agents.hooks import GraphHooks
from src.agents.registry import AgentRegistry
from src.agents.runner import AgentRunner

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT", bound=BaseModel)


@dataclass
class GraphResult(Generic[StateT]):
    """图执行的最终结果。"""

    output: Any
    state: StateT
    trace: list[TraceEvent] = field(default_factory=list)


class GraphEngine:
    """异步图执行器。"""

    def __init__(
        self,
        registry: AgentRegistry,
        hooks: GraphHooks | None = None,
        max_handoff_depth: int = 10,
    ):
        self.registry = registry
        self.hooks = hooks or GraphHooks()
        self.max_handoff_depth = max_handoff_depth
        self._runner = AgentRunner(registry=registry)

    async def run(self, graph: CompiledGraph, context: RunContext) -> GraphResult:
        """执行编译后的图。"""
        await self.hooks.on_graph_start(context)

        # 为所有 AgentNode 注入 runner
        for node in graph.nodes.values():
            if isinstance(node, AgentNode) and node.runner is None:
                node.runner = self._runner

        last_output: Any = None
        pending: list[str] = [graph.entry]
        visited: set[str] = set()

        while pending:
            # 检查是否有并行组匹配当前 pending
            parallel_group = self._find_parallel_group(pending, graph.parallel_groups)

            if parallel_group:
                # 并行执行
                nodes_to_run = parallel_group.nodes
                results = await self._run_parallel(nodes_to_run, graph, context)

                for name, node_result in results.items():
                    last_output = node_result.output
                    self._write_state(context, name, node_result.output)
                    visited.add(name)

                # 汇聚到 then 节点
                pending = [parallel_group.then]
            else:
                # 顺序执行当前节点
                current_name = pending.pop(0)
                if current_name in visited and current_name != graph.entry:
                    continue

                node = graph.nodes.get(current_name)
                if node is None:
                    logger.warning(f"Node '{current_name}' not found, skipping")
                    continue

                node_result = await self._execute_node(node, context)
                last_output = node_result.output
                self._write_state(context, current_name, node_result.output)
                visited.add(current_name)

                # 处理 handoff
                if node_result.handoff:
                    target = node_result.handoff.target
                    context.depth += 1
                    if context.depth > self.max_handoff_depth:
                        logger.warning(f"Max handoff depth reached ({self.max_handoff_depth})")
                    elif target in graph.nodes:
                        # handoff 到图中已有节点
                        context.input = node_result.handoff.task
                        pending = [target]
                        continue
                    elif self.registry.has(target):
                        # 动态创建临时节点
                        agent = self.registry.get(target)
                        temp_node = AgentNode(agent=agent, runner=self._runner)
                        context.input = node_result.handoff.task
                        temp_result = await self._execute_node(temp_node, context)
                        last_output = temp_result.output
                        self._write_state(context, target, temp_result.output)
                        visited.add(target)
                    else:
                        logger.error(f"Handoff target '{target}' not found")
                        context.trace.append(TraceEvent(
                            node=current_name,
                            event="error",
                            timestamp=time.time(),
                            data={"error": f"Handoff target '{target}' not found"},
                        ))
                    continue

                # 处理显式 next
                if node_result.next is not None:
                    if isinstance(node_result.next, list):
                        pending = node_result.next
                    else:
                        pending = [node_result.next]
                    continue

                # 按边路由
                next_nodes = self._resolve_edges(current_name, graph, context)
                pending = next_nodes

        result = GraphResult(
            output=last_output,
            state=context.state,
            trace=list(context.trace),
        )
        await self.hooks.on_graph_end(context, result)
        return result

    async def _execute_node(self, node: GraphNode, context: RunContext) -> NodeResult:
        """执行单个节点，带 hooks 和 tracing。"""
        await self.hooks.on_node_start(node.name, context)
        context.trace.append(TraceEvent(
            node=node.name, event="start", timestamp=time.time(),
        ))

        try:
            result = await node.execute(context)
        except Exception as e:
            context.trace.append(TraceEvent(
                node=node.name, event="error", timestamp=time.time(),
                data={"error": str(e)},
            ))
            raise

        context.trace.append(TraceEvent(
            node=node.name, event="end", timestamp=time.time(),
        ))
        await self.hooks.on_node_end(node.name, context, result)
        return result

    async def _run_parallel(
        self,
        node_names: list[str],
        graph: CompiledGraph,
        context: RunContext,
    ) -> dict[str, NodeResult]:
        """并行执行多个节点。"""
        async def _run_one(name: str) -> tuple[str, NodeResult]:
            node = graph.nodes[name]
            if isinstance(node, AgentNode) and node.runner is None:
                node.runner = self._runner
            result = await self._execute_node(node, context)
            return name, result

        tasks = [_run_one(name) for name in node_names]
        results = await asyncio.gather(*tasks)
        return dict(results)

    def _resolve_edges(
        self, source: str, graph: CompiledGraph, context: RunContext,
    ) -> list[str]:
        """根据边的条件选择下一批节点。"""
        next_nodes = []
        for edge in graph.edges:
            if edge.source != source:
                continue
            if edge.condition is None or edge.condition(context):
                next_nodes.append(edge.target)
        return next_nodes

    def _find_parallel_group(
        self, pending: list[str], groups: list[ParallelGroup],
    ) -> ParallelGroup | None:
        """检查 pending 中是否有并行组的入口节点。"""
        pending_set = set(pending)
        for group in groups:
            if pending_set & set(group.nodes):
                return group
        return None

    def _write_state(self, context: RunContext, node_name: str, output: Any) -> None:
        """将节点输出写入 context.state。"""
        try:
            setattr(context.state, node_name, output)
        except (AttributeError, ValueError):
            logger.debug(f"Cannot set state.{node_name}, state type may not support it")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/agents/graph/test_engine.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/agents/graph/engine.py tests/agents/graph/test_engine.py
git commit -m "feat(agents): add GraphEngine with sequential, parallel, conditional, handoff execution"
```

---

### Task 10: Package __init__.py — 公共 API 导出

**Files:**
- Modify: `src/agents/__init__.py`

- [ ] **Step 1: Write the new __init__.py**

Overwrite `src/agents/__init__.py`:

```python
"""Agents 模块 — 图引擎 + Agent Runner 混合架构。

对外导出所有公共接口。
"""

from src.agents.agent import Agent, AgentResult, HandoffRequest
from src.agents.registry import AgentRegistry
from src.agents.runner import AgentRunner
from src.agents.context import RunContext, TraceEvent, DictState, EmptyDeps
from src.agents.guardrails import Guardrail, GuardrailResult, run_guardrails
from src.agents.hooks import AgentHooks, GraphHooks
from src.agents.graph.types import (
    GraphNode,
    AgentNode,
    FunctionNode,
    NodeResult,
    Edge,
    CompiledGraph,
    ParallelGroup,
)
from src.agents.graph.builder import GraphBuilder
from src.agents.graph.engine import GraphEngine, GraphResult

__all__ = [
    # Agent
    "Agent",
    "AgentResult",
    "HandoffRequest",
    # Registry
    "AgentRegistry",
    # Runner
    "AgentRunner",
    # Context
    "RunContext",
    "TraceEvent",
    "DictState",
    "EmptyDeps",
    # Guardrails
    "Guardrail",
    "GuardrailResult",
    "run_guardrails",
    # Hooks
    "AgentHooks",
    "GraphHooks",
    # Graph types
    "GraphNode",
    "AgentNode",
    "FunctionNode",
    "NodeResult",
    "Edge",
    "CompiledGraph",
    "ParallelGroup",
    # Graph builder & engine
    "GraphBuilder",
    "GraphEngine",
    "GraphResult",
]
```

- [ ] **Step 2: Run all agent tests to verify nothing breaks**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/agents/ -v`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add src/agents/__init__.py
git commit -m "feat(agents): update __init__.py with full public API exports"
```

---

### Task 11: 删除旧代码

**Files:**
- Delete: `src/agents/orchestrator.py`
- Delete: `src/agents/specialist_runner.py`
- Delete: `src/agents/specialists.py`
- Delete: `src/core/fsm.py`
- Delete: `src/flows/` (entire directory)
- Delete: `tests/core/test_fsm.py`
- Delete: `tests/flows/` (entire directory)
- Modify: `pyproject.toml` (remove python-statemachine)

- [ ] **Step 1: Delete old agent files**

```bash
rm src/agents/orchestrator.py src/agents/specialist_runner.py src/agents/specialists.py
```

- [ ] **Step 2: Delete FSM and flows**

```bash
rm src/core/fsm.py
rm -rf src/flows/
```

- [ ] **Step 3: Delete old tests**

```bash
rm -f tests/core/test_fsm.py
rm -rf tests/flows/
```

- [ ] **Step 4: Remove python-statemachine from pyproject.toml**

Edit `pyproject.toml`: remove the line `"python-statemachine>=3.0.0",` from dependencies.

- [ ] **Step 5: Update main.py imports**

Remove all references to `MultiAgentFlow`, `FSMRunner`, `FlowModel`, flows imports. Replace with the new agents API. The exact changes depend on current `main.py` state — read `main.py` and update all FSM/Flow imports and usages to use `GraphEngine`, `GraphBuilder`, `RunContext` instead.

- [ ] **Step 6: Run remaining tests to check for breakage**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/ -v --ignore=tests/flows --ignore=tests/core/test_fsm.py`
Expected: All non-deleted tests pass. Fix any import errors from other modules that referenced FSM/flows.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(agents): remove FSM, flows, old agent code; drop python-statemachine dep"
```

---

### Task 12: main.py 集成

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Read current main.py**

Read `main.py` to understand the current wiring.

- [ ] **Step 2: Rewrite handle_input to use GraphEngine**

Replace the FSM-based `handle_input` with the new graph engine approach:

1. Define `AgentDeps` (Pydantic model with `tool_router`, `memory`, `store`)
2. Define specialist agents (`weather_agent`, `calendar_agent`, `email_agent`) and orchestrator
3. Register all agents in `AgentRegistry`
4. Build default graph: single orchestrator node with handoff support
5. Create `GraphEngine` instance
6. In `handle_input`: create `RunContext` → `engine.run()` → return result

Key pattern:

```python
from src.agents import (
    Agent, AgentRegistry, GraphBuilder, GraphEngine,
    RunContext, DictState,
)

# At startup
registry = AgentRegistry()
# ... register agents ...
default_graph = GraphBuilder().add_agent("orchestrator", orchestrator).set_entry("orchestrator").compile()
engine = GraphEngine(registry=registry)

# Per request
async def handle_input(user_input, ...):
    ctx = RunContext(input=user_input, state=DictState(), deps=deps)
    result = await engine.run(default_graph, ctx)
    return result.output
```

- [ ] **Step 3: Run the application manually to smoke-test**

Run: `cd /Users/dingdalong/github/agent && python main.py`
Test with a simple input. Verify the agent responds.

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: integrate GraphEngine into main.py, replace FSM-based handle_input"
```

---

### Task 13: 运行全部测试 + 清理

**Files:**
- Possibly modify: any files with broken imports

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 2: Fix any broken imports or references**

If any tests fail due to imports of deleted modules (`fsm`, `flows`, `MultiAgentFlow`, etc.), update or remove those test files.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: fix broken imports and clean up after agents refactor"
```
