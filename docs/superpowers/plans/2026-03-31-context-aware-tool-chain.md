# Context-Aware Tool Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make RunContext flow through the entire tool execution chain so that delegation, handoff, and tool routing automatically adapt to the current execution environment (main graph, skill graph, plan graph).

**Architecture:** Add optional `context` parameter to `ToolProvider.execute()` and `ToolRouter.route()`. Remove construction-time state captures from `DelegateToolProvider`, `AgentRunner`, and `AgentNode` — they read registry/runner/deps from `RunContext.deps` at runtime. Share `category_resolver` with skill graphs so category agents can be lazy-loaded.

**Tech Stack:** Python 3.13, asyncio, Pydantic BaseModel, Protocol typing

**Spec:** `docs/superpowers/specs/2026-03-31-context-aware-tool-chain-design.md`

---

### Task 1: AgentDeps — add runner field

**Files:**
- Modify: `src/agents/deps.py:8-17`
- Test: `tests/agents/test_deps.py` (create)

- [ ] **Step 1: Write the test**

```python
# tests/agents/test_deps.py
"""AgentDeps 字段完整性测试。"""
from unittest.mock import MagicMock

from src.agents.deps import AgentDeps


def test_agent_deps_has_runner_field():
    """AgentDeps 应包含 runner 字段。"""
    mock_runner = MagicMock()
    deps = AgentDeps(runner=mock_runner)
    assert deps.runner is mock_runner


def test_agent_deps_runner_defaults_to_none():
    """runner 字段默认为 None。"""
    deps = AgentDeps()
    assert deps.runner is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_deps.py -v`
Expected: FAIL — `runner` field not recognized

- [ ] **Step 3: Add runner field to AgentDeps**

```python
# src/agents/deps.py
"""AgentDeps — Agent 运行时外部依赖模型。"""

from typing import Any

from pydantic import BaseModel, ConfigDict


class AgentDeps(BaseModel):
    """外部依赖：传递给 AgentRunner、PlanFlow 等组件。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Any = None              # LLMProvider
    tool_router: Any = None      # ToolRouter
    agent_registry: Any = None   # AgentRegistry
    graph_engine: Any = None     # GraphEngine
    ui: Any = None               # UserInterface
    memory: Any = None           # MemoryProvider
    runner: Any = None           # AgentRunner
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agents/test_deps.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/deps.py tests/agents/test_deps.py
git commit -m "feat(deps): add runner field to AgentDeps"
```

---

### Task 2: ToolProvider protocol + ToolRouter — add context param, remove set_delegate_depth

**Files:**
- Modify: `src/tools/router.py:1-69`
- Test: `tests/tools/test_router.py` (create)

- [ ] **Step 1: Write the tests**

```python
# tests/tools/test_router.py
"""ToolRouter context 透传测试。"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.tools.router import ToolRouter, LocalToolProvider


class FakeProvider:
    """测试用 provider，记录 execute 收到的 context。"""

    def __init__(self):
        self.received_context = None

    def can_handle(self, tool_name: str) -> bool:
        return tool_name == "test_tool"

    async def execute(self, tool_name: str, arguments: dict, context=None) -> str:
        self.received_context = context
        return "ok"

    def get_schemas(self):
        return []


@pytest.mark.asyncio
async def test_route_passes_context_to_provider():
    """ToolRouter.route() 应将 context 透传给 provider.execute()。"""
    router = ToolRouter()
    provider = FakeProvider()
    router.add_provider(provider)

    sentinel = object()
    result = await router.route("test_tool", {}, context=sentinel)

    assert result == "ok"
    assert provider.received_context is sentinel


@pytest.mark.asyncio
async def test_route_without_context_passes_none():
    """不传 context 时，provider 收到 None。"""
    router = ToolRouter()
    provider = FakeProvider()
    router.add_provider(provider)

    await router.route("test_tool", {})

    assert provider.received_context is None


def test_set_delegate_depth_removed():
    """ToolRouter 不应有 set_delegate_depth 方法。"""
    router = ToolRouter()
    assert not hasattr(router, "set_delegate_depth")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tools/test_router.py -v`
Expected: FAIL — `context` param not accepted, `set_delegate_depth` still exists

- [ ] **Step 3: Update ToolProvider protocol, ToolRouter, and LocalToolProvider**

```python
# src/tools/router.py
"""ToolProvider 协议 + ToolRouter 统一路由 + LocalToolProvider。"""

import logging
from typing import Any, Protocol, runtime_checkable

from .executor import ToolExecutor
from .middleware import Middleware, NextFn, build_pipeline
from .registry import ToolRegistry
from .schemas import ToolDict

logger = logging.getLogger(__name__)


@runtime_checkable
class ToolProvider(Protocol):
    """工具来源的统一接口。

    ToolRouter 通过此协议查询和执行工具。每个 provider 管理一组工具，
    通过 can_handle 判断是否能处理某个工具名，通过 get_schemas 暴露
    工具的 JSON Schema 供 LLM 选择。

    实现者：LocalToolProvider（本地 @tool）、MCPToolProvider（MCP）、
    SkillToolProvider（技能）、DelegateToolProvider（委派）。
    """

    def can_handle(self, tool_name: str) -> bool: ...
    async def execute(self, tool_name: str, arguments: dict, context: Any = None) -> str: ...
    def get_schemas(self) -> list[ToolDict]: ...


class ToolRouter:
    """按注册顺序查询 provider，找到第一个能处理的执行。"""

    def __init__(self):
        self._providers: list[ToolProvider] = []

    def add_provider(self, provider: ToolProvider) -> None:
        self._providers.append(provider)

    async def route(self, tool_name: str, arguments: dict, context: Any = None) -> str:
        for provider in self._providers:
            if provider.can_handle(tool_name):
                return await provider.execute(tool_name, arguments, context)
        return f"错误：未找到工具 '{tool_name}'"

    async def ensure_tools(self, tool_names: list[str]) -> None:
        """通知各 provider 预加载指定工具（如按需连接 MCP server）。"""
        for provider in self._providers:
            if hasattr(provider, "ensure_tools"):
                await provider.ensure_tools(tool_names)

    def get_all_schemas(self) -> list[ToolDict]:
        schemas: list[ToolDict] = []
        for provider in self._providers:
            schemas.extend(provider.get_schemas())
        return schemas

    def is_sensitive(self, tool_name: str) -> bool:
        for provider in self._providers:
            if provider.can_handle(tool_name):
                if hasattr(provider, "is_sensitive"):
                    return provider.is_sensitive(tool_name)
                return False
        return False


class LocalToolProvider:
    """本地工具的 Provider 实现，桥接 ToolExecutor + 中间件。"""

    def __init__(
        self,
        registry: ToolRegistry,
        executor: ToolExecutor,
        middlewares: list[Middleware],
    ):
        self.registry = registry
        self._pipeline: NextFn = build_pipeline(executor.execute, middlewares)

    def can_handle(self, tool_name: str) -> bool:
        return self.registry.has(tool_name)

    async def execute(self, tool_name: str, arguments: dict, context: Any = None) -> str:
        return await self._pipeline(tool_name, arguments)

    def get_schemas(self) -> list[ToolDict]:
        return self.registry.get_schemas()

    def is_sensitive(self, tool_name: str) -> bool:
        entry = self.registry.get(tool_name)
        return entry.sensitive if entry else False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tools/test_router.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/router.py tests/tools/test_router.py
git commit -m "feat(router): add context param to ToolProvider.execute and ToolRouter.route, remove set_delegate_depth"
```

---

### Task 3: MCPToolProvider + SkillToolProvider — adapt execute signature

**Files:**
- Modify: `src/mcp/provider.py:15`
- Modify: `src/skills/provider.py:15`

- [ ] **Step 1: Update MCPToolProvider**

```python
# src/mcp/provider.py
"""MCPToolProvider — 将 MCPManager 适配为 ToolProvider。"""

from typing import Any

from src.tools.schemas import ToolDict


class MCPToolProvider:
    """MCP 工具的 Provider 实现"""

    def __init__(self, mcp_manager):
        self._manager = mcp_manager

    def can_handle(self, tool_name: str) -> bool:
        return tool_name.startswith("mcp_")

    async def execute(self, tool_name: str, arguments: dict, context: Any = None) -> str:
        return await self._manager.call_tool(tool_name, arguments)

    async def ensure_tools(self, tool_names: list[str]) -> None:
        """按需连接工具名对应的 MCP Server。"""
        await self._manager.ensure_servers_for_tools(tool_names)

    def get_schemas(self) -> list[ToolDict]:
        return self._manager.get_tools_schemas()
```

- [ ] **Step 2: Update SkillToolProvider**

```python
# src/skills/provider.py
"""SkillToolProvider — 将 SkillManager 适配为 ToolProvider。"""

from typing import Any

from src.tools.schemas import ToolDict


class SkillToolProvider:
    """Skill 工具的 Provider 实现"""

    def __init__(self, skill_manager):
        self._manager = skill_manager

    def can_handle(self, tool_name: str) -> bool:
        return tool_name == "activate_skill"

    async def execute(self, tool_name: str, arguments: dict, context: Any = None) -> str:
        result = self._manager.activate(arguments.get("name", ""))
        return result if result else "未找到指定的 Skill"

    def get_schemas(self) -> list[ToolDict]:
        schema = self._manager.build_activate_tool_schema()
        return [schema] if schema else []
```

- [ ] **Step 3: Run existing tests to verify nothing breaks**

Run: `uv run pytest tests/ -v --timeout=30`
Expected: PASS（可能有部分 runner 测试因 `set_delegate_depth` 被删而失败，后续 task 修复）

- [ ] **Step 4: Commit**

```bash
git add src/mcp/provider.py src/skills/provider.py
git commit -m "feat(providers): adapt MCPToolProvider and SkillToolProvider execute signature for context param"
```

---

### Task 4: AgentRunner — remove registry, pass context to route

**Files:**
- Modify: `src/agents/runner.py`
- Modify: `tests/agents/test_runner.py`

- [ ] **Step 1: Update test file — remove registry from AgentRunner, remove set_delegate_depth test, fix route assertions**

```python
# tests/agents/test_runner.py
"""AgentRunner 测试 — mock deps.llm.chat 和 ToolRouter。"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.agent import Agent, AgentResult, HandoffRequest
from src.agents.context import RunContext, DynamicState
from src.agents.deps import AgentDeps
from src.agents.registry import AgentRegistry
from src.llm.types import LLMResponse


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    return llm


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
def handoff_agent():
    return Agent(
        name="orchestrator",
        description="Orchestrator",
        instructions="You orchestrate.",
        handoffs=["calendar_agent"],
    )


@pytest.mark.asyncio
async def test_runner_simple_response(simple_agent, mock_router, mock_llm):
    from src.agents.runner import AgentRunner

    mock_llm.chat = AsyncMock(return_value=LLMResponse(content="Hello back!", tool_calls={}))

    ctx = RunContext(
        input="hello",
        state=DynamicState(),
        deps=AgentDeps(llm=mock_llm, tool_router=mock_router),
    )

    runner = AgentRunner()
    result = await runner.run(simple_agent, ctx)

    assert result.text == "Hello back!"
    assert result.handoff is None


@pytest.mark.asyncio
async def test_runner_tool_call_loop(simple_agent, mock_router, mock_llm):
    from src.agents.runner import AgentRunner

    mock_llm.chat = AsyncMock(side_effect=[
        LLMResponse(
            content="",
            tool_calls={0: {"id": "call_1", "name": "get_weather", "arguments": '{"city": "Beijing"}'}},
        ),
        LLMResponse(content="Beijing is sunny, 25°C.", tool_calls={}),
    ])

    ctx = RunContext(
        input="weather in Beijing",
        state=DynamicState(),
        deps=AgentDeps(llm=mock_llm, tool_router=mock_router),
    )

    runner = AgentRunner()
    result = await runner.run(simple_agent, ctx)

    assert "25" in result.text
    # route 现在接收 3 个参数：name, args, context
    mock_router.route.assert_called_once_with("get_weather", {"city": "Beijing"}, ctx)


@pytest.mark.asyncio
async def test_runner_handoff_detection(handoff_agent, mock_router, mock_llm, registry):
    from src.agents.runner import AgentRunner

    mock_llm.chat = AsyncMock(return_value=LLMResponse(
        content="",
        tool_calls={0: {
            "id": "call_1",
            "name": "transfer_to_calendar_agent",
            "arguments": json.dumps({"task": "Book meeting tomorrow"}),
        }},
    ))

    ctx = RunContext(
        input="book a meeting",
        state=DynamicState(),
        deps=AgentDeps(llm=mock_llm, tool_router=mock_router, agent_registry=registry),
    )

    runner = AgentRunner()
    result = await runner.run(handoff_agent, ctx)

    assert result.handoff is not None
    assert result.handoff.target == "calendar_agent"
    assert result.handoff.task == "Book meeting tomorrow"


@pytest.mark.asyncio
async def test_runner_max_rounds(simple_agent, mock_router, mock_llm):
    from src.agents.runner import AgentRunner

    mock_llm.chat = AsyncMock(side_effect=[
        LLMResponse(
            content="",
            tool_calls={0: {"id": "call_1", "name": "get_weather", "arguments": "{}"}},
        ),
        LLMResponse(
            content="",
            tool_calls={0: {"id": "call_2", "name": "get_weather", "arguments": "{}"}},
        ),
        LLMResponse(content="Fallback response after max rounds", tool_calls={}),
    ])

    ctx = RunContext(
        input="loop",
        state=DynamicState(),
        deps=AgentDeps(llm=mock_llm, tool_router=mock_router),
    )

    runner = AgentRunner(max_tool_rounds=2)
    result = await runner.run(simple_agent, ctx)

    assert mock_llm.chat.call_count == 3  # 2 rounds + 1 final
    assert result.text == "Fallback response after max rounds"


@pytest.mark.asyncio
async def test_runner_dynamic_instructions(mock_router, mock_llm):
    from src.agents.runner import AgentRunner

    mock_llm.chat = AsyncMock(return_value=LLMResponse(content="OK", tool_calls={}))

    def make_instructions(ctx):
        return f"Handle input: {ctx.input}"

    agent = Agent(
        name="dynamic",
        description="Dynamic",
        instructions=make_instructions,
    )
    ctx = RunContext(
        input="test input",
        state=DynamicState(),
        deps=AgentDeps(llm=mock_llm, tool_router=mock_router),
    )

    runner = AgentRunner()
    await runner.run(agent, ctx)

    messages = mock_llm.chat.call_args[0][0]
    assert "Handle input: test input" in messages[0]["content"]


@pytest.mark.asyncio
async def test_runner_passes_context_to_route(mock_llm):
    """runner 应在 tool_router.route() 调用时透传 context。"""
    from src.agents.runner import AgentRunner

    mock_llm.chat = AsyncMock(side_effect=[
        LLMResponse(
            content="",
            tool_calls={0: {"id": "call_1", "name": "delegate_tool_calc", "arguments": '{"task": "1+1"}'}},
        ),
        LLMResponse(content="2", tool_calls={}),
    ])

    mock_router = AsyncMock()
    mock_router.route = AsyncMock(return_value="2")
    mock_router.get_all_schemas = MagicMock(return_value=[
        {
            "type": "function",
            "function": {
                "name": "delegate_tool_calc",
                "description": "委派任务给计算专家",
                "parameters": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]},
            },
        }
    ])

    agent = Agent(
        name="tool_terminal",
        description="终端操作",
        instructions="终端专家。",
        tools=["delegate_tool_calc"],
    )
    ctx = RunContext(
        input="calculate 1+1",
        state=DynamicState(),
        deps=AgentDeps(llm=mock_llm, tool_router=mock_router),
        delegate_depth=0,
    )

    runner = AgentRunner()
    await runner.run(agent, ctx)

    # 验证 route 被调用时传入了 context
    call_args = mock_router.route.call_args
    assert call_args[0][0] == "delegate_tool_calc"       # tool_name
    assert call_args[0][1] == {"task": "1+1"}             # arguments
    assert call_args[0][2] is ctx                         # context


def test_build_tools_includes_delegates_at_depth_0():
    """delegate_depth=0 时，_build_tools 应包含 delegate 工具。"""
    from src.agents.runner import AgentRunner

    mock_router = MagicMock()
    mock_router.get_all_schemas = MagicMock(return_value=[
        {"type": "function", "function": {"name": "exec", "description": "Execute", "parameters": {}}},
        {"type": "function", "function": {"name": "delegate_tool_calc", "description": "委派计算", "parameters": {}}},
    ])

    agent = Agent(name="test", description="Test", instructions="Test.", tools=["exec", "delegate_tool_calc"])
    ctx = RunContext(input="test", state=DynamicState(), deps=AgentDeps(tool_router=mock_router), delegate_depth=0)

    runner = AgentRunner()
    tools = runner._build_tools(agent, ctx)

    names = [t["function"]["name"] for t in tools]
    assert "exec" in names
    assert "delegate_tool_calc" in names


def test_build_tools_excludes_delegates_at_depth_1():
    """delegate_depth>=1 时，_build_tools 应过滤掉所有 delegate 工具。"""
    from src.agents.runner import AgentRunner

    mock_router = MagicMock()
    mock_router.get_all_schemas = MagicMock(return_value=[
        {"type": "function", "function": {"name": "exec", "description": "Execute", "parameters": {}}},
        {"type": "function", "function": {"name": "delegate_tool_calc", "description": "委派计算", "parameters": {}}},
    ])

    agent = Agent(name="test", description="Test", instructions="Test.", tools=["exec", "delegate_tool_calc"])
    ctx = RunContext(input="test", state=DynamicState(), deps=AgentDeps(tool_router=mock_router), delegate_depth=1)

    runner = AgentRunner()
    tools = runner._build_tools(agent, ctx)

    names = [t["function"]["name"] for t in tools]
    assert "exec" in names
    assert "delegate_tool_calc" not in names


def test_build_handoff_tools_uses_context_registry():
    """_build_handoff_tools 应从 context.deps.agent_registry 获取 agent 描述。"""
    from src.agents.runner import AgentRunner

    registry = AgentRegistry()
    registry.register(Agent(name="target", description="目标 agent", instructions=""))
    agent = Agent(name="test", description="Test", instructions="", handoffs=["target"])
    ctx = RunContext(
        input="test",
        state=DynamicState(),
        deps=AgentDeps(agent_registry=registry),
    )

    runner = AgentRunner()
    tools = runner._build_handoff_tools(agent, ctx)

    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "transfer_to_target"
    assert "目标 agent" in tools[0]["function"]["description"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/test_runner.py -v`
Expected: FAIL — `AgentRunner()` missing `registry`, `route` called with wrong args, `_build_handoff_tools` wrong signature

- [ ] **Step 3: Update AgentRunner implementation**

```python
# src/agents/runner.py
"""AgentRunner — 驱动单个 Agent 完成任务的工具调用循环。"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.agents.agent import Agent, AgentResult, HandoffRequest
from src.agents.context import RunContext, TraceEvent, AppState
from src.agents.registry import AgentRegistry
from src.guardrails import run_guardrails
from src.llm.structured import build_output_schema, parse_output

logger = logging.getLogger(__name__)

HANDOFF_PREFIX = "transfer_to_"


class AgentRunner:
    """驱动单个 Agent 完成任务的循环。

    职责边界：只负责一个 agent 的工具调用循环和 handoff 检测。
    不关心图的拓扑 — 图引擎决定收到 HandoffRequest 后怎么做。
    """

    def __init__(
        self,
        max_tool_rounds: int = 10,
        max_result_length: int = 4000,
    ):
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
        ]

        # 注入长期记忆上下文和对话历史（AppState 有显式字段，其他 state 类型走 getattr）
        if isinstance(context.state, AppState):
            memory_context = context.state.memory_context
            conversation_history = context.state.conversation_history
        else:
            memory_context = getattr(context.state, "memory_context", None)
            conversation_history = getattr(context.state, "conversation_history", None)

        if memory_context:
            messages.append({
                "role": "system",
                "content": f"[相关记忆]\n{memory_context}",
            })
        if conversation_history:
            for msg in conversation_history:
                if msg.get("role") == "system":
                    continue
                messages.append(msg)
            # 避免重复添加当前用户消息
            if not conversation_history or conversation_history[-1].get("content") != task:
                messages.append({"role": "user", "content": task})
        else:
            messages.append({"role": "user", "content": task})

        # 5. 按需连接 MCP server，然后构建工具列表
        tool_router = getattr(context.deps, "tool_router", None)
        if tool_router:
            if agent.tools:
                await tool_router.ensure_tools(agent.tools)
        tools = self._build_tools(agent, context)
        handoff_tools = self._build_handoff_tools(agent, context)
        all_tools = tools + handoff_tools
        if not all_tools:
            all_tools = None

        # 6. 工具调用循环
        final_text = ""
        for round_idx in range(self.max_tool_rounds):
            response = await context.deps.llm.chat(
                messages,
                tools=all_tools,
                silent=True,
            )
            content, tool_calls = response.content, response.tool_calls

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

            # 普通工具调用
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

                tool_router = getattr(context.deps, "tool_router", None)
                if tool_router:
                    result_text = await tool_router.route(tool_name, args, context)
                else:
                    result_text = "Error: no tool_router in deps"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": str(result_text),
                })
        else:
            # 超过 max_tool_rounds
            response = await context.deps.llm.chat(messages, silent=True)
            final_text = response.content

        # 截断
        if len(final_text) > self.max_result_length:
            final_text = final_text[: self.max_result_length] + "...(已截断)"

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
            struct_response = await context.deps.llm.chat(
                messages + [{"role": "user", "content": "请将结果整理为结构化数据。"}],
                tools=[output_schema],
                silent=True,
            )
            parsed = parse_output(struct_response.tool_calls, "agent_output", agent.output_model)
            if parsed is not None:
                structured_data = parsed.model_dump()

        result = AgentResult(text=final_text, data=structured_data)

        # 9. hooks.on_end
        if hooks:
            await hooks.on_end(agent, context, result)

        return result

    def _build_tools(self, agent: Agent, context: RunContext) -> list[dict]:
        """从 deps.tool_router 过滤 agent 允许的工具。

        当 context.delegate_depth >= 1 时，过滤掉所有 delegate_ 前缀的工具，
        防止被委派的 agent 再次委派（递归深度限制）。
        """
        tool_router = getattr(context.deps, "tool_router", None)
        if not tool_router or not agent.tools:
            return []
        all_schemas = tool_router.get_all_schemas()
        allowed = set(agent.tools)
        # 委派深度 >= 1 时，移除所有 delegate 工具
        if context.delegate_depth >= 1:
            allowed = {name for name in allowed if not name.startswith("delegate_")}
        return [s for s in all_schemas if s["function"]["name"] in allowed]

    def _build_handoff_tools(self, agent: Agent, context: RunContext) -> list[dict]:
        """为 agent.handoffs 生成 transfer_to_<name> 工具。"""
        registry = getattr(context.deps, "agent_registry", None)
        tools = []
        for target_name in agent.handoffs:
            target = registry.get(target_name) if registry else None
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

Run: `uv run pytest tests/agents/test_runner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/runner.py tests/agents/test_runner.py
git commit -m "refactor(runner): remove registry param, pass context to route, use context.deps for handoff tools"
```

---

### Task 5: AgentNode — get runner from context.deps

**Files:**
- Modify: `src/agents/node.py`
- Test: `tests/agents/test_node.py` (create)

- [ ] **Step 1: Write the test**

```python
# tests/agents/test_node.py
"""AgentNode 测试 — 从 context.deps.runner 获取 runner。"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.agent import Agent, AgentResult
from src.agents.context import RunContext, DynamicState
from src.agents.deps import AgentDeps
from src.agents.node import AgentNode
from src.graph.types import NodeResult


@pytest.mark.asyncio
async def test_agent_node_uses_runner_from_context():
    """AgentNode 应从 context.deps.runner 获取 runner。"""
    agent = Agent(name="test", description="Test", instructions="test")
    mock_runner = AsyncMock()
    mock_runner.run = AsyncMock(return_value=AgentResult(text="ok", data={}))

    ctx = RunContext(
        input="hello",
        state=DynamicState(),
        deps=AgentDeps(runner=mock_runner),
    )
    node = AgentNode(agent=agent)
    result = await node.execute(ctx)

    mock_runner.run.assert_called_once_with(agent, ctx)
    assert result.output == {"text": "ok", "data": {}}


@pytest.mark.asyncio
async def test_agent_node_raises_when_no_runner():
    """deps.runner 为 None 时应抛出 RuntimeError。"""
    agent = Agent(name="test", description="Test", instructions="test")
    ctx = RunContext(
        input="hello",
        state=DynamicState(),
        deps=AgentDeps(),  # runner=None
    )
    node = AgentNode(agent=agent)
    with pytest.raises(RuntimeError, match="deps.runner is None"):
        await node.execute(ctx)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_node.py -v`
Expected: FAIL — `AgentNode` still expects `runner` param

- [ ] **Step 3: Update AgentNode**

```python
# src/agents/node.py
"""AgentNode — 将 Agent 适配为 GraphNode。"""

from __future__ import annotations

from typing import Any

from src.graph.types import NodeResult


class AgentNode:
    """包装一个 Agent，通过 context.deps.runner 驱动。"""

    def __init__(self, agent: Any):
        self.name: str = agent.name
        self.agent = agent

    async def execute(self, context: Any) -> NodeResult:
        runner = getattr(context.deps, "runner", None)
        if runner is None:
            raise RuntimeError(f"AgentNode({self.name}): deps.runner is None")
        result = await runner.run(self.agent, context)
        return NodeResult(
            output={"text": result.text, "data": result.data},
            handoff=result.handoff,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agents/test_node.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/node.py tests/agents/test_node.py
git commit -m "refactor(node): AgentNode gets runner from context.deps instead of constructor"
```

---

### Task 6: DelegateToolProvider — context-aware, drop captured state

**Files:**
- Modify: `src/tools/delegate.py`
- Modify: `tests/tools/test_delegate_integration.py`

- [ ] **Step 1: Update integration tests for new constructor and context-aware execute**

```python
# tests/tools/test_delegate_integration.py
"""跨分类委派集成测试。"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.agent import Agent, AgentResult
from src.agents.context import RunContext, DynamicState
from src.agents.deps import AgentDeps
from src.agents.registry import AgentRegistry
from src.agents.runner import AgentRunner
from src.tools.categories import CategoryResolver
from src.tools.delegate import DelegateToolProvider
from src.tools.router import ToolRouter
from src.llm.types import LLMResponse


@pytest.fixture
def categories():
    return {
        "tool_terminal": {"description": "终端操作", "tools": {"exec_cmd": "Execute command"}},
        "tool_calc": {"description": "数学计算", "tools": {"calculate": "Calculate math"}},
    }


@pytest.fixture
def resolver(categories):
    return CategoryResolver(categories)


@pytest.fixture
def registry(resolver):
    reg = AgentRegistry()
    reg.set_category_resolver(resolver)
    return reg


@pytest.mark.asyncio
async def test_delegate_end_to_end(resolver, registry):
    """Agent_A 通过 delegate 调用 Agent_B，获取结果并继续完成任务。"""
    mock_llm = AsyncMock()

    call_count = 0

    async def mock_chat(messages, tools=None, silent=True):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(
                content="",
                tool_calls={0: {
                    "id": "call_delegate",
                    "name": "delegate_tool_calc",
                    "arguments": json.dumps({
                        "objective": "执行计算任务，获取 1+1 的结果",
                        "task": "计算 1+1",
                        "context": "用户要求执行计算任务",
                        "expected_result": "计算结果数值",
                    }),
                }},
            )
        if call_count == 2:
            return LLMResponse(content="计算结果是 2", tool_calls={})
        return LLMResponse(content="命令执行完毕，1+1=2", tool_calls={})

    mock_llm.chat = mock_chat

    runner = AgentRunner()
    router = ToolRouter()

    # DelegateToolProvider 只接收 resolver 和 mcp_manager
    delegate_provider = DelegateToolProvider(
        resolver=resolver,
    )
    router.add_provider(delegate_provider)

    agent_a = registry.get("tool_terminal")
    assert "delegate_tool_calc" in agent_a.tools

    # runner 和 registry 通过 deps 传递
    ctx = RunContext(
        input="执行计算任务",
        state=DynamicState(),
        deps=AgentDeps(
            llm=mock_llm,
            tool_router=router,
            agent_registry=registry,
            runner=runner,
        ),
        delegate_depth=0,
    )
    result = await runner.run(agent_a, ctx)

    assert "2" in result.text
    assert call_count == 3


@pytest.mark.asyncio
async def test_delegated_agent_cannot_delegate_further(resolver, registry):
    """被 delegate 调用的 Agent_B 不应看到任何 delegate 工具。"""
    mock_llm = AsyncMock()

    tools_seen_by_b = []

    async def mock_chat(messages, tools=None, silent=True):
        if tools:
            tools_seen_by_b.extend([t["function"]["name"] for t in tools])
        return LLMResponse(content="结果", tool_calls={})

    mock_llm.chat = mock_chat

    runner = AgentRunner()
    router = ToolRouter()

    deps = AgentDeps(
        llm=mock_llm,
        tool_router=router,
        agent_registry=registry,
        runner=runner,
    )
    delegate_provider = DelegateToolProvider(
        resolver=resolver,
    )
    router.add_provider(delegate_provider)

    agent_b = registry.get("tool_calc")

    ctx = RunContext(
        input="计算 1+1",
        state=DynamicState(),
        deps=deps,
        delegate_depth=1,
    )
    await runner.run(agent_b, ctx)

    delegate_tools = [t for t in tools_seen_by_b if t.startswith("delegate_")]
    assert delegate_tools == [], f"Agent_B should not see delegate tools, but saw: {delegate_tools}"


@pytest.mark.asyncio
async def test_delegate_execute_without_context_returns_error(resolver):
    """context=None 时应返回错误信息。"""
    provider = DelegateToolProvider(resolver=resolver)
    result = await provider.execute(
        "delegate_tool_calc",
        {"objective": "test", "task": "test"},
        context=None,
    )
    assert "错误" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tools/test_delegate_integration.py -v`
Expected: FAIL — `DelegateToolProvider` constructor expects `runner`, `registry`, `deps`

- [ ] **Step 3: Update DelegateToolProvider**

```python
# src/tools/delegate.py
"""DelegateToolProvider — 将 Tool Agent 包装为可调用工具。

业务 Agent 可通过 delegate_<name>(objective, task, context?, expected_result?)
调用对应的 Tool Agent。委托时通过结构化的四字段 schema 强制发送方说清楚
任务意图，接收方通过 prompt 模板获得完整的任务上下文。

协议设计详见 docs/superpowers/specs/2026-03-31-structured-delegation-protocol-design.md

本模块位于 Layer 1（src/tools/），对 Layer 2 的依赖
（AgentRunner、AgentRegistry、AgentDeps）仅在 TYPE_CHECKING
或 execute() 运行时才导入，不违反分层约束。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.tools.schemas import ToolDict

if TYPE_CHECKING:
    from src.mcp.manager import MCPManager
    from src.tools.categories import CategoryResolver

DELEGATE_PREFIX = "delegate_"

DELEGATE_DESCRIPTION_TEMPLATE = (
    "委托任务给{description}专家。"
    "请基于当前对话上下文，清晰完整地填写以下字段，"
    "确保对方无需额外信息就能执行任务。"
)

RECEIVING_TEMPLATE = (
    "你收到了一个委托任务：\n"
    "最终目标：{objective}\n"
    "具体任务：{task}\n"
    "{context_line}"
    "{expected_result_line}"
    "\n"
    "完成后请按以下格式返回：\n"
    "第一行标注任务状态：已完成 / 信息不足 / 失败\n"
    "之后是具体结果或需要补充的信息。\n"
    "不要猜测或假设缺失的信息。"
)


def _build_receiving_input(
    objective: str,
    task: str,
    context: str | None = None,
    expected_result: str | None = None,
) -> str:
    """用接收方模板组装委托任务的 input 文本。"""
    context_line = f"相关上下文：{context}\n" if context and context.strip() else ""
    expected_result_line = f"期望结果：{expected_result}\n" if expected_result and expected_result.strip() else ""
    return RECEIVING_TEMPLATE.format(
        objective=objective,
        task=task,
        context_line=context_line,
        expected_result_line=expected_result_line,
    )


class DelegateToolProvider:
    """将 Tool Agent 包装为可调用工具的 ToolProvider。

    构造时只绑定只读依赖（resolver 用于 schema 生成，mcp_manager 用于按需连接）。
    运行时从 context.deps 获取 registry、runner 等作用域状态。
    """

    def __init__(
        self,
        resolver: CategoryResolver,
        mcp_manager: MCPManager | None = None,
    ) -> None:
        self._resolver = resolver
        self._mcp_manager = mcp_manager

    def can_handle(self, tool_name: str) -> bool:
        """判断 tool_name 是否为已知的 delegate 工具。"""
        if not tool_name.startswith(DELEGATE_PREFIX):
            return False
        agent_name = tool_name[len(DELEGATE_PREFIX):]
        return self._resolver.can_resolve(agent_name)

    def get_schemas(self) -> list[ToolDict]:
        """为每个可委派的 Tool Agent 生成结构化委托 schema。"""
        schemas: list[ToolDict] = []
        for summary in self._resolver.get_all_summaries():
            name = summary["name"]
            desc = summary["description"]
            schemas.append({
                "type": "function",
                "function": {
                    "name": f"{DELEGATE_PREFIX}{name}",
                    "description": DELEGATE_DESCRIPTION_TEMPLATE.format(description=desc),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "objective": {
                                "type": "string",
                                "description": "你的最终目标是什么（为什么需要这次委托）",
                            },
                            "task": {
                                "type": "string",
                                "description": "你需要对方具体做什么",
                            },
                            "context": {
                                "type": "string",
                                "description": "当前已知的相关信息。只填你确定知道的，不要猜测。",
                            },
                            "expected_result": {
                                "type": "string",
                                "description": "你期望对方完成后告诉你什么。如果不确定，可简要描述即可。",
                            },
                        },
                        "required": ["objective", "task"],
                    },
                },
            })
        return schemas

    async def execute(self, tool_name: str, arguments: dict[str, Any], context: Any = None) -> str:
        """委派执行：从 context.deps 获取 registry/runner，创建子 RunContext。"""
        if context is None:
            return "错误：delegate 调用缺少执行上下文"

        from src.agents.context import DynamicState, RunContext

        agent_name = tool_name[len(DELEGATE_PREFIX):]

        registry = getattr(context.deps, "agent_registry", None)
        if registry is None:
            return f"错误：deps 中缺少 agent_registry"

        agent = registry.get(agent_name)
        if agent is None:
            return f"错误：找不到 agent {agent_name}"

        # 按需连接该 agent 所需的 MCP server
        if self._mcp_manager:
            mcp_tools = [t for t in agent.tools if t.startswith("mcp_")]
            if mcp_tools:
                await self._mcp_manager.ensure_servers_for_tools(mcp_tools)

        # 从结构化参数构建接收方 input
        task = arguments.get("task", "")
        objective = arguments.get("objective", task)
        ctx_str = arguments.get("context")
        expected_result = arguments.get("expected_result")
        receiving_input = _build_receiving_input(
            objective=objective,
            task=task,
            context=ctx_str,
            expected_result=expected_result,
        )

        runner = getattr(context.deps, "runner", None)
        if runner is None:
            return "错误：deps 中缺少 runner"

        sub_ctx: RunContext = RunContext(
            input=receiving_input,
            state=DynamicState(),
            deps=context.deps,
            delegate_depth=context.delegate_depth + 1,
        )
        result = await runner.run(agent, sub_ctx)
        return result.text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tools/test_delegate_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/delegate.py tests/tools/test_delegate_integration.py
git commit -m "refactor(delegate): context-aware execute, drop captured runner/registry/deps"
```

---

### Task 7: presets.py — remove runner param from graph builders

**Files:**
- Modify: `src/app/presets.py`

- [ ] **Step 1: Update presets.py**

```python
# src/app/presets.py
"""Agent 预设定义与图构建。

orchestrator 的 handoff 列表和路由指令根据传入的 category_summaries
与 business_agents 动态生成，不再硬编码占位 Agent。
"""

from __future__ import annotations

from src.agents.agent import Agent
from src.agents.node import AgentNode
from src.agents.registry import AgentRegistry
from src.agents.context import RunContext
from src.graph.types import NodeResult, CompiledGraph
from src.graph.builder import GraphBuilder

_ORCHESTRATOR_BASE_INSTRUCTIONS = (
    "你是一个智能助手。根据用户的请求选择合适的操作：\n"
    "{handoff_instructions}"
    "- 需要多步骤协作的复杂任务，交给 planner\n"
    "- 其他问题，直接回答用户\n"
)

_PLANNER_AGENT = Agent(
    name="planner",
    description="处理需要多步骤的复杂任务，生成计划并按步骤执行",
    instructions="",
)


def _build_handoff_instructions(
    category_summaries: list[dict[str, str]],
    business_agents: list[dict[str, str]] | None = None,
) -> str:
    """根据分类摘要和业务 Agent 列表生成 handoff 路由指令。"""
    lines: list[str] = []
    for s in category_summaries:
        lines.append(f"- {s['description']}相关，交给 {s['name']}")
    if business_agents:
        for a in business_agents:
            lines.append(f"- {a['description']}相关，交给 {a['name']}")
    return "\n".join(lines) + "\n" if lines else ""


def _make_planner_node_fn():
    async def planner_node_fn(ctx: RunContext) -> NodeResult:
        from src.plan.flow import PlanFlow

        plan_flow = PlanFlow(
            llm=ctx.deps.llm,
            tool_router=ctx.deps.tool_router,
            agent_registry=ctx.deps.agent_registry,
            engine=ctx.deps.graph_engine,
            ui=ctx.deps.ui,
        )
        result = await plan_flow.run(ctx.input)
        return NodeResult(output=result)

    return planner_node_fn


def _register_and_build(
    registry: AgentRegistry,
    skill_content: str | None = None,
    category_summaries: list[dict[str, str]] | None = None,
    business_agents: list[dict[str, str]] | None = None,
) -> CompiledGraph:
    """内部构建函数：注册 orchestrator + planner，编译图。"""
    summaries = category_summaries or []
    handoff_instructions = _build_handoff_instructions(summaries, business_agents)
    instructions = _ORCHESTRATOR_BASE_INSTRUCTIONS.format(
        handoff_instructions=handoff_instructions
    )
    if skill_content:
        instructions = f"{skill_content}\n\n{instructions}"

    # 动态构建 handoff 列表
    handoffs = [s["name"] for s in summaries]
    if business_agents:
        handoffs.extend(a["name"] for a in business_agents)
    handoffs.append("planner")

    orchestrator = Agent(
        name="orchestrator",
        description="总控 Agent，负责路由和直接回答",
        instructions=instructions,
        handoffs=handoffs,
    )
    registry.register(orchestrator)
    registry.register(_PLANNER_AGENT)

    builder = GraphBuilder()
    builder.add_node(AgentNode(agent=orchestrator))
    # 为每个 category agent 添加 graph node，使 handoff 可达
    for s in summaries:
        agent = registry.get(s["name"])
        if agent:
            builder.add_node(AgentNode(agent=agent))
    if business_agents:
        for a in business_agents:
            agent = registry.get(a["name"])
            if agent:
                builder.add_node(AgentNode(agent=agent))
    builder.add_function("planner", _make_planner_node_fn())
    builder.set_entry("orchestrator")
    return builder.compile()


def build_default_graph(
    registry: AgentRegistry,
    category_summaries: list[dict[str, str]] | None = None,
    business_agents: list[dict[str, str]] | None = None,
) -> CompiledGraph:
    """构建默认图（无 skill 前缀指令）。"""
    return _register_and_build(
        registry,
        category_summaries=category_summaries,
        business_agents=business_agents,
    )


def build_skill_graph(
    registry: AgentRegistry,
    skill_content: str,
    category_summaries: list[dict[str, str]] | None = None,
    business_agents: list[dict[str, str]] | None = None,
) -> CompiledGraph:
    """构建技能图（skill 内容作为指令前缀）。"""
    return _register_and_build(
        registry,
        skill_content=skill_content,
        category_summaries=category_summaries,
        business_agents=business_agents,
    )
```

- [ ] **Step 2: Run existing tests**

Run: `uv run pytest tests/ -v --timeout=30 -x`
Expected: PASS（或仅有 bootstrap/app 层面的 wiring 问题，后续 task 修复）

- [ ] **Step 3: Commit**

```bash
git add src/app/presets.py
git commit -m "refactor(presets): remove runner param from graph builders, AgentNode gets runner from context"
```

---

### Task 8: PlanCompiler — context-aware tool routing and agent execution

**Files:**
- Modify: `src/plan/compiler.py:115-213`
- Modify: `tests/plan/test_compiler.py`

- [ ] **Step 1: Update test assertions for context-aware routing**

在 `tests/plan/test_compiler.py` 中，更新 `test_tool_step_calls_router` 和 `test_variable_resolution_during_execution` 中的 mock 以接受 context 参数。

```python
# tests/plan/test_compiler.py — 修改部分

# 替换 TestPlanCompilerExecution 类中的两个方法：

@pytest.mark.asyncio
class TestPlanCompilerExecution:
    async def test_tool_step_calls_router(self):
        """编译后的工具步骤执行时调用 tool_router.route"""
        registry, router = _make_registry_and_router()
        router.route = AsyncMock(return_value="晴天 25°C")
        compiler = PlanCompiler(registry, router)

        plan = Plan(steps=[
            Step(id="weather", description="查天气", tool_name="get_weather", tool_args={"location": "广州"}),
        ])
        graph = compiler.compile(plan)
        engine = GraphEngine()

        from src.agents.deps import AgentDeps

        ctx = RunContext(
            input="test",
            state=DynamicState(),
            deps=AgentDeps(tool_router=router, agent_registry=registry),
        )
        result = await engine.run(graph, ctx)

        # route 现在接收 3 个参数
        router.route.assert_called_once_with("get_weather", {"location": "广州"}, ctx)
        assert result.output == "晴天 25°C"

    async def test_variable_resolution_during_execution(self):
        """执行时 $step_id.field 变量正确解析"""
        registry, router = _make_registry_and_router()

        call_count = 0
        async def mock_route(tool_name, args, context=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "sunny"
            return f"translated: {args.get('text', '')}"

        router.route = mock_route
        compiler = PlanCompiler(registry, router)

        plan = Plan(steps=[
            Step(id="search", description="搜索", tool_name="get_weather"),
            Step(
                id="translate",
                description="翻译",
                tool_name="get_weather",
                tool_args={"text": "$search"},
                depends_on=["search"],
            ),
        ])
        graph = compiler.compile(plan)
        engine = GraphEngine()

        from src.agents.deps import AgentDeps

        ctx = RunContext(
            input="test",
            state=DynamicState(),
            deps=AgentDeps(tool_router=router, agent_registry=registry),
        )
        result = await engine.run(graph, ctx)

        assert "sunny" in str(result.output)
```

- [ ] **Step 2: Update PlanCompiler**

```python
# src/plan/compiler.py — 修改 PlanCompiler 类

class PlanCompiler:
    """将 Plan 编译为 CompiledGraph。

    编译流程：
    1. 验证：检查 Step ID 唯一性、agent 存在性
    2. 分层拓扑排序：同层步骤无互相依赖，可并行执行
    3. 每个 Step 生成 FunctionNode 闭包（工具调用或智能体调用）
    4. 单步骤层 → 顺序 Edge；多步骤层 → ParallelGroup + merge 节点
    5. 支持 $step_id.field 变量引用，运行时从 context.state 解析
    """

    def __init__(self, agent_registry: AgentRegistry, tool_router: ToolRouter):
        self._registry = agent_registry
        self._router = tool_router

    def compile(self, plan: Plan) -> CompiledGraph:
        """Plan -> CompiledGraph。

        1. 验证计划合法性
        2. 分层拓扑排序
        3. 每个 Step -> FunctionNode
        4. 层间关系 -> Edge / ParallelGroup
        """
        self._validate(plan)
        layers = _topological_sort_layered(plan.steps)

        builder = GraphBuilder()
        prev_exit: str | None = None

        for layer_idx, layer in enumerate(layers):
            # 为本层每个 step 创建 FunctionNode
            for step in layer:
                fn = self._make_node_fn(step)
                builder.add_function(step.id, fn)

            if len(layer) == 1:
                step_id = layer[0].id
                if layer_idx == 0:
                    builder.set_entry(step_id)
                if prev_exit is not None:
                    builder.add_edge(prev_exit, step_id)
                prev_exit = step_id
            else:
                # 多个步骤 -> ParallelGroup + merge 节点
                merge_name = f"_merge_{layer_idx}"
                merge_fn = self._make_merge_fn()
                builder.add_function(merge_name, merge_fn)
                builder.add_parallel([s.id for s in layer], then=merge_name)

                if layer_idx == 0:
                    builder.set_entry(layer[0].id)
                if prev_exit is not None:
                    for step in layer:
                        builder.add_edge(prev_exit, step.id)
                prev_exit = merge_name

        return builder.compile()

    def _validate(self, plan: Plan) -> None:
        """编译前验证。"""
        if not plan.steps:
            raise CompileError("空计划：没有步骤")

        # 检查 ID 唯一性
        seen: set[str] = set()
        for step in plan.steps:
            if step.id in seen:
                raise CompileError(f"重复的步骤 ID: {step.id}")
            seen.add(step.id)

        # 检查 agent 存在性
        for step in plan.steps:
            if step.agent_name and not self._registry.has(step.agent_name):
                raise CompileError(f"Agent '{step.agent_name}' 不存在于注册表中")

    def _make_node_fn(self, step: Step):
        """为一个 Step 创建 FunctionNode 的执行函数。"""
        if step.tool_name is not None:
            return self._make_tool_fn(step)
        return self._make_agent_fn(step)

    def _make_tool_fn(self, step: Step):
        """工具步骤 -> 闭包函数。"""
        tool_name = step.tool_name
        tool_args = step.tool_args
        router = self._router

        async def fn(ctx: RunContext) -> NodeResult:
            resolved = resolve_variables(tool_args, _state_to_dict(ctx.state))
            result = await router.route(tool_name, resolved, ctx)
            return NodeResult(output=result)

        return fn

    def _make_agent_fn(self, step: Step):
        """Agent 步骤 -> 闭包函数。从 context.deps 获取 registry 和 runner。"""
        agent_name = step.agent_name
        agent_prompt = step.agent_prompt or step.description

        async def fn(ctx: RunContext) -> NodeResult:
            resolved_prompt = resolve_variables(agent_prompt, _state_to_dict(ctx.state))
            registry = ctx.deps.agent_registry
            runner = ctx.deps.runner
            agent = registry.get(agent_name)
            agent_ctx = replace(ctx, input=resolved_prompt)
            result = await runner.run(agent, agent_ctx)
            return NodeResult(output={"text": result.text, "data": result.data}, handoff=result.handoff)

        return fn

    @staticmethod
    def _make_merge_fn():
        """并行组合并节点 -> 空操作透传。"""

        async def fn(ctx: RunContext) -> NodeResult:
            return NodeResult(output=None)

        return fn
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/plan/test_compiler.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/plan/compiler.py tests/plan/test_compiler.py
git commit -m "refactor(compiler): remove self._runner, use context.deps for registry/runner, pass context to route"
```

---

### Task 9: AgentApp — store category_resolver, fix _handle_skill

**Files:**
- Modify: `src/app/app.py`

- [ ] **Step 1: Update AgentApp**

```python
# src/app/app.py
"""AgentApp — 消息路由和 REPL。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.interfaces.base import UserInterface
from src.guardrails import InputGuardrail
from src.agents import RunContext, DynamicState, AppState, AgentDeps, AgentRegistry
from src.agents.node import AgentNode
from src.graph import GraphEngine, CompiledGraph
from src.tools.router import ToolRouter
from src.skills.manager import SkillManager
from src.mcp.manager import MCPManager
from src.plan.flow import PlanFlow
from src.app.presets import build_skill_graph

if TYPE_CHECKING:
    from src.memory.buffer import ConversationBuffer
    from src.memory.types import MemoryRecord
    from src.tools.categories import CategoryResolver

logger = logging.getLogger(__name__)


class AgentApp:
    """应用核心 — 消息路由 + REPL 循环。

    所有组件由 bootstrap.py 注入，AgentApp 不创建任何具体实现。
    消息路由逻辑：
    - 所有输入先经过 InputGuardrail 安全检查
    - /plan 命令 → PlanFlow 多步骤规划执行
    - /skill-name → SkillManager 激活技能，构建独立图执行
    - 普通消息 → 默认图（orchestrator → 专家智能体）
    """

    def __init__(
        self,
        deps: AgentDeps,
        ui: UserInterface,
        guardrail: InputGuardrail,
        tool_router: ToolRouter,
        agent_registry: AgentRegistry,
        engine: GraphEngine,
        graph: CompiledGraph,
        skill_manager: SkillManager,
        mcp_manager: MCPManager,
        conversation_buffer: ConversationBuffer | None = None,
        category_summaries: list[dict[str, str]] | None = None,
        category_resolver: CategoryResolver | None = None,
    ):
        self.deps = deps
        self.ui = ui
        self.guardrail = guardrail
        self.tool_router = tool_router
        self.agent_registry = agent_registry
        self.engine = engine
        self.graph = graph
        self.skill_manager = skill_manager
        self.mcp_manager = mcp_manager
        self.conversation_buffer = conversation_buffer
        self._category_summaries: list[dict[str, str]] = category_summaries or []
        self._category_resolver = category_resolver

    async def process(self, user_input: str) -> None:
        """处理单条用户消息。"""
        passed, reason = self.guardrail.check(user_input)
        if not passed:
            await self.ui.display(f"\n[安全拦截] {reason}\n")
            return

        if user_input.strip().startswith("/plan"):
            await self._handle_plan(user_input)
            return

        skill_name = self.skill_manager.is_slash_command(user_input)
        if skill_name:
            await self._handle_skill(user_input, skill_name)
            return

        await self._handle_normal(user_input)

    async def _handle_plan(self, user_input: str) -> None:
        plan_request = user_input.strip()[5:].strip()
        if not plan_request:
            await self.ui.display("\n请在 /plan 后输入你的请求\n")
            return
        plan_flow = PlanFlow(
            llm=self.deps.llm,
            tool_router=self.tool_router,
            agent_registry=self.agent_registry,
            engine=self.engine,
            ui=self.ui,
        )
        result = await plan_flow.run(plan_request)
        await self.ui.display(f"\n{result}\n")

    async def _handle_skill(self, user_input: str, skill_name: str) -> None:
        skill_content = self.skill_manager.activate(skill_name)
        if not skill_content:
            return
        remaining = user_input[len(f"/{skill_name}"):].strip()
        actual_input = remaining or f"已激活 {skill_name} skill，请按指令执行。"

        # 隔离的 registry，共享只读的 category_resolver
        skill_registry = AgentRegistry()
        if self._category_resolver:
            skill_registry.set_category_resolver(self._category_resolver)

        skill_graph = build_skill_graph(
            skill_registry,
            skill_content,
            category_summaries=self._category_summaries,
        )
        skill_engine = GraphEngine()
        ctx = RunContext(
            input=actual_input,
            state=DynamicState(),
            deps=AgentDeps(
                llm=self.deps.llm,
                tool_router=self.tool_router,
                agent_registry=skill_registry,
                graph_engine=skill_engine,
                ui=self.ui,
                memory=self.deps.memory,
                runner=self.deps.runner,
            ),
        )
        result = await skill_engine.run(skill_graph, ctx)
        output = result.output
        if isinstance(output, dict):
            output = output.get("text", str(output))
        await self.ui.display(f"\n{output}\n")

    async def _handle_normal(self, user_input: str) -> None:
        state = AppState()

        # --- Pre-turn: 记忆检索 ---
        if self.conversation_buffer is not None:
            self.conversation_buffer.add_user_message(user_input)

        if self.deps.memory is not None:
            try:
                memories: list[MemoryRecord] = self.deps.memory.search(user_input, n=5)
                if memories:
                    state.memory_context = self._format_memories(memories)
            except Exception:
                logger.warning("[记忆系统] 检索失败，跳过", exc_info=True)

        if self.conversation_buffer is not None:
            state.conversation_history = self.conversation_buffer.get_messages_for_api()

        # --- Execution ---
        ctx = RunContext(
            input=user_input,
            state=state,
            deps=self.deps,
        )
        result = await self.engine.run(self.graph, ctx)
        output = result.output
        if isinstance(output, dict):
            output = output.get("text", str(output))
        await self.ui.display(f"\n{output}\n")

        # --- Post-turn: 记忆存储 ---
        if self.conversation_buffer is not None:
            self.conversation_buffer.add_assistant_message(output)

        if self.deps.memory is not None:
            try:
                await self.deps.memory.add_from_conversation(
                    user_input=user_input,
                    assistant_response=output,
                )
            except Exception:
                logger.warning("[记忆系统] 事实提取失败，跳过", exc_info=True)

        if (
            self.conversation_buffer is not None
            and self.deps.memory is not None
            and self.conversation_buffer.should_compress()
        ):
            try:
                await self.conversation_buffer.compress(
                    store=self.deps.memory,
                    llm=self.deps.llm,
                )
            except Exception:
                logger.warning("[记忆系统] 对话压缩失败，跳过", exc_info=True)

    def _format_memories(self, memories: list[MemoryRecord]) -> str:
        """将 MemoryRecord 列表格式化为 LLM 上下文字符串。"""
        lines = []
        for m in memories:
            prefix = "[事实]" if m.memory_type.value == "fact" else "[摘要]"
            lines.append(f"{prefix} {m.content}")
        return "\n".join(lines)

    async def run(self) -> None:
        """CLI 主循环。"""
        await self.ui.display("Agent 已启动，输入 'exit' 退出。\n")
        while True:
            user_input = await self.ui.prompt("\n你: ")
            if user_input.strip().lower() in ("exit", "quit"):
                break
            await self.process(user_input)

    async def shutdown(self) -> None:
        if (
            self.conversation_buffer is not None
            and self.deps.memory is not None
            and len(self.conversation_buffer.messages) > 0
        ):
            try:
                await self.conversation_buffer.compress(
                    store=self.deps.memory,
                    llm=self.deps.llm,
                )
            except Exception:
                logger.warning("[记忆系统] 退出时对话压缩失败", exc_info=True)
        await self.mcp_manager.disconnect_all()
```

- [ ] **Step 2: Commit**

```bash
git add src/app/app.py
git commit -m "refactor(app): store category_resolver, fix _handle_skill to share resolver and runner via deps"
```

---

### Task 10: bootstrap.py — wire everything together

**Files:**
- Modify: `src/app/bootstrap.py`

- [ ] **Step 1: Update bootstrap.py**

```python
# src/app/bootstrap.py
"""应用组装 — 读配置、创建组件、注入依赖。"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import load_config
from src.llm.openai import OpenAIProvider
from src.interfaces.cli import CLIInterface
from src.tools.decorator import get_registry
from src.tools.discovery import discover_tools
from src.tools.executor import ToolExecutor
from src.tools.router import ToolRouter, LocalToolProvider
from src.tools.middleware import (
    error_handler_middleware,
    sensitive_confirm_middleware,
    truncate_middleware,
)
from src.mcp.config import load_mcp_config
from src.mcp.manager import MCPManager
from src.mcp.provider import MCPToolProvider
from src.skills.manager import SkillManager
from src.skills.provider import SkillToolProvider
from src.guardrails import InputGuardrail
from src.agents import AgentRegistry
from src.agents.runner import AgentRunner
from src.graph import GraphEngine
from src.agents.deps import AgentDeps
from src.memory import ChromaMemoryStore, ConversationBuffer
from src.memory.utils import build_collection_name
from src.app.presets import build_default_graph
from src.app.app import AgentApp

logger = logging.getLogger(__name__)


async def create_app(config_path: str = "config.yaml") -> AgentApp:
    """应用组装入口 — 整个框架唯一的具体实现实例化点。

    组装流程：
    1. 加载 config.yaml + .env
    2. 创建 LLM provider（OpenAIProvider）
    3. 发现并注册本地工具，构建中间件管道
    4. 加载 MCP 配置（不连接），注册工具提供者
    5. 发现技能
    6. 注册预设智能体，构建默认图
    7. 组装 AgentDeps 依赖容器
    8. 返回 AgentApp 实例
    """
    raw = load_config(config_path)
    llm_cfg = raw.get("llm", {})
    ui = CLIInterface()

    # 1. LLM
    llm = OpenAIProvider(
        api_key=llm_cfg.get("api_key", ""),
        base_url=llm_cfg.get("base_url", ""),
        model=llm_cfg.get("model", ""),
        concurrency=llm_cfg.get("concurrency", 5),
        max_retries=llm_cfg.get("max_retries", 3),
        on_chunk=ui.display,
    )

    # 2. Tools
    discover_tools("src.tools.builtin", Path("src/tools/builtin"))
    registry = get_registry()
    executor = ToolExecutor(registry)
    middlewares = [
        error_handler_middleware(),
        sensitive_confirm_middleware(registry, ui),
        truncate_middleware(raw.get("tools", {}).get("max_output_length", 2000)),
    ]
    tool_router = ToolRouter()
    tool_router.add_provider(LocalToolProvider(registry, executor, middlewares))

    # 3. MCP — 只加载配置，不连接。连接在 DelegateToolProvider.execute 中按需触发
    mcp_config_path = raw.get("mcp", {}).get("config_path", "mcp_servers.json")
    mcp_configs = load_mcp_config(mcp_config_path)
    mcp_manager = MCPManager(configs=mcp_configs)
    if mcp_configs:
        tool_router.add_provider(MCPToolProvider(mcp_manager))

    # 4. Skills
    skill_dirs = raw.get("skills", {}).get("dirs", ["skills/", ".agents/skills/"])
    skill_manager = SkillManager(skill_dirs=skill_dirs)
    await skill_manager.discover()
    if skill_manager._skills:
        tool_router.add_provider(SkillToolProvider(skill_manager))

    # 5. Memory
    embedding_cfg = raw.get("embedding", {})
    memory_cfg = raw.get("memory", {})
    user_cfg = raw.get("user", {})
    memory_store = None
    conversation_buffer = None

    if (
        memory_cfg.get("provider") == "chroma"
        and embedding_cfg.get("model")
        and embedding_cfg.get("base_url")
    ):
        try:
            collection_name = build_collection_name(
                "memories", user_cfg.get("id", "default_user")
            )
            memory_store = ChromaMemoryStore(
                embedding_model=embedding_cfg["model"],
                embedding_url=embedding_cfg["base_url"],
                collection_name=collection_name,
                persist_dir=memory_cfg.get("path", "./chroma_data"),
                llm=llm,
            )
            logger.info("[记忆系统] ChromaMemoryStore 已初始化")
        except Exception:
            logger.warning("[记忆系统] 初始化失败，降级为无记忆模式", exc_info=True)
            memory_store = None

    if memory_store is not None:
        agent_cfg_buf = raw.get("agents", {})
        conversation_buffer = ConversationBuffer(
            max_rounds=agent_cfg_buf.get("max_conversation_rounds", 10),
            max_tokens=agent_cfg_buf.get("max_conversation_tokens", 4096),
        )

    # 5.5 Tool Categories
    from src.tools.categories import load_categories, CategoryResolver

    categories_path = raw.get("tools", {}).get(
        "categories_path", "tool_categories.json"
    )
    category_resolver = None
    category_summaries: list[dict[str, str]] = []

    categories = load_categories(categories_path)
    if categories:
        category_resolver = CategoryResolver(categories)
        category_summaries = category_resolver.get_all_summaries()
        logger.info("[工具分类] 加载 %d 个类别", len(categories))
    else:
        logger.info("[工具分类] 未找到分类配置，跳过")

    # 6. Agents
    agent_cfg = raw.get("agents", {})
    agent_registry = AgentRegistry()
    if category_resolver:
        agent_registry.set_category_resolver(category_resolver)

    runner = AgentRunner(
        max_tool_rounds=agent_cfg.get("max_tool_rounds", 10),
    )
    graph = build_default_graph(
        agent_registry,
        category_summaries=category_summaries,
    )
    engine = GraphEngine(max_handoff_depth=agent_cfg.get("max_handoffs", 10))

    # 7. Deps
    deps = AgentDeps(
        llm=llm,
        tool_router=tool_router,
        agent_registry=agent_registry,
        graph_engine=engine,
        ui=ui,
        memory=memory_store,
        runner=runner,
    )

    # 7.5 Delegate Tool Provider
    from src.tools.delegate import DelegateToolProvider

    if category_resolver:
        delegate_provider = DelegateToolProvider(
            resolver=category_resolver,
            mcp_manager=mcp_manager,
        )
        tool_router.add_provider(delegate_provider)

    return AgentApp(
        deps=deps,
        ui=ui,
        guardrail=InputGuardrail(),
        tool_router=tool_router,
        agent_registry=agent_registry,
        engine=engine,
        graph=graph,
        skill_manager=skill_manager,
        mcp_manager=mcp_manager,
        conversation_buffer=conversation_buffer,
        category_summaries=category_summaries,
        category_resolver=category_resolver,
    )
```

- [ ] **Step 2: Run all tests**

Run: `uv run pytest tests/ -v --timeout=30`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/app/bootstrap.py
git commit -m "refactor(bootstrap): wire runner in deps, simplify DelegateToolProvider construction, pass category_resolver to AgentApp"
```

---

### Task 11: Integration test — skill mode delegation end-to-end

**Files:**
- Create: `tests/app/test_skill_delegation.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/app/test_skill_delegation.py
"""Skill 模式下跨分类委派集成测试。

验证 skill 图中：
1. orchestrator → handoff → category agent 可达
2. category agent A → delegate → category agent B 可达
3. delegate_depth 限制正常工作
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.context import RunContext, DynamicState
from src.agents.deps import AgentDeps
from src.agents.registry import AgentRegistry
from src.agents.runner import AgentRunner
from src.graph import GraphEngine
from src.tools.categories import CategoryResolver
from src.tools.delegate import DelegateToolProvider
from src.tools.router import ToolRouter
from src.app.presets import build_skill_graph
from src.llm.types import LLMResponse


@pytest.fixture
def categories():
    return {
        "tool_terminal": {"description": "终端操作", "tools": {"exec_cmd": "Execute command"}},
        "tool_calc": {"description": "数学计算", "tools": {"calculate": "Calculate math"}},
    }


@pytest.fixture
def resolver(categories):
    return CategoryResolver(categories)


@pytest.mark.asyncio
async def test_skill_graph_has_category_agent_nodes(resolver):
    """skill 图应包含 category agent 节点。"""
    skill_registry = AgentRegistry()
    skill_registry.set_category_resolver(resolver)

    summaries = resolver.get_all_summaries()
    graph = build_skill_graph(
        skill_registry,
        "<skill_content>测试 skill 指令</skill_content>",
        category_summaries=summaries,
    )

    assert "orchestrator" in graph.nodes
    assert "tool_terminal" in graph.nodes
    assert "tool_calc" in graph.nodes


@pytest.mark.asyncio
async def test_skill_mode_handoff_to_category_agent(resolver):
    """skill 模式下 orchestrator 应能 handoff 到 category agent。"""
    mock_llm = AsyncMock()
    call_count = 0

    async def mock_chat(messages, tools=None, silent=True):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # orchestrator handoff 到 tool_terminal
            return LLMResponse(
                content="",
                tool_calls={0: {
                    "id": "call_1",
                    "name": "transfer_to_tool_terminal",
                    "arguments": json.dumps({"task": "执行终端命令"}),
                }},
            )
        # tool_terminal 直接返回结果
        return LLMResponse(content="命令执行完毕", tool_calls={})

    mock_llm.chat = mock_chat

    skill_registry = AgentRegistry()
    skill_registry.set_category_resolver(resolver)
    runner = AgentRunner()
    router = ToolRouter()

    delegate_provider = DelegateToolProvider(resolver=resolver)
    router.add_provider(delegate_provider)

    summaries = resolver.get_all_summaries()
    graph = build_skill_graph(
        skill_registry,
        "<skill_content>测试 skill</skill_content>",
        category_summaries=summaries,
    )
    engine = GraphEngine()

    ctx = RunContext(
        input="执行终端任务",
        state=DynamicState(),
        deps=AgentDeps(
            llm=mock_llm,
            tool_router=router,
            agent_registry=skill_registry,
            graph_engine=engine,
            runner=runner,
        ),
    )
    result = await engine.run(graph, ctx)

    assert "命令执行完毕" in str(result.output)
    assert call_count == 2  # orchestrator + tool_terminal


@pytest.mark.asyncio
async def test_skill_mode_cross_agent_delegation(resolver):
    """skill 模式下 category agent A 应能 delegate 给 category agent B。"""
    mock_llm = AsyncMock()
    call_count = 0

    async def mock_chat(messages, tools=None, silent=True):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # orchestrator handoff 到 tool_terminal
            return LLMResponse(
                content="",
                tool_calls={0: {
                    "id": "call_1",
                    "name": "transfer_to_tool_terminal",
                    "arguments": json.dumps({"task": "执行计算"}),
                }},
            )
        if call_count == 2:
            # tool_terminal delegate 给 tool_calc
            return LLMResponse(
                content="",
                tool_calls={0: {
                    "id": "call_2",
                    "name": "delegate_tool_calc",
                    "arguments": json.dumps({
                        "objective": "执行计算",
                        "task": "计算 1+1",
                    }),
                }},
            )
        if call_count == 3:
            # tool_calc 返回结果
            return LLMResponse(content="已完成\n计算结果是 2", tool_calls={})
        # tool_terminal 收到 delegate 结果后输出
        return LLMResponse(content="计算完毕，结果是 2", tool_calls={})

    mock_llm.chat = mock_chat

    skill_registry = AgentRegistry()
    skill_registry.set_category_resolver(resolver)
    runner = AgentRunner()
    router = ToolRouter()

    delegate_provider = DelegateToolProvider(resolver=resolver)
    router.add_provider(delegate_provider)

    summaries = resolver.get_all_summaries()
    graph = build_skill_graph(
        skill_registry,
        "<skill_content>测试 skill</skill_content>",
        category_summaries=summaries,
    )
    engine = GraphEngine()

    ctx = RunContext(
        input="帮我计算",
        state=DynamicState(),
        deps=AgentDeps(
            llm=mock_llm,
            tool_router=router,
            agent_registry=skill_registry,
            graph_engine=engine,
            runner=runner,
        ),
    )
    result = await engine.run(graph, ctx)

    assert "2" in str(result.output)
    # orchestrator(1) + terminal calls delegate(2) + calc responds(3) + terminal responds(4)
    assert call_count == 4
```

- [ ] **Step 2: Run integration tests**

Run: `uv run pytest tests/app/test_skill_delegation.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite to verify no regressions**

Run: `uv run pytest tests/ -v --timeout=30`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/app/test_skill_delegation.py
git commit -m "test: add integration tests for skill mode handoff and cross-agent delegation"
```

---

### Task 12: Update architecture documentation

**Files:**
- Modify: `docs/architecture/tools.md`
- Modify: `docs/architecture/agents.md`

- [ ] **Step 1: Update tools.md — reflect context-aware tool chain**

在 `docs/architecture/tools.md` 中更新调用链图和 DelegateToolProvider 说明：
- `ToolProvider.execute()` 现在接收 `context` 参数
- `DelegateToolProvider` 构造函数只需 `resolver` 和 `mcp_manager`
- `ToolRouter.set_delegate_depth()` 已删除，depth 通过 context 传递
- 调用链更新为 `route(name, args, context)` → `execute(name, args, context)`

- [ ] **Step 2: Update agents.md — reflect stateless AgentRunner and AgentNode**

在 `docs/architecture/agents.md` 中更新：
- `AgentRunner` 不再持有 `registry`，从 `context.deps` 获取
- `AgentNode` 不再持有 `runner`，从 `context.deps.runner` 获取
- `AgentDeps` 新增 `runner` 字段

- [ ] **Step 3: Commit**

```bash
git add docs/architecture/tools.md docs/architecture/agents.md
git commit -m "docs: update architecture docs for context-aware tool chain"
```
