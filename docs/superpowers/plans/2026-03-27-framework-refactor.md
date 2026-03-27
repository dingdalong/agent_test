# Framework Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the entire project: extract graph engine, create LLM abstraction, unify guardrails, add pluggable memory, create config system, and clean up module boundaries.

**Architecture:** Ports-and-adapters with Protocol-based interfaces for all pluggable components (LLM, Memory, Embedding). Graph engine independent of agents. Centralized config.yaml with per-module defaults. Application layer (src/app/) as the only place that knows concrete implementations.

**Tech Stack:** Python 3.13, asyncio, OpenAI SDK, Pydantic v2, ChromaDB, Protocol (typing), PyYAML

**Spec:** `docs/superpowers/specs/2026-03-27-framework-refactor-design.md`

---

### Task 1: Create `src/llm/` module — Protocol, types, and OpenAI implementation

**Files:**
- Create: `src/llm/__init__.py`
- Create: `src/llm/base.py`
- Create: `src/llm/types.py`
- Create: `src/llm/openai.py`
- Create: `src/llm/structured.py`
- Create: `tests/llm/__init__.py`
- Create: `tests/llm/test_types.py`
- Create: `tests/llm/test_openai.py`
- Create: `tests/llm/test_structured.py`

- [ ] **Step 1: Create `src/llm/__init__.py`**

```python
from src.llm.base import LLMProvider
from src.llm.types import LLMResponse, StreamChunk, ToolCallData
from src.llm.structured import build_output_schema, parse_output

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "StreamChunk",
    "ToolCallData",
    "build_output_schema",
    "parse_output",
]
```

- [ ] **Step 2: Create `src/llm/types.py`**

```python
"""LLM 模块类型定义。"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolCallData:
    """单个工具调用数据。"""
    id: str
    name: str
    arguments: str


@dataclass
class LLMResponse:
    """非流式 LLM 响应。"""
    content: str
    tool_calls: dict[int, dict[str, str]] = field(default_factory=dict)
    finish_reason: Optional[str] = None


@dataclass
class StreamChunk:
    """流式响应的单个 chunk。"""
    content: str = ""
    tool_calls_delta: dict[int, dict[str, str]] = field(default_factory=dict)
    finish_reason: Optional[str] = None
```

- [ ] **Step 3: Create `src/llm/base.py`**

```python
"""LLMProvider Protocol — LLM 调用的抽象接口。"""

from typing import Protocol

from src.llm.types import LLMResponse


class LLMProvider(Protocol):
    """所有 LLM 实现必须满足的协议。

    消费者（AgentRunner, planner, extractor, buffer）依赖此协议，
    不关心底层是 OpenAI、Claude 还是本地模型。
    """

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 1.0,
        tool_choice: str | None = None,
        silent: bool = False,
    ) -> LLMResponse: ...
```

- [ ] **Step 4: Create `src/llm/openai.py`**

Refactored from `src/core/async_api.py`. Key changes:
- Class-based `OpenAIProvider` implementing `LLMProvider`
- Client, model, semaphore, retries are constructor params with defaults
- `agent_output` replaced by optional `on_chunk` callback
- No global imports from `config.py`

```python
"""OpenAI SDK 实现的 LLM Provider。"""

import asyncio
import logging
import time
from typing import Callable, Awaitable, Optional

from openai import AsyncOpenAI, APIConnectionError, RateLimitError, APIError

from src.llm.types import LLMResponse

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """基于 OpenAI SDK 的 LLM Provider。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        concurrency: int = 5,
        max_retries: int = 3,
        timeout: float = 120.0,
        on_chunk: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self.model = model
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(concurrency)
        self._on_chunk = on_chunk
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=2,
        )

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 1.0,
        tool_choice: str | None = None,
        silent: bool = False,
    ) -> LLMResponse:
        """流式调用 LLM，返回完整响应。"""
        async with self._semaphore:
            for attempt in range(self.max_retries):
                try:
                    response = await self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools,
                        stream=True,
                        temperature=temperature,
                        tool_choice=tool_choice or ("auto" if tools else None),
                    )
                    return await self._parse_stream(response, silent=silent)

                except (APIConnectionError, RateLimitError, asyncio.TimeoutError) as e:
                    if attempt == self.max_retries - 1:
                        raise
                    wait_time = 2 ** attempt
                    logger.warning(f"API错误 ({type(e).__name__})，{wait_time}秒后重试...")
                    await asyncio.sleep(wait_time)

                except APIError:
                    raise

    async def _parse_stream(
        self, stream, silent: bool = False,
    ) -> LLMResponse:
        """解析流式响应。"""
        tool_calls: dict[int, dict[str, str]] = {}
        content_parts: list[str] = []
        finish_reason = None

        async for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.content:
                if not (delta.tool_calls and delta.content.isspace()):
                    content_parts.append(delta.content)
                    if not silent and self._on_chunk:
                        await self._on_chunk(delta.content)

            if delta.tool_calls:
                for tool_chunk in delta.tool_calls:
                    idx = tool_chunk.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                    if tool_chunk.id:
                        tool_calls[idx]["id"] = tool_chunk.id
                    if tool_chunk.function.name:
                        tool_calls[idx]["name"] += tool_chunk.function.name
                    if tool_chunk.function.arguments:
                        tool_calls[idx]["arguments"] += tool_chunk.function.arguments

            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        content = "".join(content_parts)
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )
```

- [ ] **Step 5: Create `src/llm/structured.py`**

Moved from `src/core/structured_output.py` — no code changes, only import path updates.

```python
"""structured_output — 结构化输出。

利用 function calling 机制约束 LLM 按 Pydantic 模型输出结构化 JSON。
"""

import json
import logging
from typing import Dict, Optional, Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


def build_output_schema(name: str, description: str, model: Type[BaseModel]) -> dict:
    """从 Pydantic 模型构建结构化输出的 tool schema。"""
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema.pop("description", None)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }


def parse_output(
    tool_calls: Dict[int, Dict[str, str]],
    name: str,
    model: Type[BaseModel],
) -> Optional[BaseModel]:
    """从 LLM 的 tool_calls 中解析结构化输出。"""
    for tc in tool_calls.values():
        if tc.get("name") == name:
            try:
                data = json.loads(tc["arguments"])
                return model(**data)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"结构化输出 '{name}' 解析失败: {e}")
                return None
    return None
```

- [ ] **Step 6: Write tests for `src/llm/`**

Create `tests/llm/__init__.py` (empty) and `tests/llm/test_structured.py`:

```python
"""Tests for src.llm.structured — migrated from tests/core/test_async_api.py."""

from pydantic import BaseModel
from src.llm.structured import build_output_schema, parse_output


class SampleOutput(BaseModel):
    score: float
    label: str


class TestBuildOutputSchema:
    def test_generates_valid_schema(self):
        schema = build_output_schema("test", "desc", SampleOutput)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test"
        assert "properties" in schema["function"]["parameters"]

    def test_strips_title_and_description(self):
        schema = build_output_schema("test", "desc", SampleOutput)
        params = schema["function"]["parameters"]
        assert "title" not in params
        assert "description" not in params


class TestParseOutput:
    def test_parses_matching_tool_call(self):
        tool_calls = {0: {"name": "test", "arguments": '{"score": 0.9, "label": "good"}'}}
        result = parse_output(tool_calls, "test", SampleOutput)
        assert result is not None
        assert result.score == 0.9
        assert result.label == "good"

    def test_returns_none_for_no_match(self):
        tool_calls = {0: {"name": "other", "arguments": "{}"}}
        assert parse_output(tool_calls, "test", SampleOutput) is None

    def test_returns_none_for_invalid_json(self):
        tool_calls = {0: {"name": "test", "arguments": "not json"}}
        assert parse_output(tool_calls, "test", SampleOutput) is None

    def test_returns_none_for_validation_error(self):
        tool_calls = {0: {"name": "test", "arguments": '{"wrong": "field"}'}}
        assert parse_output(tool_calls, "test", SampleOutput) is None
```

- [ ] **Step 7: Run tests to verify**

Run: `pytest tests/llm/ -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/llm/ tests/llm/
git commit -m "feat: create src/llm/ module with LLMProvider protocol and OpenAI implementation"
```

---

### Task 2: Create `src/graph/` module — extract from `src/agents/graph/`

**Files:**
- Create: `src/graph/__init__.py`
- Create: `src/graph/types.py`
- Create: `src/graph/builder.py`
- Create: `src/graph/engine.py`
- Create: `src/graph/hooks.py`
- Create: `tests/graph/__init__.py`
- Create: `tests/graph/test_types.py`
- Create: `tests/graph/test_builder.py`
- Create: `tests/graph/test_engine.py`

- [ ] **Step 1: Create `src/graph/__init__.py`**

```python
from src.graph.types import (
    GraphNode,
    FunctionNode,
    NodeResult,
    Edge,
    ParallelGroup,
    CompiledGraph,
)
from src.graph.builder import GraphBuilder
from src.graph.engine import GraphEngine, GraphResult
from src.graph.hooks import GraphHooks

__all__ = [
    "GraphNode",
    "FunctionNode",
    "NodeResult",
    "Edge",
    "ParallelGroup",
    "CompiledGraph",
    "GraphBuilder",
    "GraphEngine",
    "GraphResult",
    "GraphHooks",
]
```

- [ ] **Step 2: Create `src/graph/types.py`**

Moved from `src/agents/graph/types.py` with these changes:
- **Remove** `AgentNode` (stays in `src/agents/node.py`, Task 6)
- Remove import of `HandoffRequest` — `NodeResult.handoff` becomes `Any`

```python
"""图类型定义 — 节点、边、执行结果、编译后的图。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Protocol, runtime_checkable


@dataclass
class NodeResult:
    """节点执行结果。"""
    output: Any
    next: Optional[str | list[str]] = None
    handoff: Any = None


@runtime_checkable
class GraphNode(Protocol):
    """图节点协议。"""
    name: str

    async def execute(self, context: Any) -> NodeResult: ...


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

- [ ] **Step 3: Create `src/graph/hooks.py`**

Extracted from `src/agents/hooks.py` — only `GraphHooks` class.

```python
"""GraphHooks — 图级生命周期钩子。"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional


class GraphHooks:
    """图级钩子。所有钩子均为可选，未设置时调用为 no-op。"""

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

- [ ] **Step 4: Create `src/graph/builder.py`**

Moved from `src/agents/graph/builder.py`. Changes:
- Imports from `src.graph.types` instead of `src.agents.graph.types`
- **Remove** `add_agent()` method (it depends on `Agent` + `AgentNode`). `add_agent()` will be provided by a helper in `src/agents/` or `src/app/presets.py`.
- Keep `add_function()`, `set_entry()`, `add_edge()`, `add_parallel()`, `compile()`
- Add `add_node()` for directly adding any `GraphNode`

```python
"""GraphBuilder — 声明式图构建器。"""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from src.graph.types import (
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

    def add_node(self, node: GraphNode) -> GraphBuilder:
        """添加任意 GraphNode 实现。"""
        self._nodes[node.name] = node
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
        if self._entry is None:
            raise ValueError("Graph has no entry node. Call set_entry() before compile().")
        if self._entry not in self._nodes:
            raise ValueError(f"Entry node '{self._entry}' not found in registered nodes.")

        for edge in self._edges:
            if edge.source not in self._nodes:
                raise ValueError(f"Edge source '{edge.source}' not found in registered nodes.")
            if edge.target not in self._nodes:
                raise ValueError(f"Edge target '{edge.target}' not found in registered nodes.")

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

- [ ] **Step 5: Create `src/graph/engine.py`**

Moved from `src/agents/graph/engine.py`. Key changes:
- Import from `src.graph.types`, `src.graph.hooks`
- Remove import of `AgentRunner`, `AgentRegistry`, `AgentNode`, `Agent`
- `GraphEngine.__init__` no longer creates `AgentRunner` — the engine is agent-agnostic
- Nodes must have runner pre-assigned before being passed to the engine
- Remove `AgentNode`-specific runner injection from `run()`

```python
"""GraphEngine — 异步图执行器（Agent 无关）。"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from src.graph.types import CompiledGraph, GraphNode, NodeResult, ParallelGroup
from src.graph.hooks import GraphHooks

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT", bound=BaseModel)


@dataclass
class GraphResult(Generic[StateT]):
    """图执行的最终结果。"""
    output: Any
    state: StateT
    trace: list = field(default_factory=list)


class GraphEngine:
    """异步图执行器。Agent 无关 — 只负责图的遍历和执行。"""

    def __init__(
        self,
        hooks: GraphHooks | None = None,
        max_handoff_depth: int = 10,
    ):
        self.hooks = hooks or GraphHooks()
        self.max_handoff_depth = max_handoff_depth

    async def run(self, graph: CompiledGraph, context: Any) -> GraphResult:
        """执行编译后的图。"""
        await self.hooks.on_graph_start(context)

        last_output: Any = None
        pending: list[str] = [graph.entry]
        visited: set[str] = set()

        while pending:
            parallel_group = self._find_parallel_group(pending, graph.parallel_groups)

            if parallel_group:
                nodes_to_run = parallel_group.nodes
                results = await self._run_parallel(nodes_to_run, graph, context)
                for name, node_result in results.items():
                    last_output = node_result.output
                    self._write_state(context, name, node_result.output)
                    visited.add(name)
                pending = [parallel_group.then]
            else:
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
                        context.input = node_result.handoff.task
                        pending = [target]
                        continue
                    else:
                        # Handoff to unknown node — let the caller handle it
                        # via on_handoff_miss hook or by pre-registering nodes
                        logger.error(f"Handoff target '{target}' not found in graph")
                        self._add_trace(context, current_name, "error",
                                        {"error": f"Handoff target '{target}' not found"})
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
            trace=list(getattr(context, "trace", [])),
        )
        await self.hooks.on_graph_end(context, result)
        return result

    async def _execute_node(self, node: GraphNode, context: Any) -> NodeResult:
        await self.hooks.on_node_start(node.name, context)
        self._add_trace(context, node.name, "start")
        try:
            result = await node.execute(context)
        except Exception as e:
            self._add_trace(context, node.name, "error", {"error": str(e)})
            raise
        self._add_trace(context, node.name, "end")
        await self.hooks.on_node_end(node.name, context, result)
        return result

    async def _run_parallel(
        self, node_names: list[str], graph: CompiledGraph, context: Any,
    ) -> dict[str, NodeResult]:
        async def _run_one(name: str) -> tuple[str, NodeResult]:
            node = graph.nodes[name]
            result = await self._execute_node(node, context)
            return name, result

        tasks = [_run_one(name) for name in node_names]
        results = await asyncio.gather(*tasks)
        return dict(results)

    def _resolve_edges(
        self, source: str, graph: CompiledGraph, context: Any,
    ) -> list[str]:
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
        pending_set = set(pending)
        for group in groups:
            if pending_set & set(group.nodes):
                return group
        return None

    def _write_state(self, context: Any, node_name: str, output: Any) -> None:
        try:
            setattr(context.state, node_name, output)
        except (AttributeError, ValueError):
            logger.debug(f"Cannot set state.{node_name}, state type may not support it")

    def _add_trace(self, context: Any, node: str, event: str, data: dict | None = None) -> None:
        trace = getattr(context, "trace", None)
        if trace is not None and hasattr(trace, "append"):
            trace.append({
                "node": node, "event": event, "timestamp": time.time(), "data": data or {},
            })
```

- [ ] **Step 6: Write tests for `src/graph/`**

Copy `tests/agents/graph/test_types.py`, `test_builder.py`, `test_engine.py` to `tests/graph/`, updating imports:
- `from src.agents.graph.types import ...` → `from src.graph.types import ...`
- `from src.agents.graph.builder import ...` → `from src.graph.builder import ...`
- `from src.agents.graph.engine import ...` → `from src.graph.engine import ...`
- `from src.agents.hooks import GraphHooks` → `from src.graph.hooks import GraphHooks`
- Remove any `AgentNode`-specific tests from `test_types.py` (those move to `tests/agents/test_node.py` in Task 6)
- In `test_builder.py`, replace `builder.add_agent()` calls with `builder.add_node()`
- In `test_engine.py`, remove `AgentRunner`/`AgentRegistry` dependencies — test with `FunctionNode` only

- [ ] **Step 7: Run tests**

Run: `pytest tests/graph/ -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/graph/ tests/graph/
git commit -m "feat: create src/graph/ as independent graph execution engine"
```

---

### Task 3: Create `src/guardrails/` module

**Files:**
- Create: `src/guardrails/__init__.py`
- Create: `src/guardrails/base.py`
- Create: `src/guardrails/input.py`
- Create: `src/guardrails/output.py`
- Create: `src/guardrails/runner.py`
- Create: `tests/guardrails/__init__.py`
- Create: `tests/guardrails/test_guardrails.py`

- [ ] **Step 1: Create `src/guardrails/base.py`**

```python
"""Guardrail Protocol — 护栏抽象接口。"""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Protocol


@dataclass
class GuardrailResult:
    """护栏检查结果。"""
    passed: bool
    message: str = ""
    action: str = "block"  # "block" | "warn" | "rewrite"


@dataclass
class Guardrail:
    """护栏定义：名称 + 异步检查函数。"""
    name: str
    check: Callable[..., Awaitable[GuardrailResult]]
```

- [ ] **Step 2: Create `src/guardrails/runner.py`**

From `src/agents/guardrails.py`:

```python
"""Guardrail 运行器。"""

from typing import Any, Optional

from src.guardrails.base import Guardrail, GuardrailResult


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

- [ ] **Step 3: Create `src/guardrails/input.py`**

From `src/core/guardrails.py` `InputGuardrail`:

```python
"""InputGuardrail — 输入安全检查。"""

import re
from typing import Tuple


class InputGuardrail:
    """输入安全检查（关键词+正则）"""

    def __init__(self, blocked_patterns: list[str] | None = None):
        self.blocked_patterns = blocked_patterns or [
            r"忽略.*指令|忽略.*系统提示",
            r"删除.*文件|rm\s+-rf",
            r"DROP\s+TABLE",
            r"eval\s*\(",
            r"exec\s*\(",
        ]

    def check(self, user_input: str) -> Tuple[bool, str]:
        """返回 (是否通过, 拒绝理由)"""
        for pattern in self.blocked_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return False, f"输入包含不安全内容（匹配模式：{pattern}）"
        return True, ""
```

- [ ] **Step 4: Create `src/guardrails/output.py`**

From `src/core/guardrails.py` `OutputGuardrail`:

```python
"""OutputGuardrail — 输出安全检查。"""

from typing import Tuple


class OutputGuardrail:
    """输出安全检查"""

    def __init__(self, blocked_content: list[str] | None = None):
        self.blocked_content = blocked_content or [
            "rm -rf",
            "DROP TABLE",
            "eval(",
        ]

    def check(self, output: str) -> Tuple[bool, str]:
        for phrase in self.blocked_content:
            if phrase in output:
                return False, f"输出包含不安全内容：{phrase}"
        return True, ""
```

- [ ] **Step 5: Create `src/guardrails/__init__.py`**

```python
from src.guardrails.base import Guardrail, GuardrailResult
from src.guardrails.runner import run_guardrails
from src.guardrails.input import InputGuardrail
from src.guardrails.output import OutputGuardrail

__all__ = [
    "Guardrail",
    "GuardrailResult",
    "run_guardrails",
    "InputGuardrail",
    "OutputGuardrail",
]
```

- [ ] **Step 6: Write tests**

Combine tests from `tests/agents/test_guardrails.py` (for `Guardrail`, `run_guardrails`) and add tests for `InputGuardrail`, `OutputGuardrail` in `tests/guardrails/test_guardrails.py`. Update imports to `src.guardrails`.

- [ ] **Step 7: Run tests and commit**

Run: `pytest tests/guardrails/ -v`

```bash
git add src/guardrails/ tests/guardrails/
git commit -m "feat: create src/guardrails/ as independent module"
```

---

### Task 4: Move `performance.py` to `src/utils/`

**Files:**
- Create: `src/utils/performance.py`
- Modify: `src/utils/__init__.py` (if exists)

- [ ] **Step 1: Copy `src/core/performance.py` → `src/utils/performance.py`**

No code changes needed — the file has no imports from `src/core/`.

- [ ] **Step 2: Commit**

```bash
git add src/utils/performance.py
git commit -m "refactor: move performance.py from core/ to utils/"
```

---

### Task 5: Restructure `src/memory/` — add Protocol, move chroma to subdirectory

**Files:**
- Create: `src/memory/base.py`
- Create: `src/memory/chroma/__init__.py`
- Create: `src/memory/chroma/store.py`
- Create: `src/memory/chroma/embeddings.py`
- Create: `src/memory/chroma/utils.py`
- Modify: `src/memory/__init__.py`
- Modify: `src/memory/extractor.py` — depend on `LLMProvider` instead of `call_model`
- Modify: `src/memory/buffer.py` — depend on `LLMProvider` instead of `call_model`
- Modify: `src/memory/store.py` → `src/memory/chroma/store.py`

- [ ] **Step 1: Create `src/memory/base.py`**

```python
"""MemoryProvider Protocol — 记忆存储的抽象接口。"""

from typing import Protocol

from src.memory.types import MemoryRecord, MemoryType


class MemoryProvider(Protocol):
    """所有记忆存储实现必须满足的协议。"""

    def add(self, record: MemoryRecord) -> str: ...
    def search(
        self, query: str, n: int = 5,
        memory_type: MemoryType | None = None,
        type_tag: str | None = None,
    ) -> list[MemoryRecord]: ...
    def cleanup(self, min_importance: float = 0.1) -> int: ...
    def recalculate_importance(self) -> None: ...
```

- [ ] **Step 2: Create `src/memory/chroma/` directory**

Move `src/memory/store.py` → `src/memory/chroma/store.py`, with changes:
- Class renamed to `ChromaMemoryStore`
- Import `EmbeddingClient` from `.embeddings` (relative)
- Import `FactExtractor` from `src.memory.extractor`
- Import performance from `src.utils.performance`
- Constructor takes `embedding_model`, `embedding_url` as explicit params (no `os.getenv`)

Move `src/memory/embeddings.py` → `src/memory/chroma/embeddings.py` (no changes).
Move `src/memory/utils.py` → `src/memory/chroma/utils.py` (no changes).
Create `src/memory/chroma/__init__.py`:

```python
from src.memory.chroma.store import ChromaMemoryStore

__all__ = ["ChromaMemoryStore"]
```

- [ ] **Step 3: Update `src/memory/extractor.py`**

Change `from src.core.async_api import call_model` to accept `LLMProvider` via constructor:

```python
class FactExtractor:
    def __init__(self, llm: "LLMProvider"):
        self._llm = llm
        # ... existing init code ...

    async def extract(self, ...):
        # Replace: content, tool_calls, _ = await call_model(...)
        # With:    response = await self._llm.chat(...)
        #          content, tool_calls = response.content, response.tool_calls
```

- [ ] **Step 4: Update `src/memory/buffer.py`**

Change `summarize_conversation` to accept `llm` parameter:

```python
async def summarize_conversation(messages: list[dict], llm: "LLMProvider") -> str:
    # Replace: response, _, _ = await call_model(...)
    # With:    response = await llm.chat(...)
    #          return response.content
```

`ConversationBuffer.compress()` also needs `llm` param.

- [ ] **Step 5: Update tests**

Update `tests/memory/conftest.py` to mock `ChromaMemoryStore` instead of `MemoryStore`.
Update all `tests/memory/test_store.py` imports.

- [ ] **Step 6: Run tests and commit**

Run: `pytest tests/memory/ -v`

```bash
git add src/memory/ tests/memory/
git commit -m "feat: add MemoryProvider protocol, move ChromaDB impl to memory/chroma/"
```

---

### Task 6: Clean up `src/agents/` — create `node.py`, update deps, remove old graph/

**Files:**
- Create: `src/agents/node.py`
- Modify: `src/agents/hooks.py` — remove `GraphHooks`
- Modify: `src/agents/deps.py` — add proper types
- Modify: `src/agents/runner.py` — use `LLMProvider` instead of `call_model`
- Modify: `src/agents/context.py` — remove `EmptyDeps`
- Modify: `src/agents/__init__.py` — update exports
- Delete: `src/agents/graph/` (entire directory)
- Delete: `src/agents/guardrails.py`
- Delete: `src/agents/definitions.py`

- [ ] **Step 1: Create `src/agents/node.py`**

Extracted from `src/agents/graph/types.py`:

```python
"""AgentNode — 将 Agent 适配为 GraphNode。"""

from __future__ import annotations

from typing import Any

from src.graph.types import NodeResult


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
```

- [ ] **Step 2: Update `src/agents/hooks.py`**

Remove `GraphHooks` class, keep only `AgentHooks`:

```python
"""AgentHooks — Agent 级生命周期钩子。"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional


class AgentHooks:
    """Agent 级钩子。所有钩子均为可选，未设置时调用为 no-op。"""

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
```

- [ ] **Step 3: Update `src/agents/deps.py`**

Replace `Any` with Protocol types:

```python
"""AgentDeps — Agent 运行时外部依赖模型。"""

from typing import Any, Optional

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
```

Note: We keep `Any` type annotations because Pydantic + Protocol has limitations, but the field names and docstring document the expected types. The `llm` and `memory` fields are new additions.

- [ ] **Step 4: Update `src/agents/runner.py`**

Replace `call_model` with `LLMProvider` from `context.deps.llm`:

```python
# Remove these imports:
# from src.core.async_api import call_model
# from src.core.structured_output import build_output_schema, parse_output

# Add:
from src.llm.structured import build_output_schema, parse_output

# In AgentRunner.__init__, remove registry param (get it from context.deps)
# In run(), replace:
#   content, tool_calls, _ = await call_model(messages, tools=all_tools, silent=True)
# With:
#   llm = context.deps.llm
#   response = await llm.chat(messages, tools=all_tools, silent=True)
#   content, tool_calls = response.content, response.tool_calls

# Replace guardrails import:
# from src.agents.guardrails import run_guardrails
# With:
from src.guardrails import run_guardrails
```

Full runner changes: every `await call_model(...)` becomes `await context.deps.llm.chat(...)`, and the return value is `LLMResponse` instead of a tuple.

- [ ] **Step 5: Remove `src/agents/context.py` `EmptyDeps`**

Just delete the `EmptyDeps` class. Keep `DictState`, `RunContext`, `TraceEvent`.

- [ ] **Step 6: Delete old files**

```bash
rm -rf src/agents/graph/
rm src/agents/guardrails.py
rm src/agents/definitions.py
```

- [ ] **Step 7: Update `src/agents/__init__.py`**

```python
from src.agents.agent import Agent, AgentResult, HandoffRequest
from src.agents.node import AgentNode
from src.agents.runner import AgentRunner
from src.agents.context import RunContext, TraceEvent, DictState
from src.agents.deps import AgentDeps
from src.agents.registry import AgentRegistry
from src.agents.hooks import AgentHooks

__all__ = [
    "Agent", "AgentResult", "HandoffRequest",
    "AgentNode", "AgentRunner",
    "RunContext", "TraceEvent", "DictState",
    "AgentDeps", "AgentRegistry", "AgentHooks",
]
```

- [ ] **Step 8: Update agent tests**

- `tests/agents/test_runner.py` — mock `context.deps.llm.chat` instead of `call_model`
- `tests/agents/test_hooks.py` — remove `GraphHooks` tests (they're in `tests/graph/`)
- `tests/agents/test_guardrails.py` — delete (moved to `tests/guardrails/`)
- `tests/agents/test_context.py` — remove `EmptyDeps` test
- Delete `tests/agents/graph/` directory
- Delete `tests/agents/test_definitions.py` (will be recreated as `tests/app/test_presets.py`)

- [ ] **Step 9: Run tests and commit**

Run: `pytest tests/agents/ tests/graph/ tests/guardrails/ -v`

```bash
git add -A src/agents/ tests/agents/ tests/graph/ tests/guardrails/
git commit -m "refactor: clean up agents/ — extract node.py, remove graph/, use LLMProvider"
```

---

### Task 7: Update `src/plan/` — change dependencies

**Files:**
- Modify: `src/plan/planner.py` — use `LLMProvider` instead of `call_model`
- Modify: `src/plan/compiler.py` — import from `src.graph` instead of `src.agents.graph`
- Modify: `src/plan/flow.py` — import from `src.graph`
- Modify: `src/plan/exceptions.py` — remove `config` import

- [ ] **Step 1: Update `src/plan/planner.py`**

Replace all `call_model` calls with `llm.chat()`. The module-level functions now need an `llm` parameter:

```python
# Remove:
# from src.core.async_api import call_model
# from src.core.structured_output import build_output_schema, parse_output

# Add:
from src.llm.structured import build_output_schema, parse_output
from src.llm.base import LLMProvider

# Each function gets llm as first param:
async def classify_user_feedback(user_feedback: str, plan: Plan, llm: LLMProvider) -> str:
    ...
    response = await llm.chat([...], temperature=0, tools=[...], silent=True)
    ...

async def check_clarification_needed(user_input: str, gathered_info: str, llm: LLMProvider) -> Optional[str]:
    ...
    response = await llm.chat([...], temperature=0, silent=True)
    ...

async def generate_plan(user_input, available_tools, available_agents, context, llm: LLMProvider) -> Optional[Plan]:
    ...
    response = await llm.chat([...], tools=plan_tools, silent=True)
    content, tool_calls = response.content, response.tool_calls
    ...

async def adjust_plan(original_request, current_plan, feedback, available_tools, available_agents, llm: LLMProvider) -> Plan:
    ...
```

- [ ] **Step 2: Update `src/plan/compiler.py`**

```python
# Replace:
# from src.agents.context import RunContext, DictState
# from src.agents.graph.types import FunctionNode, NodeResult, CompiledGraph, Edge, ParallelGroup
# from src.agents.graph.builder import GraphBuilder
# from src.agents.registry import AgentRegistry
# from src.agents.runner import AgentRunner

# With:
from src.agents.context import RunContext, DictState
from src.agents.registry import AgentRegistry
from src.agents.runner import AgentRunner
from src.graph.types import FunctionNode, NodeResult, CompiledGraph, Edge, ParallelGroup
from src.graph.builder import GraphBuilder
from src.tools.router import ToolRouter
```

- [ ] **Step 3: Update `src/plan/flow.py`**

```python
# Replace engine import:
# from src.agents.graph.engine import GraphEngine → from src.graph import GraphEngine
# PlanFlow.__init__ receives llm param
# Pass llm to all planner functions
```

- [ ] **Step 4: Update `src/plan/exceptions.py`**

Remove `from config import PLAN_MAX_RAW_RESPONSE_LENGTH`. Hardcode the constant:

```python
_MAX_RAW_RESPONSE_LENGTH = 500

class JSONParseError(PlanError):
    def __str__(self) -> str:
        base = super().__str__()
        if self.raw_response and len(self.raw_response) < _MAX_RAW_RESPONSE_LENGTH:
            return f"{base} (原始响应: {self.raw_response})"
        ...
```

- [ ] **Step 5: Update plan tests**

Update `tests/plan/test_planner.py`:
- Mock `llm.chat` instead of `call_model`
- Pass `llm` param to all planner functions

Update `tests/plan/test_compiler.py`:
- Import from `src.graph` instead of `src.agents.graph`

Update `tests/plan/test_flow.py`:
- Pass `llm` to `PlanFlow`

- [ ] **Step 6: Run tests and commit**

Run: `pytest tests/plan/ -v`

```bash
git add src/plan/ tests/plan/
git commit -m "refactor: update plan/ to use LLMProvider and src/graph/"
```

---

### Task 8: Update `src/tools/middleware.py` — remove `io.py` dependency

**Files:**
- Modify: `src/tools/middleware.py`

- [ ] **Step 1: Update `sensitive_confirm_middleware`**

Replace `from src.core.io import agent_output, agent_input` with `UserInterface` passed as parameter:

```python
def sensitive_confirm_middleware(registry: ToolRegistry, ui) -> Middleware:
    """敏感工具执行前需要用户确认。"""

    async def middleware(name: str, args: dict, next_fn: NextFn) -> str:
        entry = registry.get(name)
        if entry and entry.sensitive:
            if entry.confirm_template:
                try:
                    msg = entry.confirm_template.format(**args)
                except KeyError:
                    msg = f"执行敏感操作: {name}"
            else:
                msg = f"执行敏感操作: {name}"

            await ui.display(f"\n⚠️  是否允许{msg}？\n")
            confirmed = await ui.confirm("")
            if not confirmed:
                return "用户取消了操作"

        return await next_fn(name, args)

    return middleware
```

- [ ] **Step 2: Update tests and commit**

Update `tests/tools/test_architecture.py` if it tests middleware.

Run: `pytest tests/tools/ -v`

```bash
git add src/tools/middleware.py tests/tools/
git commit -m "refactor: replace io.py with UserInterface in middleware"
```

---

### Task 9: Create config system

**Files:**
- Create: `config.yaml`
- Rewrite: `src/config.py`

- [ ] **Step 1: Create `config.yaml`**

```yaml
# ===== 用户配置 =====
llm:
  provider: deepseek
  model: deepseek-chat
  base_url: https://api.deepseek.com/v1

embedding:
  provider: ollama
  model: qwen3-embedding:0.6b
  base_url: http://127.0.0.1:11434

memory:
  provider: chroma
  path: ./chroma_data

user:
  id: user_001

# ===== 开发者调优（全部可选）=====
# plan:
#   max_adjustments: 3
#   max_clarification_rounds: 3
# agents:
#   max_handoffs: 3
#   max_tool_rounds: 3
# llm:
#   concurrency: 5
#   max_retries: 3
# tools:
#   max_output_length: 2000
# mcp:
#   config_path: mcp_servers.json
# skills:
#   dirs:
#     - skills/
#     - .agents/skills/
```

- [ ] **Step 2: Rewrite `src/config.py`**

```python
"""配置加载器 — 读 config.yaml + .env，返回原始 dict。"""

import os

import yaml
from dotenv import load_dotenv
from pathlib import Path


def load_config(path: str = "config.yaml") -> dict:
    """加载配置文件，返回原始 dict。文件不存在返回空 dict。"""
    load_dotenv()
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    # .env 中的 secrets 合并到 config
    if "llm" not in config:
        config["llm"] = {}
    if not config["llm"].get("api_key"):
        config["llm"]["api_key"] = os.getenv("OPENAI_API_KEY", "")
    if not config["llm"].get("base_url"):
        config["llm"]["base_url"] = os.getenv("OPENAI_BASE_URL", "")
    if not config["llm"].get("model"):
        config["llm"]["model"] = os.getenv("OPENAI_MODEL", "")

    if "embedding" not in config:
        config["embedding"] = {}
    if not config["embedding"].get("model"):
        config["embedding"]["model"] = os.getenv("OPENAI_MODEL_EMBEDDING", "")
    if not config["embedding"].get("base_url"):
        config["embedding"]["base_url"] = os.getenv("OPENAI_MODEL_EMBEDDING_URL", "")

    if "user" not in config:
        config["user"] = {}
    if not config["user"].get("id"):
        config["user"]["id"] = os.getenv("USER_ID", "default_user")

    return config
```

- [ ] **Step 3: Commit**

```bash
git add config.yaml src/config.py
git commit -m "feat: create config.yaml + config loader, replacing root config.py"
```

---

### Task 10: Create `src/app/` module

**Files:**
- Create: `src/app/__init__.py`
- Create: `src/app/bootstrap.py`
- Create: `src/app/app.py`
- Create: `src/app/presets.py`

- [ ] **Step 1: Create `src/app/presets.py`**

Moved from `src/agents/definitions.py`, adapted to new imports:

```python
"""Agent 预设定义与图构建。"""

from __future__ import annotations

from src.agents.agent import Agent
from src.agents.node import AgentNode
from src.agents.registry import AgentRegistry
from src.agents.context import RunContext
from src.graph.types import NodeResult, CompiledGraph
from src.graph.builder import GraphBuilder


_ORCHESTRATOR_BASE_INSTRUCTIONS = (
    "你是一个智能助手。根据用户的请求选择合适的操作：\n"
    "- 天气相关问题，交给 weather_agent\n"
    "- 日历/日程相关问题，交给 calendar_agent\n"
    "- 邮件相关问题，交给 email_agent\n"
    "- 需要多步骤协作的复杂任务（如查天气然后发邮件），交给 planner\n"
    "- 其他问题，直接回答用户\n"
)

_SPECIALIST_AGENTS = [
    Agent(
        name="weather_agent",
        description="处理天气查询",
        instructions="你是天气助手。使用 get_weather 工具查询天气信息并回复用户。",
        tools=["get_weather"],
    ),
    Agent(
        name="calendar_agent",
        description="管理日历事件",
        instructions="你是日历助手。使用 create_event 工具帮用户管理日历事件。",
        tools=["create_event"],
    ),
    Agent(
        name="email_agent",
        description="发送邮件",
        instructions="你是邮件助手。使用 send_email 工具帮用户发送邮件。",
        tools=["send_email"],
    ),
]

_PLANNER_AGENT = Agent(
    name="planner",
    description="处理需要多步骤的复杂任务，生成计划并按步骤执行",
    instructions="",
)


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
    runner=None,
    skill_content: str | None = None,
) -> CompiledGraph:
    for agent in _SPECIALIST_AGENTS:
        registry.register(agent)

    instructions = _ORCHESTRATOR_BASE_INSTRUCTIONS
    if skill_content:
        instructions = f"{skill_content}\n\n{instructions}"

    orchestrator = Agent(
        name="orchestrator",
        description="总控 Agent，负责路由和直接回答",
        instructions=instructions,
        handoffs=["weather_agent", "calendar_agent", "email_agent", "planner"],
    )
    registry.register(orchestrator)
    registry.register(_PLANNER_AGENT)

    builder = GraphBuilder()
    builder.add_node(AgentNode(agent=orchestrator, runner=runner))
    builder.add_function("planner", _make_planner_node_fn())
    builder.set_entry("orchestrator")
    return builder.compile()


def build_default_graph(registry: AgentRegistry, runner=None) -> CompiledGraph:
    return _register_and_build(registry, runner=runner)


def build_skill_graph(registry: AgentRegistry, skill_content: str, runner=None) -> CompiledGraph:
    return _register_and_build(registry, runner=runner, skill_content=skill_content)
```

- [ ] **Step 2: Create `src/app/bootstrap.py`**

```python
"""应用组装 — 读配置、创建组件、注入依赖。"""

from __future__ import annotations

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
from src.agents import AgentRegistry, AgentRunner
from src.graph import GraphEngine
from src.agents.deps import AgentDeps
from src.app.presets import build_default_graph
from src.app.app import AgentApp


async def create_app(config_path: str = "config.yaml") -> AgentApp:
    """读配置 → 创建所有组件 → 注入依赖 → 返回 AgentApp。"""
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

    # 3. MCP
    mcp_config_path = raw.get("mcp", {}).get("config_path", "mcp_servers.json")
    mcp_manager = MCPManager()
    await mcp_manager.connect_all(load_mcp_config(mcp_config_path))
    if mcp_manager.get_tools_schemas():
        tool_router.add_provider(MCPToolProvider(mcp_manager))

    # 4. Skills
    skill_dirs = raw.get("skills", {}).get("dirs", ["skills/", ".agents/skills/"])
    skill_manager = SkillManager(skill_dirs=skill_dirs)
    await skill_manager.discover()
    if skill_manager._skills:
        tool_router.add_provider(SkillToolProvider(skill_manager))

    # 5. Agents
    agent_cfg = raw.get("agents", {})
    agent_registry = AgentRegistry()
    runner = AgentRunner(
        registry=agent_registry,
        max_tool_rounds=agent_cfg.get("max_tool_rounds", 10),
    )
    graph = build_default_graph(agent_registry, runner=runner)
    engine = GraphEngine(max_handoff_depth=agent_cfg.get("max_handoffs", 10))

    # 6. Deps
    deps = AgentDeps(
        llm=llm,
        tool_router=tool_router,
        agent_registry=agent_registry,
        graph_engine=engine,
        ui=ui,
    )

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
        runner=runner,
    )
```

- [ ] **Step 3: Create `src/app/app.py`**

Simplified from `src/app.py` — no component creation, only routing + REPL:

```python
"""AgentApp — 消息路由和 REPL。"""

from __future__ import annotations

from src.interfaces.base import UserInterface
from src.guardrails import InputGuardrail
from src.agents import RunContext, DictState, AgentDeps, AgentRegistry, AgentRunner
from src.agents.node import AgentNode
from src.graph import GraphEngine, CompiledGraph
from src.tools.router import ToolRouter
from src.skills.manager import SkillManager
from src.mcp.manager import MCPManager
from src.plan.flow import PlanFlow
from src.app.presets import build_skill_graph


class AgentApp:
    """应用核心：消息路由 + REPL。组件由 bootstrap 注入。"""

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
        runner: AgentRunner,
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
        self.runner = runner

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
        skill_registry = AgentRegistry()
        skill_runner = AgentRunner(registry=skill_registry)
        skill_graph = build_skill_graph(skill_registry, skill_content, runner=skill_runner)
        skill_engine = GraphEngine()
        ctx = RunContext(
            input=actual_input,
            state=DictState(),
            deps=AgentDeps(
                llm=self.deps.llm,
                tool_router=self.tool_router,
                agent_registry=skill_registry,
                graph_engine=skill_engine,
                ui=self.ui,
            ),
        )
        result = await skill_engine.run(skill_graph, ctx)
        await self.ui.display(f"\n{result.output}\n")

    async def _handle_normal(self, user_input: str) -> None:
        ctx = RunContext(
            input=user_input,
            state=DictState(),
            deps=self.deps,
        )
        result = await self.engine.run(self.graph, ctx)
        output = result.output
        if isinstance(output, dict):
            output = output.get("text", str(output))
        await self.ui.display(f"\n{output}\n")

    async def run(self) -> None:
        """CLI 主循环。"""
        await self.ui.display("Agent 已启动，输入 'exit' 退出。\n")
        while True:
            user_input = await self.ui.prompt("\n你: ")
            if user_input.strip().lower() in ("exit", "quit"):
                break
            await self.process(user_input)

    async def shutdown(self) -> None:
        await self.mcp_manager.disconnect_all()
```

- [ ] **Step 4: Create `src/app/__init__.py`**

```python
from src.app.app import AgentApp
from src.app.bootstrap import create_app

__all__ = ["AgentApp", "create_app"]
```

- [ ] **Step 5: Commit**

```bash
git add src/app/
git commit -m "feat: create src/app/ with bootstrap, app, and presets"
```

---

### Task 11: Update `main.py` + delete old files

**Files:**
- Modify: `main.py`
- Delete: `config.py` (root)
- Delete: `src/core/` (entire directory)
- Delete: `src/app.py` (old file, replaced by `src/app/`)

- [ ] **Step 1: Update `main.py`**

```python
"""Agent 入口。"""

import asyncio

from src.app import create_app


async def main():
    app = await create_app()
    try:
        await app.run()
    finally:
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Delete old files**

```bash
rm config.py
rm src/app.py
rm -rf src/core/
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor: update main.py, delete config.py, src/core/, old src/app.py"
```

---

### Task 12: Update all tests

**Files:**
- Modify: `tests/conftest.py` — remove `mock_call_model`
- Modify: `tests/test_app.py` — update imports to `src.app.app`
- Delete: `tests/core/` — replaced by `tests/llm/` and `tests/guardrails/`
- Modify: `tests/memory/conftest.py` — update for `ChromaMemoryStore`
- Modify: `tests/memory/test_store.py` — update imports
- Verify all remaining test files have correct imports

- [ ] **Step 1: Update `tests/conftest.py`**

Remove the `mock_call_model` fixture (no longer patching `src.core.async_api.call_model`). Add a `mock_llm` fixture instead:

```python
import pytest
from unittest.mock import AsyncMock
from pathlib import Path
from src.llm.types import LLMResponse


@pytest.fixture
def workspace_dir():
    ws = Path("./workspace")
    ws.mkdir(exist_ok=True)
    yield ws

@pytest.fixture
def mock_llm():
    """Mock LLMProvider for tests."""
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(content="test response", tool_calls={}, finish_reason="stop")
    return llm
```

- [ ] **Step 2: Update `tests/test_app.py`**

Update imports:
```python
# from src.app import AgentApp → from src.app.app import AgentApp
# Update fixtures to construct AgentApp with new constructor signature
```

- [ ] **Step 3: Delete `tests/core/`**

```bash
rm -rf tests/core/
```

The tests in `tests/core/test_async_api.py` are replaced by `tests/llm/test_openai.py`.
The test `tests/core/test_main.py` can move to `tests/test_main.py`.

- [ ] **Step 4: Update memory tests**

In `tests/memory/conftest.py`, update `memory_store` fixture to use `ChromaMemoryStore`.
In `tests/memory/test_store.py`, update imports:
```python
# from src.memory.store import MemoryStore → from src.memory.chroma.store import ChromaMemoryStore
```

- [ ] **Step 5: Run full test suite**

Run: `pytest -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "test: update all tests for new module structure"
```

---

### Task 13: Final cleanup and verification

- [ ] **Step 1: Check for stale `__pycache__` and `.pyc` files**

```bash
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
```

- [ ] **Step 2: Run full test suite from clean state**

```bash
pytest -v --tb=short
```

- [ ] **Step 3: Verify no stale imports**

```bash
python -c "from src.app import create_app; print('OK')"
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: clean up cache files after framework refactor"
```
