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
    graph.set_entry("task_a")
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
            ("", {0: {"id": "c1", "name": "transfer_to_agent_b", "arguments": json.dumps({"task": "do B stuff"})}}, "tool_calls"),
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

    graph = GraphBuilder()
    graph.add_agent("agent_a", agent_a)
    graph.set_entry("agent_a")
    compiled = graph.compile()

    engine = GraphEngine(registry=registry)
    ctx = RunContext(input="test", state=DictState(), deps=EmptyDeps())

    with patch("src.agents.runner.call_model", new_callable=AsyncMock) as mock_call:
        import json
        mock_call.side_effect = [
            ("", {0: {"id": "c1", "name": "transfer_to_dynamic_b", "arguments": json.dumps({"task": "dynamic task"})}}, "tool_calls"),
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
