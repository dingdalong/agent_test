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
def handoff_agent():
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
        mock_call.side_effect = [
            (
                "",
                {0: {"id": "call_1", "name": "get_weather", "arguments": '{"city": "Beijing"}'}},
                "tool_calls",
            ),
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
        # Need max_tool_rounds + 1 side effects: max_tool_rounds with tool_calls, then 1 final text call
        mock_call.side_effect = [
            ("", {0: {"id": "call_1", "name": "get_weather", "arguments": "{}"}}, "tool_calls"),
            ("", {0: {"id": "call_2", "name": "get_weather", "arguments": "{}"}}, "tool_calls"),
            ("Fallback response after max rounds", {}, None),
        ]
        runner = AgentRunner(registry=AgentRegistry(), max_tool_rounds=2)
        result = await runner.run(simple_agent, ctx)

    assert mock_call.call_count == 3  # 2 rounds + 1 final
    assert result.text == "Fallback response after max rounds"


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

    messages = mock_call.call_args[0][0]
    assert "Handle input: test input" in messages[0]["content"]
