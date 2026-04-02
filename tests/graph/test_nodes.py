import pytest
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict

from src.graph.nodes import DecisionNode, SubgraphNode, TerminalNode
from src.graph.messages import AgentResponse, ResponseStatus


class DynamicState(BaseModel):
    model_config = ConfigDict(extra="allow")


@dataclass
class MockContext:
    input: str = "test"
    state: DynamicState = field(default_factory=DynamicState)
    deps: MagicMock = field(default_factory=MagicMock)
    trace: list = field(default_factory=list)
    depth: int = 0


class TestDecisionNode:
    @pytest.mark.asyncio
    async def test_returns_chosen_branch(self):
        mock_response = MagicMock()
        mock_response.content = "yes"
        mock_response.tool_calls = {}
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = mock_response
        node = DecisionNode(
            name="decide",
            question="Is it ready?",
            branches=["yes", "no"],
        )
        ctx = MockContext()
        ctx.deps.llm = mock_llm
        result = await node.execute(ctx)
        assert result.output.data["chosen_branch"] == "yes"

    @pytest.mark.asyncio
    async def test_strips_whitespace(self):
        mock_response = MagicMock()
        mock_response.content = "  no  \n"
        mock_response.tool_calls = {}
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = mock_response
        node = DecisionNode(name="d", question="?", branches=["yes", "no"])
        ctx = MockContext()
        ctx.deps.llm = mock_llm
        result = await node.execute(ctx)
        assert result.output.data["chosen_branch"] == "no"


class TestTerminalNode:
    @pytest.mark.asyncio
    async def test_passes_through_last_output(self):
        node = TerminalNode(name="end")
        ctx = MockContext()
        last = AgentResponse(text="done", data={"x": 1})
        ctx.state._last_output = last  # type: ignore[attr-defined]
        result = await node.execute(ctx)
        assert result.output is last

    @pytest.mark.asyncio
    async def test_empty_when_no_last_output(self):
        node = TerminalNode(name="end")
        ctx = MockContext()
        result = await node.execute(ctx)
        assert result.output.text == ""


class TestSubgraphNode:
    @pytest.mark.asyncio
    async def test_runs_sub_graph(self):
        from src.graph.engine import GraphResult

        mock_engine = AsyncMock()
        mock_engine.run.return_value = GraphResult(
            output=AgentResponse(text="sub result", data={"k": "v"}),
            state=DynamicState(),
        )

        sub_graph = MagicMock()
        node = SubgraphNode(name="sub", sub_graph=sub_graph)

        ctx = MockContext()
        ctx.deps.engine = mock_engine

        result = await node.execute(ctx)
        assert result.output.text == "sub result"
        mock_engine.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_depth_limit(self):
        node = SubgraphNode(name="sub", sub_graph=MagicMock(), max_subgraph_depth=2)
        ctx = MockContext(depth=2)
        ctx.deps.engine = AsyncMock()

        result = await node.execute(ctx)
        assert result.output.status == ResponseStatus.FAILED
        assert "深度超过限制" in result.output.text
        ctx.deps.engine.run.assert_not_called()
