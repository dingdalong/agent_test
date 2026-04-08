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
from src.agents.delegate import DelegateToolProvider
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
    """非工具类 agent 通过 delegate 调用工具类 agent，获取结果并继续。"""
    mock_llm = AsyncMock()

    call_count = 0

    async def mock_chat(messages, tools=None, silent=True):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # orchestrator 调用 delegate_tool_calc
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
            # tool_calc agent 执行并返回
            return LLMResponse(content="计算结果是 2", tool_calls={})
        # orchestrator 汇总结果
        return LLMResponse(content="1+1=2", tool_calls={})

    mock_llm.chat = mock_chat

    runner = AgentRunner()
    router = ToolRouter()

    delegate_provider = DelegateToolProvider(resolver=resolver)
    router.add_provider(delegate_provider)

    # orchestrator 是非工具类 agent，delegate 由 _build_tools 自动注入
    orchestrator = Agent(
        name="orchestrator",
        description="总控 Agent",
        instructions="你是总控 Agent，使用 delegate 工具完成任务。",
        tools=[],
    )
    registry.register(orchestrator)

    ctx = RunContext(
        input="计算 1+1",
        state=DynamicState(),
        deps=AgentDeps(
            llm=mock_llm,
            tool_router=router,
            agent_registry=registry,
            runner=runner,
            category_resolver=resolver,
        ),
        delegate_depth=0,
    )
    result = await runner.run(orchestrator, ctx)

    assert "2" in result.text
    assert call_count == 3


@pytest.mark.asyncio
async def test_delegated_agent_cannot_delegate_further(resolver, registry):
    """被 delegate 调用的工具类 Agent 不应看到任何 delegate 工具。"""
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
        category_resolver=resolver,
    )
    delegate_provider = DelegateToolProvider(resolver=resolver)
    router.add_provider(delegate_provider)

    agent_b = registry.get("tool_calc")

    ctx = RunContext(
        input="计算 1+1",
        state=DynamicState(),
        deps=deps,
        delegate_depth=1,
    )
    await runner.run(agent_b, ctx)

    delegate_tools = [t for t in tools_seen_by_b if t.startswith("delegate_") or t == "parallel_delegate"]
    assert delegate_tools == [], f"Tool agent should not see delegate tools, but saw: {delegate_tools}"


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
