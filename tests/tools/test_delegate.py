"""DelegateToolProvider 测试。"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.tools.categories import CategoryResolver


@pytest.fixture
def resolver():
    cats = {
        "tool_terminal": {"description": "终端操作", "tools": ["exec"]},
        "tool_calc": {"description": "数学计算", "tools": ["calc"]},
    }
    return CategoryResolver(cats)


@pytest.fixture
def mock_runner():
    return AsyncMock()


@pytest.fixture
def mock_registry(resolver):
    from src.agents.registry import AgentRegistry
    registry = AgentRegistry()
    registry.set_category_resolver(resolver)
    return registry


@pytest.fixture
def mock_deps():
    from src.agents.deps import AgentDeps
    return AgentDeps()


@pytest.fixture
def provider(resolver, mock_runner, mock_registry, mock_deps):
    from src.tools.delegate import DelegateToolProvider
    return DelegateToolProvider(
        resolver=resolver, runner=mock_runner, registry=mock_registry, deps=mock_deps,
    )


def test_can_handle(provider):
    assert provider.can_handle("delegate_tool_terminal") is True
    assert provider.can_handle("delegate_tool_calc") is True
    assert provider.can_handle("delegate_tool_unknown") is False
    assert provider.can_handle("tool_terminal") is False
    assert provider.can_handle("calculate") is False


def test_get_schemas(provider):
    schemas = provider.get_schemas()
    assert len(schemas) == 2
    names = {s["function"]["name"] for s in schemas}
    assert names == {"delegate_tool_terminal", "delegate_tool_calc"}
    terminal_schema = next(s for s in schemas if s["function"]["name"] == "delegate_tool_terminal")
    assert terminal_schema["type"] == "function"
    params = terminal_schema["function"]["parameters"]
    assert "task" in params["properties"]
    assert "task" in params["required"]


@pytest.mark.asyncio
async def test_execute_delegates_to_runner(provider, mock_runner):
    from src.agents.agent import AgentResult
    mock_runner.run = AsyncMock(return_value=AgentResult(text="执行完成"))
    result = await provider.execute("delegate_tool_terminal", {"task": "列出当前目录"})
    assert result == "执行完成"
    mock_runner.run.assert_called_once()
    call_args = mock_runner.run.call_args
    agent = call_args[0][0]
    assert agent.name == "tool_terminal"
    ctx = call_args[0][1]
    assert ctx.input == "列出当前目录"


@pytest.mark.asyncio
async def test_execute_unknown_agent(provider):
    # Replace registry with one that can't resolve the agent
    provider._registry = MagicMock()
    provider._registry.get = MagicMock(return_value=None)
    result = await provider.execute("delegate_tool_unknown", {"task": "test"})
    assert "错误" in result
