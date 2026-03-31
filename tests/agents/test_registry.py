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


# ---------------------------------------------------------------------------
# CategoryResolver 懒加载
# ---------------------------------------------------------------------------


def test_lazy_resolve_from_category_resolver(registry):
    from src.tools.categories import CategoryResolver

    cats = {"tool_terminal": {"description": "终端操作", "tools": {"exec": "Execute command"}}}
    resolver = CategoryResolver(cats)
    registry.set_category_resolver(resolver)

    assert not registry.has("tool_terminal")  # has 不触发懒加载
    agent = registry.get("tool_terminal")
    assert agent is not None
    assert agent.name == "tool_terminal"
    assert agent.description == "终端操作"
    assert "exec" in agent.tools
    assert registry.has("tool_terminal")  # 已缓存
    assert registry.get("tool_terminal") is agent  # 返回同一实例


def test_lazy_resolve_unknown_returns_none(registry):
    from src.tools.categories import CategoryResolver

    resolver = CategoryResolver({})
    registry.set_category_resolver(resolver)
    assert registry.get("nonexistent") is None


def test_no_resolver_get_returns_none(registry):
    assert registry.get("nonexistent") is None


# ---------------------------------------------------------------------------
# delegate 工具注入
# ---------------------------------------------------------------------------


def test_lazy_resolve_includes_delegate_tools(registry):
    """懒加载的 category agent 应包含其他分类的 delegate 工具名。"""
    from src.tools.categories import CategoryResolver

    cats = {
        "tool_terminal": {"description": "终端操作", "tools": {"exec": "Execute command"}},
        "tool_calc": {"description": "计算", "tools": {"calc": "Calculate math"}},
        "tool_files": {"description": "文件操作", "tools": {"read": "Read file"}},
    }
    resolver = CategoryResolver(cats)
    registry.set_category_resolver(resolver)

    agent = registry.get("tool_terminal")
    assert agent is not None

    # 自身的 MCP 工具
    assert "exec" in agent.tools
    # 其他分类的 delegate 工具
    assert "delegate_tool_calc" in agent.tools
    assert "delegate_tool_files" in agent.tools
    # 不包含自身的 delegate 工具
    assert "delegate_tool_terminal" not in agent.tools


def test_lazy_resolve_single_category_no_delegates(registry):
    """只有一个分类时，agent 不包含任何 delegate 工具。"""
    from src.tools.categories import CategoryResolver

    cats = {"tool_only": {"description": "唯一", "tools": {"t1": "Tool"}}}
    resolver = CategoryResolver(cats)
    registry.set_category_resolver(resolver)

    agent = registry.get("tool_only")
    assert agent is not None
    assert agent.tools == ["t1"]


def test_lazy_resolve_instructions_contain_delegate_info(registry):
    """懒加载的 agent 指令中应包含协作能力描述。"""
    from src.tools.categories import CategoryResolver

    cats = {
        "tool_terminal": {"description": "终端操作", "tools": {"exec": "Execute"}},
        "tool_calc": {"description": "计算", "tools": {"calc": "Calculate"}},
    }
    resolver = CategoryResolver(cats)
    registry.set_category_resolver(resolver)

    agent = registry.get("tool_terminal")
    assert "协作能力" in agent.instructions
    assert "delegate_tool_calc" in agent.instructions
