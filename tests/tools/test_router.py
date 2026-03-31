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
