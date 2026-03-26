"""测试 SkillToolProvider 路由。"""
import pytest
from unittest.mock import MagicMock
from src.skills.provider import SkillToolProvider


@pytest.mark.asyncio
async def test_provider_routes_activate_skill():
    mock_manager = MagicMock()
    mock_manager.activate.return_value = "<skill_content>skill_content</skill_content>"
    provider = SkillToolProvider(mock_manager)

    assert provider.can_handle("activate_skill")
    result = await provider.execute("activate_skill", {"name": "test"})
    assert "skill_content" in result
    mock_manager.activate.assert_called_once_with("test")


@pytest.mark.asyncio
async def test_provider_skill_not_found():
    mock_manager = MagicMock()
    mock_manager.activate.return_value = None
    provider = SkillToolProvider(mock_manager)

    result = await provider.execute("activate_skill", {"name": "missing"})
    assert "未找到" in result


def test_provider_does_not_handle_other_tools():
    mock_manager = MagicMock()
    provider = SkillToolProvider(mock_manager)
    assert not provider.can_handle("calculator")
    assert not provider.can_handle("mcp_something")
