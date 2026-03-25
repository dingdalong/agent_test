import pytest
from unittest.mock import MagicMock
from src.tools.tool_executor import ToolExecutor


@pytest.mark.asyncio
async def test_executor_routes_activate_skill():
    mock_skill_mgr = MagicMock()
    mock_skill_mgr.activate.return_value = '<skill_content name="test">body</skill_content>'
    executor = ToolExecutor({})
    executor.skill_manager = mock_skill_mgr
    result = await executor.execute("activate_skill", {"name": "test"})
    mock_skill_mgr.activate.assert_called_once_with("test")
    assert "skill_content" in result


@pytest.mark.asyncio
async def test_executor_activate_skill_not_found():
    mock_skill_mgr = MagicMock()
    mock_skill_mgr.activate.return_value = None
    executor = ToolExecutor({})
    executor.skill_manager = mock_skill_mgr
    result = await executor.execute("activate_skill", {"name": "nonexistent"})
    assert "未找到" in result


@pytest.mark.asyncio
async def test_executor_no_skill_manager_for_activate():
    executor = ToolExecutor({})
    result = await executor.execute("activate_skill", {"name": "test"})
    assert "未知工具" in result
