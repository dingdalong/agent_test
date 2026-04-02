# tests/utils/test_interaction.py
"""UserInteractionService 测试。"""
import asyncio
import pytest
from unittest.mock import AsyncMock

from src.utils.interaction import UserInteractionService


@pytest.fixture
def mock_ui():
    ui = AsyncMock()
    ui.prompt = AsyncMock(return_value="用户的回答")
    ui.display = AsyncMock()
    ui.confirm = AsyncMock(return_value=True)
    return ui


@pytest.fixture
def service(mock_ui):
    return UserInteractionService(mock_ui)


class TestAsk:
    @pytest.mark.asyncio
    async def test_ask_returns_user_input(self, service, mock_ui):
        result = await service.ask("你想要什么格式？")
        assert result == "用户的回答"
        mock_ui.display.assert_called_once()
        mock_ui.prompt.assert_called_once_with("你的回答: ")

    @pytest.mark.asyncio
    async def test_ask_with_source_shows_label(self, service, mock_ui):
        await service.ask("问题", source="weather_agent")
        display_arg = mock_ui.display.call_args[0][0]
        assert "[weather_agent]" in display_arg

    @pytest.mark.asyncio
    async def test_ask_without_source_no_label(self, service, mock_ui):
        await service.ask("问题")
        display_arg = mock_ui.display.call_args[0][0]
        assert "[" not in display_arg


class TestConfirm:
    @pytest.mark.asyncio
    async def test_confirm_returns_true(self, service, mock_ui):
        mock_ui.confirm = AsyncMock(return_value=True)
        result = await service.confirm("执行敏感操作: delete_file")
        assert result is True

    @pytest.mark.asyncio
    async def test_confirm_returns_false(self, service, mock_ui):
        mock_ui.confirm = AsyncMock(return_value=False)
        result = await service.confirm("执行敏感操作: delete_file")
        assert result is False

    @pytest.mark.asyncio
    async def test_confirm_displays_message(self, service, mock_ui):
        await service.confirm("删除文件")
        display_arg = mock_ui.display.call_args[0][0]
        assert "删除文件" in display_arg


class TestLock:
    @pytest.mark.asyncio
    async def test_concurrent_ask_serialized(self, mock_ui):
        """并行调用 ask 应被 Lock 串行化。"""
        call_order = []

        async def slow_prompt(msg):
            call_order.append("start")
            await asyncio.sleep(0.05)
            call_order.append("end")
            return "answer"

        mock_ui.prompt = slow_prompt
        service = UserInteractionService(mock_ui)

        await asyncio.gather(
            service.ask("问题1"),
            service.ask("问题2"),
        )

        # Lock 保证串行：start-end-start-end，不会交错
        assert call_order == ["start", "end", "start", "end"]
