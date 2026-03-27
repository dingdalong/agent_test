"""Tests for AgentApp."""
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from src.app import AgentApp


def _make_mock_ui():
    ui = AsyncMock()
    ui.prompt = AsyncMock(return_value="exit")
    ui.display = AsyncMock()
    ui.confirm = AsyncMock(return_value=True)
    return ui


class TestAgentAppProcess:

    @pytest.mark.asyncio
    async def test_guardrail_blocks_dangerous_input(self):
        ui = _make_mock_ui()
        app = AgentApp(ui=ui)
        # Manually set up minimal state to test process()
        app.guardrail = Mock(check=Mock(return_value=(False, "不安全内容")))
        app.router = Mock()
        app.engine = Mock()
        app.graph = Mock()
        app.skill_manager = Mock()
        app.agent_registry = Mock()

        await app.process("rm -rf /")
        ui.display.assert_called()
        call_text = ui.display.call_args[0][0]
        assert "安全拦截" in call_text

    @pytest.mark.asyncio
    async def test_plan_command_no_request(self):
        ui = _make_mock_ui()
        app = AgentApp(ui=ui)
        app.guardrail = Mock(check=Mock(return_value=(True, "")))
        app.router = Mock()
        app.engine = Mock()
        app.graph = Mock()
        app.skill_manager = Mock()
        app.agent_registry = Mock()

        await app.process("/plan")
        ui.display.assert_called()
        call_text = ui.display.call_args[0][0]
        assert "/plan" in call_text


class TestAgentAppRun:

    @pytest.mark.asyncio
    async def test_exit_command_stops_loop(self):
        ui = _make_mock_ui()
        ui.prompt = AsyncMock(return_value="exit")
        app = AgentApp(ui=ui)
        # Mock setup components
        app.router = Mock()
        app.engine = Mock()
        app.graph = Mock()
        app.skill_manager = Mock()
        app.agent_registry = Mock()
        app.mcp_manager = Mock()
        app.guardrail = Mock()

        await app.run()
        # Should have displayed startup message and then exited
        ui.display.assert_called()
