"""
测试 src/flows/meeting_booking.py — MeetingBookingFlow 状态机
"""
import pytest
from unittest.mock import AsyncMock, patch

from src.flows.meeting_booking import MeetingBookingFlow, BookingModel
from src.core.fsm import FSMRunner, OUTPUT_PREFIX


def _make_booking_flow():
    return MeetingBookingFlow()


# ==================== BookingModel 测试 ====================


class TestBookingModel:
    def test_inherits_flow_model(self):
        model = BookingModel()
        assert model.data == {}
        assert model.needs_input is True


# ==================== MeetingBookingFlow 条件方法测试 ====================


class TestBookingFlowConditions:
    def test_user_confirmed(self):
        flow = _make_booking_flow()
        flow.model.data["confirm_action"] = "confirm"
        assert flow.user_confirmed() is True

    def test_user_restart(self):
        flow = _make_booking_flow()
        flow.model.data["confirm_action"] = "restart"
        assert flow.user_restart() is True

    def test_neither(self):
        flow = _make_booking_flow()
        flow.model.data["confirm_action"] = "confirm"
        assert flow.user_restart() is False


# ==================== 完整流程测试 ====================


class TestMeetingBookingFlowIntegration:

    @pytest.mark.asyncio
    async def test_happy_path(self):
        """测试完整预订流程：日期 → 时间 → 人数 → 设备 → 确认 → 完成"""
        flow = _make_booking_flow()
        runner = FSMRunner(flow)

        with patch("src.core.fsm.agent_output", new_callable=AsyncMock) as mock_out, \
             patch("src.core.fsm.agent_input", new_callable=AsyncMock) as mock_in:
            mock_in.side_effect = [
                "2026-03-28",      # 日期
                "14:00-15:00",     # 时间
                "5",               # 人数
                "投影仪",           # 设备
                "确认",             # 确认
            ]
            result = await runner.run()

        assert "预订成功" in result
        assert "2026-03-28" in result
        assert "14:00-15:00" in result
        assert "5" in result
        assert "投影仪" in result

        # 验证 model.data
        assert flow.model.data["date"] == "2026-03-28"
        assert flow.model.data["time"] == "14:00-15:00"
        assert flow.model.data["attendees"] == "5"
        assert flow.model.data["equipment"] == "投影仪"

    @pytest.mark.asyncio
    async def test_no_equipment(self):
        """测试不需要设备的预订"""
        flow = _make_booking_flow()
        runner = FSMRunner(flow)

        with patch("src.core.fsm.agent_output", new_callable=AsyncMock), \
             patch("src.core.fsm.agent_input", new_callable=AsyncMock) as mock_in:
            mock_in.side_effect = [
                "2026-04-01",
                "10:00-11:00",
                "3",
                "无",              # 不需要设备
                "确认",
            ]
            result = await runner.run()

        assert "预订成功" in result
        assert flow.model.data["equipment"] == ""

    @pytest.mark.asyncio
    async def test_restart_from_confirm(self):
        """测试在确认阶段选择重填"""
        flow = _make_booking_flow()
        runner = FSMRunner(flow)

        with patch("src.core.fsm.agent_output", new_callable=AsyncMock), \
             patch("src.core.fsm.agent_input", new_callable=AsyncMock) as mock_in:
            mock_in.side_effect = [
                "2026-03-28",      # 第一次: 日期
                "14:00-15:00",     # 第一次: 时间
                "5",               # 第一次: 人数
                "无",              # 第一次: 设备
                "重填",             # 重填 → 回到 collect_date
                "2026-04-01",      # 第二次: 日期
                "10:00-11:00",     # 第二次: 时间
                "3",               # 第二次: 人数
                "白板",             # 第二次: 设备
                "确认",             # 确认
            ]
            result = await runner.run()

        assert "预订成功" in result
        assert flow.model.data["date"] == "2026-04-01"
        assert flow.model.data["time"] == "10:00-11:00"

    @pytest.mark.asyncio
    async def test_cancel_at_date(self):
        """测试在日期收集阶段取消"""
        flow = _make_booking_flow()
        runner = FSMRunner(flow)

        with patch("src.core.fsm.agent_output", new_callable=AsyncMock), \
             patch("src.core.fsm.agent_input", new_callable=AsyncMock) as mock_in:
            mock_in.return_value = "取消"
            result = await runner.run()

        assert result == "预订已取消。"

    @pytest.mark.asyncio
    async def test_cancel_at_equipment(self):
        """测试在设备收集阶段取消"""
        flow = _make_booking_flow()
        runner = FSMRunner(flow)

        with patch("src.core.fsm.agent_output", new_callable=AsyncMock), \
             patch("src.core.fsm.agent_input", new_callable=AsyncMock) as mock_in:
            mock_in.side_effect = [
                "2026-03-28",
                "14:00-15:00",
                "5",
                "取消",            # 在设备阶段取消
            ]
            result = await runner.run()

        assert result == "预订已取消。"

    @pytest.mark.asyncio
    async def test_cancel_at_confirm(self):
        """测试在确认阶段取消"""
        flow = _make_booking_flow()
        runner = FSMRunner(flow)

        with patch("src.core.fsm.agent_output", new_callable=AsyncMock), \
             patch("src.core.fsm.agent_input", new_callable=AsyncMock) as mock_in:
            mock_in.side_effect = [
                "2026-03-28",
                "14:00-15:00",
                "5",
                "投影仪",
                "取消",            # 在确认阶段取消
            ]
            result = await runner.run()

        assert result == "预订已取消。"

    @pytest.mark.asyncio
    async def test_confirm_variants(self):
        """测试各种确认关键词"""
        for keyword in ["确认", "确定", "ok", "yes", "y"]:
            flow = _make_booking_flow()
            runner = FSMRunner(flow)

            with patch("src.core.fsm.agent_output", new_callable=AsyncMock), \
                 patch("src.core.fsm.agent_input", new_callable=AsyncMock) as mock_in:
                mock_in.side_effect = ["2026-03-28", "14:00-15:00", "5", "无", keyword]
                result = await runner.run()

            assert "预订成功" in result, f"确认关键词 '{keyword}' 失败"

    @pytest.mark.asyncio
    async def test_output_prompts(self):
        """测试每个状态输出正确的提示"""
        flow = _make_booking_flow()
        runner = FSMRunner(flow)

        with patch("src.core.fsm.agent_output", new_callable=AsyncMock) as mock_out, \
             patch("src.core.fsm.agent_input", new_callable=AsyncMock) as mock_in:
            mock_in.side_effect = ["2026-03-28", "14:00-15:00", "5", "无", "确认"]
            await runner.run()

        output_texts = [c.args[0] for c in mock_out.call_args_list]
        assert any("日期" in t for t in output_texts)
        assert any("时间" in t for t in output_texts)
        assert any("人数" in t for t in output_texts)
        assert any("设备" in t for t in output_texts)
        assert any("确认" in t for t in output_texts)
        assert any("预订成功" in t for t in output_texts)

    @pytest.mark.asyncio
    async def test_whitespace_stripped_from_input(self):
        """测试用户输入前后空格被去除"""
        flow = _make_booking_flow()
        runner = FSMRunner(flow)

        with patch("src.core.fsm.agent_output", new_callable=AsyncMock), \
             patch("src.core.fsm.agent_input", new_callable=AsyncMock) as mock_in:
            mock_in.side_effect = [
                "  2026-03-28  ",
                "  14:00-15:00  ",
                "  5  ",
                "  无  ",
                "确认",
            ]
            await runner.run()

        assert flow.model.data["date"] == "2026-03-28"
        assert flow.model.data["time"] == "14:00-15:00"
        assert flow.model.data["attendees"] == "5"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
