"""
测试 src/flows/planning.py — PlanningFlow 状态机
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.flows.planning import (
    PlanningFlow,
    PlanningModel,
    format_plan_for_display,
    format_execution_results,
    _format_tool_args,
)
from src.plan.models import Plan, Step
from src.plan.executor import DEFERRED_PLACEHOLDER
from src.core.fsm import FSMRunner, OUTPUT_PREFIX


# ==================== 辅助函数测试 ====================


class TestFormatPlanForDisplay:
    def test_basic(self):
        plan = Plan(steps=[
            Step(id="s1", description="查询天气", action="tool", tool_name="get_weather"),
            Step(id="s2", description="发送邮件", action="tool", tool_name="send_email"),
        ])
        text = format_plan_for_display(plan)
        assert "1. 查询天气" in text
        assert "2. 发送邮件" in text

    def test_empty_plan(self):
        plan = Plan(steps=[])
        text = format_plan_for_display(plan)
        assert text == ""


class TestFormatExecutionResults:
    def test_basic(self):
        plan = Plan(steps=[
            Step(id="s1", description="查询天气", action="tool", tool_name="t1"),
            Step(id="s2", description="发送邮件", action="tool", tool_name="t2"),
        ])
        result_dict = {"s1": "北京晴天", "s2": "邮件已发送"}
        text = format_execution_results(plan, result_dict)
        assert "查询天气: 北京晴天" in text
        assert "发送邮件: 邮件已发送" in text

    def test_deferred_placeholder_skipped(self):
        plan = Plan(steps=[
            Step(id="s1", description="查询天气", action="tool", tool_name="t1"),
            Step(id="s2", description="删除文件", action="tool", tool_name="t2"),
        ])
        result_dict = {"s1": "结果1", "s2": DEFERRED_PLACEHOLDER}
        text = format_execution_results(plan, result_dict)
        assert "查询天气" in text
        assert "删除文件" not in text

    def test_missing_result(self):
        plan = Plan(steps=[
            Step(id="s1", description="步骤1", action="tool", tool_name="t1"),
        ])
        text = format_execution_results(plan, {})
        assert "无结果" in text


class TestFormatToolArgs:
    def test_basic(self):
        text = _format_tool_args({"location": "北京", "date": "2026-03-25"})
        assert "location: 北京" in text
        assert "date: 2026-03-25" in text

    def test_empty(self):
        assert _format_tool_args({}) == ""
        assert _format_tool_args(None) == ""

    def test_long_value_truncated(self):
        text = _format_tool_args({"data": "x" * 300})
        assert "..." in text
        assert len(text) < 300


# ==================== PlanningModel 测试 ====================


class TestPlanningModel:
    def test_init(self):
        model = PlanningModel(available_tools=[], tool_executor=MagicMock())
        assert model.gathered_info_parts == []
        assert model.current_plan is None
        assert model.adjustment_count == 0
        assert model.clarification_round == 0
        assert model.result_dict == {}
        assert model.deferred_steps == []


# ==================== PlanningFlow 条件方法测试 ====================


class TestPlanningFlowConditions:
    def _make_flow(self):
        return PlanningFlow(available_tools=[], tool_executor=MagicMock())

    def test_info_sufficient(self):
        flow = self._make_flow()
        flow.model.data["info_sufficient"] = True
        assert flow.info_sufficient() is True

    def test_need_more_info(self):
        flow = self._make_flow()
        flow.model.data["info_sufficient"] = False
        assert flow.need_more_info() is True

    def test_plan_generated(self):
        flow = self._make_flow()
        flow.model.current_plan = Plan(steps=[])
        assert flow.plan_generated() is True

    def test_no_plan_needed(self):
        flow = self._make_flow()
        flow.model.current_plan = None
        assert flow.no_plan_needed() is True

    def test_user_confirmed(self):
        flow = self._make_flow()
        flow.model.data["feedback_action"] = "confirm"
        assert flow.user_confirmed() is True

    def test_user_wants_adjust(self):
        flow = self._make_flow()
        flow.model.data["feedback_action"] = "adjust"
        flow.model.adjustment_count = 0
        assert flow.user_wants_adjust() is True

    def test_max_adjustments_reached(self):
        flow = self._make_flow()
        flow.model.data["feedback_action"] = "adjust"
        flow.model.adjustment_count = 999
        assert flow.max_adjustments_reached() is True

    def test_has_deferred(self):
        flow = self._make_flow()
        flow.model.deferred_steps = [MagicMock()]
        assert flow.has_deferred() is True

    def test_no_deferred(self):
        flow = self._make_flow()
        flow.model.deferred_steps = []
        assert flow.no_deferred() is True


# ==================== PlanningFlow 完整流程测试 ====================


class TestPlanningFlowIntegration:

    @pytest.mark.asyncio
    async def test_no_plan_needed(self):
        """测试 LLM 判断不需要计划 → 直接到 done"""
        flow = PlanningFlow(available_tools=[], tool_executor=MagicMock())
        flow.model.data["original_request"] = "你好"

        with patch("src.flows.planning.check_clarification_needed", new_callable=AsyncMock) as mock_clarify, \
             patch("src.flows.planning.generate_plan", new_callable=AsyncMock) as mock_gen:
            mock_clarify.return_value = None  # 不需要澄清
            mock_gen.return_value = None  # 不需要计划

            runner = FSMRunner(flow)
            with patch("src.core.fsm.agent_output", new_callable=AsyncMock), \
                 patch("src.core.fsm.agent_input", new_callable=AsyncMock):
                result = await runner.run()

        assert result is None  # None 表示不需要计划

    @pytest.mark.asyncio
    async def test_generate_and_confirm(self):
        """测试 clarify → generating → confirming → 用户确认 → executing → done"""
        plan = Plan(steps=[
            Step(id="s1", description="步骤1", action="tool", tool_name="t1"),
        ])

        flow = PlanningFlow(available_tools=[], tool_executor=MagicMock())
        flow.model.data["original_request"] = "帮我查天气"

        with patch("src.flows.planning.check_clarification_needed", new_callable=AsyncMock) as mock_clarify, \
             patch("src.flows.planning.generate_plan", new_callable=AsyncMock) as mock_gen, \
             patch("src.flows.planning.classify_user_feedback", new_callable=AsyncMock) as mock_classify, \
             patch("src.flows.planning.execute_plan", new_callable=AsyncMock) as mock_exec:

            mock_clarify.return_value = None
            mock_gen.return_value = plan
            mock_classify.return_value = "confirm"
            mock_exec.return_value = ({"s1": "北京晴天"}, [])

            runner = FSMRunner(flow)
            with patch("src.core.fsm.agent_output", new_callable=AsyncMock), \
                 patch("src.core.fsm.agent_input", new_callable=AsyncMock) as mock_in:
                mock_in.return_value = "确认"
                result = await runner.run()

        assert "北京晴天" in result

    @pytest.mark.asyncio
    async def test_clarification_round(self):
        """测试需要澄清信息 → 用户回答 → 生成计划"""
        plan = Plan(steps=[
            Step(id="s1", description="步骤1", action="tool", tool_name="t1"),
        ])

        flow = PlanningFlow(available_tools=[], tool_executor=MagicMock())
        flow.model.data["original_request"] = "帮我订票"

        with patch("src.flows.planning.check_clarification_needed", new_callable=AsyncMock) as mock_clarify, \
             patch("src.flows.planning.generate_plan", new_callable=AsyncMock) as mock_gen, \
             patch("src.flows.planning.classify_user_feedback", new_callable=AsyncMock) as mock_classify, \
             patch("src.flows.planning.execute_plan", new_callable=AsyncMock) as mock_exec:

            # 第一次需要澄清，第二次信息充足
            mock_clarify.side_effect = ["你要订哪天的票？", None]
            mock_gen.return_value = plan
            mock_classify.return_value = "confirm"
            mock_exec.return_value = ({"s1": "已订票"}, [])

            runner = FSMRunner(flow)
            with patch("src.core.fsm.agent_output", new_callable=AsyncMock) as mock_out, \
                 patch("src.core.fsm.agent_input", new_callable=AsyncMock) as mock_in:
                mock_in.side_effect = ["明天的", "确认"]
                result = await runner.run()

        assert "已订票" in result
        # 验证澄清问题被输出
        output_texts = [c.args[0] for c in mock_out.call_args_list]
        assert any("你要订哪天的票" in t for t in output_texts)
        # 验证用户回答被记录
        assert len(flow.model.gathered_info_parts) == 1
        assert "明天的" in flow.model.gathered_info_parts[0]

    @pytest.mark.asyncio
    async def test_adjust_plan(self):
        """测试 confirming → adjusting → confirming → 确认"""
        plan = Plan(steps=[
            Step(id="s1", description="原始步骤", action="tool", tool_name="t1"),
        ])
        adjusted_plan = Plan(steps=[
            Step(id="s1", description="调整后步骤", action="tool", tool_name="t1"),
        ])

        flow = PlanningFlow(available_tools=[], tool_executor=MagicMock())
        flow.model.data["original_request"] = "帮我做事"

        with patch("src.flows.planning.check_clarification_needed", new_callable=AsyncMock) as mock_clarify, \
             patch("src.flows.planning.generate_plan", new_callable=AsyncMock) as mock_gen, \
             patch("src.flows.planning.classify_user_feedback", new_callable=AsyncMock) as mock_classify, \
             patch("src.flows.planning.adjust_plan", new_callable=AsyncMock) as mock_adjust, \
             patch("src.flows.planning.execute_plan", new_callable=AsyncMock) as mock_exec:

            mock_clarify.return_value = None
            mock_gen.return_value = plan
            # 第一次：调整；第二次：确认
            mock_classify.side_effect = ["adjust", "confirm"]
            mock_adjust.return_value = adjusted_plan
            mock_exec.return_value = ({"s1": "结果"}, [])

            runner = FSMRunner(flow)
            with patch("src.core.fsm.agent_output", new_callable=AsyncMock), \
                 patch("src.core.fsm.agent_input", new_callable=AsyncMock) as mock_in:
                mock_in.side_effect = ["改一下", "确认"]
                result = await runner.run()

        assert flow.model.adjustment_count == 1
        assert "结果" in result

    @pytest.mark.asyncio
    async def test_cancel_during_confirm(self):
        """测试在确认阶段取消"""
        plan = Plan(steps=[
            Step(id="s1", description="步骤1", action="tool", tool_name="t1"),
        ])

        flow = PlanningFlow(available_tools=[], tool_executor=MagicMock())
        flow.model.data["original_request"] = "帮我做事"

        with patch("src.flows.planning.check_clarification_needed", new_callable=AsyncMock) as mock_clarify, \
             patch("src.flows.planning.generate_plan", new_callable=AsyncMock) as mock_gen:

            mock_clarify.return_value = None
            mock_gen.return_value = plan

            runner = FSMRunner(flow)
            with patch("src.core.fsm.agent_output", new_callable=AsyncMock), \
                 patch("src.core.fsm.agent_input", new_callable=AsyncMock) as mock_in:
                mock_in.return_value = "取消"
                result = await runner.run()

        assert result == "计划已取消。"

    @pytest.mark.asyncio
    async def test_plan_generation_failure(self):
        """测试计划生成失败"""
        from src.plan.exceptions import PlanError

        flow = PlanningFlow(available_tools=[], tool_executor=MagicMock())
        flow.model.data["original_request"] = "复杂任务"

        with patch("src.flows.planning.check_clarification_needed", new_callable=AsyncMock) as mock_clarify, \
             patch("src.flows.planning.generate_plan", new_callable=AsyncMock) as mock_gen:

            mock_clarify.return_value = None
            mock_gen.side_effect = PlanError("生成失败")

            runner = FSMRunner(flow)
            with patch("src.core.fsm.agent_output", new_callable=AsyncMock), \
                 patch("src.core.fsm.agent_input", new_callable=AsyncMock):
                result = await runner.run()

        assert "无法生成有效计划" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
