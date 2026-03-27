"""测试 plan.planner 模块"""
import pytest
import json
from unittest.mock import AsyncMock, patch
from src.plan.planner import generate_plan, adjust_plan, parse_plan_from_tool_calls, _SUBMIT_PLAN_SCHEMA
from src.plan.models import Plan, Step


def _make_submit_plan_tool_calls(steps_data: list) -> dict:
    """构造 submit_plan 的 tool_calls 返回格式"""
    return {
        0: {
            "id": "call_test",
            "name": "submit_plan",
            "arguments": json.dumps({"steps": steps_data}, ensure_ascii=False)
        }
    }


@pytest.mark.asyncio
async def test_generate_plan_success():
    """测试成功生成计划"""
    available_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气信息",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
            }
        },
    ]

    steps_data = [
        {
            "id": "step1",
            "description": "查询天气",
            "tool_name": "get_weather",
            "tool_args": {"location": "广州"},
            "depends_on": []
        },
        {
            "id": "step2",
            "description": "翻译结果",
            "tool_name": "get_weather",
            "tool_args": {"text": "$step1"},
            "depends_on": ["step1"]
        }
    ]

    tool_calls = _make_submit_plan_tool_calls(steps_data)

    with patch('src.plan.planner.call_model', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("", tool_calls, "stop")

        plan = await generate_plan(
            user_input="查询广州天气并翻译",
            available_tools=available_tools,
            available_agents=["weather_agent"],
        )

        assert mock_call.called
        call_kwargs = mock_call.call_args[1]
        assert "tools" in call_kwargs
        tool_names = [t["function"]["name"] for t in call_kwargs["tools"]]
        assert "get_weather" in tool_names
        assert "submit_plan" in tool_names

        assert len(plan.steps) == 2
        assert plan.steps[0].tool_name == "get_weather"
        assert plan.steps[0].tool_args == {"location": "广州"}
        assert plan.steps[1].depends_on == ["step1"]


@pytest.mark.asyncio
async def test_generate_plan_no_tool_call():
    """LLM 不调用 submit_plan → 返回 None"""
    with patch('src.plan.planner.call_model', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("简单问题", {}, "stop")
        plan = await generate_plan("你好", [], available_agents=[])
        assert plan is None


@pytest.mark.asyncio
async def test_generate_plan_empty_steps():
    """空步骤列表 → 返回 None"""
    tool_calls = _make_submit_plan_tool_calls([])
    with patch('src.plan.planner.call_model', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("", tool_calls, "stop")
        plan = await generate_plan("测试", [], available_agents=[])
        assert plan is None


@pytest.mark.asyncio
async def test_generate_plan_invalid_json():
    """无效 JSON → 返回 None"""
    tool_calls = {0: {"id": "call_test", "name": "submit_plan", "arguments": "无效JSON"}}
    with patch('src.plan.planner.call_model', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("", tool_calls, "stop")
        plan = await generate_plan("测试", [], available_agents=[])
        assert plan is None


@pytest.mark.asyncio
async def test_generate_plan_api_error():
    """API 调用失败 → 抛出 APIGenerationError"""
    from src.plan.exceptions import APIGenerationError
    with patch('src.plan.planner.call_model', new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = Exception("连接失败")
        with pytest.raises(APIGenerationError):
            await generate_plan("测试", [], available_agents=[])


@pytest.mark.asyncio
async def test_generate_plan_with_context():
    """上下文信息传入 prompt"""
    with patch('src.plan.planner.call_model', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("不需要", {}, "stop")
        await generate_plan("测试", [], available_agents=[], context="额外信息")
        user_prompt = mock_call.call_args[0][0][1]["content"]
        assert "额外信息" in user_prompt


@pytest.mark.asyncio
async def test_generate_plan_agents_in_prompt():
    """available_agents 出现在 system prompt 中"""
    with patch('src.plan.planner.call_model', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("不需要", {}, "stop")
        await generate_plan("测试", [], available_agents=["email_agent", "weather_agent"])
        system_prompt = mock_call.call_args[0][0][0]["content"]
        assert "email_agent" in system_prompt
        assert "weather_agent" in system_prompt


@pytest.mark.asyncio
async def test_adjust_plan_success():
    """成功调整计划"""
    original_plan = Plan(steps=[
        Step(id="s1", description="原始步骤", tool_name="test_tool")
    ])

    adjusted_steps = [
        {"id": "s1", "description": "修改后", "tool_name": "test_tool", "tool_args": {"p": "v"}, "depends_on": []},
        {"id": "s2", "description": "新增", "agent_name": "helper", "depends_on": ["s1"]},
    ]
    tool_calls = _make_submit_plan_tool_calls(adjusted_steps)

    with patch('src.plan.planner.call_model', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("", tool_calls, "stop")
        new_plan = await adjust_plan("原请求", original_plan, "添加步骤", [], available_agents=["helper"])
        assert len(new_plan.steps) == 2
        assert new_plan.steps[1].agent_name == "helper"


@pytest.mark.asyncio
async def test_adjust_plan_failure_keeps_original():
    """调整失败 → 返回原计划"""
    original_plan = Plan(steps=[
        Step(id="s1", description="原始", tool_name="t")
    ])
    with patch('src.plan.planner.call_model', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("无法调整", {}, "stop")
        new_plan = await adjust_plan("测试", original_plan, "修改", [], available_agents=[])
        assert new_plan == original_plan


@pytest.mark.asyncio
async def test_adjust_plan_api_error_keeps_original():
    """API 失败 → 返回原计划"""
    original_plan = Plan(steps=[
        Step(id="s1", description="原始", tool_name="t")
    ])
    with patch('src.plan.planner.call_model', new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = Exception("连接失败")
        result = await adjust_plan("测试", original_plan, "反馈", [], available_agents=[])
        assert result == original_plan


class TestParsePlanFromToolCalls:
    def test_valid(self):
        tool_calls = _make_submit_plan_tool_calls([
            {"id": "s1", "description": "步骤1", "tool_name": "t1"}
        ])
        plan = parse_plan_from_tool_calls(tool_calls)
        assert plan is not None
        assert plan.steps[0].tool_name == "t1"

    def test_no_submit_plan(self):
        tool_calls = {0: {"id": "x", "name": "other", "arguments": "{}"}}
        assert parse_plan_from_tool_calls(tool_calls) is None

    def test_empty(self):
        assert parse_plan_from_tool_calls({}) is None

    def test_invalid_json(self):
        tool_calls = {0: {"id": "x", "name": "submit_plan", "arguments": "bad"}}
        assert parse_plan_from_tool_calls(tool_calls) is None


class TestSubmitPlanSchema:
    def test_schema_structure(self):
        assert _SUBMIT_PLAN_SCHEMA["function"]["name"] == "submit_plan"
        params = _SUBMIT_PLAN_SCHEMA["function"]["parameters"]
        assert "steps" in params["properties"]

    def test_schema_has_key_fields(self):
        schema_str = json.dumps(_SUBMIT_PLAN_SCHEMA)
        assert "tool_name" in schema_str
        assert "agent_name" in schema_str
        props = _SUBMIT_PLAN_SCHEMA["function"]["parameters"]
        defs = props.get("$defs", {})
        step_props = {}
        for def_val in defs.values():
            if "tool_name" in def_val.get("properties", {}):
                step_props = def_val["properties"]
                break
        if step_props:
            assert "action" not in step_props
