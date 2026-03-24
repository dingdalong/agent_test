import logging
from typing import List, Dict, Any, Optional

from src.tools.tool_executor import ToolExecutor
from src.core.io import agent_input, agent_output
from src.plan.models import Plan
from src.plan.planner import generate_plan, adjust_plan, classify_user_feedback, check_clarification_needed
from src.tools import ToolDict
from src.plan.executor import execute_plan
from src.plan.exceptions import PlanError
from config import PLAN_MAX_ADJUSTMENTS, PLAN_MAX_CLARIFICATION_ROUNDS

logger = logging.getLogger(__name__)

USER_PROMPT_FINAL_CONFIRM_YES = "y"
INPUT_PREFIX = "\n你: "
OUTPUT_PREFIX = "助手: "


def format_plan_for_display(plan: Plan) -> str:
    """格式化计划为简洁的展示文本，只展示步骤描述"""
    lines = []
    for i, step in enumerate(plan.steps, 1):
        lines.append(f"  {i}. {step.description}")
    return "\n".join(lines)


def _log_plan_detail(plan: Plan) -> None:
    """将计划的详细信息记录到日志"""
    for i, step in enumerate(plan.steps, 1):
        detail = f"步骤{i} [{step.action}] {step.description}"
        if step.action == "tool" and step.tool_name:
            detail += f" | 工具: {step.tool_name}({step.tool_args or {}})"
        elif step.action == "subtask" and step.subtask_prompt:
            detail += f" | prompt: {step.subtask_prompt}"
        if step.depends_on:
            detail += f" | 依赖: {step.depends_on}"
        logger.debug(detail)


def format_execution_results(plan: Plan, result_dict: Dict[str, Any]) -> str:
    """格式化执行结果为易读文本"""
    output_lines = []
    for step in plan.steps:
        res = result_dict.get(step.id, "无结果")
        output_lines.append(f"{step.description}: {res}")
    return "\n".join(output_lines)

async def handle_planning_request(
    user_input: str,
    available_tools: List[ToolDict],
    tool_executor: ToolExecutor,
    max_adjustments: int = PLAN_MAX_ADJUSTMENTS,
) -> Optional[str]:
    """
    处理需要规划的用户请求，包含确认和调整循环。
    返回最终执行结果（字符串）。
    """
    current_plan = None
    original_request = user_input

    # === 阶段1：信息收集 ===
    gathered_info_parts = []
    for round_idx in range(PLAN_MAX_CLARIFICATION_ROUNDS):
        gathered_info = "\n".join(gathered_info_parts) if gathered_info_parts else ""
        # check_clarification_needed 流式输出问题给用户，返回问题文本或 None（信息充足）
        question = await check_clarification_needed(original_request, gathered_info)
        if question is None:
            break  # 信息充足，进入计划生成
        await agent_output("\n")
        user_answer = await agent_input(INPUT_PREFIX)
        gathered_info_parts.append(f"问：{question}\n答：{user_answer}")

    # 将收集到的信息拼接为上下文，传给计划生成
    clarification_context = "\n".join(gathered_info_parts)

    # === 阶段2：计划生成与确认 ===
    for cycle in range(max_adjustments):
        if current_plan is None:
            # 生成初始计划
            try:
                current_plan = await generate_plan(
                    original_request, available_tools, context=clarification_context
                )
            except PlanError as e:
                logger.error(f"计划生成失败: {e}")
                return "无法生成有效计划，请简化请求。"
            if current_plan is None:
                return None  # 模型判断不需要计划

        # 展示计划（简洁版给用户，详细版记录日志）
        _log_plan_detail(current_plan)
        plan_display = format_plan_for_display(current_plan)
        await agent_output(f"\n{OUTPUT_PREFIX}📋 我为你制定了以下计划：\n{plan_display}\n")
        await agent_output(f"{OUTPUT_PREFIX}是否执行此计划？输入 '确认' 开始执行，或输入修改意见。\n")

        # 询问用户
        user_feedback = await agent_input(INPUT_PREFIX)

        action = await classify_user_feedback(user_feedback, current_plan)
        if action == "confirm":
            # 确认执行，continue_on_error=True 避免单步失败导致整体崩溃
            result_dict = await execute_plan(
                current_plan, tool_executor, continue_on_error=True
            )
            logger.debug(f"执行结果: {result_dict}")
            return format_execution_results(current_plan, result_dict)
        else:
            # 调整计划
            current_plan = await adjust_plan(original_request, current_plan, user_feedback, available_tools)
            await agent_output(f"\n{OUTPUT_PREFIX}已根据你的意见调整计划。\n")

    # 达到最大调整次数，询问是否执行当前计划
    assert current_plan is not None
    await agent_output(f"\n{OUTPUT_PREFIX}已达到最大调整次数，是否仍要执行当前计划？(y/n)\n")
    final_confirm = await agent_input(INPUT_PREFIX)
    if final_confirm.lower() == USER_PROMPT_FINAL_CONFIRM_YES:
        result_dict = await execute_plan(
            current_plan, tool_executor, continue_on_error=True
        )
        return format_execution_results(current_plan, result_dict)
    else:
        return "计划已取消。"
