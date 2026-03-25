"""FSM 基础设施：FlowModel 和 FSMRunner，驱动 python-statemachine 的 I/O 循环。"""

import logging
from typing import Any, Dict, Optional, Set

from statemachine import StateMachine

from src.core.io import agent_input, agent_output

logger = logging.getLogger(__name__)

CANCEL_KEYWORDS: Set[str] = {"取消", "算了", "cancel", "退出"}

INPUT_PREFIX = "\n你: "
OUTPUT_PREFIX = "助手: "


class FlowModel:
    """所有 Flow 的 model 基类，携带跨状态共享数据。

    Flow 的 on_enter / on_transition 回调通过 self.model 访问此对象，
    设置 output_text 和 needs_input 来控制 FSMRunner 的 I/O 行为。
    """

    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.user_input: Optional[str] = None
        self.output_text: Optional[str] = None
        self.needs_input: bool = True
        self.result: Any = None
        self.is_cancelled: bool = False


class FSMRunner:
    """驱动 StateMachine 实例的异步 I/O 循环。

    循环逻辑：
    1. activate_initial_state() → 触发初始状态的 on_enter
    2. 输出 model.output_text（如果有）
    3. 如果 model.needs_input → agent_input() 获取用户输入
       - 取消关键词 → send('cancel')
       - 否则存入 model.user_input → send('proceed')
    4. 如果 !model.needs_input → 直接 send('proceed')
    5. 重复直到 sm.is_terminated
    6. 返回 model.result
    """

    def __init__(self, sm: StateMachine):
        self.sm = sm

    @property
    def model(self) -> FlowModel:
        return self.sm.model

    async def run(self) -> Any:
        """执行 flow 直到终态，返回 model.result。"""
        await self.sm.activate_initial_state()

        while not self.sm.is_terminated:
            # 输出
            if self.model.output_text:
                await agent_output(self.model.output_text)
                self.model.output_text = None

            if self.model.needs_input:
                user_input = await agent_input(INPUT_PREFIX)

                # 取消检测
                if user_input.strip() in CANCEL_KEYWORDS:
                    self.model.is_cancelled = True
                    try:
                        await self.sm.send("cancel")
                    except Exception:
                        # 如果当前状态没有 cancel 转移，直接退出
                        logger.debug("当前状态无 cancel 转移，强制退出")
                        await agent_output(f"\n{OUTPUT_PREFIX}已取消。\n")
                        break
                    continue

                self.model.user_input = user_input
            else:
                self.model.user_input = None

            # 重置 needs_input 为默认值（各状态 on_enter 会覆盖）
            self.model.needs_input = True

            # 预处理钩子：在 send("proceed") 前调用，
            # 用于根据 user_input 设置条件所需的数据（因为 statemachine 的条件在 on_exit 之前求值）
            if hasattr(self.sm, 'prepare_proceed'):
                await self.sm.prepare_proceed()

            try:
                await self.sm.send("proceed")
            except Exception as e:
                logger.error(f"状态转移失败: {e}")
                await agent_output(f"\n{OUTPUT_PREFIX}处理出错: {e}\n")
                break

        # 最后一次输出（终态可能设置了 output_text）
        if self.model.output_text:
            await agent_output(self.model.output_text)
            self.model.output_text = None

        return self.model.result
