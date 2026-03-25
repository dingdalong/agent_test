"""MeetingBookingFlow：多步表单示例 — 预订会议室。

状态流转：
  collect_date → collect_time → collect_attendees → collect_equipment → confirm → booking → done
  任意状态 → cancelled

触发关键词：/book、预订会议室
"""

import logging
from typing import List

from statemachine import StateMachine, State

from src.core.fsm import FlowModel, OUTPUT_PREFIX
from src.flows import register_flow

logger = logging.getLogger(__name__)


class BookingModel(FlowModel):
    """会议预订 flow 的 model。"""
    pass


class MeetingBookingFlow(StateMachine):
    """会议室预订：逐步收集日期/时间/人数/设备 → 确认 → 预订"""

    # === 状态 ===
    collect_date = State(initial=True)
    collect_time = State()
    collect_attendees = State()
    collect_equipment = State()
    confirm = State()
    booking = State()
    done = State(final=True)
    cancelled = State(final=True)

    # === 转移 ===
    proceed = (
        collect_date.to(collect_time)
        | collect_time.to(collect_attendees)
        | collect_attendees.to(collect_equipment)
        | collect_equipment.to(confirm)
        | confirm.to(booking, cond="user_confirmed")
        | confirm.to(collect_date, cond="user_restart")
        | booking.to(done)
    )

    cancel = (
        collect_date.to(cancelled)
        | collect_time.to(cancelled)
        | collect_attendees.to(cancelled)
        | collect_equipment.to(cancelled)
        | confirm.to(cancelled)
    )

    def __init__(self, **kwargs):
        model = BookingModel()
        super().__init__(model=model)

    # === 条件 ===

    def user_confirmed(self) -> bool:
        return self.model.data.get("confirm_action") == "confirm"

    def user_restart(self) -> bool:
        return self.model.data.get("confirm_action") == "restart"

    # === 状态回调 ===

    async def on_enter_collect_date(self):
        self.model.output_text = f"\n{OUTPUT_PREFIX}请输入会议日期（如 2026-03-28）：\n"
        self.model.needs_input = True

    async def on_exit_collect_date(self):
        if self.model.user_input:
            self.model.data["date"] = self.model.user_input.strip()

    async def on_enter_collect_time(self):
        self.model.output_text = f"\n{OUTPUT_PREFIX}请输入会议时间（如 14:00-15:00）：\n"
        self.model.needs_input = True

    async def on_exit_collect_time(self):
        if self.model.user_input:
            self.model.data["time"] = self.model.user_input.strip()

    async def on_enter_collect_attendees(self):
        self.model.output_text = f"\n{OUTPUT_PREFIX}请输入参会人数：\n"
        self.model.needs_input = True

    async def on_exit_collect_attendees(self):
        if self.model.user_input:
            self.model.data["attendees"] = self.model.user_input.strip()

    async def on_enter_collect_equipment(self):
        self.model.output_text = (
            f"\n{OUTPUT_PREFIX}需要哪些设备？（如：投影仪、白板、视频会议，输入'无'跳过）：\n"
        )
        self.model.needs_input = True

    async def on_exit_collect_equipment(self):
        if self.model.user_input:
            text = self.model.user_input.strip()
            self.model.data["equipment"] = text if text != "无" else ""

    async def on_enter_confirm(self):
        d = self.model.data
        summary = (
            f"  📅 日期：{d.get('date', '未填')}\n"
            f"  🕐 时间：{d.get('time', '未填')}\n"
            f"  👥 人数：{d.get('attendees', '未填')}\n"
            f"  🖥️ 设备：{d.get('equipment') or '无'}\n"
        )
        self.model.output_text = (
            f"\n{OUTPUT_PREFIX}请确认预订信息：\n{summary}\n"
            f"{OUTPUT_PREFIX}输入 '确认' 提交预订，'重填' 重新输入，或 '取消' 放弃：\n"
        )
        self.model.needs_input = True

    async def prepare_proceed(self):
        """在 send('proceed') 前根据 user_input 设置条件数据。"""
        if self.model.user_input and self.confirm in self.configuration:
            text = self.model.user_input.strip()
            if text in ("确认", "确定", "ok", "yes", "y"):
                self.model.data["confirm_action"] = "confirm"
            elif text in ("重填", "重新输入", "restart"):
                self.model.data["confirm_action"] = "restart"
            else:
                # 默认当作确认
                self.model.data["confirm_action"] = "confirm"

    async def on_enter_booking(self):
        d = self.model.data
        # 这里可以调用实际的预订工具，目前模拟
        result = (
            f"会议室已预订成功！\n"
            f"  日期：{d.get('date')}\n"
            f"  时间：{d.get('time')}\n"
            f"  人数：{d.get('attendees')}\n"
            f"  设备：{d.get('equipment') or '无'}"
        )
        self.model.output_text = f"\n{OUTPUT_PREFIX}✅ {result}\n"
        self.model.result = result
        self.model.needs_input = False

    async def on_enter_done(self):
        pass

    async def on_enter_cancelled(self):
        self.model.output_text = f"\n{OUTPUT_PREFIX}预订已取消。\n"
        self.model.result = "预订已取消。"


# === 注册触发关键词 ===

def _create_meeting_flow(**kwargs) -> MeetingBookingFlow:
    return MeetingBookingFlow(**kwargs)

register_flow("/book", _create_meeting_flow)
register_flow("预订会议室", _create_meeting_flow)
