from . import tool
from pydantic import BaseModel, Field
from typing import List

class CalendarArgs(BaseModel):
    title: str
    datetime: str  # 实际可用 datetime 类型，但这里简化
    attendees: List[str]

@tool(model=CalendarArgs, description="创建日历事件（模拟）")
async def create_event(title: str, datetime: str, attendees: List[str]) -> str:
    return f"事件 '{title}' 已创建，时间 {datetime}，参与人 {attendees}"
