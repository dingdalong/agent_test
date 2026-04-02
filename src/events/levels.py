"""EventLevel — 事件展示级别。"""

from enum import IntEnum


class EventLevel(IntEnum):
    """三级事件粒度。

    PROGRESS: 节点开始/结束、图开始/结束、错误（默认）
    DETAIL:   + 工具调用、工具结果、handoff、agent 开始/结束
    TRACE:    + thinking、token_delta
    """

    PROGRESS = 1
    DETAIL = 2
    TRACE = 3

    @classmethod
    def from_str(cls, value: str) -> "EventLevel":
        """从字符串解析级别，无效值返回 PROGRESS。"""
        try:
            return cls[value.upper()]
        except KeyError:
            return cls.PROGRESS
