"""事件类型定义 — 所有 EventBus 可发布的事件。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union

from src.events.levels import EventLevel


@dataclass
class Event:
    """事件基类。"""

    timestamp: float
    source: str
    level: EventLevel
    type: str = ""


# --- PROGRESS 级别 ---

@dataclass
class GraphStarted(Event):
    level: EventLevel = field(default=EventLevel.PROGRESS, init=False)
    type: Literal["graph_started"] = field(default="graph_started", init=False)


@dataclass
class GraphEnded(Event):
    output: Any = None
    level: EventLevel = field(default=EventLevel.PROGRESS, init=False)
    type: Literal["graph_ended"] = field(default="graph_ended", init=False)


@dataclass
class NodeStarted(Event):
    node_type: str = ""
    level: EventLevel = field(default=EventLevel.PROGRESS, init=False)
    type: Literal["node_started"] = field(default="node_started", init=False)


@dataclass
class NodeEnded(Event):
    output_summary: str = ""
    level: EventLevel = field(default=EventLevel.PROGRESS, init=False)
    type: Literal["node_ended"] = field(default="node_ended", init=False)


@dataclass
class ErrorOccurred(Event):
    error: str = ""
    level: EventLevel = field(default=EventLevel.PROGRESS, init=False)
    type: Literal["error"] = field(default="error", init=False)


# --- DETAIL 级别 ---

@dataclass
class AgentStarted(Event):
    agent_name: str = ""
    level: EventLevel = field(default=EventLevel.DETAIL, init=False)
    type: Literal["agent_started"] = field(default="agent_started", init=False)


@dataclass
class AgentEnded(Event):
    agent_name: str = ""
    level: EventLevel = field(default=EventLevel.DETAIL, init=False)
    type: Literal["agent_ended"] = field(default="agent_ended", init=False)


@dataclass
class ToolCalled(Event):
    tool_name: str = ""
    args: dict = field(default_factory=dict)
    level: EventLevel = field(default=EventLevel.DETAIL, init=False)
    type: Literal["tool_called"] = field(default="tool_called", init=False)


@dataclass
class ToolResult(Event):
    tool_name: str = ""
    result: str = ""
    level: EventLevel = field(default=EventLevel.DETAIL, init=False)
    type: Literal["tool_result"] = field(default="tool_result", init=False)


@dataclass
class Handoff(Event):
    from_agent: str = ""
    to_agent: str = ""
    task: str = ""
    level: EventLevel = field(default=EventLevel.DETAIL, init=False)
    type: Literal["handoff"] = field(default="handoff", init=False)


# --- TRACE 级别 ---

@dataclass
class TokenDelta(Event):
    delta: str = ""
    level: EventLevel = field(default=EventLevel.TRACE, init=False)
    type: Literal["token_delta"] = field(default="token_delta", init=False)


@dataclass
class ThinkingDelta(Event):
    content: str = ""
    level: EventLevel = field(default=EventLevel.TRACE, init=False)
    type: Literal["thinking_delta"] = field(default="thinking_delta", init=False)


# 联合类型
AgentEvent = Union[
    GraphStarted, GraphEnded,
    NodeStarted, NodeEnded,
    ErrorOccurred,
    AgentStarted, AgentEnded,
    ToolCalled, ToolResult,
    Handoff,
    TokenDelta, ThinkingDelta,
]
