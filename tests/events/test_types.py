"""事件类型测试。"""
import time

from src.events.types import (
    Event,
    GraphStarted,
    GraphEnded,
    NodeStarted,
    NodeEnded,
    ErrorOccurred,
    AgentStarted,
    AgentEnded,
    ToolCalled,
    ToolResult,
    Handoff,
    TokenDelta,
    ThinkingDelta,
)
from src.events.levels import EventLevel


def test_progress_events_have_correct_level():
    now = time.time()
    assert GraphStarted(timestamp=now, source="graph").level == EventLevel.PROGRESS
    assert GraphEnded(timestamp=now, source="graph", output="done").level == EventLevel.PROGRESS
    assert NodeStarted(timestamp=now, source="node1", node_type="agent").level == EventLevel.PROGRESS
    assert NodeEnded(timestamp=now, source="node1", output_summary="ok").level == EventLevel.PROGRESS
    assert ErrorOccurred(timestamp=now, source="node1", error="boom").level == EventLevel.PROGRESS


def test_detail_events_have_correct_level():
    now = time.time()
    assert AgentStarted(timestamp=now, source="agent1", agent_name="agent1").level == EventLevel.DETAIL
    assert AgentEnded(timestamp=now, source="agent1", agent_name="agent1").level == EventLevel.DETAIL
    assert ToolCalled(timestamp=now, source="agent1", tool_name="get_weather", args={"city": "北京"}).level == EventLevel.DETAIL
    assert ToolResult(timestamp=now, source="agent1", tool_name="get_weather", result="25°C").level == EventLevel.DETAIL
    assert Handoff(timestamp=now, source="router", from_agent="router", to_agent="weather", task="查天气").level == EventLevel.DETAIL


def test_trace_events_have_correct_level():
    now = time.time()
    assert TokenDelta(timestamp=now, source="llm", delta="北").level == EventLevel.TRACE
    assert ThinkingDelta(timestamp=now, source="llm", content="用户想知道...").level == EventLevel.TRACE


def test_event_type_field():
    now = time.time()
    assert GraphStarted(timestamp=now, source="g").type == "graph_started"
    assert ToolCalled(timestamp=now, source="a", tool_name="t", args={}).type == "tool_called"
    assert TokenDelta(timestamp=now, source="l", delta="x").type == "token_delta"
