"""RunContext 泛型 + TraceEvent 测试。"""
import time
import pytest
from pydantic import BaseModel, ConfigDict


class MyState(BaseModel):
    counter: int = 0
    result: str = ""


class MyDeps(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    api_key: str = "test-key"


def test_run_context_typed_state():
    from src.agents.context import RunContext

    ctx = RunContext(input="hello", state=MyState(), deps=MyDeps())
    assert ctx.state.counter == 0
    ctx.state.counter = 5
    assert ctx.state.counter == 5


def test_run_context_typed_deps():
    from src.agents.context import RunContext

    ctx = RunContext(input="hi", state=MyState(), deps=MyDeps(api_key="secret"))
    assert ctx.deps.api_key == "secret"


def test_run_context_defaults():
    from src.agents.context import RunContext, DictState, EmptyDeps

    ctx = RunContext(input="test", state=DictState(), deps=EmptyDeps())
    assert ctx.current_agent == ""
    assert ctx.depth == 0
    assert ctx.trace == []


def test_run_context_dict_state_allows_extra():
    from src.agents.context import RunContext, DictState, EmptyDeps

    ctx = RunContext(input="test", state=DictState(), deps=EmptyDeps())
    ctx.state.weather = {"temp": 25}
    assert ctx.state.weather == {"temp": 25}


def test_trace_event():
    from src.agents.context import TraceEvent

    event = TraceEvent(node="weather", event="start", timestamp=time.time())
    assert event.node == "weather"
    assert event.event == "start"
    assert event.data == {}


def test_trace_event_with_data():
    from src.agents.context import TraceEvent

    event = TraceEvent(
        node="weather",
        event="tool_call",
        timestamp=1234567890.0,
        data={"tool": "get_weather", "args": {"city": "Beijing"}},
    )
    assert event.data["tool"] == "get_weather"


def test_run_context_add_trace():
    from src.agents.context import RunContext, DictState, EmptyDeps, TraceEvent

    ctx = RunContext(input="test", state=DictState(), deps=EmptyDeps())
    ctx.trace.append(TraceEvent(node="a", event="start", timestamp=1.0))
    ctx.trace.append(TraceEvent(node="a", event="end", timestamp=2.0))
    assert len(ctx.trace) == 2
    assert ctx.trace[0].event == "start"
    assert ctx.trace[1].event == "end"
