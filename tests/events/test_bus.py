"""EventBus 测试 — emit/subscribe、级别过滤、类型过滤、多订阅者。"""
import asyncio
import time

import pytest

from src.events.bus import EventBus
from src.events.levels import EventLevel
from src.events.types import (
    NodeStarted,
    ToolCalled,
    TokenDelta,
    ErrorOccurred,
)


@pytest.mark.asyncio
async def test_emit_and_subscribe():
    """订阅者能收到 emit 的事件。"""
    bus = EventBus(level=EventLevel.DETAIL)
    received = []

    async def consumer():
        async for event in bus.subscribe():
            received.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)  # let consumer start

    event = ToolCalled(timestamp=time.time(), source="node1", tool_name="t", args={})
    await bus.emit(event)
    await asyncio.sleep(0)  # let consumer process

    bus.close()
    await task

    assert len(received) == 1
    assert received[0] is event


@pytest.mark.asyncio
async def test_level_filter():
    """高于 bus level 的事件不被广播。"""
    bus = EventBus(level=EventLevel.PROGRESS)
    received = []

    async def consumer():
        async for event in bus.subscribe():
            received.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    # PROGRESS 事件应该通过
    await bus.emit(TokenDelta(timestamp=time.time(), source="l", delta="x"))
    # DETAIL 事件应该被过滤
    await bus.emit(ToolCalled(timestamp=time.time(), source="a", tool_name="t", args={}))
    # TRACE 事件应该被过滤
    await bus.emit(NodeStarted(timestamp=time.time(), source="n", node_type="agent"))
    await asyncio.sleep(0)

    bus.close()
    await task

    assert len(received) == 1
    assert received[0].type == "token_delta"


@pytest.mark.asyncio
async def test_type_filter():
    """订阅者可只关注特定事件类型。"""
    bus = EventBus(level=EventLevel.DETAIL)
    received = []

    async def consumer():
        async for event in bus.subscribe(event_types={ToolCalled}):
            received.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    await bus.emit(NodeStarted(timestamp=time.time(), source="n", node_type="agent"))
    await bus.emit(ToolCalled(timestamp=time.time(), source="a", tool_name="t", args={}))
    await asyncio.sleep(0)

    bus.close()
    await task

    assert len(received) == 1
    assert received[0].type == "tool_called"


@pytest.mark.asyncio
async def test_multiple_subscribers():
    """多个订阅者各自独立收到事件。"""
    bus = EventBus(level=EventLevel.PROGRESS)
    received_a = []
    received_b = []

    async def consumer_a():
        async for event in bus.subscribe():
            received_a.append(event)

    async def consumer_b():
        async for event in bus.subscribe():
            received_b.append(event)

    task_a = asyncio.create_task(consumer_a())
    task_b = asyncio.create_task(consumer_b())
    await asyncio.sleep(0)

    await bus.emit(ErrorOccurred(timestamp=time.time(), source="n", error="test"))
    await asyncio.sleep(0)

    bus.close()
    await task_a
    await task_b

    assert len(received_a) == 1
    assert len(received_b) == 1


@pytest.mark.asyncio
async def test_set_level_runtime():
    """运行时调整级别立即生效。"""
    bus = EventBus(level=EventLevel.PROGRESS)
    received = []

    async def consumer():
        async for event in bus.subscribe():
            received.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    # DETAIL 事件先被过滤
    await bus.emit(ToolCalled(timestamp=time.time(), source="a", tool_name="t", args={}))
    await asyncio.sleep(0)
    assert len(received) == 0

    # 提升级别后可以收到
    bus.set_level(EventLevel.DETAIL)
    await bus.emit(ToolCalled(timestamp=time.time(), source="a", tool_name="t2", args={}))
    await asyncio.sleep(0)

    bus.close()
    await task

    assert len(received) == 1
    assert received[0].tool_name == "t2"


@pytest.mark.asyncio
async def test_error_event_always_passes():
    """ErrorOccurred 是 PROGRESS 级别，任何级别都应通过。"""
    bus = EventBus(level=EventLevel.PROGRESS)
    received = []

    async def consumer():
        async for event in bus.subscribe():
            received.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    await bus.emit(ErrorOccurred(timestamp=time.time(), source="n", error="boom"))
    await asyncio.sleep(0)

    bus.close()
    await task

    assert len(received) == 1
    assert received[0].type == "error"
