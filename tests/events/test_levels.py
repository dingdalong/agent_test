"""EventLevel 枚举测试。"""
from src.events.levels import EventLevel


def test_level_values():
    assert EventLevel.PROGRESS == 1
    assert EventLevel.DETAIL == 2
    assert EventLevel.TRACE == 3


def test_level_ordering():
    assert EventLevel.PROGRESS < EventLevel.DETAIL < EventLevel.TRACE


def test_from_str():
    assert EventLevel.from_str("progress") == EventLevel.PROGRESS
    assert EventLevel.from_str("DETAIL") == EventLevel.DETAIL
    assert EventLevel.from_str("trace") == EventLevel.TRACE


def test_from_str_default():
    assert EventLevel.from_str("invalid") == EventLevel.PROGRESS
