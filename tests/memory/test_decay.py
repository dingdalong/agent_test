"""
Unit tests for the memory decay system (decay.py).
"""

import math
from datetime import datetime, timedelta, timezone

from src.memory.decay import (
    FREQUENCY_CAP,
    RECENCY_LAMBDA,
    calculate_importance,
)
from src.memory.types import MemoryRecord, MemoryType


def _make_fact(**overrides) -> MemoryRecord:
    defaults = dict(
        memory_type=MemoryType.FACT,
        content="test",
        confidence=1.0,
        access_count=0,
    )
    defaults.update(overrides)
    return MemoryRecord(**defaults)


class TestCalculateImportance:

    def test_summary_always_returns_one(self):
        record = MemoryRecord(
            memory_type=MemoryType.SUMMARY,
            content="summary",
            confidence=0.1,
            access_count=0,
        )
        now = datetime.now(timezone.utc)
        assert calculate_importance(record, now) == 1.0

    def test_brand_new_zero_access(self):
        """A brand-new fact with 0 access should have 0 frequency weight -> importance = 0."""
        now = datetime.now(timezone.utc)
        record = _make_fact(last_accessed=now, access_count=0, confidence=1.0)
        importance = calculate_importance(record, now)
        # log(0+1)/log(20) = 0 -> importance = 0
        assert round(importance, 4) == round(0.0, 4)

    def test_one_access_produces_nonzero(self):
        now = datetime.now(timezone.utc)
        record = _make_fact(last_accessed=now, access_count=1, confidence=1.0)
        importance = calculate_importance(record, now)
        expected_freq = math.log(2) / math.log(FREQUENCY_CAP)
        assert round(importance, 4) == round(expected_freq, 4)

    def test_max_frequency_cap(self):
        """At access_count = FREQUENCY_CAP - 1, frequency weight should be ~1.0."""
        now = datetime.now(timezone.utc)
        record = _make_fact(
            last_accessed=now,
            access_count=FREQUENCY_CAP - 1,
            confidence=1.0,
        )
        importance = calculate_importance(record, now)
        # frequency_w = min(1.0, log(20)/log(20)) = 1.0, recency_w ~ 1.0
        assert round(importance, 2) == round(1.0, 2)

    def test_recency_decay_over_time(self):
        """Importance should decrease as days pass."""
        now = datetime.now(timezone.utc)
        record_recent = _make_fact(
            last_accessed=now, access_count=5, confidence=0.9,
        )
        record_old = _make_fact(
            last_accessed=now - timedelta(days=70),
            access_count=5,
            confidence=0.9,
        )
        imp_recent = calculate_importance(record_recent, now)
        imp_old = calculate_importance(record_old, now)
        assert imp_recent > imp_old

    def test_half_life_approximately_70_days(self):
        """At ~70 days, recency weight should be about 0.5."""
        now = datetime.now(timezone.utc)
        half_life_days = math.log(2) / RECENCY_LAMBDA
        record = _make_fact(
            last_accessed=now - timedelta(days=half_life_days),
            access_count=FREQUENCY_CAP,  # max frequency so it doesn't affect test
            confidence=1.0,
        )
        importance = calculate_importance(record, now)
        # Should be approximately 0.5
        assert round(importance, 1) == round(0.5, 1)

    def test_confidence_scales_linearly(self):
        now = datetime.now(timezone.utc)
        record_high = _make_fact(
            last_accessed=now, access_count=10, confidence=1.0,
        )
        record_low = _make_fact(
            last_accessed=now, access_count=10, confidence=0.5,
        )
        imp_high = calculate_importance(record_high, now)
        imp_low = calculate_importance(record_low, now)
        assert round(imp_low / imp_high, 2) == round(0.5, 2)

    def test_custom_lambda(self):
        """Custom recency_lambda should change decay rate."""
        now = datetime.now(timezone.utc)
        record = _make_fact(
            last_accessed=now - timedelta(days=10),
            access_count=10,
            confidence=1.0,
        )
        imp_default = calculate_importance(record, now)
        imp_fast = calculate_importance(record, now, recency_lambda=0.1)
        # Faster decay -> lower importance
        assert imp_default > imp_fast

    def test_now_defaults_to_utc(self):
        """If now is not provided, should use current UTC time."""
        record = _make_fact(
            last_accessed=datetime.now(timezone.utc),
            access_count=5,
            confidence=0.9,
        )
        importance = calculate_importance(record)
        assert importance > 0.0

    def test_returns_rounded_value(self):
        now = datetime.now(timezone.utc)
        record = _make_fact(
            last_accessed=now - timedelta(days=3),
            access_count=7,
            confidence=0.85,
        )
        importance = calculate_importance(record, now)
        # Should be rounded to 4 decimal places
        assert importance == round(importance, 4)
