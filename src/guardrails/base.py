"""Guardrail Protocol — 护栏抽象接口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Protocol


@dataclass
class GuardrailResult:
    """护栏检查结果。"""

    passed: bool
    message: str = ""
    action: str = "block"  # "block" | "warn" | "rewrite"


@dataclass
class Guardrail:
    """护栏定义：名称 + 异步检查函数。"""

    name: str
    check: Callable[..., Awaitable[GuardrailResult]]
