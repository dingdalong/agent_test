"""Guardrail 运行器。"""

from __future__ import annotations

from typing import Any, Optional

from src.guardrails.base import Guardrail, GuardrailResult


async def run_guardrails(
    guardrails: list[Guardrail],
    context: Any,
    text: str,
) -> Optional[GuardrailResult]:
    """依次执行护栏列表，遇到 block 立即返回，全部通过返回 None。"""
    for guard in guardrails:
        result = await guard.check(context, text)
        if not result.passed and result.action == "block":
            return result
    return None
