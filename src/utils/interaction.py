# src/utils/interaction.py
"""UserInteractionService — 统一的用户交互入口。"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.interfaces.base import UserInterface


class UserInteractionService:
    """统一的用户交互服务。

    所有需要向用户提问的组件（工具确认、agent 提问等）
    都通过此服务交互，保证并行安全和展示一致性。
    """

    def __init__(self, ui: UserInterface) -> None:
        self._ui = ui
        self._lock = asyncio.Lock()

    async def ask(self, question: str, source: str = "") -> str:
        """向用户提出开放式问题，返回自由文本回答。

        Args:
            question: 要向用户提出的问题
            source: 提问者标识（如 agent 名称），用于展示
        """
        async with self._lock:
            label = f"[{source}] " if source else ""
            await self._ui.display(f"\n🤖 {label}提问: {question}")
            return await self._ui.prompt("你的回答: ")

    async def confirm(self, message: str) -> bool:
        """向用户请求是/否确认。

        Args:
            message: 确认提示信息
        """
        async with self._lock:
            await self._ui.display(f"\n⚠️  是否允许{message}？")
            return await self._ui.confirm("")
