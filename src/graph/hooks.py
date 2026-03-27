"""GraphHooks — 图级生命周期钩子。"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional


class GraphHooks:
    """图级钩子。所有钩子均为可选，未设置时调用为 no-op。"""

    def __init__(
        self,
        on_graph_start: Optional[Callable[..., Awaitable[None]]] = None,
        on_graph_end: Optional[Callable[..., Awaitable[None]]] = None,
        on_node_start: Optional[Callable[..., Awaitable[None]]] = None,
        on_node_end: Optional[Callable[..., Awaitable[None]]] = None,
    ):
        self._on_graph_start = on_graph_start
        self._on_graph_end = on_graph_end
        self._on_node_start = on_node_start
        self._on_node_end = on_node_end

    async def on_graph_start(self, context: Any) -> None:
        if self._on_graph_start:
            await self._on_graph_start(context)

    async def on_graph_end(self, context: Any, result: Any) -> None:
        if self._on_graph_end:
            await self._on_graph_end(context, result)

    async def on_node_start(self, node_name: str, context: Any) -> None:
        if self._on_node_start:
            await self._on_node_start(node_name, context)

    async def on_node_end(self, node_name: str, context: Any, result: Any) -> None:
        if self._on_node_end:
            await self._on_node_end(node_name, context, result)
