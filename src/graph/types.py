"""图类型定义 — 节点、边、执行结果、编译后的图。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Protocol, runtime_checkable


@dataclass
class NodeResult:
    """节点执行结果。"""
    output: Any
    next: Optional[str | list[str]] = None
    handoff: Any = None


@runtime_checkable
class GraphNode(Protocol):
    """图节点协议。"""
    name: str

    async def execute(self, context: Any) -> NodeResult: ...


class FunctionNode:
    """包装一个普通 async 函数。"""

    def __init__(self, name: str, fn: Callable[..., Awaitable[NodeResult]]):
        self.name = name
        self.fn = fn

    async def execute(self, context: Any) -> NodeResult:
        return await self.fn(context)


@dataclass
class Edge:
    """节点间的连接。"""
    source: str
    target: str
    condition: Optional[Callable[..., bool]] = None


@dataclass
class ParallelGroup:
    """一组需要并行执行的节点。"""
    nodes: list[str]
    then: str


@dataclass
class CompiledGraph:
    """编译后的图，不可变，可复用。"""
    nodes: dict[str, GraphNode]
    edges: list[Edge]
    entry: str
    parallel_groups: list[ParallelGroup] = field(default_factory=list)
