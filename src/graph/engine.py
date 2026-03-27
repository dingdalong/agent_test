"""GraphEngine — 异步图执行器（Agent 无关）。"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from src.graph.types import CompiledGraph, GraphNode, NodeResult, ParallelGroup
from src.graph.hooks import GraphHooks

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT", bound=BaseModel)


@dataclass
class GraphResult(Generic[StateT]):
    """图执行的最终结果。"""
    output: Any
    state: StateT
    trace: list = field(default_factory=list)


class GraphEngine:
    """异步图执行器。Agent 无关 — 只负责图的遍历和执行。"""

    def __init__(
        self,
        hooks: GraphHooks | None = None,
        max_handoff_depth: int = 10,
    ):
        self.hooks = hooks or GraphHooks()
        self.max_handoff_depth = max_handoff_depth

    async def run(self, graph: CompiledGraph, context: Any) -> GraphResult:
        """执行编译后的图。"""
        await self.hooks.on_graph_start(context)

        last_output: Any = None
        pending: list[str] = [graph.entry]
        visited: set[str] = set()

        while pending:
            parallel_group = self._find_parallel_group(pending, graph.parallel_groups)

            if parallel_group:
                nodes_to_run = parallel_group.nodes
                results = await self._run_parallel(nodes_to_run, graph, context)
                for name, node_result in results.items():
                    last_output = node_result.output
                    self._write_state(context, name, node_result.output)
                    visited.add(name)
                pending = [parallel_group.then]
            else:
                current_name = pending.pop(0)
                if current_name in visited and current_name != graph.entry:
                    continue

                node = graph.nodes.get(current_name)
                if node is None:
                    logger.warning(f"Node '{current_name}' not found, skipping")
                    continue

                node_result = await self._execute_node(node, context)
                last_output = node_result.output
                self._write_state(context, current_name, node_result.output)
                visited.add(current_name)

                # 处理 handoff
                if node_result.handoff:
                    target = node_result.handoff.target
                    context.depth += 1
                    if context.depth > self.max_handoff_depth:
                        logger.warning(f"Max handoff depth reached ({self.max_handoff_depth})")
                    elif target in graph.nodes:
                        context.input = node_result.handoff.task
                        pending = [target]
                        continue
                    else:
                        logger.error(f"Handoff target '{target}' not found in graph")
                        self._add_trace(context, current_name, "error",
                                        {"error": f"Handoff target '{target}' not found"})
                    continue

                # 处理显式 next
                if node_result.next is not None:
                    if isinstance(node_result.next, list):
                        pending = node_result.next
                    else:
                        pending = [node_result.next]
                    continue

                # 按边路由
                next_nodes = self._resolve_edges(current_name, graph, context)
                pending = next_nodes

        result = GraphResult(
            output=last_output,
            state=context.state,
            trace=list(getattr(context, "trace", [])),
        )
        await self.hooks.on_graph_end(context, result)
        return result

    async def _execute_node(self, node: GraphNode, context: Any) -> NodeResult:
        await self.hooks.on_node_start(node.name, context)
        self._add_trace(context, node.name, "start")
        try:
            result = await node.execute(context)
        except Exception as e:
            self._add_trace(context, node.name, "error", {"error": str(e)})
            raise
        self._add_trace(context, node.name, "end")
        await self.hooks.on_node_end(node.name, context, result)
        return result

    async def _run_parallel(
        self, node_names: list[str], graph: CompiledGraph, context: Any,
    ) -> dict[str, NodeResult]:
        async def _run_one(name: str) -> tuple[str, NodeResult]:
            node = graph.nodes[name]
            result = await self._execute_node(node, context)
            return name, result

        tasks = [_run_one(name) for name in node_names]
        results = await asyncio.gather(*tasks)
        return dict(results)

    def _resolve_edges(
        self, source: str, graph: CompiledGraph, context: Any,
    ) -> list[str]:
        next_nodes = []
        for edge in graph.edges:
            if edge.source != source:
                continue
            if edge.condition is None or edge.condition(context):
                next_nodes.append(edge.target)
        return next_nodes

    def _find_parallel_group(
        self, pending: list[str], groups: list[ParallelGroup],
    ) -> ParallelGroup | None:
        pending_set = set(pending)
        for group in groups:
            if pending_set & set(group.nodes):
                return group
        return None

    def _write_state(self, context: Any, node_name: str, output: Any) -> None:
        try:
            setattr(context.state, node_name, output)
        except (AttributeError, ValueError):
            logger.debug(f"Cannot set state.{node_name}, state type may not support it")

    def _add_trace(self, context: Any, node: str, event: str, data: dict | None = None) -> None:
        trace = getattr(context, "trace", None)
        if trace is not None and hasattr(trace, "append"):
            trace.append({
                "node": node, "event": event, "timestamp": time.time(), "data": data or {},
            })
