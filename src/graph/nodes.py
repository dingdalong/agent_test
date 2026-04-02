"""附加图节点类型：决策、子图、终止。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.graph.messages import AgentResponse, ResponseStatus
from src.graph.types import NodeResult


@dataclass
class DecisionNode:
    """LLM 评估条件，选择分支。输出 chosen_branch 供条件边匹配。"""
    name: str
    question: str
    branches: list[str]

    async def execute(self, context: Any) -> NodeResult:
        options_text = ", ".join(self.branches)
        prompt = (
            f"Based on the current state, choose the next action.\n\n"
            f"Current state: {context.input}\n"
            f"Options: {options_text}\n\n"
            f"Reply with ONLY the chosen option label."
        )
        messages = [{"role": "user", "content": prompt}]
        response = await context.deps.llm.chat(messages, silent=True)
        choice = response.content.strip()
        return NodeResult(
            output=AgentResponse(
                text=choice,
                data={"chosen_branch": choice},
            ),
        )


@dataclass
class SubgraphNode:
    """嵌套执行另一个编译好的图。"""
    name: str
    sub_graph: Any  # CompiledGraph — 避免循环导入
    max_subgraph_depth: int = 3

    async def execute(self, context: Any) -> NodeResult:
        from src.agents.context import DynamicState, RunContext

        current_depth = getattr(context, "depth", 0)
        if current_depth >= self.max_subgraph_depth:
            return NodeResult(
                output=AgentResponse(
                    text=f"错误：子图嵌套深度超过限制 ({self.max_subgraph_depth})",
                    status=ResponseStatus.FAILED,
                ),
            )

        sub_ctx = RunContext(
            input=context.input,
            state=DynamicState(),
            deps=context.deps,
            depth=current_depth + 1,
        )
        result = await context.deps.engine.run(self.sub_graph, sub_ctx)
        return NodeResult(output=AgentResponse.from_graph_result(result))


@dataclass
class TerminalNode:
    """工作流终止节点，透传上一步的输出。"""
    name: str

    async def execute(self, context: Any) -> NodeResult:
        last = getattr(context.state, "_last_output", None)
        if last is None:
            last = AgentResponse(text="", data={})
        return NodeResult(output=last)
