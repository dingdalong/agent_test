# Plan 模块重构实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 plan 模块从"生成+执行一体"重构为"轻量编排层"——plan 只负责生成/管理计划，执行委托给 GraphEngine。

**Architecture:** PlanCompiler 将 Plan 转换为 CompiledGraph（每个 Step → FunctionNode，depends_on → Edge/ParallelGroup），然后由 GraphEngine 执行。planner.py 保留完整的澄清→生成→确认流程，新增 available_agents 参数。main.py 通过 /plan 命令和 orchestrator handoff 两种方式触发计划流程。

**Tech Stack:** Python 3.13, Pydantic v2, asyncio, pytest

**Spec:** `docs/superpowers/specs/2026-03-27-plan-refactor-design.md`

---

### Task 1: 重写 models.py

**Files:**
- Modify: `src/plan/models.py`
- Test: `tests/plan/test_models.py`

- [ ] **Step 1: 重写测试文件**

```python
"""测试 plan.models 模块"""
import pytest
from src.plan.models import Step, Plan


class TestStepModel:
    def test_tool_step(self):
        """工具步骤：tool_name 有值"""
        step = Step(
            id="weather",
            description="查询天气",
            tool_name="get_weather",
            tool_args={"location": "广州"},
        )
        assert step.id == "weather"
        assert step.tool_name == "get_weather"
        assert step.tool_args == {"location": "广州"}
        assert step.agent_name is None
        assert step.agent_prompt is None
        assert step.depends_on == []

    def test_agent_step(self):
        """Agent 步骤：agent_name 有值"""
        step = Step(
            id="draft",
            description="起草邮件",
            agent_name="email_agent",
            agent_prompt="根据天气信息起草一封邮件",
            depends_on=["weather"],
        )
        assert step.agent_name == "email_agent"
        assert step.agent_prompt == "根据天气信息起草一封邮件"
        assert step.tool_name is None
        assert step.depends_on == ["weather"]

    def test_tool_step_defaults(self):
        """工具步骤默认值"""
        step = Step(id="s1", description="测试", tool_name="test_tool")
        assert step.tool_args == {}
        assert step.depends_on == []

    def test_agent_step_defaults(self):
        """Agent 步骤默认值"""
        step = Step(id="s1", description="测试", agent_name="helper")
        assert step.agent_prompt is None
        assert step.depends_on == []

    def test_both_tool_and_agent_raises(self):
        """同时设置 tool_name 和 agent_name 报错"""
        with pytest.raises(ValueError, match="cannot have both"):
            Step(
                id="s1",
                description="冲突",
                tool_name="some_tool",
                agent_name="some_agent",
            )

    def test_neither_tool_nor_agent_raises(self):
        """两者都没设置报错"""
        with pytest.raises(ValueError, match="must have either"):
            Step(id="s1", description="空步骤")

    def test_variable_references_in_tool_args(self):
        """tool_args 中的 $step_id.field 变量引用"""
        step = Step(
            id="translate",
            description="翻译",
            tool_name="translate",
            tool_args={"text": "$search.results", "lang": "zh"},
            depends_on=["search"],
        )
        assert step.tool_args["text"] == "$search.results"


class TestPlanModel:
    def test_basic_plan(self):
        plan = Plan(steps=[
            Step(id="s1", description="步骤1", tool_name="t1"),
            Step(id="s2", description="步骤2", tool_name="t2", depends_on=["s1"]),
        ])
        assert len(plan.steps) == 2
        assert plan.context == {}

    def test_plan_with_context(self):
        plan = Plan(
            steps=[Step(id="s1", description="测试", tool_name="t1")],
            context={"user_id": "123"},
        )
        assert plan.context == {"user_id": "123"}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `python -m pytest tests/plan/test_models.py -v`
Expected: FAIL — 旧 Step 模型不匹配新测试

- [ ] **Step 3: 重写 models.py**

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional


class Step(BaseModel):
    """计划中的单个步骤。

    类型由字段值决定：
    - tool_name 有值 → 工具调用步骤
    - agent_name 有值 → Agent 委托步骤
    """

    id: str = Field(description="步骤唯一标识")
    description: str = Field(description="步骤描述，用于展示")
    tool_name: Optional[str] = Field(default=None, description="工具名称（工具步骤）")
    tool_args: dict = Field(default_factory=dict, description="工具参数，支持 $step_id.field 变量引用")
    agent_name: Optional[str] = Field(default=None, description="Agent 名称（委托步骤）")
    agent_prompt: Optional[str] = Field(default=None, description="Agent 指令")
    depends_on: list[str] = Field(default_factory=list, description="依赖的步骤 ID 列表")

    @model_validator(mode="after")
    def validate_step_type(self):
        has_tool = self.tool_name is not None
        has_agent = self.agent_name is not None
        if has_tool and has_agent:
            raise ValueError("Step cannot have both tool_name and agent_name")
        if not has_tool and not has_agent:
            raise ValueError("Step must have either tool_name or agent_name")
        return self


class Plan(BaseModel):
    """完整计划"""

    steps: list[Step] = Field(description="步骤列表")
    context: dict = Field(default_factory=dict, description="初始上下文")
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/plan/test_models.py -v`
Expected: ALL PASS

- [ ] **Step 5: 提交**

```bash
git add src/plan/models.py tests/plan/test_models.py
git commit -m "refactor(plan): rewrite Step model — remove action, add agent_name/agent_prompt"
```

---

### Task 2: 精简 exceptions.py

**Files:**
- Modify: `src/plan/exceptions.py`

- [ ] **Step 1: 重写 exceptions.py**

```python
"""计划模块的自定义异常类"""

from config import PLAN_MAX_RAW_RESPONSE_LENGTH


class PlanError(Exception):
    """计划系统基类异常"""
    pass


class JSONParseError(PlanError):
    """JSON 解析失败"""

    def __init__(self, message: str, raw_response: str = None):
        super().__init__(message)
        self.raw_response = raw_response

    def __str__(self) -> str:
        base = super().__str__()
        if self.raw_response and len(self.raw_response) < PLAN_MAX_RAW_RESPONSE_LENGTH:
            return f"{base} (原始响应: {self.raw_response})"
        elif self.raw_response:
            return f"{base} (原始响应过长，已截断)"
        return base


class APIGenerationError(PlanError):
    """API 生成失败"""

    def __init__(self, message: str, api_error: Exception = None):
        super().__init__(message)
        self.api_error = api_error

    def __str__(self) -> str:
        base = super().__str__()
        if self.api_error:
            return f"{base} (API错误: {self.api_error})"
        return base


class CompileError(PlanError):
    """Plan → CompiledGraph 编译失败"""

    def __init__(self, message: str, details: list[str] | None = None):
        super().__init__(message)
        self.details = details or []

    def __str__(self) -> str:
        base = super().__str__()
        if self.details:
            return f"{base} ({'; '.join(self.details)})"
        return base
```

- [ ] **Step 2: 运行已有测试确保不回归**

Run: `python -m pytest tests/plan/test_models.py -v`
Expected: ALL PASS（models 不依赖 exceptions）

- [ ] **Step 3: 提交**

```bash
git add src/plan/exceptions.py
git commit -m "refactor(plan): slim down exceptions — remove executor errors, add CompileError"
```

---

### Task 3: 实现 compiler.py

这是最核心的新增组件。Plan → CompiledGraph 的转换。

**Files:**
- Create: `src/plan/compiler.py`
- Test: `tests/plan/test_compiler.py`

- [ ] **Step 1: 编写测试文件**

```python
"""测试 plan.compiler 模块"""
import pytest
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

from src.plan.models import Step, Plan
from src.plan.compiler import PlanCompiler, resolve_variables
from src.plan.exceptions import CompileError
from src.agents import (
    AgentRegistry, Agent, RunContext, DictState, GraphEngine,
    NodeResult, FunctionNode,
)


# === resolve_variables 单元测试 ===

class TestResolveVariables:
    def test_simple_reference(self):
        ctx = {"search": "sunny 25°C"}
        assert resolve_variables("$search", ctx) == "sunny 25°C"

    def test_nested_reference(self):
        ctx = {"search": {"results": ["a", "b"]}}
        assert resolve_variables("$search.results", ctx) == ["a", "b"]

    def test_deep_nested(self):
        ctx = {"api": {"data": {"name": "test"}}}
        assert resolve_variables("$api.data.name", ctx) == "test"

    def test_dict_args(self):
        ctx = {"s1": "hello"}
        result = resolve_variables({"text": "$s1", "lang": "zh"}, ctx)
        assert result == {"text": "hello", "lang": "zh"}

    def test_list_args(self):
        ctx = {"s1": "a", "s2": "b"}
        result = resolve_variables(["$s1", "$s2", "literal"], ctx)
        assert result == ["a", "b", "literal"]

    def test_missing_reference_unchanged(self):
        ctx = {}
        assert resolve_variables("$missing", ctx) == "$missing"

    def test_non_variable_unchanged(self):
        ctx = {"s1": "val"}
        assert resolve_variables("plain text", ctx) == "plain text"
        assert resolve_variables(42, ctx) == 42
        assert resolve_variables(None, ctx) is None


# === PlanCompiler 单元测试 ===

def _make_registry_and_router():
    """创建测试用的 registry 和 router mock"""
    registry = AgentRegistry()
    registry.register(Agent(
        name="helper",
        description="帮助 agent",
        instructions="你是助手",
    ))

    router = MagicMock()
    router.get_all_schemas.return_value = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    return registry, router


class TestPlanCompilerValidation:
    def test_empty_plan_raises(self):
        registry, router = _make_registry_and_router()
        compiler = PlanCompiler(registry, router)
        with pytest.raises(CompileError, match="空计划"):
            compiler.compile(Plan(steps=[]))

    def test_duplicate_step_ids_raises(self):
        registry, router = _make_registry_and_router()
        compiler = PlanCompiler(registry, router)
        plan = Plan(steps=[
            Step(id="s1", description="a", tool_name="get_weather"),
            Step(id="s1", description="b", tool_name="get_weather"),
        ])
        with pytest.raises(CompileError, match="重复"):
            compiler.compile(plan)

    def test_missing_dependency_raises(self):
        registry, router = _make_registry_and_router()
        compiler = PlanCompiler(registry, router)
        plan = Plan(steps=[
            Step(id="s1", description="a", tool_name="get_weather", depends_on=["nonexistent"]),
        ])
        with pytest.raises(CompileError, match="不存在"):
            compiler.compile(plan)

    def test_cycle_raises(self):
        registry, router = _make_registry_and_router()
        compiler = PlanCompiler(registry, router)
        plan = Plan(steps=[
            Step(id="s1", description="a", tool_name="get_weather", depends_on=["s2"]),
            Step(id="s2", description="b", tool_name="get_weather", depends_on=["s1"]),
        ])
        with pytest.raises(CompileError, match="循环"):
            compiler.compile(plan)

    def test_unknown_agent_raises(self):
        registry, router = _make_registry_and_router()
        compiler = PlanCompiler(registry, router)
        plan = Plan(steps=[
            Step(id="s1", description="a", agent_name="nonexistent"),
        ])
        with pytest.raises(CompileError, match="Agent.*不存在"):
            compiler.compile(plan)


class TestPlanCompilerCompile:
    def test_single_tool_step(self):
        """单步工具计划 → 单节点图"""
        registry, router = _make_registry_and_router()
        compiler = PlanCompiler(registry, router)
        plan = Plan(steps=[
            Step(id="weather", description="查询天气", tool_name="get_weather"),
        ])
        graph = compiler.compile(plan)
        assert graph.entry == "weather"
        assert "weather" in graph.nodes
        assert isinstance(graph.nodes["weather"], FunctionNode)

    def test_sequential_steps(self):
        """顺序依赖 → 顺序边"""
        registry, router = _make_registry_and_router()
        compiler = PlanCompiler(registry, router)
        plan = Plan(steps=[
            Step(id="s1", description="第一步", tool_name="get_weather"),
            Step(id="s2", description="第二步", tool_name="get_weather", depends_on=["s1"]),
        ])
        graph = compiler.compile(plan)
        assert graph.entry == "s1"
        edge_pairs = [(e.source, e.target) for e in graph.edges]
        assert ("s1", "s2") in edge_pairs

    def test_parallel_steps(self):
        """无依赖的同层步骤 → ParallelGroup"""
        registry, router = _make_registry_and_router()
        compiler = PlanCompiler(registry, router)
        plan = Plan(steps=[
            Step(id="a", description="A", tool_name="get_weather"),
            Step(id="b", description="B", tool_name="get_weather"),
            Step(id="c", description="汇总", tool_name="get_weather", depends_on=["a", "b"]),
        ])
        graph = compiler.compile(plan)
        assert graph.entry == "a"
        assert len(graph.parallel_groups) == 1
        pg = graph.parallel_groups[0]
        assert set(pg.nodes) == {"a", "b"}

    def test_agent_step(self):
        """Agent 步骤 → FunctionNode（内部调用 AgentRunner）"""
        registry, router = _make_registry_and_router()
        compiler = PlanCompiler(registry, router)
        plan = Plan(steps=[
            Step(id="draft", description="起草", agent_name="helper"),
        ])
        graph = compiler.compile(plan)
        assert "draft" in graph.nodes
        assert isinstance(graph.nodes["draft"], FunctionNode)


@pytest.mark.asyncio
class TestPlanCompilerExecution:
    async def test_tool_step_calls_router(self):
        """编译后的工具步骤执行时调用 tool_router.route"""
        registry, router = _make_registry_and_router()
        router.route = AsyncMock(return_value="晴天 25°C")
        compiler = PlanCompiler(registry, router)

        plan = Plan(steps=[
            Step(id="weather", description="查天气", tool_name="get_weather", tool_args={"location": "广州"}),
        ])
        graph = compiler.compile(plan)
        engine = GraphEngine(registry=registry)

        from pydantic import ConfigDict
        from pydantic import BaseModel

        class TestDeps(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            tool_router: object = None

        ctx = RunContext(input="test", state=DictState(), deps=TestDeps(tool_router=router))
        result = await engine.run(graph, ctx)

        router.route.assert_called_once_with("get_weather", {"location": "广州"})
        assert result.output == "晴天 25°C"

    async def test_variable_resolution_during_execution(self):
        """执行时 $step_id.field 变量正确解析"""
        registry, router = _make_registry_and_router()

        call_count = 0
        async def mock_route(tool_name, args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "sunny"
            return f"translated: {args.get('text', '')}"

        router.route = mock_route
        compiler = PlanCompiler(registry, router)

        plan = Plan(steps=[
            Step(id="search", description="搜索", tool_name="get_weather"),
            Step(
                id="translate",
                description="翻译",
                tool_name="get_weather",
                tool_args={"text": "$search"},
                depends_on=["search"],
            ),
        ])
        graph = compiler.compile(plan)
        engine = GraphEngine(registry=registry)

        from pydantic import ConfigDict, BaseModel

        class TestDeps(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            tool_router: object = None

        ctx = RunContext(input="test", state=DictState(), deps=TestDeps(tool_router=router))
        result = await engine.run(graph, ctx)

        assert "sunny" in str(result.output)
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `python -m pytest tests/plan/test_compiler.py -v`
Expected: FAIL — compiler.py 不存在

- [ ] **Step 3: 实现 compiler.py**

```python
"""PlanCompiler — 将 Plan 转换为 CompiledGraph。"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from src.plan.models import Plan, Step
from src.plan.exceptions import CompileError
from src.agents.context import RunContext, DictState
from src.agents.graph.types import FunctionNode, NodeResult, CompiledGraph, Edge, ParallelGroup
from src.agents.graph.builder import GraphBuilder
from src.agents.registry import AgentRegistry
from src.agents.runner import AgentRunner
from src.tools.router import ToolRouter

logger = logging.getLogger(__name__)

VARIABLE_PREFIX = "$"


def resolve_variables(obj: Any, context: dict) -> Any:
    """解析变量引用，支持 $step_id.field 嵌套语法。

    Args:
        obj: 待解析的对象（str / dict / list / 其他原样返回）
        context: 变量上下文字典
    """
    if isinstance(obj, dict):
        return {k: resolve_variables(v, context) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_variables(v, context) for v in obj]
    if isinstance(obj, str) and obj.startswith(VARIABLE_PREFIX):
        parts = obj[len(VARIABLE_PREFIX):].split(".")
        cur: Any = context
        for part in parts:
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return obj  # 路径不存在，原样返回
        return cur
    return obj


def _state_to_dict(state: Any) -> dict:
    """将 DictState（Pydantic extra='allow'）转为普通 dict。"""
    if hasattr(state, "model_extra") and state.model_extra:
        return dict(state.model_extra)
    return {}


def _topological_sort_layered(steps: list[Step]) -> list[list[Step]]:
    """分层拓扑排序。同一层的步骤无互相依赖，可并行。

    Raises:
        CompileError: 循环依赖或缺失依赖
    """
    step_ids = {s.id for s in steps}
    step_map = {s.id: s for s in steps}

    # 检查依赖存在性
    for step in steps:
        for dep in step.depends_on:
            if dep not in step_ids:
                raise CompileError(
                    f"步骤 {step.id} 依赖不存在的步骤 {dep}",
                    details=[f"{step.id} -> {dep}"],
                )

    # 计算入度
    in_degree = {s.id: 0 for s in steps}
    dependents: dict[str, list[str]] = {s.id: [] for s in steps}
    for step in steps:
        for dep in step.depends_on:
            in_degree[step.id] += 1
            dependents[dep].append(step.id)

    # 分层 BFS
    layers: list[list[Step]] = []
    queue = [sid for sid, deg in in_degree.items() if deg == 0]

    while queue:
        layers.append([step_map[sid] for sid in queue])
        next_queue: list[str] = []
        for sid in queue:
            for dep_id in dependents[sid]:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    next_queue.append(dep_id)
        queue = next_queue

    sorted_count = sum(len(layer) for layer in layers)
    if sorted_count != len(steps):
        unsorted = step_ids - {s.id for layer in layers for s in layer}
        raise CompileError(
            f"存在循环依赖：{unsorted}",
            details=[f"循环涉及步骤: {', '.join(unsorted)}"],
        )

    return layers


class PlanCompiler:
    """将 Plan 编译为 CompiledGraph。"""

    def __init__(self, agent_registry: AgentRegistry, tool_router: ToolRouter):
        self._registry = agent_registry
        self._router = tool_router
        self._runner = AgentRunner(registry=agent_registry)

    def compile(self, plan: Plan) -> CompiledGraph:
        """Plan → CompiledGraph。

        1. 验证计划合法性
        2. 分层拓扑排序
        3. 每个 Step → FunctionNode
        4. 层间关系 → Edge / ParallelGroup
        """
        self._validate(plan)
        layers = _topological_sort_layered(plan.steps)

        builder = GraphBuilder()
        prev_exit: str | None = None

        for layer_idx, layer in enumerate(layers):
            # 为本层每个 step 创建 FunctionNode
            for step in layer:
                fn = self._make_node_fn(step)
                builder.add_function(step.id, fn)

            if len(layer) == 1:
                step_id = layer[0].id
                if layer_idx == 0:
                    builder.set_entry(step_id)
                if prev_exit is not None:
                    builder.add_edge(prev_exit, step_id)
                prev_exit = step_id
            else:
                # 多个步骤 → ParallelGroup + merge 节点
                merge_name = f"_merge_{layer_idx}"
                merge_fn = self._make_merge_fn()
                builder.add_function(merge_name, merge_fn)
                builder.add_parallel([s.id for s in layer], then=merge_name)

                if layer_idx == 0:
                    builder.set_entry(layer[0].id)
                if prev_exit is not None:
                    for step in layer:
                        builder.add_edge(prev_exit, step.id)
                prev_exit = merge_name

        return builder.compile()

    def _validate(self, plan: Plan) -> None:
        """编译前验证。"""
        if not plan.steps:
            raise CompileError("空计划：没有步骤")

        # 检查 ID 唯一性
        seen: set[str] = set()
        for step in plan.steps:
            if step.id in seen:
                raise CompileError(f"重复的步骤 ID: {step.id}")
            seen.add(step.id)

        # 检查 agent 存在性
        for step in plan.steps:
            if step.agent_name and not self._registry.has(step.agent_name):
                raise CompileError(f"Agent '{step.agent_name}' 不存在于注册表中")

    def _make_node_fn(self, step: Step):
        """为一个 Step 创建 FunctionNode 的执行函数。"""
        if step.tool_name is not None:
            return self._make_tool_fn(step)
        return self._make_agent_fn(step)

    def _make_tool_fn(self, step: Step):
        """工具步骤 → 闭包函数。"""
        tool_name = step.tool_name
        tool_args = step.tool_args
        router = self._router

        async def fn(ctx: RunContext) -> NodeResult:
            resolved = resolve_variables(tool_args, _state_to_dict(ctx.state))
            result = await router.route(tool_name, resolved)
            return NodeResult(output=result)

        return fn

    def _make_agent_fn(self, step: Step):
        """Agent 步骤 → 闭包函数。"""
        agent_name = step.agent_name
        agent_prompt = step.agent_prompt or step.description
        registry = self._registry
        runner = self._runner

        async def fn(ctx: RunContext) -> NodeResult:
            resolved_prompt = resolve_variables(agent_prompt, _state_to_dict(ctx.state))
            agent = registry.get(agent_name)
            agent_ctx = replace(ctx, input=resolved_prompt)
            result = await runner.run(agent, agent_ctx)
            return NodeResult(output={"text": result.text, "data": result.data})

        return fn

    @staticmethod
    def _make_merge_fn():
        """并行组合并节点 → 空操作透传。"""

        async def fn(ctx: RunContext) -> NodeResult:
            return NodeResult(output=None)

        return fn
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/plan/test_compiler.py -v`
Expected: ALL PASS

- [ ] **Step 5: 提交**

```bash
git add src/plan/compiler.py tests/plan/test_compiler.py
git commit -m "feat(plan): add PlanCompiler — Plan to CompiledGraph conversion"
```

---

### Task 4: 更新 planner.py

**Files:**
- Modify: `src/plan/planner.py`
- Modify: `tests/plan/test_planner.py`

- [ ] **Step 1: 更新测试——添加 available_agents 参数**

在 `tests/plan/test_planner.py` 中做以下修改：

1. 更新 `_make_submit_plan_tool_calls` 中的 steps_data 格式：`action` 字段改为 `tool_name` / `agent_name`。

2. 所有 `generate_plan` 调用添加 `available_agents=[]` 参数。

3. 所有 `adjust_plan` 调用添加 `available_agents=[]` 参数。

4. 所有 steps_data 中去掉 `action` 字段。

完整替换 `tests/plan/test_planner.py`：

```python
"""测试 plan.planner 模块"""
import pytest
import json
from unittest.mock import AsyncMock, patch
from src.plan.planner import generate_plan, adjust_plan, parse_plan_from_tool_calls, _SUBMIT_PLAN_SCHEMA
from src.plan.models import Plan, Step


def _make_submit_plan_tool_calls(steps_data: list) -> dict:
    """构造 submit_plan 的 tool_calls 返回格式"""
    return {
        0: {
            "id": "call_test",
            "name": "submit_plan",
            "arguments": json.dumps({"steps": steps_data}, ensure_ascii=False)
        }
    }


@pytest.mark.asyncio
async def test_generate_plan_success():
    """测试成功生成计划"""
    available_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气信息",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
            }
        },
    ]

    steps_data = [
        {
            "id": "step1",
            "description": "查询天气",
            "tool_name": "get_weather",
            "tool_args": {"location": "广州"},
            "depends_on": []
        },
        {
            "id": "step2",
            "description": "翻译结果",
            "tool_name": "get_weather",
            "tool_args": {"text": "$step1"},
            "depends_on": ["step1"]
        }
    ]

    tool_calls = _make_submit_plan_tool_calls(steps_data)

    with patch('src.plan.planner.call_model', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("", tool_calls, "stop")

        plan = await generate_plan(
            user_input="查询广州天气并翻译",
            available_tools=available_tools,
            available_agents=["weather_agent"],
        )

        assert mock_call.called
        call_kwargs = mock_call.call_args[1]
        assert "tools" in call_kwargs
        tool_names = [t["function"]["name"] for t in call_kwargs["tools"]]
        assert "get_weather" in tool_names
        assert "submit_plan" in tool_names

        assert len(plan.steps) == 2
        assert plan.steps[0].tool_name == "get_weather"
        assert plan.steps[0].tool_args == {"location": "广州"}
        assert plan.steps[1].depends_on == ["step1"]


@pytest.mark.asyncio
async def test_generate_plan_no_tool_call():
    """LLM 不调用 submit_plan → 返回 None"""
    with patch('src.plan.planner.call_model', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("简单问题", {}, "stop")
        plan = await generate_plan("你好", [], available_agents=[])
        assert plan is None


@pytest.mark.asyncio
async def test_generate_plan_empty_steps():
    """空步骤列表 → 返回 None"""
    tool_calls = _make_submit_plan_tool_calls([])
    with patch('src.plan.planner.call_model', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("", tool_calls, "stop")
        plan = await generate_plan("测试", [], available_agents=[])
        assert plan is None


@pytest.mark.asyncio
async def test_generate_plan_invalid_json():
    """无效 JSON → 返回 None"""
    tool_calls = {0: {"id": "call_test", "name": "submit_plan", "arguments": "无效JSON"}}
    with patch('src.plan.planner.call_model', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("", tool_calls, "stop")
        plan = await generate_plan("测试", [], available_agents=[])
        assert plan is None


@pytest.mark.asyncio
async def test_generate_plan_api_error():
    """API 调用失败 → 抛出 APIGenerationError"""
    from src.plan.exceptions import APIGenerationError
    with patch('src.plan.planner.call_model', new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = Exception("连接失败")
        with pytest.raises(APIGenerationError):
            await generate_plan("测试", [], available_agents=[])


@pytest.mark.asyncio
async def test_generate_plan_with_context():
    """上下文信息传入 prompt"""
    with patch('src.plan.planner.call_model', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("不需要", {}, "stop")
        await generate_plan("测试", [], available_agents=[], context="额外信息")
        user_prompt = mock_call.call_args[0][0][1]["content"]
        assert "额外信息" in user_prompt


@pytest.mark.asyncio
async def test_generate_plan_agents_in_prompt():
    """available_agents 出现在 system prompt 中"""
    with patch('src.plan.planner.call_model', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("不需要", {}, "stop")
        await generate_plan("测试", [], available_agents=["email_agent", "weather_agent"])
        system_prompt = mock_call.call_args[0][0][0]["content"]
        assert "email_agent" in system_prompt
        assert "weather_agent" in system_prompt


@pytest.mark.asyncio
async def test_adjust_plan_success():
    """成功调整计划"""
    original_plan = Plan(steps=[
        Step(id="s1", description="原始步骤", tool_name="test_tool")
    ])

    adjusted_steps = [
        {"id": "s1", "description": "修改后", "tool_name": "test_tool", "tool_args": {"p": "v"}, "depends_on": []},
        {"id": "s2", "description": "新增", "agent_name": "helper", "depends_on": ["s1"]},
    ]
    tool_calls = _make_submit_plan_tool_calls(adjusted_steps)

    with patch('src.plan.planner.call_model', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("", tool_calls, "stop")
        new_plan = await adjust_plan("原请求", original_plan, "添加步骤", [], available_agents=["helper"])
        assert len(new_plan.steps) == 2
        assert new_plan.steps[1].agent_name == "helper"


@pytest.mark.asyncio
async def test_adjust_plan_failure_keeps_original():
    """调整失败 → 返回原计划"""
    original_plan = Plan(steps=[
        Step(id="s1", description="原始", tool_name="t")
    ])
    with patch('src.plan.planner.call_model', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("无法调整", {}, "stop")
        new_plan = await adjust_plan("测试", original_plan, "修改", [], available_agents=[])
        assert new_plan == original_plan


@pytest.mark.asyncio
async def test_adjust_plan_api_error_keeps_original():
    """API 失败 → 返回原计划"""
    original_plan = Plan(steps=[
        Step(id="s1", description="原始", tool_name="t")
    ])
    with patch('src.plan.planner.call_model', new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = Exception("连接失败")
        result = await adjust_plan("测试", original_plan, "反馈", [], available_agents=[])
        assert result == original_plan


class TestParsePlanFromToolCalls:
    def test_valid(self):
        tool_calls = _make_submit_plan_tool_calls([
            {"id": "s1", "description": "步骤1", "tool_name": "t1"}
        ])
        plan = parse_plan_from_tool_calls(tool_calls)
        assert plan is not None
        assert plan.steps[0].tool_name == "t1"

    def test_no_submit_plan(self):
        tool_calls = {0: {"id": "x", "name": "other", "arguments": "{}"}}
        assert parse_plan_from_tool_calls(tool_calls) is None

    def test_empty(self):
        assert parse_plan_from_tool_calls({}) is None

    def test_invalid_json(self):
        tool_calls = {0: {"id": "x", "name": "submit_plan", "arguments": "bad"}}
        assert parse_plan_from_tool_calls(tool_calls) is None


class TestSubmitPlanSchema:
    def test_schema_structure(self):
        assert _SUBMIT_PLAN_SCHEMA["function"]["name"] == "submit_plan"
        params = _SUBMIT_PLAN_SCHEMA["function"]["parameters"]
        assert "steps" in params["properties"]

    def test_schema_has_key_fields(self):
        schema_str = json.dumps(_SUBMIT_PLAN_SCHEMA)
        assert "tool_name" in schema_str
        assert "agent_name" in schema_str
        # action 字段应该已被移除
        # 注意：description 中可能包含 action 一词，所以只检查 properties 层
        props = _SUBMIT_PLAN_SCHEMA["function"]["parameters"]
        defs = props.get("$defs", {})
        step_props = {}
        for def_val in defs.values():
            if "tool_name" in def_val.get("properties", {}):
                step_props = def_val["properties"]
                break
        if step_props:
            assert "action" not in step_props
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `python -m pytest tests/plan/test_planner.py -v`
Expected: FAIL — 旧 planner 签名不匹配

- [ ] **Step 3: 更新 planner.py**

修改点：
1. `generate_plan` 签名加 `available_agents: list[str]`
2. `adjust_plan` 签名加 `available_agents: list[str]`
3. `PLAN_GENERATION_SYSTEM_PROMPT` 更新：去掉 `action` 字段说明，改为 `tool_name` / `agent_name` 区分，加可用 agent 列表
4. `PLAN_ADJUSTMENT_SYSTEM_PROMPT` 同步更新
5. `_build_plan_tools` 加 agent 列表信息

替换 `src/plan/planner.py` 全文：

```python
import json
import logging
from typing import Literal, Optional

from pydantic import BaseModel
from src.core.async_api import call_model
from src.plan.models import Plan, Step
from src.plan.exceptions import JSONParseError, APIGenerationError, PlanError
from src.tools import ToolDict
from src.core.structured_output import build_output_schema, parse_output

logger = logging.getLogger(__name__)


# === submit_plan 虚拟工具 ===

_SUBMIT_PLAN_SCHEMA = build_output_schema(
    "submit_plan",
    "提交执行计划。每个步骤用 tool_name 或 agent_name 指定执行方式。如果请求是简单问答或闲聊，不要调用此工具。",
    Plan
)


# === classify_feedback 虚拟工具 ===

class FeedbackClassification(BaseModel):
    action: Literal["confirm", "adjust"]

_CLASSIFY_FEEDBACK_TOOL = build_output_schema(
    "classify_feedback",
    "输出用户反馈的分类结果：confirm=确认执行, adjust=要求调整",
    FeedbackClassification
)


def _build_plan_tools(available_tools: list[ToolDict]) -> list:
    """构建计划生成用的工具列表：真实工具 + submit_plan"""
    return list(available_tools) + [_SUBMIT_PLAN_SCHEMA]


def _build_agent_list_text(available_agents: list[str]) -> str:
    """构建可用 Agent 列表的文本段落"""
    if not available_agents:
        return ""
    agent_list = ", ".join(available_agents)
    return f"\n\n可用的 Agent（可通过 agent_name 委托任务）：{agent_list}"


# === Prompt 模板 ===

CLARIFICATION_SYSTEM_PROMPT = """你是一个任务规划助手。在制定计划之前，你需要判断用户的请求是否提供了足够的信息来生成一个具体可执行的计划。

请根据"用户原始请求"和"已收集的信息"，判断当前掌握的信息是否足够生成一个具体、可执行的计划。

输出规则：
- 如果信息已经充足，只回复一个词：READY
- 如果还缺少关键信息，直接向用户提问（自然语言，不要输出 JSON）

判断标准：
- 对照用户的原始请求，思考要完成这件事需要哪些关键信息（如时间、地点、人数、预算、偏好、约束条件等）
- 检查"已收集的信息"中是否已经覆盖了这些关键要素
- 如果已收集的信息已经涵盖了主要要素，即使不完美也应该返回 READY，不要过度追问
- 只针对尚未回答的关键信息提问，绝对不要重复问已经回答过的问题
- 提问要自然友好，像朋友对话一样，每次聚焦 2-3 个问题

重要：关注执行所需的信息
- 如果计划涉及发邮件、发消息等操作，必须确认收件人/接收方信息
- 如果涉及文件保存，确认文件名或保存位置偏好
- 绝不要凭空编造用户未提供的信息（如邮箱地址、电话号码、姓名等）"""

PLAN_GENERATION_SYSTEM_PROMPT_TEMPLATE = """你是一个任务规划助手。根据用户请求，判断是否需要生成多步骤计划。

规则：
- 如果请求是简单的问答、闲聊或单步操作，直接回复文本，不要调用 submit_plan
- 如果请求需要多个步骤协作完成，调用 submit_plan 工具提交计划

步骤类型（二选一）：
- tool_name: 调用工具执行操作，tool_args 指定参数。tool_name 和 tool_args 必须严格匹配可用工具定义
- agent_name: 委托给指定 Agent 执行。agent_prompt 描述委托任务和上下文

依赖与变量：
- 步骤之间通过 depends_on 声明依赖关系
- 后续步骤可用 $step_id 或 $step_id.field 引用前序步骤的结果

严格禁止：
- 绝不要在 tool_args 中编造用户未提供的信息（如邮箱地址、电话号码等）{agent_list}"""

PLAN_ADJUSTMENT_SYSTEM_PROMPT_TEMPLATE = """你是一个任务规划助手。用户对当前计划提出了修改意见，请调整计划，然后调用 submit_plan 提交。

步骤类型（二选一）：
- tool_name + tool_args: 工具调用
- agent_name + agent_prompt: Agent 委托

规则：
- 如果用户要求删除某个步骤，从列表中移除
- 如果要求添加步骤，合理插入
- 如果要求修改步骤，更新相应步骤
- tool_name 和 tool_args 必须严格匹配可用工具定义{agent_list}"""

CONFIRM_CLASSIFICATION_PROMPT = """判断用户的回复是"确认执行计划"还是"要求调整计划"。

当前计划：
{plan_summary}

用户回复：{user_feedback}

请调用 classify_feedback 工具输出分类结果：
- 如果用户表示同意、确认、执行、没问题等意思，action 为 confirm
- 如果用户提出修改意见、补充要求、质疑等，action 为 adjust"""


# === 解析函数 ===

def parse_plan_from_tool_calls(tool_calls: dict[int, dict[str, str]]) -> Optional[Plan]:
    """从 tool_calls 中解析 submit_plan 调用，返回 Plan 对象。"""
    return parse_output(tool_calls, "submit_plan", Plan)


# === 核心功能 ===

async def classify_user_feedback(user_feedback: str, plan: Plan) -> str:
    """使用 LLM 判断用户反馈是确认还是调整"""
    plan_summary = "\n".join(
        f"{i}. {step.description}" for i, step in enumerate(plan.steps, 1)
    )
    prompt = CONFIRM_CLASSIFICATION_PROMPT.format(
        plan_summary=plan_summary, user_feedback=user_feedback
    )

    try:
        _, tool_calls, _ = await call_model(
            [{"role": "user", "content": prompt}],
            temperature=0,
            tools=[_CLASSIFY_FEEDBACK_TOOL],
            silent=True,
        )
        result = parse_output(tool_calls, "classify_feedback", FeedbackClassification)
        if result and result.action in ("confirm", "adjust"):
            return result.action
        return "adjust"
    except Exception as e:
        logger.warning(f"分类用户反馈失败: {e}, 默认为调整")
        return "adjust"


async def check_clarification_needed(user_input: str, gathered_info: str = "") -> Optional[str]:
    """判断信息是否充足。充足返回 None，否则返回追问内容。"""
    user_message = f"用户原始请求：{user_input}\n\n已收集的信息：\n{gathered_info or '（暂无）'}"

    try:
        response, _, _ = await call_model([
            {"role": "system", "content": CLARIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ], temperature=0)
        if response.strip().upper() == "READY":
            return None
        return response
    except Exception as e:
        logger.warning(f"信息充分性判断失败: {e}，跳过")
        return None


async def generate_plan(
    user_input: str,
    available_tools: list[ToolDict],
    available_agents: list[str],
    context: str = "",
) -> Optional[Plan]:
    """生成计划。LLM 判断不需要时返回 None。"""
    agent_list_text = _build_agent_list_text(available_agents)
    system_prompt = PLAN_GENERATION_SYSTEM_PROMPT_TEMPLATE.format(agent_list=agent_list_text)

    plan_tools = _build_plan_tools(available_tools)
    user_prompt = f"用户请求：{user_input}\n{context}"

    try:
        content, tool_calls, _ = await call_model(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=plan_tools,
            silent=True,
        )
    except Exception as e:
        raise APIGenerationError(f"API调用失败: {e}", api_error=e) from e

    if not tool_calls:
        return None

    plan = parse_plan_from_tool_calls(tool_calls)
    if plan is None or not plan.steps:
        return None
    return plan


async def adjust_plan(
    original_request: str,
    current_plan: Plan,
    feedback: str,
    available_tools: list[ToolDict],
    available_agents: list[str],
) -> Plan:
    """根据反馈调整计划。失败时返回原计划。"""
    agent_list_text = _build_agent_list_text(available_agents)
    system_prompt = PLAN_ADJUSTMENT_SYSTEM_PROMPT_TEMPLATE.format(agent_list=agent_list_text)

    plan_json = json.dumps(
        {"steps": [s.model_dump(exclude_none=True) for s in current_plan.steps]},
        ensure_ascii=False, indent=2,
    )
    plan_tools = _build_plan_tools(available_tools)

    user_prompt = f"""原任务：{original_request}

当前计划：
{plan_json}

用户反馈：{feedback}"""

    try:
        content, tool_calls, _ = await call_model(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=plan_tools,
            silent=True,
        )
    except Exception as e:
        logger.error(f"API调用失败: {e}")
        return current_plan

    try:
        plan = parse_plan_from_tool_calls(tool_calls or {})
        if plan is None or not plan.steps:
            logger.warning("调整后未获得有效计划，保留原计划")
            return current_plan
        return plan
    except Exception as e:
        logger.error(f"解析调整后计划失败: {e}, 原计划保留")
        return current_plan
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/plan/test_planner.py -v`
Expected: ALL PASS

- [ ] **Step 5: 提交**

```bash
git add src/plan/planner.py tests/plan/test_planner.py
git commit -m "refactor(plan): update planner with available_agents param and new Step format"
```

---

### Task 5: 更新 __init__.py

**Files:**
- Modify: `src/plan/__init__.py`

- [ ] **Step 1: 更新导出**

```python
"""计划模块 — 任务规划与编译"""

from src.plan.models import Plan, Step
from src.plan.planner import (
    generate_plan,
    adjust_plan,
    classify_user_feedback,
    check_clarification_needed,
)
from src.plan.compiler import PlanCompiler, resolve_variables
from src.plan.exceptions import PlanError, JSONParseError, APIGenerationError, CompileError

__all__ = [
    "Plan",
    "Step",
    "generate_plan",
    "adjust_plan",
    "classify_user_feedback",
    "check_clarification_needed",
    "PlanCompiler",
    "resolve_variables",
    "PlanError",
    "JSONParseError",
    "APIGenerationError",
    "CompileError",
]
```

- [ ] **Step 2: 运行所有 plan 测试**

Run: `python -m pytest tests/plan/ -v`
Expected: ALL PASS

- [ ] **Step 3: 提交**

```bash
git add src/plan/__init__.py
git commit -m "refactor(plan): update __init__.py exports for new module structure"
```

---

### Task 6: 清理 config.py 和删除 executor

**Files:**
- Modify: `config.py`
- Delete: `src/plan/executor.py`
- Delete: `tests/plan/test_executor.py`

- [ ] **Step 1: 删除 config.py 中不再需要的常量**

在 `config.py` 中删除：

```python
PLAN_DEFAULT_TIMEOUT = 120.0
PLAN_MAX_VARIABLE_DEPTH = 10
```

保留：

```python
PLAN_MAX_ADJUSTMENTS = 3
PLAN_MAX_CLARIFICATION_ROUNDS = 3
PLAN_MAX_RAW_RESPONSE_LENGTH = 500
```

- [ ] **Step 2: 删除 executor.py 和 test_executor.py**

```bash
rm src/plan/executor.py tests/plan/test_executor.py
```

- [ ] **Step 3: 运行所有 plan 测试确认无回归**

Run: `python -m pytest tests/plan/ -v`
Expected: ALL PASS（executor 测试已删除，剩余测试不依赖 executor）

- [ ] **Step 4: 提交**

```bash
git add -A src/plan/executor.py tests/plan/test_executor.py config.py
git commit -m "refactor(plan): remove executor.py and unused config constants"
```

---

### Task 7: 集成到 main.py

**Files:**
- Modify: `main.py`
- Modify: `config.py` (如果需要新增常量)

- [ ] **Step 1: 扩展 AgentDeps 模型**

在 `main.py` 中修改 `AgentDeps`，添加 `agent_registry` 和 `graph_engine` 字段：

```python
class AgentDeps(BaseModel):
    """外部依赖：传递给 AgentRunner 和 plan 流程。"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tool_router: Any = None
    agent_registry: Any = None
    graph_engine: Any = None
```

- [ ] **Step 2: 实现 run_plan_flow 函数**

在 `main.py` 中，`handle_input` 之前添加：

```python
from src.plan import (
    generate_plan, adjust_plan, classify_user_feedback,
    check_clarification_needed, PlanCompiler,
)
from config import PLAN_MAX_CLARIFICATION_ROUNDS, PLAN_MAX_ADJUSTMENTS


def _format_plan(plan) -> str:
    """格式化计划用于展示"""
    lines = []
    for i, step in enumerate(plan.steps, 1):
        deps = f" (依赖: {', '.join(step.depends_on)})" if step.depends_on else ""
        if step.tool_name:
            lines.append(f"  {i}. [工具] {step.description} -> {step.tool_name}{deps}")
        elif step.agent_name:
            lines.append(f"  {i}. [Agent] {step.description} -> {step.agent_name}{deps}")
    return "\n".join(lines)


async def run_plan_flow(
    user_input: str,
    tool_router: ToolRouter,
    agent_registry: AgentRegistry,
    engine: GraphEngine,
) -> str:
    """完整的计划流程：澄清 → 生成 → 确认 → 编译 → 执行。"""
    available_tools = tool_router.get_all_schemas()
    available_agents = [a.name for a in agent_registry.all_agents()]

    # 1. 澄清循环
    gathered = ""
    for _ in range(PLAN_MAX_CLARIFICATION_ROUNDS):
        question = await check_clarification_needed(user_input, gathered)
        if question is None:
            break
        await agent_output(f"\n{question}\n")
        answer = await agent_input("\n你: ")
        gathered += f"\n{question}\n回答: {answer}"

    # 2. 生成计划
    context = gathered if gathered else ""
    plan = await generate_plan(user_input, available_tools, available_agents, context)
    if plan is None:
        return "这个请求不需要多步计划，我直接回答。"

    # 3. 确认/调整循环
    for _ in range(PLAN_MAX_ADJUSTMENTS):
        plan_display = _format_plan(plan)
        await agent_output(f"\n执行计划：\n{plan_display}\n")
        feedback_input = await agent_input("\n确认执行？(输入 '确认' 或修改意见): ")

        action = await classify_user_feedback(feedback_input, plan)
        if action == "confirm":
            break
        plan = await adjust_plan(
            user_input, plan, feedback_input, available_tools, available_agents
        )

    # 4. 编译并执行
    compiler = PlanCompiler(agent_registry, tool_router)
    compiled_graph = compiler.compile(plan)

    ctx = RunContext(
        input=user_input,
        state=DictState(),
        deps=AgentDeps(tool_router=tool_router),
    )
    result = await engine.run(compiled_graph, ctx)

    # 提取输出
    output = result.output
    if isinstance(output, dict):
        output = output.get("text", str(output))
    return str(output) if output else "计划执行完成。"
```

- [ ] **Step 3: 添加 planner FunctionNode 到图中**

修改 `_build_agents_and_graph`，添加 planner 节点和 orchestrator handoff：

```python
def _build_agents_and_graph(registry: AgentRegistry, skill_content: str | None = None):
    """创建 agent 定义、注册到 registry，构建并编译图。"""
    # Specialist agents（不变）
    weather_agent = Agent(
        name="weather_agent",
        description="处理天气查询",
        instructions="你是天气助手。使用 get_weather 工具查询天气信息并回复用户。",
        tools=["get_weather"],
    )
    calendar_agent = Agent(
        name="calendar_agent",
        description="管理日历事件",
        instructions="你是日历助手。使用 create_event 工具帮用户管理日历事件。",
        tools=["create_event"],
    )
    email_agent = Agent(
        name="email_agent",
        description="发送邮件",
        instructions="你是邮件助手。使用 send_email 工具帮用户发送邮件。",
        tools=["send_email"],
    )

    # Orchestrator — 新增 planner handoff
    base_instructions = (
        "你是一个智能助手。根据用户的请求选择合适的操作：\n"
        "- 天气相关问题，交给 weather_agent\n"
        "- 日历/日程相关问题，交给 calendar_agent\n"
        "- 邮件相关问题，交给 email_agent\n"
        "- 需要多步骤协作的复杂任务（如查天气然后发邮件），交给 planner\n"
        "- 其他问题，直接回答用户\n"
    )
    if skill_content:
        base_instructions = f"{skill_content}\n\n{base_instructions}"

    orchestrator = Agent(
        name="orchestrator",
        description="总控 Agent，负责路由和直接回答",
        instructions=base_instructions,
        handoffs=["weather_agent", "calendar_agent", "email_agent", "planner"],
    )

    # Planner agent（用于 handoff 工具生成的描述）
    planner_agent = Agent(
        name="planner",
        description="处理需要多步骤的复杂任务，生成计划并按步骤执行",
        instructions="",  # 不会被 AgentRunner 使用
    )

    # Register
    for ag in [weather_agent, calendar_agent, email_agent, orchestrator, planner_agent]:
        registry.register(ag)

    # Build graph — planner 作为 FunctionNode
    async def planner_node_fn(ctx: RunContext) -> NodeResult:
        router = ctx.deps.tool_router
        agent_reg = ctx.deps.agent_registry
        engine = ctx.deps.graph_engine
        result = await run_plan_flow(ctx.input, router, agent_reg, engine)
        return NodeResult(output=result)

    graph = (
        GraphBuilder()
        .add_agent("orchestrator", orchestrator)
        .add_function("planner", planner_node_fn)
        .set_entry("orchestrator")
        .compile()
    )
    return graph
```

- [ ] **Step 4: 更新 handle_input — 添加 /plan 检测**

修改 `handle_input`，在 skill 检测之前添加 /plan 处理：

```python
async def handle_input(
    user_input: str,
    router: ToolRouter,
    engine: GraphEngine,
    graph,
    skill_manager=None,
    agent_registry=None,
):
    """统一入口：护栏 → /plan → Skill 斜杠命令 → GraphEngine 执行"""
    # 1. 护栏检查
    passed, reason = input_guard.check(user_input)
    if not passed:
        await agent_output(f"\n[安全拦截] {reason}\n")
        return

    # 2. /plan 命令检测
    if user_input.strip().startswith("/plan"):
        plan_request = user_input.strip()[5:].strip()
        if not plan_request:
            await agent_output("\n请在 /plan 后输入你的请求，例如：/plan 查询广州天气并发邮件给同事\n")
            return
        result = await run_plan_flow(plan_request, router, agent_registry, engine)
        await agent_output(f"\n{result}\n")
        return

    # 3. Skill 斜杠命令检测
    skill_content = None
    actual_input = user_input
    if skill_manager:
        skill_name = skill_manager.is_slash_command(user_input)
        if skill_name:
            skill_content = skill_manager.activate(skill_name)
            if skill_content:
                remaining = user_input[len(f"/{skill_name}"):].strip()
                actual_input = remaining or f"已激活 {skill_name} skill，请按指令执行。"
                skill_registry = AgentRegistry()
                skill_graph = _build_agents_and_graph(skill_registry, skill_content)
                skill_engine = GraphEngine(registry=skill_registry)
                ctx = RunContext(
                    input=actual_input,
                    state=DictState(),
                    deps=AgentDeps(tool_router=router),
                )
                result = await skill_engine.run(skill_graph, ctx)
                await agent_output(f"\n{result.output}\n")
                return

    # 4. 正常执行
    ctx = RunContext(
        input=actual_input,
        state=DictState(),
        deps=AgentDeps(
            tool_router=router,
            agent_registry=agent_registry,
            graph_engine=engine,
        ),
    )
    result = await engine.run(graph, ctx)

    output = result.output
    if isinstance(output, dict):
        output = output.get("text", str(output))
    await agent_output(f"\n{output}\n")
```

- [ ] **Step 5: 更新 main() 函数，传递 agent_registry**

修改 REPL 循环中的 `handle_input` 调用：

```python
    try:
        while True:
            user_input = await agent_input("\n你: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            await handle_input(
                user_input, router, engine, default_graph,
                skill_manager, agent_registry=agent_reg,
            )
            await agent_output("\n")
    finally:
        await mcp_manager.disconnect_all()
```

- [ ] **Step 6: 运行所有测试确认无回归**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 7: 提交**

```bash
git add main.py
git commit -m "feat(plan): integrate plan flow into main.py — /plan command + orchestrator handoff"
```

---

### Task 8: 移除 SkillManager 中的 plan 保留命令

**Files:**
- Modify: `src/skills/manager.py`

- [ ] **Step 1: 从 _RESERVED_COMMANDS 中移除 "plan"**

`/plan` 现在由 `handle_input` 直接处理，不再需要 SkillManager 保留。

在 `src/skills/manager.py` 中：

```python
# 旧
_RESERVED_COMMANDS = {"plan", "book"}

# 新
_RESERVED_COMMANDS = {"book"}
```

- [ ] **Step 2: 运行所有测试**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 3: 提交**

```bash
git add src/skills/manager.py
git commit -m "refactor(skills): remove 'plan' from reserved commands — now handled by main.py"
```
