# Cross-Agent Delegate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable category agents to invoke other category agents via lightweight delegate tools, so cross-category tool access works without breaking the token-saving categorization.

**Architecture:** Each category agent's `tools` list is extended with `delegate_tool_<other_category>` names (excluding self). `DelegateToolProvider` (already in `ToolRouter`) handles execution. Recursion is prevented by propagating `delegate_depth` through `RunContext` — at depth ≥ 1, delegate tools are stripped from the LLM's tool list.

**Tech Stack:** Python 3.13, existing Agent/Tool framework, pytest + unittest.mock

---

### File Structure

| File | Role | Action |
|---|---|---|
| `src/agents/context.py` | RunContext dataclass | Modify: add `delegate_depth: int = 0` field |
| `src/tools/categories.py` | CategoryResolver | Modify: add `get_delegate_names(exclude)` method, update `build_instructions()` signature and template |
| `src/agents/registry.py` | AgentRegistry lazy-load | Modify: pass delegate tool names + summaries when creating category agents |
| `src/tools/delegate.py` | DelegateToolProvider | Modify: propagate `delegate_depth` in sub-context |
| `src/agents/runner.py` | AgentRunner tool loop | Modify: filter delegate tools when `delegate_depth >= 1` |
| `tests/agents/test_context.py` | Context tests | Modify: add test for new field |
| `tests/tools/test_categories.py` | CategoryResolver tests | Modify: add tests for `get_delegate_names()` and updated `build_instructions()` |
| `tests/agents/test_registry.py` | Registry tests | Modify: add test for delegate tools in lazy-loaded agents |
| `tests/tools/test_delegate.py` | Delegate tests | Modify: add test for `delegate_depth` propagation |
| `tests/agents/test_runner.py` | Runner tests | Modify: add test for delegate tool filtering by depth |

---

### Task 1: Add `delegate_depth` to RunContext

**Files:**
- Modify: `src/agents/context.py:47-61`
- Test: `tests/agents/test_context.py`

- [ ] **Step 1: Write the failing test**

In `tests/agents/test_context.py`, add:

```python
def test_run_context_delegate_depth_default():
    from src.agents.context import RunContext, DynamicState

    ctx = RunContext(input="test", state=DynamicState(), deps=AgentDeps())
    assert ctx.delegate_depth == 0


def test_run_context_delegate_depth_custom():
    from src.agents.context import RunContext, DynamicState

    ctx = RunContext(input="test", state=DynamicState(), deps=AgentDeps(), delegate_depth=2)
    assert ctx.delegate_depth == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/test_context.py::test_run_context_delegate_depth_default tests/agents/test_context.py::test_run_context_delegate_depth_custom -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'delegate_depth'`

- [ ] **Step 3: Add `delegate_depth` field to RunContext**

In `src/agents/context.py`, add the field to the `RunContext` dataclass after the `depth` field:

```python
@dataclass
class RunContext(Generic[StateT, DepsT]):
    """贯穿整个图执行的共享上下文。

    两个泛型参数：
    - StateT: 共享状态结构，节点间传递数据的唯一通道。
    - DepsT: 外部依赖，由使用者定义，框架只负责传递。
    """

    input: str
    state: StateT
    deps: DepsT
    trace: list[TraceEvent] = field(default_factory=list)
    current_agent: str = ""
    depth: int = 0
    delegate_depth: int = 0  # 委派深度：0=顶层，≥1=被 delegate 调用
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_context.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/context.py tests/agents/test_context.py
git commit -m "feat: add delegate_depth field to RunContext"
```

---

### Task 2: Add `get_delegate_names()` to CategoryResolver

**Files:**
- Modify: `src/tools/categories.py:154-193`
- Test: `tests/tools/test_categories.py`

- [ ] **Step 1: Write the failing tests**

In `tests/tools/test_categories.py`, add:

```python
def test_category_resolver_get_delegate_names_excludes_self():
    """get_delegate_names 返回其他分类的 delegate 工具名，排除自身。"""
    from src.tools.categories import CategoryResolver

    cats = {
        "tool_terminal": {"description": "终端操作", "tools": {"exec": "Execute"}},
        "tool_calc": {"description": "计算", "tools": {"calc": "Calculate"}},
        "tool_files": {"description": "文件操作", "tools": {"read": "Read file"}},
    }
    resolver = CategoryResolver(cats)
    names = resolver.get_delegate_names(exclude="tool_terminal")

    assert "delegate_tool_calc" in names
    assert "delegate_tool_files" in names
    assert "delegate_tool_terminal" not in names
    assert len(names) == 2


def test_category_resolver_get_delegate_names_single_category():
    """只有一个分类时，排除自身后返回空列表。"""
    from src.tools.categories import CategoryResolver

    cats = {"tool_only": {"description": "唯一", "tools": {"t1": "Tool"}}}
    resolver = CategoryResolver(cats)
    names = resolver.get_delegate_names(exclude="tool_only")
    assert names == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tools/test_categories.py::test_category_resolver_get_delegate_names_excludes_self tests/tools/test_categories.py::test_category_resolver_get_delegate_names_single_category -v`
Expected: FAIL with `AttributeError: 'CategoryResolver' object has no attribute 'get_delegate_names'`

- [ ] **Step 3: Implement `get_delegate_names()`**

In `src/tools/categories.py`, add the method to `CategoryResolver`:

```python
def get_delegate_names(self, exclude: str) -> list[str]:
    """返回除 exclude 外所有分类的 delegate 工具名。

    例如 exclude="tool_terminal" 时，返回
    ["delegate_tool_calc", "delegate_tool_files", ...]。
    """
    return [
        f"delegate_{name}"
        for name in self._categories
        if name != exclude
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tools/test_categories.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/categories.py tests/tools/test_categories.py
git commit -m "feat: add CategoryResolver.get_delegate_names()"
```

---

### Task 3: Update `build_instructions()` with delegate descriptions

**Files:**
- Modify: `src/tools/categories.py:146-186`
- Test: `tests/tools/test_categories.py`

- [ ] **Step 1: Write the failing tests**

In `tests/tools/test_categories.py`, add:

```python
def test_category_resolver_build_instructions_with_delegates():
    """传入 delegate_summaries 时，指令中包含协作能力段落。"""
    from src.tools.categories import CategoryResolver

    cats = {
        "tool_terminal": {"description": "终端操作", "tools": {"exec": "Execute"}},
        "tool_calc": {"description": "计算", "tools": {"calc": "Calculate"}},
    }
    resolver = CategoryResolver(cats)
    delegate_summaries = [{"name": "tool_calc", "description": "计算"}]
    instructions = resolver.build_instructions("tool_terminal", delegate_summaries=delegate_summaries)

    assert "终端操作" in instructions
    assert "exec" in instructions
    assert "协作能力" in instructions
    assert "delegate_tool_calc" in instructions
    assert "计算" in instructions


def test_category_resolver_build_instructions_without_delegates():
    """不传 delegate_summaries 时，不包含协作能力段落（向后兼容）。"""
    from src.tools.categories import CategoryResolver

    cats = {"tool_terminal": {"description": "终端操作", "tools": {"exec": "Execute"}}}
    resolver = CategoryResolver(cats)
    instructions = resolver.build_instructions("tool_terminal")

    assert "终端操作" in instructions
    assert "协作能力" not in instructions


def test_category_resolver_build_instructions_custom_ignores_delegates():
    """有自定义 instructions 时，delegate_summaries 不影响结果。"""
    from src.tools.categories import CategoryResolver

    cats = {
        "tool_terminal": {
            "description": "终端操作",
            "tools": {"exec": "Execute"},
            "instructions": "自定义指令",
        }
    }
    resolver = CategoryResolver(cats)
    instructions = resolver.build_instructions(
        "tool_terminal",
        delegate_summaries=[{"name": "tool_calc", "description": "计算"}],
    )
    assert instructions == "自定义指令"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tools/test_categories.py::test_category_resolver_build_instructions_with_delegates tests/tools/test_categories.py::test_category_resolver_build_instructions_without_delegates tests/tools/test_categories.py::test_category_resolver_build_instructions_custom_ignores_delegates -v`
Expected: FAIL with `TypeError: build_instructions() got an unexpected keyword argument 'delegate_summaries'`

- [ ] **Step 3: Update template and `build_instructions()`**

In `src/tools/categories.py`, replace the template and method:

```python
_TOOL_AGENT_INSTRUCTIONS_TEMPLATE = (
    "你是{description}方面的专家。\n\n"
    "## 你的工具\n"
    "{tool_names}\n\n"
    "{delegate_section}"
    "完成任务后返回结果摘要。"
)

_DELEGATE_SECTION_TEMPLATE = (
    "## 协作能力\n"
    "如果任务需要你不具备的能力，可以通过以下委派工具请求其他专家协助：\n"
    "{delegate_descriptions}\n"
    "委派时，用 task 参数清晰描述你需要的具体结果，对方会返回结果供你继续工作。\n\n"
)
```

Update the `build_instructions` method:

```python
def build_instructions(
    self,
    agent_name: str,
    delegate_summaries: list[dict[str, str]] | None = None,
) -> str:
    """构建指定类别的 agent 系统指令。

    若类别条目中包含自定义 instructions 则直接使用，
    否则根据模板自动生成。可选传入 delegate_summaries
    生成协作能力段落。

    Raises:
        KeyError: agent_name 不在已知类别中。
    """
    cat = self._categories[agent_name]
    if cat.get("instructions"):
        return cat["instructions"]

    delegate_section = ""
    if delegate_summaries:
        lines = [
            f"- delegate_{s['name']}: {s['description']}专家"
            for s in delegate_summaries
        ]
        delegate_section = _DELEGATE_SECTION_TEMPLATE.format(
            delegate_descriptions="\n".join(lines),
        )

    return _TOOL_AGENT_INSTRUCTIONS_TEMPLATE.format(
        description=cat["description"],
        tool_names="、".join(cat["tools"].keys()),
        delegate_section=delegate_section,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tools/test_categories.py -v`
Expected: ALL PASS (including the existing `test_category_resolver_build_instructions_default` which should still pass since the template output still contains description and tool names)

- [ ] **Step 5: Commit**

```bash
git add src/tools/categories.py tests/tools/test_categories.py
git commit -m "feat: update build_instructions() with delegate collaboration section"
```

---

### Task 4: Update AgentRegistry to include delegate tools in category agents

**Files:**
- Modify: `src/agents/registry.py:41-66`
- Test: `tests/agents/test_registry.py`

- [ ] **Step 1: Write the failing tests**

In `tests/agents/test_registry.py`, add:

```python
def test_lazy_resolve_includes_delegate_tools(registry):
    """懒加载的 category agent 应包含其他分类的 delegate 工具名。"""
    from src.tools.categories import CategoryResolver

    cats = {
        "tool_terminal": {"description": "终端操作", "tools": {"exec": "Execute command"}},
        "tool_calc": {"description": "计算", "tools": {"calc": "Calculate math"}},
        "tool_files": {"description": "文件操作", "tools": {"read": "Read file"}},
    }
    resolver = CategoryResolver(cats)
    registry.set_category_resolver(resolver)

    agent = registry.get("tool_terminal")
    assert agent is not None

    # 自身的 MCP 工具
    assert "exec" in agent.tools
    # 其他分类的 delegate 工具
    assert "delegate_tool_calc" in agent.tools
    assert "delegate_tool_files" in agent.tools
    # 不包含自身的 delegate 工具
    assert "delegate_tool_terminal" not in agent.tools


def test_lazy_resolve_single_category_no_delegates(registry):
    """只有一个分类时，agent 不包含任何 delegate 工具。"""
    from src.tools.categories import CategoryResolver

    cats = {"tool_only": {"description": "唯一", "tools": {"t1": "Tool"}}}
    resolver = CategoryResolver(cats)
    registry.set_category_resolver(resolver)

    agent = registry.get("tool_only")
    assert agent is not None
    assert agent.tools == ["t1"]


def test_lazy_resolve_instructions_contain_delegate_info(registry):
    """懒加载的 agent 指令中应包含协作能力描述。"""
    from src.tools.categories import CategoryResolver

    cats = {
        "tool_terminal": {"description": "终端操作", "tools": {"exec": "Execute"}},
        "tool_calc": {"description": "计算", "tools": {"calc": "Calculate"}},
    }
    resolver = CategoryResolver(cats)
    registry.set_category_resolver(resolver)

    agent = registry.get("tool_terminal")
    assert "协作能力" in agent.instructions
    assert "delegate_tool_calc" in agent.instructions
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/test_registry.py::test_lazy_resolve_includes_delegate_tools tests/agents/test_registry.py::test_lazy_resolve_single_category_no_delegates tests/agents/test_registry.py::test_lazy_resolve_instructions_contain_delegate_info -v`
Expected: FAIL — `agent.tools` only contains `["exec"]`, no delegate names

- [ ] **Step 3: Update `AgentRegistry.get()` to include delegate tools**

In `src/agents/registry.py`, update the lazy-load block in `get()`:

```python
def get(self, name: str) -> Optional[Agent]:
    """根据名称获取 Agent。

    查找顺序：
    1. 本地缓存 ``_agents``
    2. 若设置了 CategoryResolver 且能解析该名称，则创建 Agent 并缓存
    3. 都未命中则返回 None
    """
    agent = self._agents.get(name)
    if agent is not None:
        return agent

    # 尝试从 CategoryResolver 懒加载
    if self._category_resolver and self._category_resolver.can_resolve(name):
        cat = self._category_resolver.get_category(name)

        # 收集其他分类的 delegate 工具名和摘要
        delegate_names = self._category_resolver.get_delegate_names(exclude=name)
        delegate_summaries = [
            s for s in self._category_resolver.get_all_summaries()
            if s["name"] != name
        ]

        instructions = self._category_resolver.build_instructions(
            name, delegate_summaries=delegate_summaries,
        )

        tools = list(cat["tools"].keys()) + delegate_names  # type: ignore[index]

        agent = Agent(
            name=name,
            description=cat["description"],  # type: ignore[index]
            instructions=instructions,
            tools=tools,
            handoffs=[],
        )
        self.register(agent)
        return agent

    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_registry.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/registry.py tests/agents/test_registry.py
git commit -m "feat: include delegate tools when lazy-loading category agents"
```

---

### Task 5: Propagate `delegate_depth` in DelegateToolProvider

**Files:**
- Modify: `src/tools/delegate.py:79-100`
- Test: `tests/tools/test_delegate.py`

- [ ] **Step 1: Write the failing test**

In `tests/tools/test_delegate.py`, add:

```python
@pytest.mark.asyncio
async def test_execute_propagates_delegate_depth():
    """execute 创建的子 RunContext 应将 delegate_depth 递增。"""
    from src.tools.delegate import DelegateToolProvider
    from src.agents.agent import AgentResult

    cats = {
        "tool_terminal": {"description": "终端操作", "tools": {"exec": "Execute"}},
    }
    test_resolver = CategoryResolver(cats)
    test_registry = AgentRegistry()
    test_registry.set_category_resolver(test_resolver)
    test_runner = AsyncMock()
    test_runner.run = AsyncMock(return_value=AgentResult(text="done"))
    test_deps = AgentDeps()

    provider = DelegateToolProvider(
        resolver=test_resolver,
        runner=test_runner,
        registry=test_registry,
        deps=test_deps,
    )
    await provider.execute("delegate_tool_terminal", {"task": "run ls"})

    # 验证子 RunContext 的 delegate_depth 为 1（默认父级 depth=0 + 1）
    call_args = test_runner.run.call_args
    sub_ctx = call_args[0][1]
    assert sub_ctx.delegate_depth == 1


@pytest.mark.asyncio
async def test_execute_propagates_parent_delegate_depth():
    """当从已有 delegate_depth 的上下文调用时，子 RunContext 应继续递增。"""
    from src.tools.delegate import DelegateToolProvider
    from src.agents.agent import AgentResult
    from src.agents.context import DynamicState, RunContext

    cats = {
        "tool_terminal": {"description": "终端操作", "tools": {"exec": "Execute"}},
    }
    test_resolver = CategoryResolver(cats)
    test_registry = AgentRegistry()
    test_registry.set_category_resolver(test_resolver)
    test_runner = AsyncMock()
    test_runner.run = AsyncMock(return_value=AgentResult(text="done"))
    test_deps = AgentDeps()

    # 创建一个带有 parent_context 的 provider
    provider = DelegateToolProvider(
        resolver=test_resolver,
        runner=test_runner,
        registry=test_registry,
        deps=test_deps,
    )
    # 模拟从 depth=1 的上下文调用
    provider._current_delegate_depth = 1
    await provider.execute("delegate_tool_terminal", {"task": "run ls"})

    call_args = test_runner.run.call_args
    sub_ctx = call_args[0][1]
    assert sub_ctx.delegate_depth == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tools/test_delegate.py::test_execute_propagates_delegate_depth tests/tools/test_delegate.py::test_execute_propagates_parent_delegate_depth -v`
Expected: FAIL — `RunContext` will have `delegate_depth=0` (not set)

- [ ] **Step 3: Update DelegateToolProvider to propagate depth**

In `src/tools/delegate.py`, update `__init__` and `execute()`:

```python
class DelegateToolProvider:
    """将 Tool Agent 包装为可调用工具的 ToolProvider。

    实现 ToolProvider 协议（can_handle / execute / get_schemas），
    使 ToolRouter 能像路由普通工具一样路由 delegate 调用。
    """

    def __init__(
        self,
        resolver: CategoryResolver,
        runner: AgentRunner,
        registry: AgentRegistry,
        deps: AgentDeps,
        mcp_manager: MCPManager | None = None,
    ) -> None:
        self._resolver = resolver
        self._runner = runner
        self._registry = registry
        self._deps = deps
        self._mcp_manager = mcp_manager
        self._current_delegate_depth: int = 0  # 当前委派深度，由 runner 在调用前设置

    # can_handle, get_schemas — 不变

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """委派执行：按需连接 MCP server，创建子 RunContext 并驱动 AgentRunner。"""
        from src.agents.context import DynamicState, RunContext

        agent_name = tool_name[len(DELEGATE_PREFIX):]
        agent = self._registry.get(agent_name)
        if agent is None:
            return f"错误：找不到 agent {agent_name}"

        # 按需连接该 agent 所需的 MCP server
        if self._mcp_manager:
            mcp_tools = [t for t in agent.tools if t.startswith("mcp_")]
            if mcp_tools:
                await self._mcp_manager.ensure_servers_for_tools(mcp_tools)

        sub_ctx: RunContext = RunContext(
            input=arguments.get("task", ""),
            state=DynamicState(),
            deps=self._deps,
            delegate_depth=self._current_delegate_depth + 1,
        )
        result = await self._runner.run(agent, sub_ctx)
        return result.text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tools/test_delegate.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/delegate.py tests/tools/test_delegate.py
git commit -m "feat: propagate delegate_depth in DelegateToolProvider"
```

---

### Task 6: Set `_current_delegate_depth` from runner context before delegate calls

**Files:**
- Modify: `src/agents/runner.py:87-92`
- Test: `tests/agents/test_runner.py`

The `AgentRunner` tool-call loop routes calls through `tool_router.route()`. Before routing, it needs to inform the `DelegateToolProvider` about the current context's `delegate_depth` so it can propagate correctly.

- [ ] **Step 1: Write the failing test**

In `tests/agents/test_runner.py`, add:

```python
@pytest.mark.asyncio
async def test_runner_sets_delegate_depth_on_router(mock_llm):
    """runner 应在工具调用前将 context.delegate_depth 同步到 tool_router。"""
    from src.agents.runner import AgentRunner

    mock_llm.chat = AsyncMock(side_effect=[
        LLMResponse(
            content="",
            tool_calls={0: {"id": "call_1", "name": "delegate_tool_calc", "arguments": '{"task": "1+1"}'}},
        ),
        LLMResponse(content="2", tool_calls={}),
    ])

    mock_router = AsyncMock()
    mock_router.route = AsyncMock(return_value="2")
    mock_router.get_all_schemas = MagicMock(return_value=[
        {
            "type": "function",
            "function": {
                "name": "delegate_tool_calc",
                "description": "委派任务给计算专家",
                "parameters": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]},
            },
        }
    ])

    agent = Agent(
        name="tool_terminal",
        description="终端操作",
        instructions="终端专家。",
        tools=["delegate_tool_calc"],
    )
    ctx = RunContext(
        input="calculate 1+1",
        state=DynamicState(),
        deps=AgentDeps(llm=mock_llm, tool_router=mock_router),
        delegate_depth=0,
    )

    runner = AgentRunner(registry=AgentRegistry())
    await runner.run(agent, ctx)

    # 验证 router 上的 delegate_depth 被设置为 context 的值
    assert mock_router.set_delegate_depth.called
    mock_router.set_delegate_depth.assert_called_with(0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_runner.py::test_runner_sets_delegate_depth_on_router -v`
Expected: FAIL — `set_delegate_depth` not called

- [ ] **Step 3: Add depth sync in AgentRunner**

In `src/agents/runner.py`, after the `ensure_tools` block (line 91), add:

```python
        # 5. 按需连接 MCP server，然后构建工具列表
        if agent.tools:
            tool_router = getattr(context.deps, "tool_router", None)
            if tool_router:
                await tool_router.ensure_tools(agent.tools)
        # 同步 delegate 深度到 tool_router，供 DelegateToolProvider 使用
        tool_router = getattr(context.deps, "tool_router", None)
        if tool_router and hasattr(tool_router, "set_delegate_depth"):
            tool_router.set_delegate_depth(context.delegate_depth)
        tools = self._build_tools(agent, context)
```

- [ ] **Step 4: Add `set_delegate_depth` to ToolRouter**

In `src/tools/router.py`, read the current file first, then add the method:

```python
def set_delegate_depth(self, depth: int) -> None:
    """同步 delegate 深度到 DelegateToolProvider。"""
    for provider in self._providers:
        if hasattr(provider, "_current_delegate_depth"):
            provider._current_delegate_depth = depth
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_runner.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/agents/runner.py src/tools/router.py tests/agents/test_runner.py
git commit -m "feat: sync delegate_depth from RunContext to ToolRouter"
```

---

### Task 7: Filter delegate tools by depth in `_build_tools()`

**Files:**
- Modify: `src/agents/runner.py:216-222`
- Test: `tests/agents/test_runner.py`

- [ ] **Step 1: Write the failing tests**

In `tests/agents/test_runner.py`, add:

```python
def test_build_tools_includes_delegates_at_depth_0():
    """delegate_depth=0 时，_build_tools 应包含 delegate 工具。"""
    from src.agents.runner import AgentRunner

    mock_router = MagicMock()
    mock_router.get_all_schemas = MagicMock(return_value=[
        {"type": "function", "function": {"name": "exec", "description": "Execute", "parameters": {}}},
        {"type": "function", "function": {"name": "delegate_tool_calc", "description": "委派计算", "parameters": {}}},
    ])

    agent = Agent(name="test", description="Test", instructions="Test.", tools=["exec", "delegate_tool_calc"])
    ctx = RunContext(input="test", state=DynamicState(), deps=AgentDeps(tool_router=mock_router), delegate_depth=0)

    runner = AgentRunner(registry=AgentRegistry())
    tools = runner._build_tools(agent, ctx)

    names = [t["function"]["name"] for t in tools]
    assert "exec" in names
    assert "delegate_tool_calc" in names


def test_build_tools_excludes_delegates_at_depth_1():
    """delegate_depth>=1 时，_build_tools 应过滤掉所有 delegate 工具。"""
    from src.agents.runner import AgentRunner

    mock_router = MagicMock()
    mock_router.get_all_schemas = MagicMock(return_value=[
        {"type": "function", "function": {"name": "exec", "description": "Execute", "parameters": {}}},
        {"type": "function", "function": {"name": "delegate_tool_calc", "description": "委派计算", "parameters": {}}},
    ])

    agent = Agent(name="test", description="Test", instructions="Test.", tools=["exec", "delegate_tool_calc"])
    ctx = RunContext(input="test", state=DynamicState(), deps=AgentDeps(tool_router=mock_router), delegate_depth=1)

    runner = AgentRunner(registry=AgentRegistry())
    tools = runner._build_tools(agent, ctx)

    names = [t["function"]["name"] for t in tools]
    assert "exec" in names
    assert "delegate_tool_calc" not in names
```

- [ ] **Step 2: Run tests to verify `test_build_tools_excludes_delegates_at_depth_1` fails**

Run: `uv run pytest tests/agents/test_runner.py::test_build_tools_includes_delegates_at_depth_0 tests/agents/test_runner.py::test_build_tools_excludes_delegates_at_depth_1 -v`
Expected: `test_build_tools_excludes_delegates_at_depth_1` FAILS — delegate tool is still included

- [ ] **Step 3: Update `_build_tools()` to filter by depth**

In `src/agents/runner.py`, update the `_build_tools` method:

```python
def _build_tools(self, agent: Agent, context: RunContext) -> list[dict]:
    """从 deps.tool_router 过滤 agent 允许的工具。

    当 context.delegate_depth >= 1 时，过滤掉所有 delegate_ 前缀的工具，
    防止被委派的 agent 再次委派（递归深度限制）。
    """
    tool_router = getattr(context.deps, "tool_router", None)
    if not tool_router or not agent.tools:
        return []
    all_schemas = tool_router.get_all_schemas()
    allowed = set(agent.tools)
    # 委派深度 >= 1 时，移除所有 delegate 工具
    if context.delegate_depth >= 1:
        allowed = {name for name in allowed if not name.startswith("delegate_")}
    return [s for s in all_schemas if s["function"]["name"] in allowed]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_runner.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/runner.py tests/agents/test_runner.py
git commit -m "feat: filter delegate tools by delegate_depth in _build_tools()"
```

---

### Task 8: Run full test suite and verify no regressions

**Files:** None (verification only)

- [ ] **Step 1: Run all tests**

Run: `uv run pytest -v`
Expected: ALL PASS, no regressions in existing tests

- [ ] **Step 2: If any test fails, investigate and fix**

Common risks:
- `test_category_resolver_build_instructions_default` may fail if the new template output format changed — verify it still contains the description and tool names
- `test_lazy_resolve_from_category_resolver` expects `agent.tools == ["exec"]` — this will now include delegate tools. Update the assertion:

In `tests/agents/test_registry.py`, find and update the existing test:

```python
def test_lazy_resolve_from_category_resolver(registry):
    from src.tools.categories import CategoryResolver

    cats = {"tool_terminal": {"description": "终端操作", "tools": {"exec": "Execute command"}}}
    resolver = CategoryResolver(cats)
    registry.set_category_resolver(resolver)

    assert not registry.has("tool_terminal")
    agent = registry.get("tool_terminal")
    assert agent is not None
    assert agent.name == "tool_terminal"
    assert agent.description == "终端操作"
    assert "exec" in agent.tools  # 自身工具仍在列表中
    assert registry.has("tool_terminal")
    assert registry.get("tool_terminal") is agent
```

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: update existing tests for cross-agent delegate compatibility"
```
