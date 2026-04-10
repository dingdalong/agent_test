# Delegate 分配策略重新设计 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 delegate 工具从工具类 agent 中移除，改由 `AgentRunner._build_tools()` 为非工具类 agent 自动注入。

**Architecture:** 在 `AgentDeps` 中新增 `category_resolver` 字段，`_build_tools()` 通过它判断 agent 是否为工具类（`resolver.can_resolve(agent.name)`），工具类 agent 只保留自身工具，非工具类 agent 自动获得所有 delegate schemas。

**Tech Stack:** Python 3.13, pytest, asyncio

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/agents/deps.py` | Modify | 新增 `category_resolver` 字段 |
| `src/app/bootstrap.py` | Modify | 组装时传入 `category_resolver` |
| `src/agents/runner.py` | Modify | 重写 `_build_tools()` 自动注入 delegate |
| `src/agents/registry.py` | Modify | 懒创建时不再添加 delegate |
| `src/tools/categories.py` | Modify | 移除 delegate 模板，简化 `build_instructions()` |
| `tests/agents/test_runner.py` | Modify | 新增 4 个测试 |
| `tests/agents/test_registry.py` | Modify | 更新 2 个测试 |
| `tests/tools/test_categories.py` | Modify | 移除 2 个测试 |
| `tests/tools/test_delegate_integration.py` | Modify | 重写 2 个测试 |

---

### Task 1: `AgentDeps` 新增字段 + `bootstrap.py` 传入

**Files:**
- Modify: `src/agents/deps.py:27-33`
- Modify: `src/app/bootstrap.py:210-218`

- [ ] **Step 1: 在 `AgentDeps` 添加 `category_resolver` 字段**

在 `src/agents/deps.py` 的 `AgentDeps` 类中，在 `runner` 字段后添加：

```python
    runner: Any = None           # AgentRunner — 必需
    category_resolver: Any = None  # CategoryResolver — 可选
```

- [ ] **Step 2: 在 `bootstrap.py` 中传入 `category_resolver`**

在 `src/app/bootstrap.py` 的 `AgentDeps` 构造处（第 210-218 行），添加 `category_resolver`：

```python
    deps = AgentDeps(
        llm=llm,
        tool_router=tool_router,
        agent_registry=agent_registry,
        graph_engine=engine,
        ui=ui,
        memory=memory_store,
        runner=runner,
        category_resolver=category_resolver,
    )
```

- [ ] **Step 3: 运行测试确认无破坏**

Run: `uv run pytest tests/agents/test_runner.py tests/tools/test_delegate_integration.py -v`
Expected: ALL PASS（纯新增字段，无行为变更）

- [ ] **Step 4: Commit**

```bash
git add src/agents/deps.py src/app/bootstrap.py
git commit -m "feat(deps): add category_resolver field to AgentDeps"
```

---

### Task 2: 重写 `AgentRunner._build_tools()` 自动注入 delegate

**Files:**
- Modify: `src/agents/runner.py:312-332`
- Modify: `tests/agents/test_runner.py`

- [ ] **Step 1: 新增 4 个测试用例**

在 `tests/agents/test_runner.py` 末尾追加：

```python
def test_build_tools_tool_agent_no_delegates():
    """工具类 agent（CategoryResolver 可解析）不应获得 delegate 工具。"""
    from src.agents.runner import AgentRunner
    from src.tools.categories import CategoryResolver

    cats = {
        "tool_terminal": {"description": "终端操作", "tools": {"exec": "Execute"}},
        "tool_calc": {"description": "计算", "tools": {"calc": "Calculate"}},
    }
    resolver = CategoryResolver(cats)

    mock_router = MagicMock()
    mock_router.get_all_schemas = MagicMock(return_value=[
        {"type": "function", "function": {"name": "exec", "description": "Execute", "parameters": {}}},
        {"type": "function", "function": {"name": "delegate_tool_calc", "description": "委派计算", "parameters": {}}},
        {"type": "function", "function": {"name": "parallel_delegate", "description": "并行", "parameters": {}}},
        {"type": "function", "function": {"name": "ask_user", "description": "Ask user", "parameters": {}}},
    ])

    agent = Agent(name="tool_terminal", description="终端", instructions="终端专家.", tools=["exec"])
    ctx = RunContext(
        input="test", state=DynamicState(),
        deps=AgentDeps(tool_router=mock_router, category_resolver=resolver),
        delegate_depth=0,
    )

    runner = AgentRunner()
    tools = runner._build_tools(agent, ctx)

    names = [t["function"]["name"] for t in tools]
    assert "exec" in names
    assert "ask_user" in names
    assert "delegate_tool_calc" not in names
    assert "parallel_delegate" not in names


def test_build_tools_non_tool_agent_auto_injects_delegates():
    """非工具类 agent（tools=[]）自动获得所有 delegate 工具。"""
    from src.agents.runner import AgentRunner
    from src.tools.categories import CategoryResolver

    cats = {
        "tool_terminal": {"description": "终端操作", "tools": {"exec": "Execute"}},
    }
    resolver = CategoryResolver(cats)

    mock_router = MagicMock()
    mock_router.get_all_schemas = MagicMock(return_value=[
        {"type": "function", "function": {"name": "exec", "description": "Execute", "parameters": {}}},
        {"type": "function", "function": {"name": "delegate_tool_terminal", "description": "委派终端", "parameters": {}}},
        {"type": "function", "function": {"name": "parallel_delegate", "description": "并行委托", "parameters": {}}},
        {"type": "function", "function": {"name": "ask_user", "description": "Ask user", "parameters": {}}},
    ])

    agent = Agent(name="orchestrator", description="总控", instructions="路由.", tools=[])
    ctx = RunContext(
        input="test", state=DynamicState(),
        deps=AgentDeps(tool_router=mock_router, category_resolver=resolver),
        delegate_depth=0,
    )

    runner = AgentRunner()
    tools = runner._build_tools(agent, ctx)

    names = [t["function"]["name"] for t in tools]
    assert "delegate_tool_terminal" in names
    assert "parallel_delegate" in names
    assert "ask_user" in names
    assert "exec" not in names


def test_build_tools_non_tool_agent_with_tools_gets_delegates():
    """非工具类 agent 即使有声明工具，也自动获得 delegate。"""
    from src.agents.runner import AgentRunner
    from src.tools.categories import CategoryResolver

    cats = {
        "tool_terminal": {"description": "终端操作", "tools": {"exec": "Execute"}},
    }
    resolver = CategoryResolver(cats)

    mock_router = MagicMock()
    mock_router.get_all_schemas = MagicMock(return_value=[
        {"type": "function", "function": {"name": "some_tool", "description": "Some", "parameters": {}}},
        {"type": "function", "function": {"name": "delegate_tool_terminal", "description": "委派终端", "parameters": {}}},
        {"type": "function", "function": {"name": "ask_user", "description": "Ask user", "parameters": {}}},
    ])

    agent = Agent(name="my_agent", description="自定义", instructions="自定义.", tools=["some_tool"])
    ctx = RunContext(
        input="test", state=DynamicState(),
        deps=AgentDeps(tool_router=mock_router, category_resolver=resolver),
        delegate_depth=0,
    )

    runner = AgentRunner()
    tools = runner._build_tools(agent, ctx)

    names = [t["function"]["name"] for t in tools]
    assert "some_tool" in names
    assert "delegate_tool_terminal" in names
    assert "ask_user" in names


def test_build_tools_filters_parallel_delegate_at_depth_1():
    """delegate_depth>=1 时，parallel_delegate 也应被过滤。"""
    from src.agents.runner import AgentRunner

    mock_router = MagicMock()
    mock_router.get_all_schemas = MagicMock(return_value=[
        {"type": "function", "function": {"name": "exec", "description": "Execute", "parameters": {}}},
        {"type": "function", "function": {"name": "delegate_tool_calc", "description": "Delegate", "parameters": {}}},
        {"type": "function", "function": {"name": "parallel_delegate", "description": "Parallel", "parameters": {}}},
        {"type": "function", "function": {"name": "ask_user", "description": "Ask user", "parameters": {}}},
    ])

    agent = Agent(name="test", description="Test", instructions="Test.", tools=["exec", "delegate_tool_calc", "parallel_delegate"])
    ctx = RunContext(input="test", state=DynamicState(), deps=AgentDeps(tool_router=mock_router), delegate_depth=1)

    runner = AgentRunner()
    tools = runner._build_tools(agent, ctx)

    names = [t["function"]["name"] for t in tools]
    assert "exec" in names
    assert "ask_user" in names
    assert "delegate_tool_calc" not in names
    assert "parallel_delegate" not in names
```

- [ ] **Step 2: 运行新增测试确认它们失败**

Run: `uv run pytest tests/agents/test_runner.py::test_build_tools_tool_agent_no_delegates tests/agents/test_runner.py::test_build_tools_non_tool_agent_auto_injects_delegates tests/agents/test_runner.py::test_build_tools_non_tool_agent_with_tools_gets_delegates tests/agents/test_runner.py::test_build_tools_filters_parallel_delegate_at_depth_1 -v`
Expected: 前 3 个 FAIL（当前代码不区分 tool/non-tool agent），第 4 个 FAIL（`parallel_delegate` 未被过滤）

- [ ] **Step 3: 添加 `_is_delegate_tool` 辅助函数，重写 `_build_tools()`**

在 `src/agents/runner.py` 中，在 `SYSTEM_TOOLS` 定义后添加辅助函数：

```python
SYSTEM_TOOLS = {"ask_user"}


def _is_delegate_tool(name: str) -> bool:
    """判断工具名是否为 delegate 工具。"""
    return name.startswith("delegate_") or name == "parallel_delegate"
```

替换 `_build_tools` 方法（第 312-332 行）为：

```python
    def _build_tools(self, agent: Agent, context: RunContext) -> list[dict]:
        """从 deps.tool_router 过滤 agent 允许的工具。

        规则：
        - 系统工具（SYSTEM_TOOLS）始终包含
        - 工具类 agent（CategoryResolver 可解析）：只包含自身声明的工具
        - 非工具类 agent：自动注入所有 delegate 工具
        - delegate_depth >= 1 时，过滤掉所有 delegate 工具（安全网）
        """
        tool_router = getattr(context.deps, "tool_router", None)
        if not tool_router:
            return []
        all_schemas = tool_router.get_all_schemas()

        resolver = getattr(context.deps, "category_resolver", None)
        is_tool_agent = resolver is not None and resolver.can_resolve(agent.name)

        # 1. 构建基础 allowed 集合
        if not agent.tools:
            allowed = set(SYSTEM_TOOLS)
        else:
            allowed = set(agent.tools) | SYSTEM_TOOLS

        # 2. 按 agent 类型处理 delegate
        if is_tool_agent:
            # 工具类 agent — 确保不包含 delegate（防御性过滤）
            allowed = {n for n in allowed if not _is_delegate_tool(n)}
        else:
            # 非工具类 agent — 追加所有 delegate
            for s in all_schemas:
                fname = s["function"]["name"]
                if _is_delegate_tool(fname):
                    allowed.add(fname)

        # 3. 安全网：delegate 链中不允许再次 delegate
        if context.delegate_depth >= 1:
            allowed = {n for n in allowed if not _is_delegate_tool(n)}

        return [s for s in all_schemas if s["function"]["name"] in allowed]
```

- [ ] **Step 4: 运行全部 `_build_tools` 测试**

Run: `uv run pytest tests/agents/test_runner.py -k "build_tools" -v`
Expected: ALL PASS（新增 4 个 + 原有 5 个）

- [ ] **Step 5: 运行完整 runner 测试确认无回归**

Run: `uv run pytest tests/agents/test_runner.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/agents/runner.py tests/agents/test_runner.py
git commit -m "feat(runner): auto-inject delegate tools for non-tool agents in _build_tools"
```

---

### Task 3: 简化 `AgentRegistry.get()` — 移除 delegate 注入

**Files:**
- Modify: `src/agents/registry.py:53-74`
- Modify: `tests/agents/test_registry.py:106-157`

- [ ] **Step 1: 更新测试用例**

在 `tests/agents/test_registry.py` 中，替换以下三个测试：

将 `test_lazy_resolve_includes_delegate_tools`（第 106-127 行）替换为：

```python
def test_lazy_resolve_excludes_delegate_tools(registry):
    """懒加载的 category agent 不应包含其他分类的 delegate 工具名。"""
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

    # 只有自身工具
    assert "exec" in agent.tools
    # 不包含任何 delegate 工具
    assert "delegate_tool_calc" not in agent.tools
    assert "delegate_tool_files" not in agent.tools
    assert "delegate_tool_terminal" not in agent.tools
    assert agent.tools == ["exec"]
```

将 `test_lazy_resolve_instructions_contain_delegate_info`（第 143-157 行）替换为：

```python
def test_lazy_resolve_instructions_no_delegate_info(registry):
    """懒加载的 agent 指令中不应包含协作能力描述。"""
    from src.tools.categories import CategoryResolver

    cats = {
        "tool_terminal": {"description": "终端操作", "tools": {"exec": "Execute"}},
        "tool_calc": {"description": "计算", "tools": {"calc": "Calculate"}},
    }
    resolver = CategoryResolver(cats)
    registry.set_category_resolver(resolver)

    agent = registry.get("tool_terminal")
    assert "协作能力" not in agent.instructions
    assert "delegate_tool_calc" not in agent.instructions
```

- [ ] **Step 2: 运行更新后的测试确认失败**

Run: `uv run pytest tests/agents/test_registry.py::test_lazy_resolve_excludes_delegate_tools tests/agents/test_registry.py::test_lazy_resolve_instructions_no_delegate_info -v`
Expected: FAIL（当前 registry 仍注入 delegate）

- [ ] **Step 3: 简化 `AgentRegistry.get()` 实现**

在 `src/agents/registry.py` 中，替换懒加载分支（第 53-76 行）为：

```python
        # 尝试从 CategoryResolver 懒加载
        if self._category_resolver and self._category_resolver.can_resolve(name):
            cat = self._category_resolver.get_category(name)

            instructions = self._category_resolver.build_instructions(name)

            tools = list(cat["tools"].keys())  # 只包含自身工具

            agent = Agent(
                name=name,
                description=cat["description"],  # type: ignore[index]
                instructions=instructions,
                tools=tools,
                handoffs=[],
            )
            self.register(agent)
            return agent
```

- [ ] **Step 4: 运行 registry 测试**

Run: `uv run pytest tests/agents/test_registry.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/registry.py tests/agents/test_registry.py
git commit -m "refactor(registry): remove delegate injection from lazy agent creation"
```

---

### Task 4: 清理 `CategoryResolver` — 移除 delegate 模板

**Files:**
- Modify: `src/tools/categories.py:200-304`
- Modify: `tests/tools/test_categories.py:275-344`

- [ ] **Step 1: 移除 2 个测试，更新注释**

在 `tests/tools/test_categories.py` 中：

删除整个 `test_category_resolver_build_instructions_with_delegates` 函数（第 280-296 行）。

删除整个 `test_category_resolver_build_instructions_custom_appends_delegates` 函数（第 311-329 行）。

将第 275-278 行的注释区块替换为：

```python
# ---------------------------------------------------------------------------
# build_instructions() 测试（delegate 相关已移除）
# ---------------------------------------------------------------------------
```

- [ ] **Step 2: 运行测试确认通过**

Run: `uv run pytest tests/tools/test_categories.py -v`
Expected: ALL PASS（这些测试调用旧签名，但移除后不会被执行）

注意：此时 `build_instructions` 的 `delegate_summaries` 参数仍存在但不再被 `registry.py` 调用。下一步移除。

- [ ] **Step 3: 简化 `categories.py` 实现**

在 `src/tools/categories.py` 中：

删除 `_DELEGATE_SECTION_TEMPLATE`（第 208-213 行）。

替换 `_TOOL_AGENT_INSTRUCTIONS_TEMPLATE`（第 200-206 行）为：

```python
_TOOL_AGENT_INSTRUCTIONS_TEMPLATE = (
    "你是{description}方面的专家。\n\n"
    "## 你的工具\n"
    "{tool_names}\n\n"
    "完成任务后返回结果摘要。"
)
```

删除 `get_delegate_info` 方法（第 247-259 行）。

替换 `build_instructions` 方法（第 261-304 行）为：

```python
    def build_instructions(self, agent_name: str) -> str:
        """构建指定类别的 agent 系统指令。

        若类别条目中包含自定义 instructions 则直接使用，
        否则根据模板自动生成。

        Raises:
            KeyError: agent_name 不在已知类别中。
        """
        cat = self._categories[agent_name]
        custom = cat.get("instructions")
        if custom:
            return custom

        return _TOOL_AGENT_INSTRUCTIONS_TEMPLATE.format(
            description=cat["description"],
            tool_names="、".join(cat["tools"].keys()),
        )
```

- [ ] **Step 4: 运行全部 categories 和 registry 测试**

Run: `uv run pytest tests/tools/test_categories.py tests/agents/test_registry.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/categories.py tests/tools/test_categories.py
git commit -m "refactor(categories): remove delegate template from build_instructions"
```

---

### Task 5: 更新集成测试

**Files:**
- Modify: `tests/tools/test_delegate_integration.py`

- [ ] **Step 1: 重写 `test_delegate_end_to_end`**

替换 `test_delegate_end_to_end`（第 38-94 行）为：

```python
@pytest.mark.asyncio
async def test_delegate_end_to_end(resolver, registry):
    """非工具类 agent 通过 delegate 调用工具类 agent，获取结果并继续。"""
    mock_llm = AsyncMock()

    call_count = 0

    async def mock_chat(messages, tools=None, silent=True):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # orchestrator 调用 delegate_tool_calc
            return LLMResponse(
                content="",
                tool_calls={0: {
                    "id": "call_delegate",
                    "name": "delegate_tool_calc",
                    "arguments": json.dumps({
                        "objective": "执行计算任务，获取 1+1 的结果",
                        "task": "计算 1+1",
                        "context": "用户要求执行计算任务",
                        "expected_result": "计算结果数值",
                    }),
                }},
            )
        if call_count == 2:
            # tool_calc agent 执行并返回
            return LLMResponse(content="计算结果是 2", tool_calls={})
        # orchestrator 汇总结果
        return LLMResponse(content="1+1=2", tool_calls={})

    mock_llm.chat = mock_chat

    runner = AgentRunner()
    router = ToolRouter()

    delegate_provider = DelegateToolProvider(resolver=resolver)
    router.add_provider(delegate_provider)

    # orchestrator 是非工具类 agent，delegate 由 _build_tools 自动注入
    orchestrator = Agent(
        name="orchestrator",
        description="总控 Agent",
        instructions="你是总控 Agent，使用 delegate 工具完成任务。",
        tools=[],
    )
    registry.register(orchestrator)

    ctx = RunContext(
        input="计算 1+1",
        state=DynamicState(),
        deps=AgentDeps(
            llm=mock_llm,
            tool_router=router,
            agent_registry=registry,
            runner=runner,
            category_resolver=resolver,
        ),
        delegate_depth=0,
    )
    result = await runner.run(orchestrator, ctx)

    assert "2" in result.text
    assert call_count == 3
```

- [ ] **Step 2: 更新 `test_delegated_agent_cannot_delegate_further`**

替换 `test_delegated_agent_cannot_delegate_further`（第 97-136 行）为：

```python
@pytest.mark.asyncio
async def test_delegated_agent_cannot_delegate_further(resolver, registry):
    """被 delegate 调用的工具类 Agent 不应看到任何 delegate 工具。"""
    mock_llm = AsyncMock()

    tools_seen_by_b = []

    async def mock_chat(messages, tools=None, silent=True):
        if tools:
            tools_seen_by_b.extend([t["function"]["name"] for t in tools])
        return LLMResponse(content="结果", tool_calls={})

    mock_llm.chat = mock_chat

    runner = AgentRunner()
    router = ToolRouter()

    deps = AgentDeps(
        llm=mock_llm,
        tool_router=router,
        agent_registry=registry,
        runner=runner,
        category_resolver=resolver,
    )
    delegate_provider = DelegateToolProvider(resolver=resolver)
    router.add_provider(delegate_provider)

    agent_b = registry.get("tool_calc")

    ctx = RunContext(
        input="计算 1+1",
        state=DynamicState(),
        deps=deps,
        delegate_depth=1,
    )
    await runner.run(agent_b, ctx)

    delegate_tools = [t for t in tools_seen_by_b if t.startswith("delegate_") or t == "parallel_delegate"]
    assert delegate_tools == [], f"Tool agent should not see delegate tools, but saw: {delegate_tools}"
```

- [ ] **Step 3: 运行集成测试**

Run: `uv run pytest tests/tools/test_delegate_integration.py -v`
Expected: ALL PASS

- [ ] **Step 4: 运行全量测试确认无回归**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add tests/tools/test_delegate_integration.py
git commit -m "test(integration): update delegate tests for new distribution strategy"
```
