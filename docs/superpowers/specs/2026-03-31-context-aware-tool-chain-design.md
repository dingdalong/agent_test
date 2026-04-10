# Context-Aware Tool Chain 设计

## 问题

### 直接问题

Skill 模式下，`_handle_skill` 创建的 `skill_registry` 没有 `category_resolver`，导致 `build_skill_graph` 无法懒加载 category agent —— orchestrator 的 handoff 列表指向不存在的节点，skill 图只有 orchestrator + planner 两个节点。

### 根本原因

`DelegateToolProvider` 在启动时通过构造函数捕获了执行环境（`runner`、`registry`、`deps`），但运行时可能处于不同的执行上下文（主图、skill 图、plan 图）。同样，`AgentRunner` 通过构造函数持有 `registry`，`AgentNode` 通过构造函数持有 `runner`。这种「构造时绑定」模式使得多上下文复用不可能。

### 当前调用链与断点

```
GraphEngine.run(graph, context: RunContext)
  └─ AgentNode.execute(context)           ← context 可达
       └─ self.runner.run(agent, context) ← runner 在构造时绑定
            └─ tool_router.route(name, args)       ← context 丢失
                 └─ DelegateToolProvider.execute(name, args)  ← 用构造时捕获的 registry/runner/deps
```

`RunContext` 在 `AgentRunner` → `ToolRouter.route()` 之间断裂，ToolProvider 无法感知当前执行上下文。

## 设计原则

**构造时绑定只读配置，运行时从 context 取环境。**

- 只读配置（`category_resolver`、`max_tool_rounds`）在构造时绑定，生命周期与应用一致
- 作用域状态（`registry`、`runner`、`deps`、`delegate_depth`）从 `RunContext` 获取，自动匹配当前执行环境
- 不同图（主图、skill 图、plan 图）通过各自的 `RunContext.deps` 携带不同的 registry/runner，共享同一套 ToolProvider 实例

## 详细设计

### 1. ToolProvider 协议扩展

```python
# src/tools/router.py
class ToolProvider(Protocol):
    def can_handle(self, tool_name: str) -> bool: ...
    def get_schemas(self) -> list[ToolDict]: ...
    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any | None = None,      # RunContext，用 Any 避免循环导入
    ) -> str: ...
```

`context` 为 `Optional`，默认 `None`：
- `LocalToolProvider`、`MCPToolProvider`、`SkillToolProvider` 忽略此参数，签名兼容即可
- `DelegateToolProvider` 从 context 获取 registry、runner、deps

### 2. ToolRouter.route() 透传 context

```python
# src/tools/router.py
class ToolRouter:
    async def route(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> str:
        for provider in self._providers:
            if provider.can_handle(tool_name):
                return await provider.execute(tool_name, arguments, context)
        return f"未知工具: {tool_name}"
```

删除 `set_delegate_depth()` 方法 —— depth 通过 context 自然流入。

### 3. AgentRunner 无状态化

**变更**：移除构造函数中的 `registry` 参数，从 `context.deps` 读取。

```python
# src/agents/runner.py
class AgentRunner:
    def __init__(
        self,
        max_tool_rounds: int = 10,
        max_result_length: int = 4000,
    ):
        self.max_tool_rounds = max_tool_rounds
        self.max_result_length = max_result_length
        # 不再持有 registry

    async def run(self, agent: Agent, context: RunContext) -> AgentResult:
        ...
        # 不再调用 tool_router.set_delegate_depth()
        tools = self._build_tools(agent, context)
        handoff_tools = self._build_handoff_tools(agent, context)
        ...
        # 工具调用时透传 context
        result_text = await tool_router.route(tool_name, args, context)
        ...

    def _build_tools(self, agent: Agent, context: RunContext) -> list[dict]:
        # 不变：仍从 context.deps.tool_router 获取 schemas
        # 不变：delegate_depth >= 1 时过滤 delegate 工具
        ...

    def _build_handoff_tools(self, agent: Agent, context: RunContext) -> list[dict]:
        registry = context.deps.agent_registry   # 从 context 取
        tools = []
        for target_name in agent.handoffs:
            target = registry.get(target_name)
            ...
        return tools
```

### 4. AgentDeps 新增 runner 字段

```python
# src/agents/deps.py
class AgentDeps(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Any = None
    tool_router: Any = None
    agent_registry: Any = None
    graph_engine: Any = None
    ui: Any = None
    memory: Any = None
    runner: Any = None               # 新增
```

### 5. AgentNode 从 context 取 runner

```python
# src/agents/node.py
class AgentNode:
    def __init__(self, agent: Any):
        self.name: str = agent.name
        self.agent = agent
        # 不再持有 runner

    async def execute(self, context: Any) -> NodeResult:
        runner = context.deps.runner
        if runner is None:
            raise RuntimeError(f"AgentNode({self.name}): deps.runner is None")
        result = await runner.run(self.agent, context)
        return NodeResult(
            output={"text": result.text, "data": result.data},
            handoff=result.handoff,
        )
```

### 6. DelegateToolProvider 只保留只读依赖

```python
# src/tools/delegate.py
class DelegateToolProvider:
    def __init__(
        self,
        resolver: CategoryResolver,
        mcp_manager: MCPManager | None = None,
    ):
        self._resolver = resolver           # 只读，用于 can_handle / get_schemas
        self._mcp_manager = mcp_manager
        # 移除：_runner, _registry, _deps, _delegate_depth

    def can_handle(self, tool_name: str) -> bool:
        # 不变
        ...

    def get_schemas(self) -> list[ToolDict]:
        # 不变
        ...

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> str:
        if context is None:
            return "错误：delegate 调用缺少执行上下文"

        agent_name = tool_name[len(DELEGATE_PREFIX):]
        registry = context.deps.agent_registry    # 从 context 取
        agent = registry.get(agent_name)
        if agent is None:
            return f"错误：找不到 agent {agent_name}"

        # 按需连接 MCP server
        if self._mcp_manager:
            mcp_tools = [t for t in agent.tools if t.startswith("mcp_")]
            if mcp_tools:
                await self._mcp_manager.ensure_servers_for_tools(mcp_tools)

        # 构建接收方 input
        receiving_input = _build_receiving_input(
            objective=arguments.get("objective", arguments.get("task", "")),
            task=arguments.get("task", ""),
            context=arguments.get("context"),
            expected_result=arguments.get("expected_result"),
        )

        runner = context.deps.runner
        sub_ctx = RunContext(
            input=receiving_input,
            state=DynamicState(),
            deps=context.deps,                          # 继承当前上下文的 deps
            delegate_depth=context.delegate_depth + 1,
        )
        result = await runner.run(agent, sub_ctx)
        return result.text
```

### 7. build_skill_graph / build_default_graph 去除 runner 参数

```python
# src/app/presets.py
def _register_and_build(
    registry: AgentRegistry,
    skill_content: str | None = None,
    category_summaries: list[dict[str, str]] | None = None,
    business_agents: list[dict[str, str]] | None = None,
) -> CompiledGraph:
    ...
    builder.add_node(AgentNode(agent=orchestrator))       # 不传 runner
    for s in summaries:
        agent = registry.get(s["name"])
        if agent:
            builder.add_node(AgentNode(agent=agent))      # 不传 runner
    ...
```

### 8. AgentApp 存储 category_resolver

```python
# src/app/app.py
class AgentApp:
    def __init__(self, ..., category_resolver=None):
        ...
        self._category_resolver = category_resolver
```

### 9. _handle_skill 共享 category_resolver

```python
# src/app/app.py
async def _handle_skill(self, user_input: str, skill_name: str) -> None:
    skill_content = self.skill_manager.activate(skill_name)
    if not skill_content:
        return
    remaining = user_input[len(f"/{skill_name}"):].strip()
    actual_input = remaining or f"已激活 {skill_name} skill，请按指令执行。"

    skill_registry = AgentRegistry()
    if self._category_resolver:
        skill_registry.set_category_resolver(self._category_resolver)

    skill_graph = build_skill_graph(
        skill_registry,
        skill_content,
        category_summaries=self._category_summaries,
    )
    skill_engine = GraphEngine()

    ctx = RunContext(
        input=actual_input,
        state=DynamicState(),
        deps=AgentDeps(
            llm=self.deps.llm,
            tool_router=self.tool_router,
            agent_registry=skill_registry,
            graph_engine=skill_engine,
            ui=self.ui,
            memory=self.deps.memory,
            runner=self.deps.runner,           # 共享无状态 runner
        ),
    )
    result = await skill_engine.run(skill_graph, ctx)
    ...
```

### 10. PlanCompiler 适配

```python
# src/plan/compiler.py
class PlanCompiler:
    def __init__(self, agent_registry: AgentRegistry, tool_router: ToolRouter):
        self._registry = agent_registry   # 仅用于 _validate 检查 agent 存在性
        self._router = tool_router
        # 移除：self._runner = AgentRunner(registry=agent_registry)

    def _make_tool_fn(self, step: Step):
        router = self._router

        async def fn(ctx: RunContext) -> NodeResult:
            resolved = resolve_variables(tool_args, _state_to_dict(ctx.state))
            result = await router.route(tool_name, resolved, ctx)   # 透传 context
            return NodeResult(output=result)

        return fn

    def _make_agent_fn(self, step: Step):
        agent_name = step.agent_name
        agent_prompt = step.agent_prompt or step.description

        async def fn(ctx: RunContext) -> NodeResult:
            resolved_prompt = resolve_variables(agent_prompt, _state_to_dict(ctx.state))
            registry = ctx.deps.agent_registry   # 从 context 取
            runner = ctx.deps.runner             # 从 context 取
            agent = registry.get(agent_name)
            agent_ctx = replace(ctx, input=resolved_prompt)
            result = await runner.run(agent, agent_ctx)
            return NodeResult(
                output={"text": result.text, "data": result.data},
                handoff=result.handoff,
            )

        return fn
```

### 11. bootstrap.py 适配

```python
# src/app/bootstrap.py

# AgentRunner 不再需要 registry
runner = AgentRunner(
    max_tool_rounds=agent_cfg.get("max_tool_rounds", 10),
)

# build_default_graph 不再传 runner
graph = build_default_graph(
    agent_registry,
    category_summaries=category_summaries,
)

# AgentDeps 加 runner
deps = AgentDeps(
    llm=llm,
    tool_router=tool_router,
    agent_registry=agent_registry,
    graph_engine=engine,
    ui=ui,
    memory=memory_store,
    runner=runner,
)

# DelegateToolProvider 简化构造
if category_resolver:
    delegate_provider = DelegateToolProvider(
        resolver=category_resolver,
        mcp_manager=mcp_manager,
    )
    tool_router.add_provider(delegate_provider)

# AgentApp 接收 category_resolver
return AgentApp(
    ...,
    category_resolver=category_resolver,
)
```

## 改后调用链

```
GraphEngine.run(graph, context)
  └─ AgentNode.execute(context)
       └─ context.deps.runner.run(agent, context)         # runner 从 context 取
            └─ context.deps.tool_router.route(name, args, context)  # context 透传
                 └─ DelegateToolProvider.execute(name, args, context)
                      └─ context.deps.agent_registry.get(name)       # 自动匹配当前环境
                      └─ context.deps.runner.run(sub_agent, sub_ctx) # sub_ctx 继承 deps
```

**主图**：`context.deps.agent_registry` = 主 registry
**Skill 图**：`context.deps.agent_registry` = skill_registry（带 category_resolver）
**Plan 图**：`context.deps.agent_registry` = plan 的 registry

无需任何分支判断，context 携带的 deps 自动决定运行环境。

## 删除清单

| 删除项 | 文件 | 原因 |
|---|---|---|
| `AgentRunner.__init__(registry)` | runner.py | 从 context.deps 读取 |
| `AgentRunner` 中 `tool_router.set_delegate_depth(...)` | runner.py | context 透传 depth |
| `AgentNode.__init__(runner)` | node.py | 从 context.deps 读取 |
| `DelegateToolProvider._runner` | delegate.py | 从 context.deps 读取 |
| `DelegateToolProvider._registry` | delegate.py | 从 context.deps 读取 |
| `DelegateToolProvider._deps` | delegate.py | 从 context.deps 读取 |
| `DelegateToolProvider._delegate_depth` | delegate.py | 从 context 读取 |
| `DelegateToolProvider.set_delegate_depth()` | delegate.py | 不再需要 |
| `ToolRouter.set_delegate_depth()` | router.py | 不再需要 |
| `build_skill_graph(runner=...)` 参数 | presets.py | 节点从 context 取 runner |
| `build_default_graph(runner=...)` 参数 | presets.py | 同上 |
| `_register_and_build(runner=...)` 参数 | presets.py | 同上 |
| `_handle_skill` 中 `skill_runner = AgentRunner(...)` | app.py | runner 共享 |
| `PlanCompiler.__init__` 中 `self._runner = AgentRunner(...)` | compiler.py | 从 context.deps 读取 |
| `PlanCompiler._make_agent_fn` 中捕获 `registry` / `runner` | compiler.py | 从 context.deps 读取 |

## 变更文件清单

| 文件 | 变更类型 | 说明 |
|---|---|---|
| `src/tools/router.py` | 修改 | `ToolProvider.execute()` 加 context 参数；`ToolRouter.route()` 加 context 参数并透传；删除 `set_delegate_depth()`；`LocalToolProvider.execute()` 签名适配 |
| `src/tools/delegate.py` | 修改 | 构造函数只保留 resolver + mcp_manager；`execute()` 从 context 取 registry/runner/deps；删除 `set_delegate_depth()` |
| `src/agents/runner.py` | 修改 | 构造函数去掉 registry；`_build_handoff_tools()` 加 context 参数从 deps 取 registry；`run()` 中 `tool_router.route()` 透传 context；删除 depth 同步逻辑 |
| `src/agents/node.py` | 修改 | 构造函数去掉 runner；`execute()` 从 context.deps.runner 获取 |
| `src/agents/deps.py` | 修改 | 新增 `runner` 字段 |
| `src/app/presets.py` | 修改 | `_register_and_build` / `build_default_graph` / `build_skill_graph` 去掉 runner 参数；`AgentNode()` 构造不传 runner |
| `src/app/app.py` | 修改 | `__init__` 接收 `category_resolver`；`_handle_skill` 中给 skill_registry 设置 category_resolver，去掉 skill_runner 创建 |
| `src/app/bootstrap.py` | 修改 | `AgentRunner()` 去掉 registry 参数；`AgentDeps` 加 runner；`DelegateToolProvider` 简化构造；`AgentApp` 传入 category_resolver |
| `src/plan/compiler.py` | 修改 | 移除 `self._runner`；`_make_tool_fn` 透传 context 到 `router.route()`；`_make_agent_fn` 从 context.deps 取 registry/runner |
| `src/skills/provider.py` | 修改 | `SkillToolProvider.execute()` 签名适配（加 context，忽略） |
| `src/mcp/provider.py` | 修改 | `MCPToolProvider.execute()` 签名适配（加 context，忽略） |
| `tests/` | 修改 | 所有涉及 AgentRunner、AgentNode、DelegateToolProvider 的测试适配新签名 |

## 测试策略

1. **单元测试**：`ToolRouter.route()` 正确透传 context 到 provider.execute()
2. **单元测试**：`AgentRunner._build_handoff_tools()` 从 context.deps.agent_registry 读取
3. **单元测试**：`AgentNode.execute()` 从 context.deps.runner 获取 runner
4. **单元测试**：`DelegateToolProvider.execute()` 从 context 取 registry/runner，不依赖构造时状态
5. **单元测试**：`DelegateToolProvider.execute()` 在 context=None 时返回错误信息
6. **集成测试**：skill 模式下 orchestrator → handoff → category agent 可达
7. **集成测试**：skill 模式下 category agent A → delegate → category agent B 可达
8. **集成测试**：delegate_depth 限制在 skill 模式下正常工作
9. **回归测试**：普通模式（非 skill）的 handoff 和 delegation 不受影响
