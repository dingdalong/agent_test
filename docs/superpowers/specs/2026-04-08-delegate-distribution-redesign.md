# Delegate 分配策略重新设计

## 背景

当前架构中，工具类 agent（如 `tool_terminal`）在懒创建时会被分配 **自身类别的工具 + 所有其他类别的 delegate 工具**。这导致：

1. 工具类 agent 既是执行者又能委派，职责不够清晰
2. 需要 `delegate_depth` 机制防止无限委托链
3. 工具类 agent 的 system prompt 中嵌入了大量 delegate 描述，增加 token 消耗

## 目标

**将 delegate 工具从工具类 agent 中移除，改由非工具类 agent 自动持有。**

- 工具类 agent：纯执行者，只持有自身类别的工具
- 所有非工具类 agent（orchestrator、skill step agent、未来新增的任何 agent）：自动获得全部 delegate 工具

## 架构变更

### 变更前

```
Orchestrator（handoff only）
  └─ handoff → tool_terminal agent
                  ├─ 自身工具: run_command, run_shell
                  └─ delegate: delegate_tool_files, delegate_tool_calc, ...
                        └─ tool_files agent（delegate_depth=1, delegate 被过滤）
```

### 变更后

```
Orchestrator（handoff + delegate 工具，自动注入）
  ├─ handoff → planner
  ├─ delegate_tool_terminal → tool_terminal agent（只有: run_command, run_shell）
  ├─ delegate_tool_files   → tool_files agent（只有: read_file, write_file, ...）
  └─ parallel_delegate     → 并发调用

Skill step agent（delegate 工具，自动注入）
  ├─ delegate_tool_terminal → ...
  └─ delegate_tool_files   → ...
```

## 核心设计：在 `_build_tools()` 自动注入

**注入点**：`AgentRunner._build_tools()`——所有 agent 运行前组装工具列表的唯一入口。

**规则**：
1. 获取 agent 声明的工具 schemas
2. 若 agent **不是**工具类 agent（`CategoryResolver.can_resolve(agent.name)` 为 False）→ 自动附加所有 delegate schemas
3. `delegate_depth >= 1` 时仍过滤 delegate（安全网，防止 delegate 链）

**判断方式**：通过 `CategoryResolver.can_resolve(agent.name)` 判断是否为工具类 agent。`AgentRunner` 通过 `context.deps` 中新增的 `category_resolver` 字段获取 resolver 引用。

### 伪代码

```python
def _build_tools(self, agent, context):
    all_schemas = tool_router.get_all_schemas()

    resolver = context.deps.category_resolver
    is_tool_agent = resolver and resolver.can_resolve(agent.name)

    if not agent.tools and not is_tool_agent:
        # 无声明工具的非工具类 agent → system tools + 所有 delegate
        delegate_schemas = [s for s in all_schemas if s["function"]["name"].startswith("delegate_") or s["function"]["name"] == "parallel_delegate"]
        system_schemas = [s for s in all_schemas if s["function"]["name"] in SYSTEM_TOOLS]
        return system_schemas + delegate_schemas

    if not agent.tools:
        # 无声明工具的工具类 agent（不应发生，但防御性处理）
        return [s for s in all_schemas if s["function"]["name"] in SYSTEM_TOOLS]

    allowed = set(agent.tools) | SYSTEM_TOOLS

    if not is_tool_agent:
        # 非工具类 agent → 追加所有 delegate
        for s in all_schemas:
            fname = s["function"]["name"]
            if fname.startswith("delegate_") or fname == "parallel_delegate":
                allowed.add(fname)

    # 安全网：delegate 链中不允许再次 delegate
    if context.delegate_depth >= 1:
        allowed = {n for n in allowed if not n.startswith("delegate_") and n != "parallel_delegate"}

    return [s for s in all_schemas if s["function"]["name"] in allowed]
```

## 逐文件改动

### 1. `src/agents/deps.py` — 新增 `category_resolver` 字段

```python
class AgentDeps(BaseModel):
    ...
    category_resolver: Any = None  # CategoryResolver — 可选
```

### 2. `src/agents/runner.py` — 重写 `_build_tools()`

按上述伪代码实现。关键变化：
- 非工具类 agent 自动获得 delegate schemas
- 工具类 agent 不再包含 delegate
- `delegate_depth` 检查保留作为安全网
- `DELEGATE_TOOLS` 常量定义 delegate 工具名的判定逻辑

### 3. `src/agents/registry.py` — 懒创建不再添加 delegate

```python
def get(self, name):
    ...
    if self._category_resolver and self._category_resolver.can_resolve(name):
        cat = self._category_resolver.get_category(name)
        instructions = self._category_resolver.build_instructions(name)
        tools = list(cat["tools"].keys())  # 只有自身工具，不再加 delegate_names
        agent = Agent(name=name, description=cat["description"],
                      instructions=instructions, tools=tools, handoffs=[])
        self.register(agent)
        return agent
```

### 4. `src/tools/categories.py` — 简化 `build_instructions()`

- `build_instructions()` 不再接收 `delegate_summaries` 参数
- 移除 `_DELEGATE_SECTION_TEMPLATE` 模板
- 移除 `get_delegate_info()` 方法（不再需要）
- `get_delegate_names()` 保留（可能被其他地方使用）

简化后的模板：
```python
_TOOL_AGENT_INSTRUCTIONS_TEMPLATE = (
    "你是{description}方面的专家。\n\n"
    "## 你的工具\n"
    "{tool_names}\n\n"
    "完成任务后返回结果摘要。"
)
```

### 5. `src/app/presets.py` — orchestrator 无需手动添加 delegate

因为 `_build_tools()` 自动注入，orchestrator 的 `tools` 字段无需变更。orchestrator 的 instructions 中已有 handoff 路由指令，delegate 描述由 `_build_tools()` 注入的 schemas 自动提供给 LLM。

### 6. `src/app/app.py` — skill step agent 无需手动改造

`make_step_agent()` 创建 agent 时 `tools=[]`，`_build_tools()` 检测到非工具类 agent 且 `tools` 为空，会自动注入 system tools + 所有 delegate schemas。无需改动。

### 7. `src/app/bootstrap.py` — 组装时传入 `category_resolver`

在构建 `AgentDeps` 时增加 `category_resolver=resolver`。

## 不变的部分

- `DelegateToolProvider`：schema 生成和执行逻辑不变
- `ToolRouter`：路由逻辑不变
- Handoff 机制：完全不变
- `parallel_delegate`：不变
- `delegate_depth` 递增逻辑：不变（在 `DelegateToolProvider.execute()` 中）

## 测试要点

1. **工具类 agent**：`_build_tools()` 返回的 schemas 中不含 `delegate_*`
2. **Orchestrator**：`_build_tools()` 返回的 schemas 中包含所有 `delegate_*` + `parallel_delegate`
3. **Skill step agent**（`tools=[]`）：`_build_tools()` 返回 system tools + 所有 delegate schemas
4. **delegate_depth >= 1**：任何 agent 都不返回 delegate schemas
5. **新建 agent 不声明 tools**：自动获得 delegate（验证"默认带 delegate"的承诺）
6. **回归测试**：工具类 agent 懒创建后的 `tools` 列表不含 `delegate_*` 名称
