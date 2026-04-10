# 跨分类智能体委派（Cross-Agent Delegate）设计

## 问题

当前架构中，工具按分类分配给不同的 category agent，每个 agent 只能使用自己分类下的工具（`handoffs=[]`）。当子智能体执行任务时发现需要其他分类的能力（如"写代码前先搜索文件"），无法跨分类访问，导致任务失败或 graph engine 报错 `Handoff target 'xxx' not found in graph`。

## 方案选择

评估了三种方案：

| 方案 | 核心思路 | token 开销 | 实现复杂度 |
|---|---|---|---|
| A. 子智能体 delegate（采用） | 每个 agent 获得其他分类的轻量 delegate 调用能力 | +520 tokens/agent | 低 |
| B. 回退 orchestrator | 子智能体回退给 orchestrator 重新调度 | 每次协作 +4 次 LLM 调用 | 中 |
| C. 运行时动态扩展 | 拦截失败的工具调用，动态注入工具 | 不确定 | 高 |

选择方案 A，理由：
1. 复用已有 `DelegateToolProvider`，改动最小
2. delegate schema 极轻量（~50 tokens/个），保住了分类节省 token 的核心价值
3. 直接 A→B 调用，比回退 orchestrator 少多次 LLM roundtrip

## 架构设计

### 改后调用链

```
用户 → Orchestrator → transfer_to Agent_A
  → Agent_A 调用 delegate_tool_B(task="搜索X")
    → DelegateToolProvider 创建子 RunContext (delegate_depth=1)
    → AgentRunner 驱动 Agent_B
    → 结果返回 Agent_A
  → Agent_A 继续完成任务 → 返回 Orchestrator
```

### 变更文件清单

| 文件 | 变更内容 |
|---|---|
| `src/agents/context.py` | `RunContext` 新增 `delegate_depth: int = 0` |
| `src/tools/categories.py` | `CategoryResolver` 新增 `get_delegate_names(exclude)` 方法；`build_instructions()` 接收 `delegate_summaries` 参数生成协作能力段落 |
| `src/agents/registry.py` | `get()` 创建 category agent 时，将其他分类的 delegate 工具名加入 `tools`，调用 `build_instructions()` 时传入 delegate 描述 |
| `src/tools/delegate.py` | `execute()` 创建子 RunContext 时设置 `delegate_depth = parent_depth + 1` |
| `src/agents/runner.py` | `_build_tools()` 当 `context.delegate_depth >= 1` 时过滤掉所有 `delegate_` 前缀工具 |

### 递归深度控制

**最大 delegate 深度：1**（只允许一层委派）。

实现方式：不是在调用时报错，而是在源头切断——当 `delegate_depth >= 1` 时，`_build_tools()` 不向 LLM 暴露 delegate 工具。LLM 看不到就不会调用，避免了"调了再拒绝"的 token 浪费。

```
Orchestrator → Agent_A (delegate_depth=0)
  → delegate_tool_B → 子 RunContext (delegate_depth=1)
    → Agent_B 的工具列表中不含任何 delegate 工具 → 只能用自己的 MCP 工具
```

深度信息通过 `RunContext.delegate_depth` 传递：
- `DelegateToolProvider.execute()` 从当前 context 读取 depth，创建子 context 时 +1
- `AgentRunner._build_tools()` 检查 depth，决定是否包含 delegate 工具

### 自身排除

Agent_A 的 delegate 工具列表不包含 `delegate_tool_A`（委派给自己无意义）。`CategoryResolver.get_delegate_names(exclude=agent_name)` 负责此过滤。

### 指令模板

改前：
```
你是{description}方面的专家。
使用你拥有的工具完成用户交给你的任务。
可用工具：{tool_names}。
只使用你拥有的工具，完成任务后返回结果摘要。
```

改后：
```
你是{description}方面的专家。

## 你的工具
{tool_names}

## 协作能力
如果任务需要你不具备的能力，可以通过以下委派工具请求其他专家协助：
{delegate_tool_descriptions}
委派时，用 task 参数清晰描述你需要的具体结果，对方会返回结果供你继续工作。

完成任务后返回结果摘要。
```

`delegate_tool_descriptions` 示例：
```
- delegate_tool_file_search_operations: 文件搜索专家
- delegate_tool_process_management: 进程管理专家
```

### Token 开销分析

以当前 9 个分类为例：

| 项目 | 数值 |
|---|---|
| 每个 delegate schema | ~50 tokens |
| 每个 agent 额外 delegate 数 | 8 个（9 - 自身） |
| 指令模板中 delegate 描述 | ~120 tokens |
| **每个 agent 额外总开销** | **~520 tokens** |

对比打平全部工具 schema 的 ~16,200 tokens，分类 + delegate 方案约 ~2,300 tokens/agent，节省 85%+。

## 不涉及的变更

- `GraphEngine` / `GraphBuilder`：无需改动，delegate 走 `ToolRouter` 路径而非 graph handoff
- `tool_categories.json`：无需改动，分类数据结构不变
- `DelegateToolProvider.get_schemas()` / `can_handle()`：无需改动，已经为所有分类生成了 delegate schema
- `MCPManager`：无需改动，按需连接逻辑已有
- Orchestrator 的 handoff 机制：不受影响，orchestrator 仍通过 `transfer_to_` 调度

## 测试策略

1. **单元测试**：`CategoryResolver.get_delegate_names()` 正确排除自身
2. **单元测试**：`AgentRegistry.get()` 创建的 agent 包含 delegate 工具名
3. **单元测试**：`_build_tools()` 在 `delegate_depth=0` 时包含 delegate 工具，`depth>=1` 时排除
4. **集成测试**：端到端 delegate 调用链——Agent_A 通过 delegate 调用 Agent_B 并获取结果
5. **集成测试**：递归深度限制——被 delegate 调用的 Agent_B 不能再 delegate
