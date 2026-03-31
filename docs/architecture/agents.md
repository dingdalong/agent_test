# 智能体系统

## 职责

定义智能体模型，驱动工具调用循环，管理智能体间的 handoff 协作。

## 核心组件

### Agent 模型（`src/agents/agent.py`）

```python
@dataclass
class Agent:
    name: str
    description: str = ""
    instructions: str = ""
    tools: list[str] = field(default_factory=list)
    handoffs: list[str] = field(default_factory=list)
    output_model: type[BaseModel] | None = None
    guardrails: list = field(default_factory=list)
    hooks: AgentHooks | None = None
```

### AgentRunner（`src/agents/runner.py`）

驱动单个智能体的工具调用循环：
1. 发送消息给 LLM（包含工具 schema）
2. 如果 LLM 返回工具调用 → 执行工具 → 将结果加入消息 → 回到 1
3. 如果检测到 `transfer_to_<name>` 调用 → 返回 `HandoffRequest`
4. 如果 LLM 返回文本 → 运行输出守卫 → 返回结果
5. 达到 `max_tool_rounds` → 强制结束

`AgentRunner` 构造时只接受配置参数（`max_tool_rounds`, `max_result_length`），不持有 `registry`。运行时从 `context.deps.agent_registry` 读取 registry，用于构建 handoff 工具和查找目标 Agent。

### AgentRegistry（`src/agents/registry.py`）

名称 → Agent 的映射表。AgentRunner 通过 registry 查找 handoff 目标。

### AgentNode（`src/agents/node.py`）

将 Agent 适配为 `GraphNode` Protocol，使智能体可以作为图节点参与图执行。构造时只持有 `agent`，不绑定 `runner`；执行时从 `context.deps.runner` 读取 runner：

```python
async def execute(self, context: Any) -> NodeResult:
    runner = context.deps.runner   # 运行时从 deps 读取
    result = await runner.run(self.agent, context)
    ...
```

### RunContext（`src/agents/context.py`）

执行上下文，贯穿整个调用链：
- `input: str` — 当前输入
- `state: DictState` — 可写状态（Pydantic `extra='allow'`）
- `deps: AgentDeps` — 依赖注入容器
- `trace: list` — 执行追踪记录

### AgentDeps（`src/agents/deps.py`）

依赖注入容器，持有所有共享组件的引用：
- `llm`, `tool_router`, `agent_registry`, `graph_engine`, `ui`, `memory`（可选）
- `runner` — `AgentRunner` 实例，供 `AgentNode` 和 `DelegateToolProvider` 在运行时读取

## 设计原则

**构造时绑定只读配置，运行时从 context 取环境。**

- `AgentNode`、`DelegateToolProvider` 等组件构造时只接受静态配置（如 `agent`、`resolver`），不在构造期注入 `runner`、`registry` 等运行时依赖。
- 运行时依赖统一通过 `context.deps` 传递，保持组件的可测试性和低耦合。

## 预设智能体（`src/app/presets.py`）

| 智能体 | 职责 |
|--------|------|
| `orchestrator` | 总控，根据用户请求路由到专家或直接回答 |
| `weather_agent` | 天气查询（工具占位） |
| `calendar_agent` | 日历管理（工具占位） |
| `email_agent` | 邮件发送（工具占位） |
| `planner` | 转发到 PlanFlow 执行多步骤任务 |

## 数据流

```
用户输入 → RunContext → GraphEngine
  → orchestrator (AgentNode)
    → AgentRunner.run()
      → LLM chat（带工具 schema）
      → 工具调用循环
      → HandoffRequest / 最终回复
  → handoff 目标智能体（如果有）
  → 最终输出
```
