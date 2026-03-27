# main.py 重构设计

## 背景

当前 `main.py` (~345行) 承担了 7 个不同的职责：依赖模型定义、辅助函数、全局状态、Agent 定义与图构建、计划流程编排、输入分发处理、应用启动与主循环。这导致：

1. **耦合严重** — 新增 Agent、命令、流程都要改 main.py
2. **I/O 绑死 CLI** — `run_plan_flow` 直接调用 `agent_input`/`agent_output`，无法接入 Web
3. **未使用的全局状态** — `store`、`buffer` 定义了但没被引用
4. **可测试性差** — 逻辑与 I/O 混合，难以单元测试

## 目标

- 职责分离：每个文件只做一件事
- I/O 解耦：核心逻辑不依赖具体界面，CLI 和 Web 通过协议注入
- main.py 瘦身到 ~15 行，只做启动

## 架构概览

```
main.py                         # 入口 (~15行)
src/
  app.py                        # AgentApp — 应用核心
  interfaces/
    __init__.py
    base.py                     # UserInterface 协议
    cli.py                      # CLI 实现
  agents/
    deps.py                     # AgentDeps 运行时依赖模型
    definitions.py              # Agent 定义 + 图构建
    ...                         # 其余不变
  plan/
    flow.py                     # PlanFlow 计划编排
    ...                         # 其余不变
  memory/
    utils.py                    # _build_collection_name 移入
    ...
```

## 模块设计

### 1. UserInterface 协议 (`src/interfaces/base.py`)

抽象 I/O 操作，CLI 和 Web 各自实现。

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class UserInterface(Protocol):
    async def prompt(self, message: str) -> str:
        """获取用户输入，message 作为提示语"""
        ...

    async def display(self, message: str) -> None:
        """展示信息给用户"""
        ...

    async def confirm(self, message: str) -> bool:
        """请求用户确认，返回 True/False"""
        ...
```

### 2. CLI 实现 (`src/interfaces/cli.py`)

封装现有 `src/core/io.py` 的控制台 I/O 逻辑为类实现。

```python
import asyncio

class CLIInterface:
    async def prompt(self, message: str) -> str:
        return await asyncio.to_thread(input, message)

    async def display(self, message: str) -> None:
        print(message, end="", flush=True)

    async def confirm(self, message: str) -> bool:
        response = await self.prompt(f"{message} (y/n): ")
        return response.strip().lower() in ("y", "yes", "确认")
```

### 3. AgentDeps (`src/agents/deps.py`)

从 main.py 移出，新增 `ui` 字段用于 I/O 操作。

```python
from typing import Any
from pydantic import BaseModel, ConfigDict

class AgentDeps(BaseModel):
    """外部依赖：传递给 AgentRunner 和 plan 流程。"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tool_router: Any = None
    agent_registry: Any = None
    graph_engine: Any = None
    ui: Any = None  # UserInterface
```

### 4. Agent 定义 (`src/agents/definitions.py`)

将所有 Agent 定义和图构建逻辑从 main.py 移出。

包含：
- `build_default_agents(registry)` — 注册 weather/calendar/email/orchestrator/planner
- `build_default_graph(registry)` — 构建默认图，返回 CompiledGraph
- orchestrator 的 instructions 构建逻辑
- `planner_node_fn` — planner 的 FunctionNode 实现（调用 `PlanFlow`）

Skill 激活时重建图的逻辑也在这里：
- `build_skill_graph(registry, skill_content)` — 带 skill 注入的图构建

### 5. PlanFlow (`src/plan/flow.py`)

将 `run_plan_flow` 和 `_format_plan` 从 main.py 移到 plan 模块。

关键变更：所有 I/O 通过 `ui`（从 `ctx.deps.ui` 获取）而非直接调用 `agent_input`/`agent_output`。

```python
class PlanFlow:
    """计划编排流程：澄清 -> 生成 -> 确认 -> 编译 -> 执行。"""

    def __init__(
        self,
        tool_router: ToolRouter,
        agent_registry: AgentRegistry,
        engine: GraphEngine,
        ui: UserInterface,
    ):
        self.tool_router = tool_router
        self.agent_registry = agent_registry
        self.engine = engine
        self.ui = ui

    async def run(self, user_input: str) -> str:
        """完整计划流程"""
        # 1. 澄清循环 — 用 self.ui.display / self.ui.prompt
        # 2. 生成计划
        # 3. 确认/调整循环
        # 4. 编译并执行
        ...

    @staticmethod
    def format_plan(plan: Plan) -> str:
        """格式化计划用于展示"""
        ...
```

### 6. AgentApp (`src/app.py`)

应用核心，组装所有组件，提供消息处理接口。

```python
class AgentApp:
    def __init__(self, ui: UserInterface):
        self.ui = ui
        # 以下在 setup() 中初始化
        self.router: ToolRouter
        self.engine: GraphEngine
        self.graph: CompiledGraph
        self.agent_registry: AgentRegistry
        self.skill_manager: SkillManager
        self.mcp_manager: MCPManager

    async def setup(self) -> None:
        """初始化所有组件"""
        # 1. 发现并注册本地工具
        # 2. 构建工具执行管道 + 路由器
        # 3. 初始化 MCP
        # 4. 初始化 Skills
        # 5. 构建 Agent 注册表、图、引擎

    async def process(self, user_input: str) -> None:
        """处理单条用户消息"""
        # 1. 护栏检查
        # 2. /plan 命令 -> PlanFlow
        # 3. /skill 命令 -> skill 图执行
        # 4. 正常 -> GraphEngine 执行

    async def run(self) -> None:
        """CLI 主循环"""
        self.ui.display("Agent 已启动，输入 'exit' 退出。\n")
        while True:
            user_input = await self.ui.prompt("\n你: ")
            if user_input.lower() in ("exit", "quit"):
                break
            await self.process(user_input)

    async def shutdown(self) -> None:
        """清理资源"""
        await self.mcp_manager.disconnect_all()
```

Web 接入时用法：
```python
# Web handler 示例
app = AgentApp(ui=WebSocketInterface(ws))
await app.setup()
# 每条消息调用 process()，不用 run()
await app.process(user_message)
```

### 7. main.py — 瘦入口

```python
import asyncio
from src.app import AgentApp
from src.interfaces.cli import CLIInterface

async def main():
    app = AgentApp(ui=CLIInterface())
    await app.setup()
    try:
        await app.run()
    finally:
        await app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### 8. 清理

- **删除** main.py 中未使用的 `store`、`buffer` 全局变量
- **删除** main.py 中的 `input_guard` 全局实例（移到 `AgentApp` 内部）
- **移动** `_build_collection_name` 到 `src/memory/utils.py`
- **保留** `src/core/io.py`（其他模块可能仍在使用，不在本次重构范围）

## 数据流

```
用户输入
  │
  ▼
main.py → AgentApp.run() → AgentApp.process(input)
  │
  ├─ InputGuardrail.check() ─── 拦截 → ui.display(reason)
  │
  ├─ /plan → PlanFlow.run()
  │           ├─ ui.display() / ui.prompt()  ← 澄清/确认
  │           ├─ generate_plan / adjust_plan  ← LLM
  │           ├─ PlanCompiler.compile()       ← 图编译
  │           └─ engine.run()                 ← 执行
  │
  ├─ /skill → skill_manager.activate()
  │           ├─ build_skill_graph()
  │           └─ skill_engine.run()
  │
  └─ 正常 → engine.run(default_graph, ctx)
              └─ orchestrator → specialist handoff
```

## 不变的部分

以下模块不在本次重构范围内，保持原样：
- `src/agents/agent.py` — Agent 数据模型
- `src/agents/runner.py` — AgentRunner 工具调用循环
- `src/agents/graph/` — 图类型、构建器、引擎
- `src/tools/` — 工具系统
- `src/plan/planner.py` — 计划生成/调整
- `src/plan/compiler.py` — PlanCompiler
- `src/mcp/` — MCP 客户端
- `src/skills/` — Skill 管理
- `src/memory/` — 记忆系统
- `src/core/` — 核心工具（async_api, guardrails, structured_output 等）
- `config.py` — 配置
