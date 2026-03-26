# Tools 模块重构设计

## 目标

对 `src/tools/` 模块进行整体架构重新设计，不考虑向后兼容性。解决当前的核心问题：

1. `__init__.py` 承担了装饰器、注册表、自动发现、模块导出四重职责
2. `ToolExecutor` 混合了 skill 路由、MCP 路由、敏感确认、校验、执行等逻辑
3. 工具文件与框架代码混在同一层级

## 架构：分层设计

采用分层架构，将注册、发现、路由、执行、中间件拆成独立模块。

### 目录结构

```
src/tools/
├── __init__.py        # 纯导出
├── registry.py        # ToolEntry + ToolRegistry
├── decorator.py       # @tool 装饰器
├── discovery.py       # discover_tools()
├── router.py          # ToolProvider 协议 + ToolRouter
├── executor.py        # ToolExecutor（纯校验+执行）
├── middleware.py       # 中间件链
├── schemas.py         # ToolDict 类型定义
├── tool_call.py       # execute_tool_calls（批量异步）
└── builtin/
    ├── __init__.py
    ├── calculator.py   # 简单无状态工具示例
    └── file.py         # 有状态、敏感工具示例
```

删除 `weather.py`、`calendar.py`、`email.py`。

---

## 模块详细设计

### 1. schemas.py — 类型定义

定义整个模块共用的类型，避免循环导入。

```python
from typing import TypedDict, Any

class ToolDict(TypedDict):
    """OpenAI function-calling 格式的工具 schema"""
    type: str           # 固定为 "function"
    function: dict      # {"name", "description", "parameters"}
```

### 2. registry.py — 注册表

纯数据存储，不包含发现或执行逻辑。

```python
from dataclasses import dataclass, field
from typing import Any, Callable
from pydantic import BaseModel

@dataclass
class ToolEntry:
    """工具的完整元数据"""
    name: str
    func: Callable
    model: type[BaseModel]
    description: str
    parameters_schema: dict[str, Any]
    sensitive: bool = False
    confirm_template: str | None = None

class ToolRegistry:
    """工具注册表 — 存储和查询工具定义"""

    def __init__(self):
        self._entries: dict[str, ToolEntry] = {}

    def register(self, entry: ToolEntry) -> None:
        """注册一个工具。重名时 warn 并跳过。"""
        if entry.name in self._entries:
            logger.warning(f"工具 '{entry.name}' 已注册，跳过")
            return
        self._entries[entry.name] = entry

    def get(self, name: str) -> ToolEntry | None:
        return self._entries.get(name)

    def list_entries(self) -> list[ToolEntry]:
        return list(self._entries.values())

    def has(self, name: str) -> bool:
        return name in self._entries

    def get_schemas(self) -> list[ToolDict]:
        """返回所有工具的 OpenAI 格式 schema 列表"""
        return [
            {
                "type": "function",
                "function": {
                    "name": entry.name,
                    "description": entry.description,
                    "parameters": entry.parameters_schema,
                }
            }
            for entry in self._entries.values()
        ]
```

### 3. decorator.py — @tool 装饰器

只做一件事：构造 ToolEntry 并注册到全局 registry。

```python
from .registry import ToolRegistry, ToolEntry

# 全局 registry 实例
_registry = ToolRegistry()

def get_registry() -> ToolRegistry:
    return _registry

def tool(
    model: type[BaseModel],
    description: str,
    name: str | None = None,
    sensitive: bool = False,
    confirm_template: str | None = None,
):
    """工具注册装饰器"""
    def decorator(func):
        tool_name = name or func.__name__
        entry = ToolEntry(
            name=tool_name,
            func=func,
            model=model,
            description=description,
            parameters_schema=model.model_json_schema(),
            sensitive=sensitive,
            confirm_template=confirm_template,
        )
        _registry.register(entry)
        return func
    return decorator
```

### 4. discovery.py — 自动发现

独立的发现逻辑，扫描指定目录下的 Python 模块以触发 @tool 注册。

```python
import importlib
from pathlib import Path

def discover_tools(package: str, package_dir: Path, skip: set[str] | None = None) -> None:
    """
    扫描 package_dir 下的 .py 文件并导入，以触发 @tool 装饰器注册。

    Args:
        package: 包的完整名称（如 "src.tools.builtin"）
        package_dir: 包的目录路径
        skip: 要跳过的模块名集合
    """
    skip = skip or {"__init__"}
    for item in sorted(package_dir.glob("*.py")):
        module_name = item.stem
        if module_name in skip:
            continue
        importlib.import_module(f".{module_name}", package=package)
```

### 5. router.py — 统一路由层

通过 `ToolProvider` 协议统一所有工具来源的分发。

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ToolProvider(Protocol):
    """所有工具来源的统一接口"""

    def can_handle(self, tool_name: str) -> bool:
        """判断此 provider 能否处理该工具"""
        ...

    async def execute(self, tool_name: str, arguments: dict) -> str:
        """执行工具调用"""
        ...

    def get_schemas(self) -> list[ToolDict]:
        """返回此 provider 提供的所有工具 schema"""
        ...

class ToolRouter:
    """
    按注册顺序依次查询 provider，找到第一个能处理的执行。
    同时聚合所有 provider 的 schema。
    """

    def __init__(self):
        self._providers: list[ToolProvider] = []

    def add_provider(self, provider: ToolProvider) -> None:
        self._providers.append(provider)

    async def route(self, tool_name: str, arguments: dict) -> str:
        """路由到合适的 provider 执行"""
        for provider in self._providers:
            if provider.can_handle(tool_name):
                return await provider.execute(tool_name, arguments)
        return f"错误：未找到工具 '{tool_name}'"

    def get_all_schemas(self) -> list[ToolDict]:
        """聚合所有 provider 的 schema"""
        schemas = []
        for provider in self._providers:
            schemas.extend(provider.get_schemas())
        return schemas

    def is_sensitive(self, tool_name: str) -> bool:
        """检查工具是否为敏感工具（用于 plan executor）"""
        for provider in self._providers:
            if provider.can_handle(tool_name):
                if hasattr(provider, 'is_sensitive'):
                    return provider.is_sensitive(tool_name)
                return False
        return False
```

### 6. executor.py — 纯执行

只做 Pydantic 校验和函数调用，不包含路由或中间件逻辑。

```python
import asyncio
from .registry import ToolRegistry

class ToolExecutor:
    """校验参数并调用工具函数"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def execute(self, tool_name: str, arguments: dict) -> str:
        """校验参数并执行工具函数"""
        entry = self.registry.get(tool_name)
        if not entry:
            raise ValueError(f"未注册的工具: {tool_name}")

        # Pydantic 校验
        validated = entry.model(**arguments).model_dump()

        # 执行
        return await self._run_func(entry.func, validated)

    async def _run_func(self, func, kwargs: dict) -> str:
        if asyncio.iscoroutinefunction(func):
            result = await func(**kwargs)
        else:
            result = await asyncio.to_thread(func, **kwargs)
        return str(result)
```

### 7. middleware.py — 中间件链

横切关注点以中间件形式组织，支持自由组合。

```python
from typing import Callable, Awaitable

# 类型定义
NextFn = Callable[[str, dict], Awaitable[str]]
Middleware = Callable[[str, dict, NextFn], Awaitable[str]]

def build_pipeline(execute_fn: NextFn, middlewares: list[Middleware]) -> NextFn:
    """
    构建中间件链。middlewares 按顺序从外到内包裹 execute_fn。
    列表中第一个中间件最先执行（最外层）。
    """
    pipeline = execute_fn
    for mw in reversed(middlewares):
        prev = pipeline
        async def wrapped(name, args, _prev=prev, _mw=mw):
            return await _mw(name, args, _prev)
        pipeline = wrapped
    return pipeline


# --- 内置中间件 ---

def sensitive_confirm_middleware(registry: ToolRegistry) -> Middleware:
    """敏感工具执行前需要用户确认"""
    async def middleware(name: str, args: dict, next_fn: NextFn) -> str:
        entry = registry.get(name)
        if entry and entry.sensitive:
            # 构造确认消息
            if entry.confirm_template:
                msg = entry.confirm_template.format(**args)
            else:
                msg = f"即将执行敏感操作: {name}"

            from src.core.io import agent_output, agent_input
            agent_output(f"\n⚠️  {msg}")
            confirm = agent_input("确认执行？(y/n): ").strip().lower()
            if confirm != 'y':
                return "操作已取消"

        return await next_fn(name, args)
    return middleware


def truncate_middleware(max_length: int = 2000) -> Middleware:
    """截断过长的结果"""
    async def middleware(name: str, args: dict, next_fn: NextFn) -> str:
        result = await next_fn(name, args)
        if len(result) > max_length:
            return result[:max_length] + f"\n...(结果已截断，共 {len(result)} 字符)"
        return result
    return middleware


def error_handler_middleware() -> Middleware:
    """捕获异常，返回错误字符串而非抛出"""
    async def middleware(name: str, args: dict, next_fn: NextFn) -> str:
        try:
            return await next_fn(name, args)
        except Exception as e:
            return f"工具 '{name}' 执行出错: {e}"
    return middleware
```

### 8. LocalToolProvider — 桥接 Router 和 Executor

将 ToolExecutor + 中间件组合成一个 ToolProvider。

放在 `router.py` 中，与 `ToolProvider` 协议和 `ToolRouter` 同文件。

```python
# router.py 中

class LocalToolProvider:
    """本地工具的 Provider 实现"""

    def __init__(self, registry: ToolRegistry, executor: ToolExecutor, middlewares: list[Middleware]):
        self.registry = registry
        self.pipeline = build_pipeline(executor.execute, middlewares)

    def can_handle(self, tool_name: str) -> bool:
        return self.registry.has(tool_name)

    async def execute(self, tool_name: str, arguments: dict) -> str:
        return await self.pipeline(tool_name, arguments)

    def get_schemas(self) -> list[ToolDict]:
        return self.registry.get_schemas()

    def is_sensitive(self, tool_name: str) -> bool:
        entry = self.registry.get(tool_name)
        return entry.sensitive if entry else False
```

### 9. tool_call.py — 批量异步执行

接口不变，但参数改为接收 `ToolRouter`。

```python
import asyncio
import json

async def execute_tool_calls(content, tool_calls, router: ToolRouter) -> list[dict]:
    """
    并行执行 LLM 返回的 tool_calls，通过 router 分发。
    返回包含 assistant 消息和 tool 结果消息的列表。
    """
    messages = [{"role": "assistant", "content": content, "tool_calls": tool_calls}]

    async def run_one(tc):
        args = json.loads(tc["function"]["arguments"])
        result = await router.route(tc["function"]["name"], args)
        return {
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": result,
        }

    tasks = [asyncio.create_task(run_one(tc)) for tc in tool_calls]
    results = await asyncio.gather(*tasks)

    messages.extend(results)
    return messages
```

### 10. __init__.py — 纯导出

```python
from .schemas import ToolDict
from .registry import ToolRegistry, ToolEntry
from .decorator import tool, get_registry
from .discovery import discover_tools
from .router import ToolRouter, ToolProvider, LocalToolProvider
from .executor import ToolExecutor
from .middleware import (
    build_pipeline,
    sensitive_confirm_middleware,
    truncate_middleware,
    error_handler_middleware,
)
from .tool_call import execute_tool_calls

__all__ = [
    "ToolDict", "ToolRegistry", "ToolEntry",
    "tool", "get_registry",
    "discover_tools",
    "ToolRouter", "ToolProvider", "LocalToolProvider",
    "ToolExecutor",
    "build_pipeline", "sensitive_confirm_middleware",
    "truncate_middleware", "error_handler_middleware",
    "execute_tool_calls",
]
```

### 11. builtin/ — 示例工具

保留 `calculator.py` 和 `file.py`，代码基本不变，仅更新导入路径：

```python
# builtin/calculator.py
from src.tools import tool  # 或 from .. import tool
```

---

## 启动流程（main.py 变化）

```python
from src.tools import (
    get_registry, discover_tools,
    ToolExecutor, ToolRouter, LocalToolProvider,
    sensitive_confirm_middleware, truncate_middleware, error_handler_middleware,
)

# 1. 发现并注册本地工具
discover_tools("src.tools.builtin", Path("src/tools/builtin"))

# 2. 构建本地工具执行管道
registry = get_registry()
executor = ToolExecutor(registry)
middlewares = [
    error_handler_middleware(),
    sensitive_confirm_middleware(registry),
    truncate_middleware(2000),
]
local_provider = LocalToolProvider(registry, executor, middlewares)

# 3. 构建路由器
router = ToolRouter()
router.add_provider(local_provider)

# 4. 注册 MCP provider（如果有）
if mcp_manager:
    router.add_provider(MCPToolProvider(mcp_manager))

# 5. 注册 Skill provider（如果有）
if skill_manager:
    router.add_provider(SkillToolProvider(skill_manager))

# 6. 获取所有工具 schema
all_tools = router.get_all_schemas()
```

---

## 外部模块适配

| 文件 | 当前用法 | 重构后 |
|---|---|---|
| `main.py` | `from src.tools import tools, tool_executor` | 使用 `ToolRouter`，见上方启动流程 |
| `specialist_runner.py` | `execute_tool_calls(..., tool_executor)` | `execute_tool_calls(..., router)` |
| `orchestrator.py` | `from src.tools import ToolDict` | 不变 |
| `plan/executor.py` | `tool_executor.execute()` / `tool_executor.is_sensitive()` | `router.route()` / `router.is_sensitive()` |
| `flows/planning.py` | 接收 `ToolExecutor` | 改为接收 `ToolRouter` |

---

## MCPToolProvider 和 SkillToolProvider 的位置

这两个 Provider 分别放在各自的模块中：
- `MCPToolProvider` → `src/mcp/provider.py`（包装 MCPManager，实现 ToolProvider 协议）
- `SkillToolProvider` → `src/skills/provider.py`（包装 SkillManager，实现 ToolProvider 协议）

这样 `src/tools/` 不依赖 `src/mcp/` 和 `src/skills/`，依赖方向是反过来的：MCP 和 Skills 模块依赖 tools 的 `ToolProvider` 协议。

---

## 设计原则

1. **单一职责**：每个模块只做一件事
2. **依赖方向**：decorator → registry ← executor ← middleware ← router（不存在循环依赖）
3. **可扩展**：新增工具来源只需实现 `ToolProvider` 协议
4. **可测试**：每层可独立测试，不需要 mock 整个系统
5. **中间件可组合**：敏感确认、截断、错误处理可自由增减和排列
