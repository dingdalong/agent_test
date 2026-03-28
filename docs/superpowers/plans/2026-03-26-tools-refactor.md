# Tools 模块重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `src/tools/` 模块从职责混杂的单层结构重构为分层架构（registry / decorator / discovery / router / executor / middleware），不考虑向后兼容性。

**Architecture:** 注册表存储工具元数据，装饰器负责注册，发现模块扫描 builtin 子包，ToolExecutor 只做校验+调用，中间件链处理横切关注点（敏感确认/截断/错误处理），ToolRouter 通过 ToolProvider 协议统一分发 local/MCP/skill 工具。

**Tech Stack:** Python 3.12, Pydantic v2, asyncio, pytest + pytest-asyncio

---

## 文件结构

| 操作 | 文件 | 职责 |
|------|------|------|
| Create | `src/tools/schemas.py` | ToolDict 类型定义 |
| Create | `src/tools/registry.py` | ToolEntry dataclass + ToolRegistry 类 |
| Create | `src/tools/decorator.py` | @tool 装饰器 + get_registry() |
| Create | `src/tools/discovery.py` | discover_tools() 自动扫描 |
| Create | `src/tools/executor.py` | ToolExecutor（纯校验+执行） |
| Create | `src/tools/middleware.py` | 中间件链 + 内置中间件 |
| Create | `src/tools/router.py` | ToolProvider 协议 + ToolRouter + LocalToolProvider |
| Create | `src/tools/builtin/__init__.py` | 空包 |
| Move+Edit | `src/tools/builtin/calculator.py` | 从 src/tools/calculator.py 迁移 |
| Move+Edit | `src/tools/builtin/file.py` | 从 src/tools/file.py 迁移 |
| Rewrite | `src/tools/tool_call.py` | 改为使用 ToolRouter |
| Rewrite | `src/tools/__init__.py` | 纯导出 |
| Create | `src/mcp/provider.py` | MCPToolProvider |
| Create | `src/skills/provider.py` | SkillToolProvider |
| Modify | `main.py` | 使用新的启动流程 |
| Modify | `src/agents/specialist_runner.py` | ToolExecutor → ToolRouter |
| Modify | `src/agents/orchestrator.py` | ToolExecutor → ToolRouter |
| Modify | `src/flows/planning.py` | ToolExecutor → ToolRouter |
| Modify | `src/plan/executor.py` | ToolExecutor → ToolRouter |
| Rewrite | `tests/test_tools.py` | 新架构测试 |
| Delete | `src/tools/calculator.py` | 迁移到 builtin/ |
| Delete | `src/tools/file.py` | 迁移到 builtin/ |
| Delete | `src/tools/weather.py` | 不再保留 |
| Delete | `src/tools/calendar.py` | 不再保留 |
| Delete | `src/tools/email.py` | 不再保留 |
| Delete | `src/tools/tool_executor.py` | 被 executor.py 替代 |

---

### Task 1: schemas.py — 类型定义

**Files:**
- Create: `src/tools/schemas.py`
- Test: `tests/test_tools_refactor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tools_refactor.py
from src.tools.schemas import ToolDict


def test_tool_dict_type():
    """ToolDict 可以正确构造"""
    td: ToolDict = {
        "type": "function",
        "function": {
            "name": "test",
            "description": "a test tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    assert td["type"] == "function"
    assert td["function"]["name"] == "test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/test_tools_refactor.py::test_tool_dict_type -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tools.schemas'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/tools/schemas.py
"""工具类型定义，供整个 tools 模块和外部使用。"""

from typing import Any, Dict, TypedDict


class ToolDict(TypedDict):
    """OpenAI function-calling 格式的工具 schema"""
    type: str
    function: Dict[str, Any]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/test_tools_refactor.py::test_tool_dict_type -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/schemas.py tests/test_tools_refactor.py
git commit -m "refactor(tools): add schemas.py with ToolDict type"
```

---

### Task 2: registry.py — ToolEntry + ToolRegistry

**Files:**
- Create: `src/tools/registry.py`
- Test: `tests/test_tools_refactor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tools_refactor.py （追加）
from pydantic import BaseModel, Field
from src.tools.registry import ToolEntry, ToolRegistry


class DummyModel(BaseModel):
    value: str = Field(description="test value")


def _make_entry(name: str = "dummy", sensitive: bool = False) -> ToolEntry:
    async def dummy_func(value: str) -> str:
        return value
    return ToolEntry(
        name=name,
        func=dummy_func,
        model=DummyModel,
        description="A dummy tool",
        parameters_schema=DummyModel.model_json_schema(),
        sensitive=sensitive,
        confirm_template=None,
    )


def test_registry_register_and_get():
    reg = ToolRegistry()
    entry = _make_entry("test_tool")
    reg.register(entry)
    assert reg.has("test_tool")
    assert reg.get("test_tool") is entry
    assert reg.get("nonexistent") is None


def test_registry_duplicate_skips():
    reg = ToolRegistry()
    entry1 = _make_entry("dup")
    entry2 = _make_entry("dup")
    reg.register(entry1)
    reg.register(entry2)
    assert reg.get("dup") is entry1
    assert len(reg.list_entries()) == 1


def test_registry_get_schemas():
    reg = ToolRegistry()
    reg.register(_make_entry("tool_a"))
    reg.register(_make_entry("tool_b"))
    schemas = reg.get_schemas()
    assert len(schemas) == 2
    names = {s["function"]["name"] for s in schemas}
    assert names == {"tool_a", "tool_b"}
    assert all(s["type"] == "function" for s in schemas)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/test_tools_refactor.py::test_registry_register_and_get tests/test_tools_refactor.py::test_registry_duplicate_skips tests/test_tools_refactor.py::test_registry_get_schemas -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tools.registry'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/tools/registry.py
"""工具注册表 — 存储和查询工具定义。"""

import logging
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from .schemas import ToolDict

logger = logging.getLogger(__name__)


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
        if entry.name in self._entries:
            logger.warning(f"工具 '{entry.name}' 已注册，跳过")
            return
        self._entries[entry.name] = entry

    def get(self, name: str) -> ToolEntry | None:
        return self._entries.get(name)

    def has(self, name: str) -> bool:
        return name in self._entries

    def list_entries(self) -> list[ToolEntry]:
        return list(self._entries.values())

    def get_schemas(self) -> list[ToolDict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": entry.name,
                    "description": entry.description,
                    "parameters": entry.parameters_schema,
                },
            }
            for entry in self._entries.values()
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/test_tools_refactor.py -v -k "registry"`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tools/registry.py tests/test_tools_refactor.py
git commit -m "refactor(tools): add ToolEntry and ToolRegistry"
```

---

### Task 3: decorator.py — @tool 装饰器

**Files:**
- Create: `src/tools/decorator.py`
- Test: `tests/test_tools_refactor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tools_refactor.py （追加）
from src.tools.decorator import tool, get_registry


def test_tool_decorator_registers():
    registry = get_registry()
    initial_count = len(registry.list_entries())

    class TestParams(BaseModel):
        x: int = Field(description="a number")

    @tool(model=TestParams, description="test decorator tool")
    async def _test_decorator_func(x: int) -> str:
        return str(x)

    assert registry.has("_test_decorator_func")
    entry = registry.get("_test_decorator_func")
    assert entry.description == "test decorator tool"
    assert entry.sensitive is False
    assert len(registry.list_entries()) == initial_count + 1


def test_tool_decorator_sensitive():
    class SensParams(BaseModel):
        target: str = Field(description="target")

    @tool(model=SensParams, description="sensitive tool", sensitive=True,
          confirm_template="操作 {target}")
    async def _test_sensitive_func(target: str) -> str:
        return target

    registry = get_registry()
    entry = registry.get("_test_sensitive_func")
    assert entry.sensitive is True
    assert entry.confirm_template == "操作 {target}"


def test_tool_decorator_custom_name():
    class NameParams(BaseModel):
        v: str = Field(description="value")

    @tool(model=NameParams, description="custom name", name="my_custom_tool")
    async def _some_internal_func(v: str) -> str:
        return v

    registry = get_registry()
    assert registry.has("my_custom_tool")
    assert not registry.has("_some_internal_func")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/test_tools_refactor.py -v -k "decorator"`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tools.decorator'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/tools/decorator.py
"""@tool 装饰器 — 将函数注册为工具。"""

from typing import Callable

from pydantic import BaseModel

from .registry import ToolRegistry, ToolEntry

# 全局 registry 实例
_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """获取全局工具注册表"""
    return _registry


def tool(
    model: type[BaseModel],
    description: str,
    name: str | None = None,
    sensitive: bool = False,
    confirm_template: str | None = None,
) -> Callable:
    """工具注册装饰器"""
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__

        # 生成参数 schema
        model_schema = model.model_json_schema()
        if model_schema.get("type") == "object":
            parameters_schema = model_schema
        else:
            parameters_schema = {
                "type": "object",
                "properties": {"input": model_schema},
                "required": ["input"],
            }
        parameters_schema.pop("description", None)

        entry = ToolEntry(
            name=tool_name,
            func=func,
            model=model,
            description=description,
            parameters_schema=parameters_schema,
            sensitive=sensitive,
            confirm_template=confirm_template,
        )
        _registry.register(entry)
        return func

    return decorator
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/test_tools_refactor.py -v -k "decorator"`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tools/decorator.py tests/test_tools_refactor.py
git commit -m "refactor(tools): add @tool decorator with global registry"
```

---

### Task 4: discovery.py — 自动发现

**Files:**
- Create: `src/tools/discovery.py`
- Test: `tests/test_tools_refactor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tools_refactor.py （追加）
import tempfile
import os
from pathlib import Path
from src.tools.discovery import discover_tools


def test_discover_tools(tmp_path):
    """discover_tools 能导入模块并触发注册"""
    # 创建临时包
    pkg_dir = tmp_path / "fake_tools"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "sample.py").write_text(
        "LOADED = True\n"
    )

    # 将 tmp_path 加入 sys.path 以便 importlib 找到
    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        discover_tools("fake_tools", pkg_dir)
        import fake_tools.sample
        assert fake_tools.sample.LOADED is True
    finally:
        sys.path.remove(str(tmp_path))


def test_discover_tools_skips_init(tmp_path):
    """discover_tools 跳过 __init__.py"""
    pkg_dir = tmp_path / "skip_test"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("INIT_LOADED = True\n")
    (pkg_dir / "real.py").write_text("REAL_LOADED = True\n")

    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        # 不应该因为 __init__ 而报错
        discover_tools("skip_test", pkg_dir)
        import skip_test.real
        assert skip_test.real.REAL_LOADED is True
    finally:
        sys.path.remove(str(tmp_path))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/test_tools_refactor.py -v -k "discover"`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tools.discovery'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/tools/discovery.py
"""自动发现工具模块 — 扫描目录下的 .py 文件并导入。"""

import importlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def discover_tools(
    package: str,
    package_dir: Path,
    skip: set[str] | None = None,
) -> None:
    """扫描 package_dir 下的 .py 文件并导入，以触发 @tool 装饰器注册。

    Args:
        package: 包的完整名称（如 "src.tools.builtin"）
        package_dir: 包的目录路径
        skip: 要跳过的模块名集合，默认跳过 __init__
    """
    skip = skip or {"__init__"}
    for item in sorted(Path(package_dir).glob("*.py")):
        module_name = item.stem
        if module_name in skip:
            continue
        try:
            importlib.import_module(f".{module_name}", package=package)
            logger.debug(f"已发现工具模块: {module_name}")
        except Exception as e:
            logger.error(f"导入工具模块 {module_name} 失败: {e}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/test_tools_refactor.py -v -k "discover"`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tools/discovery.py tests/test_tools_refactor.py
git commit -m "refactor(tools): add discover_tools for auto module scanning"
```

---

### Task 5: executor.py — 纯执行器

**Files:**
- Create: `src/tools/executor.py`
- Test: `tests/test_tools_refactor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tools_refactor.py （追加）
import pytest


@pytest.mark.asyncio
async def test_executor_validates_and_runs():
    """ToolExecutor 校验参数并执行异步函数"""
    from src.tools.executor import ToolExecutor
    from src.tools.registry import ToolRegistry, ToolEntry

    class AddModel(BaseModel):
        a: int = Field(description="first number")
        b: int = Field(description="second number")

    async def add_func(a: int, b: int) -> str:
        return f"result:{a + b}"

    reg = ToolRegistry()
    reg.register(ToolEntry(
        name="add",
        func=add_func,
        model=AddModel,
        description="Add two numbers",
        parameters_schema=AddModel.model_json_schema(),
    ))

    executor = ToolExecutor(reg)
    result = await executor.execute("add", {"a": 3, "b": 4})
    assert result == "result:7"


@pytest.mark.asyncio
async def test_executor_validation_error():
    """ToolExecutor 在参数校验失败时抛出 ValueError"""
    from src.tools.executor import ToolExecutor
    from src.tools.registry import ToolRegistry, ToolEntry

    class StrictModel(BaseModel):
        count: int = Field(description="must be int")

    async def noop(count: int) -> str:
        return "ok"

    reg = ToolRegistry()
    reg.register(ToolEntry(
        name="strict",
        func=noop,
        model=StrictModel,
        description="strict tool",
        parameters_schema=StrictModel.model_json_schema(),
    ))

    executor = ToolExecutor(reg)
    with pytest.raises(ValueError, match="参数验证失败"):
        await executor.execute("strict", {"count": "not_a_number"})


@pytest.mark.asyncio
async def test_executor_unknown_tool():
    """ToolExecutor 对未注册的工具抛出 ValueError"""
    from src.tools.executor import ToolExecutor
    from src.tools.registry import ToolRegistry

    executor = ToolExecutor(ToolRegistry())
    with pytest.raises(ValueError, match="未注册的工具"):
        await executor.execute("nonexistent", {})


@pytest.mark.asyncio
async def test_executor_sync_function():
    """ToolExecutor 支持同步函数"""
    from src.tools.executor import ToolExecutor
    from src.tools.registry import ToolRegistry, ToolEntry

    class EchoModel(BaseModel):
        msg: str = Field(description="message")

    def sync_echo(msg: str) -> str:
        return f"echo:{msg}"

    reg = ToolRegistry()
    reg.register(ToolEntry(
        name="sync_echo",
        func=sync_echo,
        model=EchoModel,
        description="sync echo",
        parameters_schema=EchoModel.model_json_schema(),
    ))

    executor = ToolExecutor(reg)
    result = await executor.execute("sync_echo", {"msg": "hello"})
    assert result == "echo:hello"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/test_tools_refactor.py -v -k "executor"`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tools.executor'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/tools/executor.py
"""ToolExecutor — 纯工具执行器，只做参数校验和函数调用。"""

import asyncio
import inspect

from pydantic import ValidationError

from .registry import ToolRegistry


class ToolExecutor:
    """校验参数并调用工具函数。

    不包含路由、敏感确认、截断等逻辑，这些由中间件处理。
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def execute(self, tool_name: str, arguments: dict) -> str:
        """校验参数并执行工具函数。

        Raises:
            ValueError: 工具未注册或参数校验失败
        """
        entry = self.registry.get(tool_name)
        if not entry:
            raise ValueError(f"未注册的工具: {tool_name}")

        try:
            validated = entry.model(**arguments).model_dump()
        except ValidationError as e:
            messages = []
            for err in e.errors()[:3]:
                loc = ".".join(str(x) for x in err["loc"])
                messages.append(f"{loc}: {err['msg']}")
            raise ValueError(f"参数验证失败: {'; '.join(messages)}") from e

        return await self._run_func(entry.func, validated)

    async def _run_func(self, func, kwargs: dict) -> str:
        if asyncio.iscoroutinefunction(func):
            result = await func(**kwargs)
        else:
            result = await asyncio.to_thread(func, **kwargs)
        return str(result)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/test_tools_refactor.py -v -k "executor"`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tools/executor.py tests/test_tools_refactor.py
git commit -m "refactor(tools): add ToolExecutor with pure validation and execution"
```

---

### Task 6: middleware.py — 中间件链

**Files:**
- Create: `src/tools/middleware.py`
- Test: `tests/test_tools_refactor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tools_refactor.py （追加）
from src.tools.middleware import (
    build_pipeline,
    truncate_middleware,
    error_handler_middleware,
)


@pytest.mark.asyncio
async def test_truncate_middleware():
    """truncate 中间件截断过长结果"""
    async def fake_execute(name: str, args: dict) -> str:
        return "x" * 100

    pipeline = build_pipeline(fake_execute, [truncate_middleware(max_length=50)])
    result = await pipeline("test", {})
    assert len(result) < 100
    assert "截断" in result


@pytest.mark.asyncio
async def test_truncate_middleware_short_result():
    """truncate 中间件不截断短结果"""
    async def fake_execute(name: str, args: dict) -> str:
        return "short"

    pipeline = build_pipeline(fake_execute, [truncate_middleware(max_length=50)])
    result = await pipeline("test", {})
    assert result == "short"


@pytest.mark.asyncio
async def test_error_handler_middleware():
    """error_handler 中间件捕获异常"""
    async def failing_execute(name: str, args: dict) -> str:
        raise RuntimeError("boom")

    pipeline = build_pipeline(failing_execute, [error_handler_middleware()])
    result = await pipeline("test", {})
    assert "执行出错" in result
    assert "boom" in result


@pytest.mark.asyncio
async def test_middleware_chain_order():
    """中间件按正确顺序执行：error_handler 在外层捕获异常"""
    async def failing_execute(name: str, args: dict) -> str:
        raise RuntimeError("inner error")

    pipeline = build_pipeline(
        failing_execute,
        [error_handler_middleware(), truncate_middleware(max_length=50)],
    )
    # error_handler 在最外层，应该捕获异常并返回错误字符串
    result = await pipeline("test", {})
    assert "执行出错" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/test_tools_refactor.py -v -k "middleware"`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tools.middleware'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/tools/middleware.py
"""中间件链 — 工具执行的横切关注点。"""

from typing import Awaitable, Callable

from .registry import ToolRegistry

# 类型定义
NextFn = Callable[[str, dict], Awaitable[str]]
Middleware = Callable[[str, dict, NextFn], Awaitable[str]]


def build_pipeline(execute_fn: NextFn, middlewares: list[Middleware]) -> NextFn:
    """构建中间件链。

    middlewares 列表中第一个中间件最先执行（最外层）。
    """
    pipeline = execute_fn
    for mw in reversed(middlewares):
        prev = pipeline

        async def wrapped(name, args, _prev=prev, _mw=mw):
            return await _mw(name, args, _prev)

        pipeline = wrapped
    return pipeline


def sensitive_confirm_middleware(registry: ToolRegistry) -> Middleware:
    """敏感工具执行前需要用户确认"""

    async def middleware(name: str, args: dict, next_fn: NextFn) -> str:
        entry = registry.get(name)
        if entry and entry.sensitive:
            if entry.confirm_template:
                try:
                    msg = entry.confirm_template.format(**args)
                except KeyError:
                    msg = f"执行敏感操作: {name}"
            else:
                msg = f"执行敏感操作: {name}"

            from src.core.io import agent_output, agent_input

            await agent_output(f"\n⚠️  是否允许{msg}？\n")
            confirm = await agent_input("(y/n): ")
            if confirm.strip().lower() != "y":
                return "用户取消了操作"

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
    """捕获异常，返回错误字符串"""

    async def middleware(name: str, args: dict, next_fn: NextFn) -> str:
        try:
            return await next_fn(name, args)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            return f"工具 '{name}' 执行出错: {error_msg}"

    return middleware
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/test_tools_refactor.py -v -k "middleware"`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tools/middleware.py tests/test_tools_refactor.py
git commit -m "refactor(tools): add middleware chain with truncate and error handler"
```

---

### Task 7: router.py — ToolProvider + ToolRouter + LocalToolProvider

**Files:**
- Create: `src/tools/router.py`
- Test: `tests/test_tools_refactor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tools_refactor.py （追加）
from src.tools.router import ToolRouter, LocalToolProvider
from src.tools.executor import ToolExecutor as NewToolExecutor
from src.tools.middleware import build_pipeline, error_handler_middleware, truncate_middleware


@pytest.mark.asyncio
async def test_local_provider_can_handle():
    """LocalToolProvider 只处理注册表中的工具"""
    from src.tools.registry import ToolRegistry, ToolEntry

    class M(BaseModel):
        v: str = Field(description="v")

    async def fn(v: str) -> str:
        return v

    reg = ToolRegistry()
    reg.register(ToolEntry(
        name="local_tool", func=fn, model=M,
        description="test", parameters_schema=M.model_json_schema(),
    ))
    executor = NewToolExecutor(reg)
    provider = LocalToolProvider(reg, executor, [error_handler_middleware()])

    assert provider.can_handle("local_tool") is True
    assert provider.can_handle("unknown") is False


@pytest.mark.asyncio
async def test_local_provider_execute():
    """LocalToolProvider 通过中间件链执行工具"""
    from src.tools.registry import ToolRegistry, ToolEntry

    class M(BaseModel):
        v: str = Field(description="v")

    async def fn(v: str) -> str:
        return f"got:{v}"

    reg = ToolRegistry()
    reg.register(ToolEntry(
        name="echo", func=fn, model=M,
        description="echo", parameters_schema=M.model_json_schema(),
    ))
    executor = NewToolExecutor(reg)
    provider = LocalToolProvider(reg, executor, [error_handler_middleware()])

    result = await provider.execute("echo", {"v": "hello"})
    assert result == "got:hello"


@pytest.mark.asyncio
async def test_router_routes_to_provider():
    """ToolRouter 路由到正确的 provider"""
    from src.tools.registry import ToolRegistry, ToolEntry

    class M(BaseModel):
        v: str = Field(description="v")

    async def fn(v: str) -> str:
        return f"routed:{v}"

    reg = ToolRegistry()
    reg.register(ToolEntry(
        name="routed_tool", func=fn, model=M,
        description="test", parameters_schema=M.model_json_schema(),
    ))
    executor = NewToolExecutor(reg)
    provider = LocalToolProvider(reg, executor, [error_handler_middleware()])

    router = ToolRouter()
    router.add_provider(provider)

    result = await router.route("routed_tool", {"v": "test"})
    assert result == "routed:test"


@pytest.mark.asyncio
async def test_router_unknown_tool():
    """ToolRouter 对未知工具返回错误"""
    router = ToolRouter()
    result = await router.route("nonexistent", {})
    assert "未找到" in result


def test_router_get_all_schemas():
    """ToolRouter 聚合所有 provider 的 schema"""
    from src.tools.registry import ToolRegistry, ToolEntry

    class M(BaseModel):
        v: str = Field(description="v")

    async def fn(v: str) -> str:
        return v

    reg = ToolRegistry()
    reg.register(ToolEntry(
        name="s1", func=fn, model=M,
        description="tool 1", parameters_schema=M.model_json_schema(),
    ))
    executor = NewToolExecutor(reg)
    provider = LocalToolProvider(reg, executor, [])

    router = ToolRouter()
    router.add_provider(provider)

    schemas = router.get_all_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "s1"


def test_router_is_sensitive():
    """ToolRouter.is_sensitive 正确委托到 LocalToolProvider"""
    from src.tools.registry import ToolRegistry, ToolEntry

    class M(BaseModel):
        v: str = Field(description="v")

    async def fn(v: str) -> str:
        return v

    reg = ToolRegistry()
    reg.register(ToolEntry(
        name="safe", func=fn, model=M,
        description="safe", parameters_schema=M.model_json_schema(),
        sensitive=False,
    ))
    reg.register(ToolEntry(
        name="danger", func=fn, model=M,
        description="danger", parameters_schema=M.model_json_schema(),
        sensitive=True,
    ))
    executor = NewToolExecutor(reg)
    provider = LocalToolProvider(reg, executor, [])

    router = ToolRouter()
    router.add_provider(provider)

    assert router.is_sensitive("safe") is False
    assert router.is_sensitive("danger") is True
    assert router.is_sensitive("nonexistent") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/test_tools_refactor.py -v -k "router or provider"`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tools.router'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/tools/router.py
"""ToolProvider 协议 + ToolRouter 统一路由 + LocalToolProvider。"""

import logging
from typing import Protocol, runtime_checkable

from .executor import ToolExecutor
from .middleware import Middleware, NextFn, build_pipeline
from .registry import ToolRegistry
from .schemas import ToolDict

logger = logging.getLogger(__name__)


@runtime_checkable
class ToolProvider(Protocol):
    """所有工具来源的统一接口"""

    def can_handle(self, tool_name: str) -> bool: ...
    async def execute(self, tool_name: str, arguments: dict) -> str: ...
    def get_schemas(self) -> list[ToolDict]: ...


class ToolRouter:
    """按注册顺序查询 provider，找到第一个能处理的执行。"""

    def __init__(self):
        self._providers: list[ToolProvider] = []

    def add_provider(self, provider: ToolProvider) -> None:
        self._providers.append(provider)

    async def route(self, tool_name: str, arguments: dict) -> str:
        for provider in self._providers:
            if provider.can_handle(tool_name):
                return await provider.execute(tool_name, arguments)
        return f"错误：未找到工具 '{tool_name}'"

    def get_all_schemas(self) -> list[ToolDict]:
        schemas: list[ToolDict] = []
        for provider in self._providers:
            schemas.extend(provider.get_schemas())
        return schemas

    def is_sensitive(self, tool_name: str) -> bool:
        for provider in self._providers:
            if provider.can_handle(tool_name):
                if hasattr(provider, "is_sensitive"):
                    return provider.is_sensitive(tool_name)
                return False
        return False


class LocalToolProvider:
    """本地工具的 Provider 实现，桥接 ToolExecutor + 中间件。"""

    def __init__(
        self,
        registry: ToolRegistry,
        executor: ToolExecutor,
        middlewares: list[Middleware],
    ):
        self.registry = registry
        self._pipeline: NextFn = build_pipeline(executor.execute, middlewares)

    def can_handle(self, tool_name: str) -> bool:
        return self.registry.has(tool_name)

    async def execute(self, tool_name: str, arguments: dict) -> str:
        return await self._pipeline(tool_name, arguments)

    def get_schemas(self) -> list[ToolDict]:
        return self.registry.get_schemas()

    def is_sensitive(self, tool_name: str) -> bool:
        entry = self.registry.get(tool_name)
        return entry.sensitive if entry else False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/test_tools_refactor.py -v -k "router or provider"`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tools/router.py tests/test_tools_refactor.py
git commit -m "refactor(tools): add ToolProvider protocol, ToolRouter, and LocalToolProvider"
```

---

### Task 8: builtin/ — 迁移 calculator 和 file 工具

**Files:**
- Create: `src/tools/builtin/__init__.py`
- Create: `src/tools/builtin/calculator.py` (从 `src/tools/calculator.py` 迁移)
- Create: `src/tools/builtin/file.py` (从 `src/tools/file.py` 迁移)

- [ ] **Step 1: Create builtin package**

```python
# src/tools/builtin/__init__.py
```

（空文件）

- [ ] **Step 2: 迁移 calculator.py**

将 `src/tools/calculator.py` 复制到 `src/tools/builtin/calculator.py`，修改导入：

```python
# src/tools/builtin/calculator.py
import ast
import operator
import asyncio
from pydantic import BaseModel, Field
from src.tools.decorator import tool

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    """递归求值 AST 节点，只允许数字和基本算术运算"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"不支持的运算符: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"不支持的运算符: {op_type.__name__}")
        return SAFE_OPERATORS[op_type](_safe_eval(node.operand))
    else:
        raise ValueError(f"不允许的表达式类型: {type(node).__name__}")


def safe_calc(expression: str):
    """安全计算数学表达式"""
    tree = ast.parse(expression, mode="eval")
    return _safe_eval(tree)


class AsyncCalculator(BaseModel):
    """计算数学表达式，例如 '2 + 3 * 4'"""
    expression: str = Field(description="要计算的数学表达式")


@tool(model=AsyncCalculator, description="安全计算数学表达式")
async def calculator(expression: str) -> str:
    await asyncio.sleep(0.1)
    try:
        result = safe_calc(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"
```

- [ ] **Step 3: 迁移 file.py**

将 `src/tools/file.py` 复制到 `src/tools/builtin/file.py`，只修改导入行：

```python
# src/tools/builtin/file.py 第 4 行
# 旧：from . import tool
# 新：
from src.tools.decorator import tool
```

其余代码完全不变。

- [ ] **Step 4: 验证 builtin 工具能正确注册**

Run: `cd /Users/dingdalong/github/agent && python -c "
from src.tools.decorator import get_registry
from src.tools.discovery import discover_tools
from pathlib import Path
discover_tools('src.tools.builtin', Path('src/tools/builtin'))
reg = get_registry()
names = [e.name for e in reg.list_entries()]
print('Registered tools:', names)
assert 'calculator' in names, 'calculator not found'
assert 'write_file' in names, 'write_file not found'
print('OK')
"`
Expected: 打印已注册工具列表并输出 OK

- [ ] **Step 5: Commit**

```bash
git add src/tools/builtin/
git commit -m "refactor(tools): migrate calculator and file tools to builtin/"
```

---

### Task 9: 重写 tool_call.py — 使用 ToolRouter

**Files:**
- Modify: `src/tools/tool_call.py`

- [ ] **Step 1: 重写 tool_call.py**

```python
# src/tools/tool_call.py
"""批量异步执行 LLM 返回的 tool_calls。"""

import asyncio
import json
from typing import Any, Dict, List

from .router import ToolRouter


async def execute_tool_calls(
    content: str,
    tool_calls: Dict[int, Dict[str, str]],
    router: ToolRouter,
) -> List[Dict[str, Any]]:
    """并行执行工具调用，通过 router 分发。"""
    if not tool_calls:
        return []

    new_messages: list[dict] = []

    # 构造 assistant 消息
    assistant_msg = {
        "role": "assistant",
        "content": content if content else None,
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                },
            }
            for tc in tool_calls.values()
        ],
    }
    new_messages.append(assistant_msg)

    # 并行执行所有工具调用
    tool_tasks: list[tuple[int, asyncio.Task]] = []
    results: list[tuple[int, str]] = []

    for idx, tc in tool_calls.items():
        try:
            args = json.loads(tc["arguments"])
        except json.JSONDecodeError as e:
            results.append((idx, f"参数 JSON 解析失败: {e}"))
            continue
        task = asyncio.create_task(router.route(tc["name"], args))
        tool_tasks.append((idx, task))

    for idx, task in tool_tasks:
        try:
            result = await task
            results.append((idx, result))
        except Exception as e:
            results.append((idx, f"工具执行异常: {e}"))

    # 按原始顺序构造 tool 消息
    for idx, result in sorted(results, key=lambda x: x[0]):
        new_messages.append({
            "role": "tool",
            "tool_call_id": tool_calls[idx]["id"],
            "content": str(result),
        })

    return new_messages
```

- [ ] **Step 2: Commit**

```bash
git add src/tools/tool_call.py
git commit -m "refactor(tools): rewrite tool_call.py to use ToolRouter"
```

---

### Task 10: 重写 __init__.py — 纯导出

**Files:**
- Modify: `src/tools/__init__.py`

- [ ] **Step 1: 重写 __init__.py**

```python
# src/tools/__init__.py
"""Tools 模块 — 分层架构的工具系统。

对外导出所有公共接口，不包含业务逻辑。
"""

from .schemas import ToolDict
from .registry import ToolEntry, ToolRegistry
from .decorator import tool, get_registry
from .discovery import discover_tools
from .executor import ToolExecutor
from .middleware import (
    Middleware,
    NextFn,
    build_pipeline,
    error_handler_middleware,
    sensitive_confirm_middleware,
    truncate_middleware,
)
from .router import LocalToolProvider, ToolProvider, ToolRouter
from .tool_call import execute_tool_calls

__all__ = [
    "ToolDict",
    "ToolEntry",
    "ToolRegistry",
    "tool",
    "get_registry",
    "discover_tools",
    "ToolExecutor",
    "Middleware",
    "NextFn",
    "build_pipeline",
    "error_handler_middleware",
    "sensitive_confirm_middleware",
    "truncate_middleware",
    "LocalToolProvider",
    "ToolProvider",
    "ToolRouter",
    "execute_tool_calls",
]
```

- [ ] **Step 2: 验证导入正常**

Run: `cd /Users/dingdalong/github/agent && python -c "
from src.tools import (
    ToolDict, ToolEntry, ToolRegistry,
    tool, get_registry, discover_tools,
    ToolExecutor, ToolRouter, LocalToolProvider,
    execute_tool_calls,
)
print('All imports OK')
"`
Expected: `All imports OK`

- [ ] **Step 3: Commit**

```bash
git add src/tools/__init__.py
git commit -m "refactor(tools): rewrite __init__.py as pure re-exports"
```

---

### Task 11: MCP 和 Skill Provider

**Files:**
- Create: `src/mcp/provider.py`
- Create: `src/skills/provider.py`

- [ ] **Step 1: Create MCPToolProvider**

```python
# src/mcp/provider.py
"""MCPToolProvider — 将 MCPManager 适配为 ToolProvider。"""

from src.tools.schemas import ToolDict


class MCPToolProvider:
    """MCP 工具的 Provider 实现"""

    def __init__(self, mcp_manager):
        self._manager = mcp_manager

    def can_handle(self, tool_name: str) -> bool:
        return tool_name.startswith("mcp_")

    async def execute(self, tool_name: str, arguments: dict) -> str:
        return await self._manager.call_tool(tool_name, arguments)

    def get_schemas(self) -> list[ToolDict]:
        return self._manager.get_tools_schemas()
```

- [ ] **Step 2: Create SkillToolProvider**

```python
# src/skills/provider.py
"""SkillToolProvider — 将 SkillManager 适配为 ToolProvider。"""

from src.tools.schemas import ToolDict


class SkillToolProvider:
    """Skill 工具的 Provider 实现"""

    def __init__(self, skill_manager):
        self._manager = skill_manager

    def can_handle(self, tool_name: str) -> bool:
        return tool_name == "activate_skill"

    async def execute(self, tool_name: str, arguments: dict) -> str:
        result = self._manager.activate(arguments.get("name", ""))
        return result if result else "未找到指定的 Skill"

    def get_schemas(self) -> list[ToolDict]:
        schema = self._manager.build_activate_tool_schema()
        return [schema] if schema else []
```

- [ ] **Step 3: Commit**

```bash
git add src/mcp/provider.py src/skills/provider.py
git commit -m "refactor(tools): add MCPToolProvider and SkillToolProvider"
```

---

### Task 12: 更新 main.py — 新启动流程

**Files:**
- Modify: `main.py`

- [ ] **Step 1: 重写 main.py 的导入和 main() 函数**

替换 `main.py` 中的工具相关代码。

旧导入（删除）:
```python
from src.tools import tools, tool_executor
```

新导入:
```python
from pathlib import Path
from src.tools import (
    get_registry, discover_tools,
    ToolExecutor, ToolRouter, LocalToolProvider,
    sensitive_confirm_middleware, truncate_middleware, error_handler_middleware,
)
from src.mcp.provider import MCPToolProvider
from src.skills.provider import SkillToolProvider
```

替换 `handle_input` 函数签名和内部的 `tool_executor` 引用 → `router`:

```python
async def handle_input(user_input: str, router: ToolRouter, skill_manager=None):
    """统一入口：护栏 → Skill 斜杠命令 → Flow 路由 → 执行"""
    all_tools = router.get_all_schemas()

    # 1. 护栏检查
    passed, reason = input_guard.check(user_input)
    if not passed:
        await agent_output(f"\n[安全拦截] {reason}\n")
        return

    # 2. Skill 斜杠命令检测
    if skill_manager:
        skill_name = skill_manager.is_slash_command(user_input)
        if skill_name:
            skill_content = skill_manager.activate(skill_name)
            if skill_content:
                remaining = user_input[len(f"/{skill_name}"):].strip()
                actual_input = remaining or f"已激活 {skill_name} skill，请按指令执行。"
                multi_agent_flow = MultiAgentFlow(
                    registry=agent_registry,
                    memory=buffer,
                    store=store,
                    all_tools=all_tools,
                    tool_executor=router,
                )
                multi_agent_flow.model.data["user_input"] = actual_input
                multi_agent_flow.model.data["skill_content"] = skill_content
                runner = FSMRunner(multi_agent_flow)
                await runner.run()
                return

    # 3. 关键词触发的特殊 Flow
    flow = detect_flow(user_input, tool_executor=router)
    if flow:
        runner = FSMRunner(flow)
        await runner.run()
        return

    # 4. 复杂请求 → PlanningFlow
    if await is_complex_request(user_input):
        planning_flow = PlanningFlow(
            available_tools=all_tools,
            tool_executor=router,
        )
        planning_flow.model.data["original_request"] = user_input
        runner = FSMRunner(planning_flow)
        result = await runner.run()
        if result is not None:
            return

    # 5. 普通对话 → MultiAgentFlow
    multi_agent_flow = MultiAgentFlow(
        registry=agent_registry,
        memory=buffer,
        store=store,
        all_tools=all_tools,
        tool_executor=router,
    )
    multi_agent_flow.model.data["user_input"] = user_input
    runner = FSMRunner(multi_agent_flow)
    await runner.run()
```

替换 `main()` 函数:

```python
async def main():
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

    # 4. 初始化 MCP
    mcp_manager = MCPManager()
    await mcp_manager.connect_all(load_mcp_config(MCP_CONFIG_PATH))
    mcp_schemas = mcp_manager.get_tools_schemas()
    if mcp_schemas:
        router.add_provider(MCPToolProvider(mcp_manager))

    # 5. 初始化 Skills
    skill_manager = SkillManager(skill_dirs=SKILLS_DIRS)
    await skill_manager.discover()
    skill_count = len(skill_manager._skills)
    if skill_count:
        router.add_provider(SkillToolProvider(skill_manager))

    print("Agent 已启动，输入 'exit' 退出。")
    if mcp_schemas:
        print(f"已加载 {len(mcp_schemas)} 个 MCP 工具")
    if skill_count:
        print(f"已发现 {skill_count} 个 Skill")

    try:
        while True:
            user_input = await agent_input("\n你: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            await handle_input(user_input, router, skill_manager)
            await agent_output("\n")
    finally:
        await mcp_manager.disconnect_all()
```

- [ ] **Step 2: Commit**

```bash
git add main.py
git commit -m "refactor(tools): update main.py to use ToolRouter startup flow"
```

---

### Task 13: 更新 specialist_runner.py

**Files:**
- Modify: `src/agents/specialist_runner.py`

- [ ] **Step 1: 更新导入和类型注解**

替换:
```python
from src.tools.tool_call import execute_tool_calls
from src.tools.tool_executor import ToolExecutor
```

为:
```python
from src.tools.tool_call import execute_tool_calls
from src.tools.router import ToolRouter
```

替换 `run_specialist` 函数签名中的 `tool_executor: ToolExecutor` → `tool_executor: ToolRouter`（参数名保持 `tool_executor` 以减少调用方改动）。

函数体内的 `execute_tool_calls(content, tool_calls, tool_executor)` 调用不需要改——因为 `tool_call.py` 的参数类型已改为 `ToolRouter`，而变量名刚好也是 `tool_executor`。

- [ ] **Step 2: Commit**

```bash
git add src/agents/specialist_runner.py
git commit -m "refactor(tools): update specialist_runner to use ToolRouter"
```

---

### Task 14: 更新 orchestrator.py

**Files:**
- Modify: `src/agents/orchestrator.py`

- [ ] **Step 1: 更新导入**

替换:
```python
from src.tools import ToolDict
from src.tools.tool_executor import ToolExecutor
```

为:
```python
from src.tools import ToolDict
from src.tools.router import ToolRouter
```

在类中将所有 `tool_executor: ToolExecutor` 的类型注解替换为 `tool_executor: ToolRouter`。注意：只改类型注解，不改变量名。搜索文件中所有 `ToolExecutor` 引用并替换。

- [ ] **Step 2: Commit**

```bash
git add src/agents/orchestrator.py
git commit -m "refactor(tools): update orchestrator to use ToolRouter"
```

---

### Task 15: 更新 flows/planning.py

**Files:**
- Modify: `src/flows/planning.py`

- [ ] **Step 1: 更新导入**

替换:
```python
from src.tools import ToolDict
from src.tools.tool_executor import ToolExecutor
```

为:
```python
from src.tools import ToolDict
from src.tools.router import ToolRouter
```

在 `PlanningModel.__init__` 中将 `tool_executor: ToolExecutor` → `tool_executor: ToolRouter`。

搜索文件中所有 `ToolExecutor` 引用（类型注解）并替换为 `ToolRouter`。

注意：`planning.py` 中 `executing_sensitive` 状态使用了 `tool_executor._confirm_sensitive()` 和 `tool_executor.execute(..., skip_confirm=True)`。这些现在需要改为通过 `router.route()` 调用。敏感确认已由中间件处理，所以可以直接调用 `router.route()`。找到所有 `_confirm_sensitive` 和 `skip_confirm` 的使用并替换为直接调用 `router.route(step.tool_name, resolved_args)`。

- [ ] **Step 2: Commit**

```bash
git add src/flows/planning.py
git commit -m "refactor(tools): update planning flow to use ToolRouter"
```

---

### Task 16: 更新 plan/executor.py

**Files:**
- Modify: `src/plan/executor.py`

- [ ] **Step 1: 更新导入和类型注解**

替换:
```python
from src.tools.tool_executor import ToolExecutor
```

为:
```python
from src.tools.router import ToolRouter
```

替换所有函数签名中的 `tool_executor: ToolExecutor` → `tool_executor: ToolRouter`，包括:
- `execute_step(step, context, tool_executor, timeout)`
- `_is_sensitive_tool_step(step, tool_executor)`
- `execute_plan(plan, tool_executor, ...)`

函数体内的调用保持不变：
- `tool_executor.is_sensitive(...)` → ToolRouter 已有此方法
- `tool_executor.execute(...)` → 改为 `tool_executor.route(...)`

具体替换:
- 第 179 行 `tool_executor.execute(step.tool_name, resolved_args)` → `tool_executor.route(step.tool_name, resolved_args)`

- [ ] **Step 2: Commit**

```bash
git add src/plan/executor.py
git commit -m "refactor(tools): update plan executor to use ToolRouter"
```

---

### Task 17: 删除旧文件

**Files:**
- Delete: `src/tools/calculator.py`
- Delete: `src/tools/file.py`
- Delete: `src/tools/weather.py`
- Delete: `src/tools/calendar.py`
- Delete: `src/tools/email.py`
- Delete: `src/tools/tool_executor.py`

- [ ] **Step 1: 删除旧文件**

```bash
cd /Users/dingdalong/github/agent
rm src/tools/calculator.py src/tools/file.py src/tools/weather.py src/tools/calendar.py src/tools/email.py src/tools/tool_executor.py
```

- [ ] **Step 2: 验证没有残留引用**

```bash
cd /Users/dingdalong/github/agent
grep -r "from src.tools.tool_executor" src/ main.py --include="*.py"
grep -r "from src.tools.calculator" src/ main.py --include="*.py"
grep -r "from src.tools.weather" src/ main.py --include="*.py"
grep -r "from src.tools.calendar" src/ main.py --include="*.py"
grep -r "from src.tools.email" src/ main.py --include="*.py"
grep -r "from src.tools.file import" src/ main.py --include="*.py"
```

Expected: 所有 grep 无输出

- [ ] **Step 3: Commit**

```bash
git add -A src/tools/calculator.py src/tools/file.py src/tools/weather.py src/tools/calendar.py src/tools/email.py src/tools/tool_executor.py
git commit -m "refactor(tools): remove old tool files replaced by new architecture"
```

---

### Task 18: 更新测试文件

**Files:**
- Modify: `tests/test_tools.py`
- Modify: `tests/test_skills/test_tool_executor_skill.py`
- Modify: `tests/test_skills/test_integration.py`

- [ ] **Step 1: 更新 test_tools.py**

重写为测试 builtin calculator 通过新架构：

```python
# tests/test_tools.py
"""测试内置 calculator 工具（通过新架构）。"""
import pytest
from src.tools.builtin.calculator import calculator


@pytest.mark.asyncio
async def test_calculator():
    result = await calculator(expression="2 + 2")
    assert "计算结果: 4" in result


@pytest.mark.asyncio
async def test_calculator_error():
    result = await calculator(expression="abc")
    assert "计算错误" in result


@pytest.mark.asyncio
async def test_calculator_complex():
    result = await calculator(expression="3 * 4 + 5")
    assert "计算结果: 17" in result
```

- [ ] **Step 2: 更新 test_tool_executor_skill.py**

`ToolExecutor` 不再直接路由 skill。这些测试需要改为测试 `SkillToolProvider`：

```python
# tests/test_skills/test_tool_executor_skill.py
"""测试 SkillToolProvider 路由。"""
import pytest
from unittest.mock import MagicMock
from src.skills.provider import SkillToolProvider


@pytest.mark.asyncio
async def test_provider_routes_activate_skill():
    mock_manager = MagicMock()
    mock_manager.activate.return_value = "<skill_content>skill_content</skill_content>"
    provider = SkillToolProvider(mock_manager)

    assert provider.can_handle("activate_skill")
    result = await provider.execute("activate_skill", {"name": "test"})
    assert "skill_content" in result
    mock_manager.activate.assert_called_once_with("test")


@pytest.mark.asyncio
async def test_provider_skill_not_found():
    mock_manager = MagicMock()
    mock_manager.activate.return_value = None
    provider = SkillToolProvider(mock_manager)

    result = await provider.execute("activate_skill", {"name": "missing"})
    assert "未找到" in result


def test_provider_does_not_handle_other_tools():
    mock_manager = MagicMock()
    provider = SkillToolProvider(mock_manager)
    assert not provider.can_handle("calculator")
    assert not provider.can_handle("mcp_something")
```

- [ ] **Step 3: 更新 test_integration.py**

更新导入：将 `from src.tools.tool_executor import ToolExecutor` 替换为 `from src.tools.router import ToolRouter, LocalToolProvider` 和 `from src.tools.executor import ToolExecutor`，并将 `test_tool_executor_skill_routing` 测试改为通过 `ToolRouter` 路由：

```python
# tests/test_skills/test_integration.py 中 test_tool_executor_skill_routing 替换为：
@pytest.mark.asyncio
async def test_router_skill_routing(tmp_path):
    """ToolRouter + SkillToolProvider 端到端路由"""
    _make_skill(tmp_path, "test-skill", "## Body\nContent here")
    mgr = SkillManager(skill_dirs=[str(tmp_path)])
    await mgr.discover()

    from src.tools.router import ToolRouter
    from src.skills.provider import SkillToolProvider

    router = ToolRouter()
    router.add_provider(SkillToolProvider(mgr))

    result = await router.route("activate_skill", {"name": "test-skill"})
    assert "skill_content" in result
    assert "## Body" in result
```

其他测试（`test_discover_activate_end_to_end`, `test_slash_command_full_flow`, `test_activate_tool_schema_used_by_executor`）不涉及 ToolExecutor，只需确认导入无误即可。

- [ ] **Step 4: Run all tests**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/ -v --tb=short`
Expected: 全部通过

- [ ] **Step 5: Commit**

```bash
git add tests/test_tools.py tests/test_skills/test_tool_executor_skill.py tests/test_skills/test_integration.py
git commit -m "refactor(tools): update all tests for new tools architecture"
```

---

### Task 19: 全量验证

- [ ] **Step 1: 运行完整测试套件**

Run: `cd /Users/dingdalong/github/agent && python -m pytest tests/ -v --tb=short`
Expected: 全部通过

- [ ] **Step 2: 验证导入链完整**

Run: `cd /Users/dingdalong/github/agent && python -c "
from pathlib import Path
from src.tools import (
    get_registry, discover_tools,
    ToolExecutor, ToolRouter, LocalToolProvider,
    sensitive_confirm_middleware, truncate_middleware, error_handler_middleware,
    execute_tool_calls,
)
from src.mcp.provider import MCPToolProvider
from src.skills.provider import SkillToolProvider

# 模拟启动流程
discover_tools('src.tools.builtin', Path('src/tools/builtin'))
reg = get_registry()
executor = ToolExecutor(reg)
local = LocalToolProvider(reg, executor, [
    error_handler_middleware(),
    truncate_middleware(2000),
])
router = ToolRouter()
router.add_provider(local)

schemas = router.get_all_schemas()
print(f'Registered {len(schemas)} tools')
tool_names = [s['function']['name'] for s in schemas]
print(f'Tools: {tool_names}')
assert 'calculator' in tool_names
assert 'write_file' in tool_names
print('Full startup flow OK')
"`
Expected: 输出工具列表和 `Full startup flow OK`

- [ ] **Step 3: 验证旧文件已清理**

```bash
ls src/tools/calculator.py src/tools/weather.py src/tools/calendar.py src/tools/email.py src/tools/tool_executor.py 2>&1
```

Expected: 所有文件 `No such file or directory`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "refactor(tools): complete tools module restructure to layered architecture"
```
