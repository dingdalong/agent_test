# UserInteractionService + UserInputToolProvider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let agents ask users questions mid-tool-loop, while unifying all user interaction (including sensitive tool confirms) behind a shared parallel-safe service.

**Architecture:** New `UserInteractionService` (Layer 0) wraps `UserInterface` with an `asyncio.Lock` for parallel safety. `UserInputToolProvider` (Layer 1) exposes `ask_user` as a tool. Existing `sensitive_confirm_middleware` is refactored to use the same service. `AgentRunner._build_tools` gains a `SYSTEM_TOOLS` set so `ask_user` is always available.

**Tech Stack:** Python 3.13, asyncio, pytest, existing `UserInterface` protocol

**Spec:** `docs/superpowers/specs/2026-04-02-user-interaction-service-design.md`

---

### Task 1: UserInteractionService

**Files:**
- Create: `src/utils/interaction.py`
- Test: `tests/utils/test_interaction.py`
- Create: `tests/utils/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/utils/__init__.py
(empty)
```

```python
# tests/utils/test_interaction.py
"""UserInteractionService 测试。"""
import asyncio
import pytest
from unittest.mock import AsyncMock

from src.utils.interaction import UserInteractionService


@pytest.fixture
def mock_ui():
    ui = AsyncMock()
    ui.prompt = AsyncMock(return_value="用户的回答")
    ui.display = AsyncMock()
    ui.confirm = AsyncMock(return_value=True)
    return ui


@pytest.fixture
def service(mock_ui):
    return UserInteractionService(mock_ui)


class TestAsk:
    @pytest.mark.asyncio
    async def test_ask_returns_user_input(self, service, mock_ui):
        result = await service.ask("你想要什么格式？")
        assert result == "用户的回答"
        mock_ui.display.assert_called_once()
        mock_ui.prompt.assert_called_once_with("你的回答: ")

    @pytest.mark.asyncio
    async def test_ask_with_source_shows_label(self, service, mock_ui):
        await service.ask("问题", source="weather_agent")
        display_arg = mock_ui.display.call_args[0][0]
        assert "[weather_agent]" in display_arg

    @pytest.mark.asyncio
    async def test_ask_without_source_no_label(self, service, mock_ui):
        await service.ask("问题")
        display_arg = mock_ui.display.call_args[0][0]
        assert "[" not in display_arg


class TestConfirm:
    @pytest.mark.asyncio
    async def test_confirm_returns_true(self, service, mock_ui):
        mock_ui.confirm = AsyncMock(return_value=True)
        result = await service.confirm("执行敏感操作: delete_file")
        assert result is True

    @pytest.mark.asyncio
    async def test_confirm_returns_false(self, service, mock_ui):
        mock_ui.confirm = AsyncMock(return_value=False)
        result = await service.confirm("执行敏感操作: delete_file")
        assert result is False

    @pytest.mark.asyncio
    async def test_confirm_displays_message(self, service, mock_ui):
        await service.confirm("删除文件")
        display_arg = mock_ui.display.call_args[0][0]
        assert "删除文件" in display_arg


class TestLock:
    @pytest.mark.asyncio
    async def test_concurrent_ask_serialized(self, mock_ui):
        """并行调用 ask 应被 Lock 串行化。"""
        call_order = []

        async def slow_prompt(msg):
            call_order.append("start")
            await asyncio.sleep(0.05)
            call_order.append("end")
            return "answer"

        mock_ui.prompt = slow_prompt
        service = UserInteractionService(mock_ui)

        await asyncio.gather(
            service.ask("问题1"),
            service.ask("问题2"),
        )

        # Lock 保证串行：start-end-start-end，不会交错
        assert call_order == ["start", "end", "start", "end"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/utils/test_interaction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utils.interaction'`

- [ ] **Step 3: Write the implementation**

```python
# src/utils/interaction.py
"""UserInteractionService — 统一的用户交互入口。"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.interfaces.base import UserInterface


class UserInteractionService:
    """统一的用户交互服务。

    所有需要向用户提问的组件（工具确认、agent 提问等）
    都通过此服务交互，保证并行安全和展示一致性。
    """

    def __init__(self, ui: UserInterface) -> None:
        self._ui = ui
        self._lock = asyncio.Lock()

    async def ask(self, question: str, source: str = "") -> str:
        """向用户提出开放式问题，返回自由文本回答。

        Args:
            question: 要向用户提出的问题
            source: 提问者标识（如 agent 名称），用于展示
        """
        async with self._lock:
            label = f"[{source}] " if source else ""
            await self._ui.display(f"\n🤖 {label}提问: {question}")
            return await self._ui.prompt("你的回答: ")

    async def confirm(self, message: str) -> bool:
        """向用户请求是/否确认。

        Args:
            message: 确认提示信息
        """
        async with self._lock:
            await self._ui.display(f"\n⚠️  是否允许{message}？")
            return await self._ui.confirm("")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/utils/test_interaction.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/interaction.py tests/utils/__init__.py tests/utils/test_interaction.py
git commit -m "feat: add UserInteractionService with Lock-based parallel safety"
```

---

### Task 2: UserInputToolProvider

**Files:**
- Create: `src/tools/user_input.py`
- Test: `tests/tools/test_user_input.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/tools/test_user_input.py
"""UserInputToolProvider 测试。"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass

from src.tools.user_input import UserInputToolProvider


@pytest.fixture
def mock_interaction():
    interaction = AsyncMock()
    interaction.ask = AsyncMock(return_value="用户的回答")
    return interaction


@pytest.fixture
def provider(mock_interaction):
    return UserInputToolProvider(mock_interaction)


class TestCanHandle:
    def test_handles_ask_user(self, provider):
        assert provider.can_handle("ask_user") is True

    def test_rejects_other_tools(self, provider):
        assert provider.can_handle("get_weather") is False
        assert provider.can_handle("ask_user_v2") is False


class TestGetSchemas:
    def test_returns_ask_user_schema(self, provider):
        schemas = provider.get_schemas()
        assert len(schemas) == 1
        func = schemas[0]["function"]
        assert func["name"] == "ask_user"
        assert "question" in func["parameters"]["properties"]
        assert "question" in func["parameters"]["required"]


class TestExecute:
    @pytest.mark.asyncio
    async def test_returns_user_answer(self, provider, mock_interaction):
        result = await provider.execute("ask_user", {"question": "你想要什么？"})
        assert result == "用户的回答"
        mock_interaction.ask.assert_called_once_with("你想要什么？", source="")

    @pytest.mark.asyncio
    async def test_extracts_source_from_context(self, provider, mock_interaction):
        @dataclass
        class FakeContext:
            current_agent: str = "weather_agent"

        await provider.execute("ask_user", {"question": "城市？"}, context=FakeContext())
        mock_interaction.ask.assert_called_once_with("城市？", source="weather_agent")

    @pytest.mark.asyncio
    async def test_empty_question_returns_error(self, provider, mock_interaction):
        result = await provider.execute("ask_user", {"question": ""})
        assert "错误" in result
        mock_interaction.ask.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_question_returns_error(self, provider, mock_interaction):
        result = await provider.execute("ask_user", {})
        assert "错误" in result
        mock_interaction.ask.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_context_uses_empty_source(self, provider, mock_interaction):
        await provider.execute("ask_user", {"question": "问题"}, context=None)
        mock_interaction.ask.assert_called_once_with("问题", source="")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tools/test_user_input.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tools.user_input'`

- [ ] **Step 3: Write the implementation**

```python
# src/tools/user_input.py
"""UserInputToolProvider — 让 agent 能主动向用户提问。"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from src.tools.schemas import ToolDict

if TYPE_CHECKING:
    from src.utils.interaction import UserInteractionService


class UserInputToolProvider:
    """让 agent 能主动向用户提问的 ToolProvider。

    实现 ToolProvider 协议，注册到 ToolRouter 后，
    agent 可通过调用 ask_user 工具向用户提出问题。
    """

    def __init__(self, interaction: UserInteractionService) -> None:
        self._interaction = interaction

    def can_handle(self, tool_name: str) -> bool:
        return tool_name == "ask_user"

    def get_schemas(self) -> list[ToolDict]:
        return [{
            "type": "function",
            "function": {
                "name": "ask_user",
                "description": (
                    "当你需要用户提供额外信息、做出选择或确认时调用此工具。"
                    "请确保问题清晰具体，避免模糊的提问。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "要向用户提出的问题",
                        },
                    },
                    "required": ["question"],
                },
            },
        }]

    async def execute(
        self, tool_name: str, arguments: dict[str, Any], context: Any = None,
    ) -> str:
        question = arguments.get("question", "")
        if not question:
            return "错误：question 参数不能为空"
        source = ""
        if context is not None:
            source = getattr(context, "current_agent", "")
        return await self._interaction.ask(question, source=source)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tools/test_user_input.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/user_input.py tests/tools/test_user_input.py
git commit -m "feat: add UserInputToolProvider with ask_user tool"
```

---

### Task 3: Refactor sensitive_confirm_middleware to use UserInteractionService

**Files:**
- Modify: `src/tools/middleware.py:28-49`
- Test: `tests/tools/test_middleware.py` (create — currently no dedicated test file)

- [ ] **Step 1: Write the failing tests**

```python
# tests/tools/test_middleware.py
"""sensitive_confirm_middleware 测试。"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.tools.middleware import sensitive_confirm_middleware, build_pipeline
from src.tools.registry import ToolRegistry, ToolEntry


def _make_registry_with_sensitive_tool() -> ToolRegistry:
    """构建包含一个 sensitive=True 工具的 registry。"""
    registry = ToolRegistry()
    entry = ToolEntry(
        name="delete_file",
        func=AsyncMock(),
        description="删除文件",
        parameters_schema={},
        model=None,
        sensitive=True,
        confirm_template="删除文件 {path}",
    )
    registry._entries["delete_file"] = entry
    return registry


@pytest.fixture
def mock_interaction():
    interaction = AsyncMock()
    interaction.confirm = AsyncMock(return_value=True)
    return interaction


class TestSensitiveConfirmMiddleware:
    @pytest.mark.asyncio
    async def test_confirm_approved_calls_next(self, mock_interaction):
        """用户确认后应继续执行下游。"""
        registry = _make_registry_with_sensitive_tool()
        mock_interaction.confirm = AsyncMock(return_value=True)
        mw = sensitive_confirm_middleware(registry, mock_interaction)

        next_fn = AsyncMock(return_value="deleted")
        result = await mw("delete_file", {"path": "/tmp/test"}, next_fn)

        assert result == "deleted"
        next_fn.assert_called_once()
        mock_interaction.confirm.assert_called_once()

    @pytest.mark.asyncio
    async def test_confirm_rejected_returns_cancel(self, mock_interaction):
        """用户拒绝后应返回取消消息，不调用下游。"""
        registry = _make_registry_with_sensitive_tool()
        mock_interaction.confirm = AsyncMock(return_value=False)
        mw = sensitive_confirm_middleware(registry, mock_interaction)

        next_fn = AsyncMock(return_value="deleted")
        result = await mw("delete_file", {"path": "/tmp/test"}, next_fn)

        assert "取消" in result
        next_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_sensitive_tool_skips_confirm(self, mock_interaction):
        """非敏感工具应直接执行，不触发确认。"""
        registry = ToolRegistry()
        entry = ToolEntry(
            name="get_weather",
            func=AsyncMock(),
            description="获取天气",
            parameters_schema={},
            model=None,
            sensitive=False,
            confirm_template=None,
        )
        registry._entries["get_weather"] = entry
        mw = sensitive_confirm_middleware(registry, mock_interaction)

        next_fn = AsyncMock(return_value="sunny")
        result = await mw("get_weather", {}, next_fn)

        assert result == "sunny"
        mock_interaction.confirm.assert_not_called()

    @pytest.mark.asyncio
    async def test_confirm_message_uses_template(self, mock_interaction):
        """确认消息应使用 confirm_template 格式化。"""
        registry = _make_registry_with_sensitive_tool()
        mock_interaction.confirm = AsyncMock(return_value=True)
        mw = sensitive_confirm_middleware(registry, mock_interaction)

        next_fn = AsyncMock(return_value="ok")
        await mw("delete_file", {"path": "/tmp/test"}, next_fn)

        confirm_arg = mock_interaction.confirm.call_args[0][0]
        assert "删除文件 /tmp/test" in confirm_arg
```

- [ ] **Step 2: Run tests to verify current state**

Run: `uv run pytest tests/tools/test_middleware.py -v`
Expected: FAIL — `sensitive_confirm_middleware` still expects `ui` param, not `interaction`

- [ ] **Step 3: Modify sensitive_confirm_middleware**

In `src/tools/middleware.py`, change `sensitive_confirm_middleware` to accept `interaction` instead of `ui`:

Replace:
```python
def sensitive_confirm_middleware(registry: ToolRegistry, ui) -> Middleware:
    """敏感工具执行前需要用户确认。ui 为 UserInterface 实例。"""

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

            await ui.display(f"\n⚠️  是否允许{msg}？\n")
            confirmed = await ui.confirm("")
            if not confirmed:
                return "用户取消了操作"

        return await next_fn(name, args)

    return middleware
```

With:
```python
def sensitive_confirm_middleware(registry: ToolRegistry, interaction) -> Middleware:
    """敏感工具执行前需要用户确认。interaction 为 UserInteractionService 实例。"""

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

            confirmed = await interaction.confirm(msg)
            if not confirmed:
                return "用户取消了操作"

        return await next_fn(name, args)

    return middleware
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tools/test_middleware.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Run existing tests to verify no regressions**

Run: `uv run pytest tests/ -v --timeout=30`
Expected: All existing tests still pass (bootstrap tests may fail — fixed in Task 5)

- [ ] **Step 6: Commit**

```bash
git add src/tools/middleware.py tests/tools/test_middleware.py
git commit -m "refactor: sensitive_confirm_middleware uses UserInteractionService"
```

---

### Task 4: AgentRunner._build_tools supports SYSTEM_TOOLS

**Files:**
- Modify: `src/agents/runner.py:248-262`
- Modify: `tests/agents/test_runner.py` (add new tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/agents/test_runner.py`:

```python
def test_build_tools_includes_system_tools_when_agent_has_tools():
    """agent.tools 非空时，_build_tools 应自动包含系统工具 ask_user。"""
    from src.agents.runner import AgentRunner

    mock_router = MagicMock()
    mock_router.get_all_schemas = MagicMock(return_value=[
        {"type": "function", "function": {"name": "exec", "description": "Execute", "parameters": {}}},
        {"type": "function", "function": {"name": "ask_user", "description": "Ask user", "parameters": {}}},
    ])

    agent = Agent(name="test", description="Test", instructions="Test.", tools=["exec"])
    ctx = RunContext(input="test", state=DynamicState(), deps=AgentDeps(tool_router=mock_router), delegate_depth=0)

    runner = AgentRunner()
    tools = runner._build_tools(agent, ctx)

    names = [t["function"]["name"] for t in tools]
    assert "exec" in names
    assert "ask_user" in names


def test_build_tools_includes_system_tools_when_agent_has_no_tools():
    """agent.tools 为空时，_build_tools 应只返回系统工具。"""
    from src.agents.runner import AgentRunner

    mock_router = MagicMock()
    mock_router.get_all_schemas = MagicMock(return_value=[
        {"type": "function", "function": {"name": "exec", "description": "Execute", "parameters": {}}},
        {"type": "function", "function": {"name": "ask_user", "description": "Ask user", "parameters": {}}},
    ])

    agent = Agent(name="orchestrator", description="Orch", instructions="Route.", tools=[])
    ctx = RunContext(input="test", state=DynamicState(), deps=AgentDeps(tool_router=mock_router), delegate_depth=0)

    runner = AgentRunner()
    tools = runner._build_tools(agent, ctx)

    names = [t["function"]["name"] for t in tools]
    assert "ask_user" in names
    assert "exec" not in names


def test_build_tools_system_tools_not_filtered_by_delegate_depth():
    """delegate_depth >= 1 时，系统工具 ask_user 不应被过滤。"""
    from src.agents.runner import AgentRunner

    mock_router = MagicMock()
    mock_router.get_all_schemas = MagicMock(return_value=[
        {"type": "function", "function": {"name": "exec", "description": "Execute", "parameters": {}}},
        {"type": "function", "function": {"name": "delegate_tool_calc", "description": "Delegate", "parameters": {}}},
        {"type": "function", "function": {"name": "ask_user", "description": "Ask user", "parameters": {}}},
    ])

    agent = Agent(name="test", description="Test", instructions="Test.", tools=["exec", "delegate_tool_calc"])
    ctx = RunContext(input="test", state=DynamicState(), deps=AgentDeps(tool_router=mock_router), delegate_depth=1)

    runner = AgentRunner()
    tools = runner._build_tools(agent, ctx)

    names = [t["function"]["name"] for t in tools]
    assert "exec" in names
    assert "ask_user" in names
    assert "delegate_tool_calc" not in names
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `uv run pytest tests/agents/test_runner.py::test_build_tools_includes_system_tools_when_agent_has_tools tests/agents/test_runner.py::test_build_tools_includes_system_tools_when_agent_has_no_tools tests/agents/test_runner.py::test_build_tools_system_tools_not_filtered_by_delegate_depth -v`
Expected: FAIL — `ask_user` not in result

- [ ] **Step 3: Modify _build_tools in runner.py**

In `src/agents/runner.py`, add `SYSTEM_TOOLS` constant after existing `HANDOFF_PREFIX` and rewrite `_build_tools`:

After line 27 (`HANDOFF_PREFIX = "transfer_to_"`), add:
```python
SYSTEM_TOOLS = {"ask_user"}
```

Replace the `_build_tools` method (lines 248-262):
```python
    def _build_tools(self, agent: Agent, context: RunContext) -> list[dict]:
        """从 deps.tool_router 过滤 agent 允许的工具。

        - 系统工具（SYSTEM_TOOLS）始终包含，所有 agent 可用
        - 当 context.delegate_depth >= 1 时，过滤掉所有 delegate_ 前缀的工具
        """
        tool_router = getattr(context.deps, "tool_router", None)
        if not tool_router:
            return []
        all_schemas = tool_router.get_all_schemas()

        if not agent.tools:
            return [s for s in all_schemas if s["function"]["name"] in SYSTEM_TOOLS]

        allowed = set(agent.tools) | SYSTEM_TOOLS
        if context.delegate_depth >= 1:
            allowed = {
                name for name in allowed
                if not name.startswith("delegate_")
            }
        return [s for s in all_schemas if s["function"]["name"] in allowed]
```

- [ ] **Step 4: Run all runner tests to verify they pass**

Run: `uv run pytest tests/agents/test_runner.py -v`
Expected: All tests PASS (existing + 3 new)

- [ ] **Step 5: Commit**

```bash
git add src/agents/runner.py tests/agents/test_runner.py
git commit -m "feat: _build_tools always includes SYSTEM_TOOLS (ask_user)"
```

---

### Task 5: Wire everything in bootstrap.py

**Files:**
- Modify: `src/app/bootstrap.py:52-74`

- [ ] **Step 1: Modify bootstrap.py**

In `src/app/bootstrap.py`, add the import after existing imports (after line 19):

```python
from src.utils.interaction import UserInteractionService
from src.tools.user_input import UserInputToolProvider
```

After `ui = CLIInterface()` (line 52), add:

```python
    interaction = UserInteractionService(ui)
```

Change the `middlewares` list (line 68-72) from:

```python
    middlewares = [
        error_handler_middleware(),
        sensitive_confirm_middleware(registry, ui),
        truncate_middleware(raw.get("tools", {}).get("max_output_length", 2000)),
    ]
```

To:

```python
    middlewares = [
        error_handler_middleware(),
        sensitive_confirm_middleware(registry, interaction),
        truncate_middleware(raw.get("tools", {}).get("max_output_length", 2000)),
    ]
```

After `tool_router.add_provider(LocalToolProvider(registry, executor, middlewares))` (line 74), add:

```python
    tool_router.add_provider(UserInputToolProvider(interaction))
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest tests/ -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 3: Quick manual smoke test**

Run: `uv run python -c "from src.utils.interaction import UserInteractionService; from src.tools.user_input import UserInputToolProvider; print('imports OK')"`
Expected: `imports OK`

- [ ] **Step 4: Commit**

```bash
git add src/app/bootstrap.py
git commit -m "feat: wire UserInteractionService and UserInputToolProvider in bootstrap"
```

---

### Task 6: Integration verification

**Files:**
- No new files — verify everything works together

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=30`
Expected: All tests PASS, no regressions

- [ ] **Step 2: Verify tool schema visibility**

Run:
```bash
uv run python -c "
from unittest.mock import MagicMock
from src.agents.agent import Agent
from src.agents.context import RunContext, DynamicState
from src.agents.deps import AgentDeps
from src.agents.runner import AgentRunner

mock_router = MagicMock()
mock_router.get_all_schemas = MagicMock(return_value=[
    {'type': 'function', 'function': {'name': 'ask_user', 'description': 'Ask', 'parameters': {}}},
    {'type': 'function', 'function': {'name': 'get_weather', 'description': 'Weather', 'parameters': {}}},
])

# agent 有工具 -> 包含 ask_user
agent = Agent(name='test', description='T', instructions='T', tools=['get_weather'])
ctx = RunContext(input='t', state=DynamicState(), deps=AgentDeps(tool_router=mock_router))
runner = AgentRunner()
tools = runner._build_tools(agent, ctx)
names = [t['function']['name'] for t in tools]
assert 'ask_user' in names, f'ask_user missing: {names}'
assert 'get_weather' in names, f'get_weather missing: {names}'

# agent 无工具 -> 只有 ask_user
agent2 = Agent(name='orch', description='O', instructions='O', tools=[])
tools2 = runner._build_tools(agent2, RunContext(input='t', state=DynamicState(), deps=AgentDeps(tool_router=mock_router)))
names2 = [t['function']['name'] for t in tools2]
assert 'ask_user' in names2, f'ask_user missing: {names2}'
assert 'get_weather' not in names2, f'get_weather should not be in: {names2}'

print('All visibility checks passed')
"
```
Expected: `All visibility checks passed`

- [ ] **Step 3: Final commit — update docs**

Update spec status if needed, otherwise just verify the spec is committed and matches the implementation.

Run: `git log --oneline -5`
Expected: 5 commits from this implementation plan
