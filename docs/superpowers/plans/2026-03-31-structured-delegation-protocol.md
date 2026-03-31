# 结构化委托协议 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 扩展 DelegateToolProvider 的委托 schema 和 input 组装，使 Agent 间委托包含结构化的 objective/task/context/expected_result 四字段，并在接收方模板中引导任务状态标记。

**Architecture:** 仅修改 `src/tools/delegate.py`。发送方通过扩展的 function calling schema 强制填写结构化字段，接收方通过 prompt 模板获得清晰的任务上下文。返回值保持 `str` 不变。

**Tech Stack:** Python 3.13, Pydantic, pytest, asyncio

---

## File Structure

| 文件 | 操作 | 职责 |
|------|------|------|
| `src/tools/delegate.py` | 修改 | 新增模板常量、重写 `get_schemas()` 和 `execute()` |
| `tests/tools/test_delegate.py` | 修改 | 更新现有测试 + 新增结构化委托测试 |
| `tests/tools/test_delegate_integration.py` | 修改 | 更新集成测试适配新 schema |

---

### Task 1: 新增模板常量和 _build_receiving_input 方法

**Files:**
- Modify: `src/tools/delegate.py:1-24`
- Test: `tests/tools/test_delegate.py`

- [ ] **Step 1: 写 _build_receiving_input 的测试**

在 `tests/tools/test_delegate.py` 末尾新增：

```python
def test_build_receiving_input_all_fields():
    """所有字段都有值时，接收方 input 包含全部信息。"""
    from src.tools.delegate import _build_receiving_input

    result = _build_receiving_input(
        objective="判断明天是否适合去故宫",
        task="查询北京明天天气",
        context="用户计划明天去故宫",
        expected_result="天气状况、温度、出行建议",
    )
    assert "最终目标：判断明天是否适合去故宫" in result
    assert "具体任务：查询北京明天天气" in result
    assert "相关上下文：用户计划明天去故宫" in result
    assert "期望结果：天气状况、温度、出行建议" in result
    assert "不要猜测或假设" in result
    assert "已完成 / 信息不足 / 失败" in result


def test_build_receiving_input_optional_fields_missing():
    """可选字段为空时，不显示对应行。"""
    from src.tools.delegate import _build_receiving_input

    result = _build_receiving_input(
        objective="帮用户查天气",
        task="查询天气预报",
        context=None,
        expected_result=None,
    )
    assert "最终目标：帮用户查天气" in result
    assert "具体任务：查询天气预报" in result
    assert "相关上下文" not in result
    assert "期望结果" not in result
    assert "不要猜测或假设" in result


def test_build_receiving_input_empty_string_treated_as_missing():
    """空字符串应与 None 同样处理，不显示对应行。"""
    from src.tools.delegate import _build_receiving_input

    result = _build_receiving_input(
        objective="目标",
        task="任务",
        context="",
        expected_result="",
    )
    assert "相关上下文" not in result
    assert "期望结果" not in result
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/tools/test_delegate.py::test_build_receiving_input_all_fields tests/tools/test_delegate.py::test_build_receiving_input_optional_fields_missing tests/tools/test_delegate.py::test_build_receiving_input_empty_string_treated_as_missing -v`
Expected: FAIL — `ImportError: cannot import name '_build_receiving_input'`

- [ ] **Step 3: 实现模板常量和 _build_receiving_input**

在 `src/tools/delegate.py` 中，`DELEGATE_PREFIX` 之后新增：

```python
DELEGATE_DESCRIPTION_TEMPLATE = (
    "委托任务给{description}专家。"
    "请基于当前对话上下文，清晰完整地填写以下字段，"
    "确保对方无需额外信息就能执行任务。"
)

RECEIVING_TEMPLATE = (
    "你收到了一个委托任务：\n"
    "最终目标：{objective}\n"
    "具体任务：{task}\n"
    "{context_line}"
    "{expected_result_line}"
    "\n"
    "完成后请按以下格式返回：\n"
    "第一行标注任务状态：已完成 / 信息不足 / 失败\n"
    "之后是具体结果或需要补充的信息。\n"
    "不要猜测或假设缺失的信息。"
)


def _build_receiving_input(
    objective: str,
    task: str,
    context: str | None = None,
    expected_result: str | None = None,
) -> str:
    """用接收方模板组装委托任务的 input 文本。"""
    context_line = f"相关上下文：{context}\n" if context else ""
    expected_result_line = f"期望结果：{expected_result}\n" if expected_result else ""
    return RECEIVING_TEMPLATE.format(
        objective=objective,
        task=task,
        context_line=context_line,
        expected_result_line=expected_result_line,
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/tools/test_delegate.py::test_build_receiving_input_all_fields tests/tools/test_delegate.py::test_build_receiving_input_optional_fields_missing tests/tools/test_delegate.py::test_build_receiving_input_empty_string_treated_as_missing -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/tools/delegate.py tests/tools/test_delegate.py
git commit -m "feat: add delegation prompt templates and _build_receiving_input"
```

---

### Task 2: 重写 get_schemas() 使用四字段 schema

**Files:**
- Modify: `src/tools/delegate.py:59-82`（`get_schemas` 方法）
- Test: `tests/tools/test_delegate.py`

- [ ] **Step 1: 写新 schema 结构的测试**

在 `tests/tools/test_delegate.py` 末尾新增：

```python
def test_get_schemas_has_structured_fields(provider):
    """schema 应包含 objective/task/context/expected_result 四个字段。"""
    schemas = provider.get_schemas()
    terminal_schema = next(s for s in schemas if s["function"]["name"] == "delegate_tool_terminal")
    params = terminal_schema["function"]["parameters"]
    props = params["properties"]

    assert "objective" in props
    assert "task" in props
    assert "context" in props
    assert "expected_result" in props
    # objective 和 task 必填，context 和 expected_result 可选
    assert set(params["required"]) == {"objective", "task"}


def test_get_schemas_description_uses_template(provider):
    """schema description 应包含委托引导语。"""
    schemas = provider.get_schemas()
    terminal_schema = next(s for s in schemas if s["function"]["name"] == "delegate_tool_terminal")
    desc = terminal_schema["function"]["description"]
    assert "确保对方无需额外信息就能执行任务" in desc
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/tools/test_delegate.py::test_get_schemas_has_structured_fields tests/tools/test_delegate.py::test_get_schemas_description_uses_template -v`
Expected: FAIL — `objective` not in params

- [ ] **Step 3: 重写 get_schemas()**

将 `src/tools/delegate.py` 中的 `get_schemas` 方法替换为：

```python
    def get_schemas(self) -> list[ToolDict]:
        """为每个可委派的 Tool Agent 生成结构化委托 schema。"""
        schemas: list[ToolDict] = []
        for summary in self._resolver.get_all_summaries():
            name = summary["name"]
            desc = summary["description"]
            schemas.append({
                "type": "function",
                "function": {
                    "name": f"{DELEGATE_PREFIX}{name}",
                    "description": DELEGATE_DESCRIPTION_TEMPLATE.format(description=desc),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "objective": {
                                "type": "string",
                                "description": "你的最终目标是什么（为什么需要这次委托）",
                            },
                            "task": {
                                "type": "string",
                                "description": "你需要对方具体做什么",
                            },
                            "context": {
                                "type": "string",
                                "description": "当前已知的相关信息。只填你确定知道的，不要猜测。",
                            },
                            "expected_result": {
                                "type": "string",
                                "description": "你期望对方完成后告诉你什么。如果不确定，可简要描述即可。",
                            },
                        },
                        "required": ["objective", "task"],
                    },
                },
            })
        return schemas
```

- [ ] **Step 4: 运行新测试确认通过**

Run: `uv run pytest tests/tools/test_delegate.py::test_get_schemas_has_structured_fields tests/tools/test_delegate.py::test_get_schemas_description_uses_template -v`
Expected: 2 PASSED

- [ ] **Step 5: 修复受影响的旧测试**

`test_get_schemas` 断言了 `"task" in params["required"]`，需要更新：

将 `tests/tools/test_delegate.py` 中的 `test_get_schemas` 替换为：

```python
def test_get_schemas(provider):
    schemas = provider.get_schemas()
    assert len(schemas) == 2
    names = {s["function"]["name"] for s in schemas}
    assert names == {"delegate_tool_terminal", "delegate_tool_calc"}
    terminal_schema = next(s for s in schemas if s["function"]["name"] == "delegate_tool_terminal")
    assert terminal_schema["type"] == "function"
    params = terminal_schema["function"]["parameters"]
    assert "objective" in params["properties"]
    assert "task" in params["properties"]
    assert set(params["required"]) == {"objective", "task"}
```

- [ ] **Step 6: 运行全部 delegate 测试确认通过**

Run: `uv run pytest tests/tools/test_delegate.py -v`
Expected: ALL PASSED

- [ ] **Step 7: Commit**

```bash
git add src/tools/delegate.py tests/tools/test_delegate.py
git commit -m "feat: structured four-field delegation schema in get_schemas()"
```

---

### Task 3: 重写 execute() 使用接收方模板组装 input

**Files:**
- Modify: `src/tools/delegate.py:84-111`（`execute` 方法）
- Test: `tests/tools/test_delegate.py`

- [ ] **Step 1: 写 execute 使用结构化参数的测试**

在 `tests/tools/test_delegate.py` 末尾新增：

```python
@pytest.mark.asyncio
async def test_execute_builds_structured_input(provider, mock_runner):
    """execute 应用接收方模板组装 input，包含 objective/task/context/expected_result。"""
    from src.agents.agent import AgentResult

    mock_runner.run = AsyncMock(return_value=AgentResult(text="已完成\n结果"))
    await provider.execute("delegate_tool_terminal", {
        "objective": "帮用户管理文件",
        "task": "列出当前目录",
        "context": "用户在 /home/user 目录下",
        "expected_result": "文件列表",
    })

    call_args = mock_runner.run.call_args
    ctx = call_args[0][1]
    assert "最终目标：帮用户管理文件" in ctx.input
    assert "具体任务：列出当前目录" in ctx.input
    assert "相关上下文：用户在 /home/user 目录下" in ctx.input
    assert "期望结果：文件列表" in ctx.input


@pytest.mark.asyncio
async def test_execute_optional_fields_omitted(provider, mock_runner):
    """只传 objective 和 task 时，input 不包含 context 和 expected_result 行。"""
    from src.agents.agent import AgentResult

    mock_runner.run = AsyncMock(return_value=AgentResult(text="已完成\n结果"))
    await provider.execute("delegate_tool_terminal", {
        "objective": "查天气",
        "task": "查询天气预报",
    })

    call_args = mock_runner.run.call_args
    ctx = call_args[0][1]
    assert "最终目标：查天气" in ctx.input
    assert "具体任务：查询天气预报" in ctx.input
    assert "相关上下文" not in ctx.input
    assert "期望结果" not in ctx.input


@pytest.mark.asyncio
async def test_execute_backward_compat_task_only(provider, mock_runner):
    """兼容旧格式：只传 task 时，objective 用 task 兜底。"""
    from src.agents.agent import AgentResult

    mock_runner.run = AsyncMock(return_value=AgentResult(text="已完成\n42"))
    await provider.execute("delegate_tool_terminal", {"task": "计算 1+1"})

    call_args = mock_runner.run.call_args
    ctx = call_args[0][1]
    assert "具体任务：计算 1+1" in ctx.input
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/tools/test_delegate.py::test_execute_builds_structured_input tests/tools/test_delegate.py::test_execute_optional_fields_omitted tests/tools/test_delegate.py::test_execute_backward_compat_task_only -v`
Expected: FAIL — ctx.input 仍然是裸 task 字符串

- [ ] **Step 3: 重写 execute()**

将 `src/tools/delegate.py` 中 `execute` 方法的 input 构造部分替换：

```python
    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """委派执行：按需连接 MCP server，创建子 RunContext 并驱动 AgentRunner。

        注意：子 RunContext 与父级共享同一 deps（含 tool_router）。
        当前 AgentRunner 串行执行工具调用，因此不存在并发问题。
        若未来支持并行工具调用，需要为每次 delegate 创建独立的 deps 副本。
        """
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

        # 从结构化参数构建接收方 input
        task = arguments.get("task", "")
        objective = arguments.get("objective", task)
        context = arguments.get("context")
        expected_result = arguments.get("expected_result")
        receiving_input = _build_receiving_input(
            objective=objective,
            task=task,
            context=context,
            expected_result=expected_result,
        )

        sub_ctx: RunContext = RunContext(
            input=receiving_input,
            state=DynamicState(),
            deps=self._deps,
            delegate_depth=self._delegate_depth + 1,
        )
        result = await self._runner.run(agent, sub_ctx)
        return result.text
```

- [ ] **Step 4: 运行新测试确认通过**

Run: `uv run pytest tests/tools/test_delegate.py::test_execute_builds_structured_input tests/tools/test_delegate.py::test_execute_optional_fields_omitted tests/tools/test_delegate.py::test_execute_backward_compat_task_only -v`
Expected: 3 PASSED

- [ ] **Step 5: 修复受影响的旧测试**

`test_execute_delegates_to_runner` 断言了 `ctx.input == "列出当前目录"`，现在 input 是模板格式。更新：

将 `tests/tools/test_delegate.py` 中 `test_execute_delegates_to_runner` 替换为：

```python
@pytest.mark.asyncio
async def test_execute_delegates_to_runner(provider, mock_runner):
    from src.agents.agent import AgentResult
    mock_runner.run = AsyncMock(return_value=AgentResult(text="已完成\n执行完成"))
    result = await provider.execute("delegate_tool_terminal", {"task": "列出当前目录"})
    assert "执行完成" in result
    mock_runner.run.assert_called_once()
    call_args = mock_runner.run.call_args
    agent = call_args[0][0]
    assert agent.name == "tool_terminal"
    ctx = call_args[0][1]
    assert "具体任务：列出当前目录" in ctx.input
```

同样更新 `test_execute_ensures_mcp_connection` 和 `test_execute_no_mcp_manager_still_works` 中传入的 arguments，将 `{"task": "..."}` 保持不变（兼容模式），但如果它们断言了 `result == "..."` 且返回值不变则无需改动。检查 `test_execute_propagates_delegate_depth` 和 `test_execute_propagates_parent_delegate_depth` 同理。

- [ ] **Step 6: 运行全部 delegate 单元测试**

Run: `uv run pytest tests/tools/test_delegate.py -v`
Expected: ALL PASSED

- [ ] **Step 7: Commit**

```bash
git add src/tools/delegate.py tests/tools/test_delegate.py
git commit -m "feat: execute() uses structured receiving template for delegation input"
```

---

### Task 4: 更新集成测试

**Files:**
- Modify: `tests/tools/test_delegate_integration.py`

- [ ] **Step 1: 更新 test_delegate_end_to_end 适配新 schema**

集成测试中 `mock_chat` 的第一次调用传递的 arguments 仍是 `{"task": "..."}` 旧格式。更新为结构化格式：

将 `tests/tools/test_delegate_integration.py` 中 `test_delegate_end_to_end` 的 mock_chat 第一次返回替换为：

```python
        if call_count == 1:
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
```

同时更新对 Agent_B 的断言 — Agent_B 收到的 input 现在是结构化模板格式，不再是裸字符串。更新 `mock_chat` 中第二次调用的检查（无需改动，因为 mock 不检查 input 内容）。

- [ ] **Step 2: 运行集成测试确认通过**

Run: `uv run pytest tests/tools/test_delegate_integration.py -v`
Expected: ALL PASSED

- [ ] **Step 3: Commit**

```bash
git add tests/tools/test_delegate_integration.py
git commit -m "test: update integration tests for structured delegation protocol"
```

---

### Task 5: 更新模块文档注释 & 运行完整测试套件

**Files:**
- Modify: `src/tools/delegate.py:1-9`（模块 docstring）

- [ ] **Step 1: 更新模块 docstring**

将 `src/tools/delegate.py` 的模块 docstring 替换为：

```python
"""DelegateToolProvider — 将 Tool Agent 包装为可调用工具。

业务 Agent 可通过 delegate_<name>(objective, task, context?, expected_result?)
调用对应的 Tool Agent。委托时通过结构化的四字段 schema 强制发送方说清楚
任务意图，接收方通过 prompt 模板获得完整的任务上下文。

协议设计详见 docs/superpowers/specs/2026-03-31-structured-delegation-protocol-design.md

本模块位于 Layer 1（src/tools/），对 Layer 2 的依赖
（AgentRunner、AgentRegistry、AgentDeps）仅在 TYPE_CHECKING
或 execute() 运行时才导入，不违反分层约束。
"""
```

- [ ] **Step 2: 运行完整测试套件**

Run: `uv run pytest -v`
Expected: ALL PASSED，无回归

- [ ] **Step 3: Commit**

```bash
git add src/tools/delegate.py
git commit -m "docs: update delegate module docstring for structured protocol"
```
