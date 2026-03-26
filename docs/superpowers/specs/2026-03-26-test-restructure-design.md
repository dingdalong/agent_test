# 测试模块重构设计规格

**日期：** 2026-03-26
**状态：** 已批准

## 目标

将 `tests/` 目录从扁平结构重构为严格镜像 `src/` 的模块化结构，建立分层 conftest 体系，统一使用纯 pytest 风格。

## 目标目录结构

```
tests/
├── __init__.py
├── conftest.py                    # 通用 fixtures
├── core/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_async_api.py          ← tests/test_async_api.py
│   ├── test_fsm.py                ← tests/test_fsm.py
│   └── test_main.py               ← test_integration.py 部分拆入（main 属性测试）
├── flows/
│   ├── __init__.py
│   ├── conftest.py                # mock io, mock memory
│   ├── test_chat.py               ← 不动
│   ├── test_meeting_booking.py    ← 不动
│   ├── test_planning.py           ← 不动
│   └── test_registry.py           ← tests/test_flows_registry.py
├── memory/
│   ├── __init__.py
│   ├── conftest.py                # mock ChromaDB collection
│   ├── test_types.py              ← tests/test_memory_types.py
│   ├── test_store.py              ← tests/test_vector_memory_extended.py + test_integration.py 部分
│   └── test_decay.py              ← tests/test_decay.py
├── mcp/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_config.py             ← tests/test_mcp_config.py
│   └── test_manager.py            ← tests/test_mcp_manager.py
├── plan/
│   ├── __init__.py                ← 已有
│   ├── test_models.py             ← 不动
│   ├── test_planner.py            ← 不动
│   └── test_executor.py           ← 不动
├── skills/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_models.py             ← tests/test_skills/test_models.py
│   ├── test_parser.py             ← tests/test_skills/test_parser.py
│   ├── test_manager.py            ← tests/test_skills/test_manager.py
│   ├── test_provider.py           ← tests/test_skills/test_tool_executor_skill.py
│   └── test_integration.py        ← tests/test_skills/test_integration.py
├── tools/
│   ├── __init__.py
│   ├── conftest.py                # workspace fixture, clean_registry fixture
│   ├── test_architecture.py       ← tests/test_tools_refactor.py
│   ├── test_builtin_calculator.py ← tests/test_tools.py
│   └── test_builtin_file.py       ← tests/test_file.py
└── performance/
    ├── __init__.py
    └── test_benchmarks.py         ← tests/performance_test.py
```

## 文件重命名映射

| 原文件 | 新位置 | 原因 |
|--------|--------|------|
| `test_flows_registry.py` | `flows/test_registry.py` | 去掉冗余前缀 |
| `test_memory_types.py` | `memory/test_types.py` | 去掉冗余前缀 |
| `test_vector_memory_extended.py` | `memory/test_store.py` | 对应 src/memory/store.py |
| `test_mcp_config.py` | `mcp/test_config.py` | 去掉冗余前缀 |
| `test_mcp_manager.py` | `mcp/test_manager.py` | 去掉冗余前缀 |
| `test_tools_refactor.py` | `tools/test_architecture.py` | 重构已完成，名字反映内容 |
| `test_tools.py` | `tools/test_builtin_calculator.py` | 明确测试对象 |
| `test_file.py` | `tools/test_builtin_file.py` | 明确测试对象 |
| `performance_test.py` | `performance/test_benchmarks.py` | 统一命名规范 |
| `test_skills/test_tool_executor_skill.py` | `skills/test_provider.py` | 对应 src/skills/provider.py |

## conftest 分层设计

### `tests/conftest.py`（根级）
- `workspace_dir` fixture — 创建/清理 workspace/ 临时目录
- `mock_call_model` fixture — 通用的 call_model mock helper

### `tests/flows/conftest.py`
- `mock_io` fixture — mock agent_input / agent_output
- `mock_memory` fixture — mock memory store 和 buffer

### `tests/memory/conftest.py`
- `mock_chroma_collection` fixture — mock ChromaDB collection 对象

### `tests/tools/conftest.py`
- `clean_registry` fixture — 每个测试前重置 ToolRegistry
- `workspace_dir` fixture — workspace 目录管理（可从根级继承或覆盖）

其余子目录 conftest 先建空文件，后续按需添加。

## root test_integration.py 拆分

- `test_build_collection_name_*` → `tests/memory/test_store.py` 追加
- `test_main_has_store` / `test_main_has_buffer` → `tests/core/test_main.py` 新建
- 拆分完成后删除根目录 `test_integration.py`

## unittest → pytest 改写

完全去掉 unittest 依赖，涉及 3 个文件：

| 文件 | 改写内容 |
|------|----------|
| `test_integration.py`（root） | 拆分时直接写成 pytest |
| `test_memory_types.py` → `memory/test_types.py` | 去掉 TestCase 继承 |
| `test_vector_memory_extended.py` → `memory/test_store.py` | 去掉 TestCase + mock.patch → fixtures |

改写规则：
- `self.assertEqual(a, b)` → `assert a == b`
- `self.assertIn(a, b)` → `assert a in b`
- `self.assertRaises` → `pytest.raises`
- `self.assertTrue/assertFalse` → `assert` / `assert not`
- `setUp/tearDown` → pytest fixtures
- 去掉所有 `import unittest` 和 `unittest.TestCase` 继承
- `unittest.mock` 保留使用但不依赖 unittest.TestCase

## 不变更的内容

- `tests/plan/` 内所有文件保持不动
- `tests/flows/` 内已有的 test_chat.py、test_meeting_booking.py、test_planning.py 保持不动
- `pyproject.toml` 的 pytest 配置保持不动
- 根目录 `test.py`（开发脚本）不动
