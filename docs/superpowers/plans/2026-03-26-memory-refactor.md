# Memory Module Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the memory module refactor by fixing the remaining bugs in consumer code (`main.py`, `test.py`, `validate_async.py`) and verifying all tests pass.

**Architecture:** The core memory module (`src/memory/`) has already been fully rewritten per the design spec — `types.py`, `store.py`, `buffer.py`, `extractor.py`, `embeddings.py`, `decay.py` are all implemented. The consumer files `chat.py` and `orchestrator.py` are already updated. The remaining work is fixing `main.py` (still references deleted `user_facts`/`conversation_summaries` variables), fixing `test.py` (calls async method synchronously), fixing `validate_async.py` (references deleted `run_agent`), and running the full test suite to verify everything passes.

**Tech Stack:** Python, Pydantic, ChromaDB, tiktoken, pytest

---

### Task 1: Fix `main.py` — old variable references in `handle_input()`

**Files:**
- Modify: `main.py:104-115`

The skill-slash-command branch (line 71-75) is already correct (`memory=buffer, store=store`), but the fallback branch (line 104-115) still uses the old `memory=memory, user_facts=user_facts, conversation_summaries=conversation_summaries` variables that no longer exist.

- [ ] **Step 1: Write the failing test**

Add a test to `test_integration.py` that verifies `handle_input` uses the new API for the normal-conversation path:

```python
class TestHandleInputNewAPI(unittest.TestCase):
    def test_normal_conversation_uses_store(self):
        """handle_input section 4 should pass buffer and store (not old vars)."""
        main_module, _, store_instance = self._load_main_with_stubs()

        # Verify no NameError: user_facts / conversation_summaries
        # We just need to confirm main.py imports and references compile correctly
        # by checking the module-level `store` and `buffer` exist
        self.assertTrue(hasattr(main_module, 'store'))
        self.assertTrue(hasattr(main_module, 'buffer'))
        # The old names must NOT exist
        self.assertFalse(hasattr(main_module, 'user_facts'))
        self.assertFalse(hasattr(main_module, 'conversation_summaries'))
        self.assertFalse(hasattr(main_module, 'memory'))
```

Note: This test reuses the existing `_load_main_with_stubs` from `TestMainMemoryIntegration`. Move it to a shared helper or duplicate the method in the new test class.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_integration.py::TestHandleInputNewAPI::test_normal_conversation_uses_store -v`
Expected: FAIL because `main.py` still references `memory`, `user_facts`, `conversation_summaries`

- [ ] **Step 3: Fix `main.py` lines 104-115**

Replace the old `MultiAgentFlow` construction in section 4 of `handle_input()`:

```python
    # 4. 普通对话 → MultiAgentFlow（总控 + 专业 Agent）
    multi_agent_flow = MultiAgentFlow(
        registry=agent_registry,
        memory=buffer,
        store=store,
        all_tools=effective_tools,
        tool_executor=tool_executor,
    )
    multi_agent_flow.model.data["user_input"] = user_input
    runner = FSMRunner(multi_agent_flow)
    await runner.run()
```

This matches the already-correct skill-slash-command branch on lines 71-75.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add main.py test_integration.py
git commit -m "fix(main): replace old memory variables with buffer/store in handle_input"
```

---

### Task 2: Fix `test.py` — async method called synchronously

**Files:**
- Modify: `test.py`

`MemoryStore.add_from_conversation()` is an async method decorated with `@async_time_function()`. The current `test.py` calls it without `await`/`asyncio.run()`.

- [ ] **Step 1: Fix `test.py` to use asyncio**

```python
import asyncio
import json
from src.memory import FactExtractor, MemoryStore

async def main():
    extractor = FactExtractor()
    store = MemoryStore()

    results = store.search("名字")
    print(results[0].content if results else "无结果")

    # 用户第一次说"我叫小明"
    await store.add_from_conversation(user_input="我叫小明", source_id="conv1")
    results = store.search("名字")
    print(results[0].content if results else "无结果")

    # 用户后来又说"我叫大明"
    await store.add_from_conversation(user_input="我叫大明", source_id="conv1")
    # 检索时只返回最新版本（is_active=True）
    results = store.search("名字")
    print(results[0].content if results else "无结果")

if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Verify syntax is correct**

Run: `python -c "import ast; ast.parse(open('test.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add test.py
git commit -m "fix(test): wrap async add_from_conversation calls in asyncio.run"
```

---

### Task 3: Fix `validate_async.py` — references deleted `run_agent`

**Files:**
- Modify: `validate_async.py:62-65`

`validate_main_program()` checks for `main.run_agent` which was replaced by `main.handle_input` in the FSM refactor.

- [ ] **Step 1: Fix `validate_async.py` to check for `handle_input` instead**

Replace lines 62-65:

```python
        if hasattr(main, 'handle_input') and asyncio.iscoroutinefunction(main.handle_input):
            print("  ✅ handle_input 是异步函数")
        else:
            print("  ❌ handle_input 不是异步函数")
            return False
```

- [ ] **Step 2: Verify syntax is correct**

Run: `python -c "import ast; ast.parse(open('validate_async.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add validate_async.py
git commit -m "fix(validate): update run_agent reference to handle_input"
```

---

### Task 4: Run full test suite and fix any remaining failures

**Files:**
- Test: `tests/test_memory_types.py`, `tests/test_decay.py`, `tests/test_vector_memory_extended.py`, `tests/flows/test_chat.py`, `test_integration.py`

- [ ] **Step 1: Run all unit tests**

Run: `python -m pytest tests/test_memory_types.py tests/test_decay.py tests/test_vector_memory_extended.py tests/flows/test_chat.py test_integration.py -v`
Expected: All PASS

- [ ] **Step 2: If any failures, fix them**

Investigate and fix any import errors or API mismatches. Common issues:
- Old test files importing deleted modules
- Method signature changes not reflected in mocks

- [ ] **Step 3: Run full project test suite**

Run: `python -m pytest tests/ test_integration.py -v --ignore=tests/performance_test.py`
Expected: All PASS

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix(tests): resolve remaining test failures after memory refactor"
```

---

### Task 5: Verify clean deleted-file state

**Files:**
- Verify deleted: `src/memory/memory.py`, `src/memory/memory_types.py`, `src/memory/serializer.py`, `src/memory/versioning.py`, `src/memory/memory_extractor.py`
- Verify deleted: `tests/test_versioning_strategies.py`

- [ ] **Step 1: Confirm old files are deleted**

Run: `ls src/memory/memory.py src/memory/memory_types.py src/memory/serializer.py src/memory/versioning.py src/memory/memory_extractor.py tests/test_versioning_strategies.py 2>&1`
Expected: All "No such file or directory"

- [ ] **Step 2: Grep for any remaining imports of deleted modules**

Run: `grep -rn "from src.memory.memory import\|from src.memory.memory_types import\|from src.memory.serializer import\|from src.memory.versioning import\|from src.memory.memory_extractor import" --include="*.py" .`
Expected: No matches (empty output)

- [ ] **Step 3: Final verification — import the entire memory package**

Run: `python -c "from src.memory import MemoryStore, ConversationBuffer, MemoryRecord, MemoryType, FactExtractor, calculate_importance; print('All imports OK')"`
Expected: `All imports OK`

---

### Summary of changes by file

| File | Status | Change |
|------|--------|--------|
| `src/memory/types.py` | Done | New — MemoryType + MemoryRecord |
| `src/memory/store.py` | Done | New — MemoryStore (single collection) |
| `src/memory/buffer.py` | Done | New — ConversationBuffer with token cache |
| `src/memory/extractor.py` | Done | Renamed + fixed include_types bug |
| `src/memory/embeddings.py` | Done | Updated — Session connection pool |
| `src/memory/decay.py` | Done | New — calculate_importance |
| `src/memory/__init__.py` | Done | Updated exports |
| `src/flows/chat.py` | Done | Uses `store: MemoryStore` |
| `src/agents/orchestrator.py` | Done | Uses `store: MemoryStore` |
| `main.py` | **Task 1** | Fix line 104-115: `buffer`/`store` instead of old vars |
| `test.py` | **Task 2** | Wrap async calls in `asyncio.run()` |
| `validate_async.py` | **Task 3** | `run_agent` → `handle_input` |
| `tests/test_memory_types.py` | Done | Tests for MemoryRecord |
| `tests/test_decay.py` | Done | Tests for calculate_importance |
| `tests/test_vector_memory_extended.py` | Done | Tests for MemoryStore |
| `tests/flows/test_chat.py` | Done | Tests for ChatFlow |
| `test_integration.py` | **Task 1** | Add test for new API usage |
| Old files (6) | Done | Deleted |
