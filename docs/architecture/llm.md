# LLM 层

## 职责

封装 LLM 调用细节，提供统一的 Protocol 接口。上层模块（AgentRunner、PlanFlow、FactExtractor）只依赖 `LLMProvider`，不关心底层是哪个模型。

## 核心组件

### LLMProvider Protocol（`src/llm/base.py`）

```python
class LLMProvider(Protocol):
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 1.0,
        tool_choice: str | None = None,
        silent: bool = False,
    ) -> LLMResponse: ...
```

### LLMResponse（`src/llm/types.py`）

返回值包含：
- `content: str` — 文本响应
- `tool_calls: dict` — 工具调用请求
- `finish_reason: str` — 结束原因

### OpenAIProvider（`src/llm/openai.py`）

唯一的具体实现，特性：
- 流式输出（`on_chunk` 回调实时显示）
- 并发控制（`asyncio.Semaphore`）
- 指数退避重试
- 支持所有 OpenAI 兼容接口（DeepSeek、Ollama 等）

### 结构化输出（`src/llm/structured.py`）

通过 function calling 技巧，强制 LLM 返回符合 Pydantic model 的 JSON：
- `build_output_schema(name, desc, model)` — Pydantic model → tool schema
- `parse_output(tool_calls, name, model)` — tool_calls → Pydantic 实例

## 扩展方式

实现 `LLMProvider` Protocol，然后在 `bootstrap.py` 中替换 `OpenAIProvider`。
