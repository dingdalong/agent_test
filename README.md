# aitest
my ai test

## 配置环境
安装`uv`:`pip install uv`
初始化环境：`uv sync`
运行：`uv run main.py`

## 异步API重构

项目已完成异步API重构，使用 `AsyncOpenAI` 客户端替代同步客户端，提升并发性能和响应速度。

### 主要特性

- **纯异步架构**: 所有API调用使用 `asyncio` 和 `AsyncOpenAI` 客户端
- **并发控制**: 使用 `asyncio.Semaphore` 限制并发请求数（默认5）
- **指数退避重试**: 网络错误和限流时自动重试
- **流式响应支持**: 实时处理流式API响应
- **异步工具执行**: 支持异步工具函数，自动检测同步/异步函数
- **性能监控**: 异步函数性能监控装饰器 `@async_time_function()`

### 核心模块

- `src/core/async_api.py`: 异步模型调用核心逻辑
- `src/core/performance.py`: 异步性能监控装饰器
- `src/tools/`: 工具模块（支持异步工具函数）
- `main.py`: 异步主程序入口

### 使用示例

```python
import asyncio
from src.core.async_api import call_model

async def main():
    messages = [{"role": "user", "content": "Hello!"}]
    content, tool_calls, stop_reason = await call_model(messages, stream=False)
    print(f"Response: {content}")

asyncio.run(main())
```

### 添加异步工具

创建新的异步工具模块（如 `src/tools/my_tool.py`）:

```python
from pydantic import BaseModel, Field
import asyncio

class MyTool(BaseModel):
    """异步工具描述"""
    param: str = Field(description="参数描述")

async def my_tool(param: str) -> str:
    """异步工具函数"""
    await asyncio.sleep(0.1)  # 模拟异步操作
    return f"Result: {param}"

ToolModel = MyTool
execute = my_tool
TOOL_NAME = "my_tool"
TOOL_DESCRIPTION = "异步工具描述"
```

工具模块会自动发现并注册到系统中。

### 测试

运行异步测试:
```bash
python -m pytest tests/test_async_api.py -v
python -m pytest tests/test_async_tools.py -v
python -m pytest test_integration.py -v
```
