import asyncio
import json
from .tool_executor import ToolExecutor
from typing import Dict, List, Any

async def execute_tool_calls(
    content: str,
    tool_calls: Dict[int, Dict[str, str]],
    tool_executor: ToolExecutor
) -> List[Dict[str, Any]]:
    """
    异步执行工具调用，支持同步/异步混合工具
    """
    if not tool_calls:
        return []

    new_messages = []

    # 构造assistant消息
    assistant_msg = {
        "role": "assistant",
        "content": content if content else None,
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": tc["arguments"]
                }
            }
            for tc in tool_calls.values()
        ]
    }
    new_messages.append(assistant_msg)

    # 并行执行所有工具调用
    tool_tasks = []
    results = []
    for idx, tc in tool_calls.items():
        # 将 arguments 解析为字典
        try:
            args = json.loads(tc["arguments"])
        except json.JSONDecodeError as e:
            # 直接返回错误，由 tool_executor 处理？但 tool_executor 期望 arguments 是字典
            # 更好的做法：构造一个错误消息作为 tool 结果
            results.append((idx, f"参数 JSON 解析失败: {e}"))
            continue
        # 调用 executor 的 execute 方法（异步）
        task = asyncio.create_task(tool_executor.execute(tc["name"], args))
        tool_tasks.append((idx, task))

    # 等待所有工具完成
    for idx, task in tool_tasks:
        try:
            result = await task
            results.append((idx, result))
        except Exception as e:
            results.append((idx, f"工具执行异常: {e}"))

    # 按原始顺序构造tool消息
    for idx, result in sorted(results, key=lambda x: x[0]):
        new_messages.append({
            "role": "tool",
            "tool_call_id": tool_calls[idx]["id"],
            "content": str(result)
        })

    return new_messages
