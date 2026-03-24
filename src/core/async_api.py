import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from openai import APIConnectionError, RateLimitError, APIError
from config import async_client, MODEL_NAME, request_semaphore
from .performance import async_time_function
from .io import agent_output
from src.utils.text import extract_json

logger = logging.getLogger(__name__)

@async_time_function()
async def call_model(
    messages: List[Dict[str, Any]],
    temperature: float = 1.0,
    tools: Optional[List[Dict]] = None,
    max_retries: int = 3,
    response_format: Optional[Dict[str, str]] = None,
    silent: bool = False,
    **kwargs
) -> Tuple[str, Dict[int, Dict[str, str]], Optional[str]]:
    """
    纯异步模型调用，统一使用流式请求，带指数退避重试和并发控制。

    流式请求只要 token 持续返回就不会超时，适合长文本生成场景。
    超时由 HTTP 客户端层面控制（连接超时、读取超时），不在应用层限制总时长。

    Args:
        silent: 为 True 时不输出到全局 IO（用于内部判断类调用）。
                默认 False，流式 chunk 会通过 agent_output 输出。
    """
    async with request_semaphore:
        for attempt in range(max_retries):
            try:
                create_kwargs = dict(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=tools,
                    stream=True,
                    temperature=temperature,
                    tool_choice="auto" if tools else None,
                )
                if response_format:
                    create_kwargs["response_format"] = response_format
                response = await async_client.chat.completions.create(**create_kwargs)

                is_json = response_format and response_format.get("type") == "json_object"
                return await parse_stream_response(response, silent=silent, clean_json=is_json)

            except (APIConnectionError, RateLimitError, asyncio.TimeoutError) as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                logger.warning(f"API错误 ({type(e).__name__})，{wait_time}秒后重试...")
                await asyncio.sleep(wait_time)

            except APIError as e:
                raise

async def parse_stream_response(
    stream,
    silent: bool = False,
    clean_json: bool = False
) -> Tuple[str, Dict[int, Dict[str, str]], Optional[str]]:
    """
    异步迭代流式响应

    Args:
        silent: 为 True 时不调用 agent_output 输出。
    """
    tool_calls = {}
    content_parts = []
    finish_reason = None

    async for chunk in stream:
        delta = chunk.choices[0].delta

        # 处理文本内容
        if delta.content:
            if not (delta.tool_calls and delta.content.isspace()):
                content_parts.append(delta.content)
                if not silent:
                    await agent_output(delta.content)

        # 处理工具调用
        if delta.tool_calls:
            for tool_chunk in delta.tool_calls:
                idx = tool_chunk.index
                if idx not in tool_calls:
                    tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                if tool_chunk.id:
                    tool_calls[idx]["id"] = tool_chunk.id
                if tool_chunk.function.name:
                    tool_calls[idx]["name"] += tool_chunk.function.name
                if tool_chunk.function.arguments:
                    tool_calls[idx]["arguments"] += tool_chunk.function.arguments

        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

    content = "".join(content_parts)
    if clean_json:
        content = extract_json(content)
    return content, tool_calls, finish_reason
