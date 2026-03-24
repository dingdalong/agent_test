import asyncio
from typing import Callable


def _default_output(text: str) -> None:
    """默认控制台输出"""
    print(text, end="", flush=True)


async def _default_input(prompt: str) -> str:
    """默认控制台输入"""
    return await asyncio.to_thread(input, prompt)


# 全局 IO 函数
_output_func: Callable = _default_output
_input_func: Callable = _default_input


def set_output(func: Callable) -> None:
    """设置全局输出函数（如切换到 socket 输出）"""
    global _output_func
    _output_func = func


def set_input(func: Callable) -> None:
    """设置全局输入函数（如切换到 socket 输入）"""
    global _input_func
    _input_func = func


async def agent_output(text: str) -> None:
    """统一输出，支持同步/异步回调"""
    if asyncio.iscoroutinefunction(_output_func):
        await _output_func(text)
    else:
        _output_func(text)


async def agent_input(prompt: str) -> str:
    """统一输入，支持同步/异步回调"""
    if asyncio.iscoroutinefunction(_input_func):
        return await _input_func(prompt)
    else:
        return await asyncio.to_thread(_input_func, prompt)
