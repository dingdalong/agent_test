"""工具类型定义，供整个 tools 模块和外部使用。"""

from typing import Any, Dict, TypedDict


class ToolDict(TypedDict):
    """OpenAI function-calling 格式的工具 schema"""
    type: str
    function: Dict[str, Any]
