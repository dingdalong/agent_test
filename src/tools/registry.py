"""工具注册表 — 存储和查询工具定义。"""

import logging
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from .schemas import ToolDict

logger = logging.getLogger(__name__)


@dataclass
class ToolEntry:
    """工具的完整元数据"""
    name: str
    func: Callable
    model: type[BaseModel]
    description: str
    parameters_schema: dict[str, Any]
    sensitive: bool = False
    confirm_template: str | None = None


class ToolRegistry:
    """工具注册表 — 存储和查询工具定义"""

    def __init__(self):
        self._entries: dict[str, ToolEntry] = {}

    def register(self, entry: ToolEntry) -> None:
        if entry.name in self._entries:
            logger.warning(f"工具 '{entry.name}' 已注册，跳过")
            return
        self._entries[entry.name] = entry

    def get(self, name: str) -> ToolEntry | None:
        return self._entries.get(name)

    def has(self, name: str) -> bool:
        return name in self._entries

    def list_entries(self) -> list[ToolEntry]:
        return list(self._entries.values())

    def get_schemas(self) -> list[ToolDict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": entry.name,
                    "description": entry.description,
                    "parameters": entry.parameters_schema,
                },
            }
            for entry in self._entries.values()
        ]
