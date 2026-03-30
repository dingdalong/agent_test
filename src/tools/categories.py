"""工具分类配置加载与校验。

从 tool_categories.json 中加载分类树，展平为叶子节点映射，
并校验分类的完整性与一致性。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_categories(config_path: str) -> dict[str, dict[str, Any]]:
    """加载 tool_categories.json，返回叶子类别映射。

    返回 dict: agent_name -> {"description": str, "tools": list[str]}
    agent_name 以 ``tool_`` 为前缀，嵌套子类别用下划线拼接，
    例如 ``tool_text_editing_code_editing``。

    如果配置文件不存在，返回空字典。
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning("分类配置 %s 不存在", config_path)
        return {}

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories", {})
    return _flatten_categories(categories, prefix="tool")


def _flatten_categories(
    categories: dict[str, Any], prefix: str
) -> dict[str, dict[str, Any]]:
    """递归展开分类树，只保留叶子节点（含 tools 字段的节点）。

    非叶子节点（含 subcategories）会继续递归，其自身不出现在结果中。
    """
    result: dict[str, dict[str, Any]] = {}
    for name, cat in categories.items():
        agent_name = f"{prefix}_{name}"
        if "subcategories" in cat:
            # 非叶子：递归处理子分类
            sub = _flatten_categories(cat["subcategories"], prefix=agent_name)
            result.update(sub)
        elif "tools" in cat:
            # 叶子节点
            result[agent_name] = {
                "description": cat["description"],
                "tools": list(cat["tools"]),
            }
            if "instructions" in cat:
                result[agent_name]["instructions"] = cat["instructions"]
    return result


def validate_categories(
    categories: dict[str, dict[str, Any]],
    all_tool_names: set[str],
) -> list[str]:
    """校验分类配置的完整性和一致性。

    校验规则：
    - 每个工具必须恰好出现在一个类别中（全覆盖、无重复）
    - 类别名（去掉 ``tool_`` 前缀后）必须是合法的 snake_case
    - description 不能为空
    - 引用的工具必须存在于 all_tool_names 中

    返回错误列表，空列表表示校验通过。
    """
    errors: list[str] = []
    seen_tools: dict[str, str] = {}  # tool_name -> category_name
    categorized_tools: set[str] = set()

    for cat_name, cat in categories.items():
        # 校验 description 非空
        if not cat.get("description"):
            errors.append(f"类别 {cat_name} 缺少 description")

        # 校验类别名 snake_case
        raw_name = cat_name.removeprefix("tool_")
        if not re.match(r"^[a-z][a-z0-9_]*$", raw_name):
            errors.append(f"类别名 {cat_name} 不合法（需要 snake_case）")

        # 校验工具：存在性与唯一性
        for tool_name in cat.get("tools", []):
            if tool_name in seen_tools:
                errors.append(
                    f"工具 {tool_name} 重复出现在 {seen_tools[tool_name]} 和 {cat_name}"
                )
            seen_tools[tool_name] = cat_name
            categorized_tools.add(tool_name)

            if tool_name not in all_tool_names:
                errors.append(f"工具 {tool_name} 不存在于当前已注册的工具中")

    # 校验全覆盖：每个已注册的工具都必须被分类
    missing = all_tool_names - categorized_tools
    for tool_name in sorted(missing):
        errors.append(f"工具 {tool_name} 未被分配到任何类别")

    return errors
