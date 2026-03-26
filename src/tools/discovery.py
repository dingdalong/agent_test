"""自动发现工具模块 — 扫描目录下的 .py 文件并导入。"""

import importlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def discover_tools(
    package: str,
    package_dir: Path,
    skip: set[str] | None = None,
) -> None:
    """扫描 package_dir 下的 .py 文件并导入，以触发 @tool 装饰器注册。

    Args:
        package: 包的完整名称（如 "src.tools.builtin"）
        package_dir: 包的目录路径
        skip: 要跳过的模块名集合，默认跳过 __init__
    """
    skip = skip or {"__init__"}
    for item in sorted(Path(package_dir).glob("*.py")):
        module_name = item.stem
        if module_name in skip:
            continue
        try:
            importlib.import_module(f".{module_name}", package=package)
            logger.debug(f"已发现工具模块: {module_name}")
        except Exception as e:
            logger.error(f"导入工具模块 {module_name} 失败: {e}")
