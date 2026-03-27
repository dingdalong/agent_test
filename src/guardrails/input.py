"""InputGuardrail — 输入安全检查。"""

from __future__ import annotations

import re
from typing import Tuple


class InputGuardrail:
    """输入安全检查（关键词+正则）"""

    def __init__(self, blocked_patterns: list[str] | None = None):
        self.blocked_patterns = blocked_patterns or [
            r"忽略.*指令|忽略.*系统提示",
            r"删除.*文件|rm\s+-rf",
            r"DROP\s+TABLE",
            r"eval\s*\(",
            r"exec\s*\(",
        ]

    def check(self, user_input: str) -> Tuple[bool, str]:
        """返回 (是否通过, 拒绝理由)"""
        for pattern in self.blocked_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return False, f"输入包含不安全内容（匹配模式：{pattern}）"
        return True, ""
