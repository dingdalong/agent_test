"""OutputGuardrail — 输出安全检查。"""

from __future__ import annotations

from typing import Tuple


class OutputGuardrail:
    """输出安全检查"""

    def __init__(self, blocked_content: list[str] | None = None):
        self.blocked_content = blocked_content or [
            "rm -rf",
            "DROP TABLE",
            "eval(",
        ]

    def check(self, output: str) -> Tuple[bool, str]:
        for phrase in self.blocked_content:
            if phrase in output:
                return False, f"输出包含不安全内容：{phrase}"
        return True, ""
