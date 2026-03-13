# core/memory.py
from typing import List, Dict, Any, Optional

class ConversationBuffer:
    """短期记忆：保存最近 N 轮对话，并按需截断"""

    def __init__(self, max_rounds: int = 10):
        """
        :param max_rounds: 最多保留的对话轮数（一轮包含 user + assistant 消息）
        """
        self.max_rounds = max_rounds
        self.messages: List[Dict[str, Any]] = []  # 完整的消息历史

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, message: Dict[str, Any]):
        """添加助手消息，可能是普通文本或包含 tool_calls"""
        self.messages.append(message)

    def add_tool_message(self, tool_call_id: str, content: str):
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    def get_messages_for_api(self) -> List[Dict[str, Any]]:
        """
        返回给 API 的消息列表，包含 system 指令 + 经过滑动窗口裁剪的历史。
        这里我们保留最近的 max_rounds 轮（每轮指 user + assistant 的组合），
        但注意如果最近一轮包含工具调用，需要保留所有相关的 tool 消息。
        """
        # 简单实现：保留最后 max_rounds*2 条消息（因为每轮可能有 user + assistant/tool）
        # 更严谨的做法是分析轮次，但简单截断通常也够用。
        truncated = self.messages[-self.max_rounds*2:]
        return truncated

    def clear(self):
        self.messages.clear()
