"""Flow 注册表：关键词触发 → 实例化对应的 StateMachine flow。"""

import logging
from typing import Callable, Dict, Optional

from statemachine import StateMachine

logger = logging.getLogger(__name__)

_FLOW_REGISTRY: Dict[str, Callable[..., StateMachine]] = {}


def register_flow(trigger: str, factory: Callable[..., StateMachine]):
    """注册一个关键词触发的 flow。

    Args:
        trigger: 触发关键词（如 "/book"），会转小写匹配
        factory: 工厂函数，接收 **kwargs 返回 StateMachine 实例
    """
    _FLOW_REGISTRY[trigger.lower()] = factory
    logger.debug(f"注册 flow: {trigger}")


def detect_flow(user_input: str, **kwargs) -> Optional[StateMachine]:
    """根据用户输入的关键词前缀匹配已注册的 flow。

    Returns:
        匹配到的 flow 实例，或 None
    """
    text = user_input.strip().lower()
    for trigger, factory in _FLOW_REGISTRY.items():
        if text.startswith(trigger):
            logger.info(f"匹配到 flow: {trigger}")
            return factory(**kwargs)
    return None


# 导入各 flow 模块以触发注册（放在最后避免循环导入）
from src.flows import meeting_booking  # noqa: E402, F401
