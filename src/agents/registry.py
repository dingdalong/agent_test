"""AgentRegistry — Agent 注册表。

支持从 CategoryResolver 懒加载：当请求的 Agent 不在注册表中、但能被
CategoryResolver 解析时，自动创建并缓存。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from src.agents.agent import Agent

if TYPE_CHECKING:
    from src.tools.categories import CategoryResolver


class AgentRegistry:
    """管理所有已注册的 Agent。

    可选地通过 :meth:`set_category_resolver` 关联一个
    :class:`CategoryResolver`，使 :meth:`get` 在本地缓存未命中时
    尝试从分类配置中懒加载 Agent。
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        self._category_resolver: CategoryResolver | None = None

    # -- resolver 配置 -------------------------------------------------------

    def set_category_resolver(self, resolver: CategoryResolver) -> None:
        """设置 CategoryResolver，启用懒加载。"""
        self._category_resolver = resolver

    # -- CRUD ----------------------------------------------------------------

    def register(self, agent: Agent) -> None:
        """注册一个 Agent（同名覆盖）。"""
        self._agents[agent.name] = agent

    def get(self, name: str) -> Optional[Agent]:
        """根据名称获取 Agent。

        查找顺序：
        1. 本地缓存 ``_agents``
        2. 若设置了 CategoryResolver 且能解析该名称，则创建 Agent 并缓存
        3. 都未命中则返回 None
        """
        agent = self._agents.get(name)
        if agent is not None:
            return agent

        # 尝试从 CategoryResolver 懒加载
        if self._category_resolver and self._category_resolver.can_resolve(name):
            cat = self._category_resolver.get_category(name)

            # 收集其他分类的 delegate 工具名和摘要
            delegate_names = self._category_resolver.get_delegate_names(exclude=name)
            delegate_summaries = [
                s for s in self._category_resolver.get_all_summaries()
                if s["name"] != name
            ]

            instructions = self._category_resolver.build_instructions(
                name, delegate_summaries=delegate_summaries,
            )

            tools = list(cat["tools"].keys()) + delegate_names  # type: ignore[index]

            agent = Agent(
                name=name,
                description=cat["description"],  # type: ignore[index]
                instructions=instructions,
                tools=tools,
                handoffs=[],
            )
            self.register(agent)
            return agent

        return None

    def has(self, name: str) -> bool:
        """检查 Agent 是否已注册（不触发懒加载）。"""
        return name in self._agents

    def all_agents(self) -> list[Agent]:
        """返回所有已注册的 Agent 列表。"""
        return list(self._agents.values())
