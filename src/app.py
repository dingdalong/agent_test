"""AgentApp — 应用核心，组装所有组件并处理用户消息。"""

from __future__ import annotations

from pathlib import Path

from src.interfaces.base import UserInterface
from src.tools import (
    get_registry,
    discover_tools,
    ToolExecutor,
    ToolRouter,
    LocalToolProvider,
    sensitive_confirm_middleware,
    truncate_middleware,
    error_handler_middleware,
)
from src.mcp.provider import MCPToolProvider
from src.mcp.config import load_mcp_config
from src.mcp.manager import MCPManager
from src.skills.provider import SkillToolProvider
from src.skills import SkillManager
from src.core.guardrails import InputGuardrail
from src.agents import AgentRegistry, GraphEngine, RunContext, DictState
from src.agents.deps import AgentDeps
from src.agents.definitions import build_default_graph, build_skill_graph
from src.plan.flow import PlanFlow
from config import MCP_CONFIG_PATH, SKILLS_DIRS


class AgentApp:
    """应用核心：初始化组件、处理消息、管理生命周期。"""

    def __init__(self, ui: UserInterface):
        self.ui = ui
        self.guardrail = InputGuardrail()

    async def setup(self) -> None:
        """初始化所有组件：工具、MCP、Skills、Agent、图引擎。"""
        # 1. 发现并注册本地工具
        discover_tools("src.tools.builtin", Path("src/tools/builtin"))

        # 2. 构建本地工具执行管道
        registry = get_registry()
        executor = ToolExecutor(registry)
        middlewares = [
            error_handler_middleware(),
            sensitive_confirm_middleware(registry, self.ui),
            truncate_middleware(2000),
        ]
        local_provider = LocalToolProvider(registry, executor, middlewares)

        # 3. 构建路由器
        self.router = ToolRouter()
        self.router.add_provider(local_provider)

        # 4. 初始化 MCP
        self.mcp_manager = MCPManager()
        await self.mcp_manager.connect_all(load_mcp_config(MCP_CONFIG_PATH))
        mcp_schemas = self.mcp_manager.get_tools_schemas()
        if mcp_schemas:
            self.router.add_provider(MCPToolProvider(self.mcp_manager))

        # 5. 初始化 Skills
        self.skill_manager = SkillManager(skill_dirs=SKILLS_DIRS)
        await self.skill_manager.discover()
        skill_count = len(self.skill_manager._skills)
        if skill_count:
            self.router.add_provider(SkillToolProvider(self.skill_manager))

        # 6. 构建 agent 注册表、图、引擎
        self.agent_registry = AgentRegistry()
        self.graph = build_default_graph(self.agent_registry)
        self.engine = GraphEngine(registry=self.agent_registry)

        # 7. 显示启动信息
        await self.ui.display("Agent 已启动，输入 'exit' 退出。\n")
        if mcp_schemas:
            await self.ui.display(f"已加载 {len(mcp_schemas)} 个 MCP 工具\n")
        if skill_count:
            await self.ui.display(f"已发现 {skill_count} 个 Skill\n")

    async def process(self, user_input: str) -> None:
        """处理单条用户消息：护栏 → /plan → /skill → 正常执行。"""
        # 1. 护栏检查
        passed, reason = self.guardrail.check(user_input)
        if not passed:
            await self.ui.display(f"\n[安全拦截] {reason}\n")
            return

        # 2. /plan 命令
        if user_input.strip().startswith("/plan"):
            plan_request = user_input.strip()[5:].strip()
            if not plan_request:
                await self.ui.display("\n请在 /plan 后输入你的请求，例如：/plan 查询广州天气并发邮件给同事\n")
                return
            plan_flow = PlanFlow(
                tool_router=self.router,
                agent_registry=self.agent_registry,
                engine=self.engine,
                ui=self.ui,
            )
            result = await plan_flow.run(plan_request)
            await self.ui.display(f"\n{result}\n")
            return

        # 3. Skill 斜杠命令
        skill_name = self.skill_manager.is_slash_command(user_input)
        if skill_name:
            skill_content = self.skill_manager.activate(skill_name)
            if skill_content:
                remaining = user_input[len(f"/{skill_name}"):].strip()
                actual_input = remaining or f"已激活 {skill_name} skill，请按指令执行。"
                skill_registry = AgentRegistry()
                skill_graph = build_skill_graph(skill_registry, skill_content)
                skill_engine = GraphEngine(registry=skill_registry)
                ctx = RunContext(
                    input=actual_input,
                    state=DictState(),
                    deps=AgentDeps(
                        tool_router=self.router,
                        agent_registry=skill_registry,
                        graph_engine=skill_engine,
                        ui=self.ui,
                    ),
                )
                result = await skill_engine.run(skill_graph, ctx)
                await self.ui.display(f"\n{result.output}\n")
                return

        # 4. 正常执行
        ctx = RunContext(
            input=user_input,
            state=DictState(),
            deps=AgentDeps(
                tool_router=self.router,
                agent_registry=self.agent_registry,
                graph_engine=self.engine,
                ui=self.ui,
            ),
        )
        result = await self.engine.run(self.graph, ctx)

        output = result.output
        if isinstance(output, dict):
            output = output.get("text", str(output))
        await self.ui.display(f"\n{output}\n")

    async def run(self) -> None:
        """CLI 主循环。Web 接入时直接调用 process() 而非 run()。"""
        await self.ui.display("欢迎！输入 'exit' 退出。\n")
        while True:
            user_input = await self.ui.prompt("\n你: ")
            if user_input.strip().lower() in ("exit", "quit"):
                break
            await self.process(user_input)

    async def shutdown(self) -> None:
        """清理资源。"""
        await self.mcp_manager.disconnect_all()
