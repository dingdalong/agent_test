"""应用组装 — 读配置、创建组件、注入依赖。"""

from __future__ import annotations

from pathlib import Path

from src.config import load_config
from src.llm.openai import OpenAIProvider
from src.interfaces.cli import CLIInterface
from src.tools.decorator import get_registry
from src.tools.discovery import discover_tools
from src.tools.executor import ToolExecutor
from src.tools.router import ToolRouter, LocalToolProvider
from src.tools.middleware import (
    error_handler_middleware,
    sensitive_confirm_middleware,
    truncate_middleware,
)
from src.mcp.config import load_mcp_config
from src.mcp.manager import MCPManager
from src.mcp.provider import MCPToolProvider
from src.skills.manager import SkillManager
from src.skills.provider import SkillToolProvider
from src.guardrails import InputGuardrail
from src.agents import AgentRegistry, AgentRunner
from src.graph import GraphEngine
from src.agents.deps import AgentDeps
from src.app.presets import build_default_graph
from src.app.app import AgentApp


async def create_app(config_path: str = "config.yaml") -> AgentApp:
    """读配置 → 创建所有组件 → 注入依赖 → 返回 AgentApp。"""
    raw = load_config(config_path)
    llm_cfg = raw.get("llm", {})
    ui = CLIInterface()

    # 1. LLM
    llm = OpenAIProvider(
        api_key=llm_cfg.get("api_key", ""),
        base_url=llm_cfg.get("base_url", ""),
        model=llm_cfg.get("model", ""),
        concurrency=llm_cfg.get("concurrency", 5),
        max_retries=llm_cfg.get("max_retries", 3),
        on_chunk=ui.display,
    )

    # 2. Tools
    discover_tools("src.tools.builtin", Path("src/tools/builtin"))
    registry = get_registry()
    executor = ToolExecutor(registry)
    middlewares = [
        error_handler_middleware(),
        sensitive_confirm_middleware(registry, ui),
        truncate_middleware(raw.get("tools", {}).get("max_output_length", 2000)),
    ]
    tool_router = ToolRouter()
    tool_router.add_provider(LocalToolProvider(registry, executor, middlewares))

    # 3. MCP
    mcp_config_path = raw.get("mcp", {}).get("config_path", "mcp_servers.json")
    mcp_manager = MCPManager()
    await mcp_manager.connect_all(load_mcp_config(mcp_config_path))
    if mcp_manager.get_tools_schemas():
        tool_router.add_provider(MCPToolProvider(mcp_manager))

    # 4. Skills
    skill_dirs = raw.get("skills", {}).get("dirs", ["skills/", ".agents/skills/"])
    skill_manager = SkillManager(skill_dirs=skill_dirs)
    await skill_manager.discover()
    if skill_manager._skills:
        tool_router.add_provider(SkillToolProvider(skill_manager))

    # 5. Agents
    agent_cfg = raw.get("agents", {})
    agent_registry = AgentRegistry()
    runner = AgentRunner(
        registry=agent_registry,
        max_tool_rounds=agent_cfg.get("max_tool_rounds", 10),
    )
    graph = build_default_graph(agent_registry, runner=runner)
    engine = GraphEngine(max_handoff_depth=agent_cfg.get("max_handoffs", 10))

    # 6. Deps
    deps = AgentDeps(
        llm=llm,
        tool_router=tool_router,
        agent_registry=agent_registry,
        graph_engine=engine,
        ui=ui,
    )

    return AgentApp(
        deps=deps,
        ui=ui,
        guardrail=InputGuardrail(),
        tool_router=tool_router,
        agent_registry=agent_registry,
        engine=engine,
        graph=graph,
        skill_manager=skill_manager,
        mcp_manager=mcp_manager,
        runner=runner,
    )
