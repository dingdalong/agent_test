# 架构总览

## 设计哲学

- **从零构建** — 不依赖 LangChain、CrewAI 等框架，所有核心功能自研
- **端口-适配器架构** — 通过 Protocol 定义接口，具体实现可替换
- **集中组装** — 所有具体类的实例化集中在 `src/app/bootstrap.py`，业务代码只依赖 Protocol
- **异步优先** — 全链路 asyncio，支持并发执行

## 分层架构

```
Layer 3: src/app/          应用层 — 组装组件、REPL、消息路由
         ↓ 依赖所有层
Layer 2: src/memory/       记忆系统 — 向量存储、事实提取、衰减
         src/agents/       智能体 — 模型定义、运行器、注册表
         src/plan/         规划系统 — 计划生成、编译、执行
         src/mcp/          MCP 客户端 — 外部工具集成
         src/skills/       技能系统 — 发现、激活、注入
         ↓ 依赖 Layer 0-1
Layer 1: src/llm/          LLM 抽象层 — Provider、流式输出、结构化输出
         src/tools/        工具系统 — 注册、执行、中间件、路由
         ↓ 依赖 Layer 0
Layer 0: src/config.py     配置加载器
         src/utils/        工具函数
         src/interfaces/   用户交互协议
         src/graph/        图执行引擎
         src/guardrails/   输入安全守卫
         （无外部依赖）
```

**依赖规则：低层模块不得导入高层模块。**

## 请求处理流程

```
用户输入
  │
  ├─ 守卫检查（InputGuardrail）
  │   └─ 拦截危险输入 → 返回安全提示
  │
  ├─ /plan 命令 → PlanFlow
  │   └─ 澄清 → 生成 → 确认 → 编译 → GraphEngine 执行
  │
  ├─ /skill-name 命令 → SkillManager
  │   └─ 激活技能 → 构建独立图 → GraphEngine 执行
  │
  └─ 普通消息 → 默认图
      └─ orchestrator → 直接回答 / handoff 到专家智能体
```

## 核心接口（Protocol）

| Protocol | 文件 | 职责 |
|----------|------|------|
| `LLMProvider` | `src/llm/base.py` | LLM 调用抽象 |
| `MemoryProvider` | `src/memory/base.py` | 记忆存储抽象 |
| `ToolProvider` | `src/tools/router.py` | 工具来源抽象 |
| `GraphNode` | `src/graph/types.py` | 图节点抽象 |
| `UserInterface` | `src/interfaces/base.py` | 用户交互抽象 |

## 模块文档

- [LLM 层](llm.md)
- [智能体系统](agents.md)
- [记忆系统](memory.md)
- [工具系统](tools.md)
- [图引擎](graph.md)
- [规划系统](plan.md)
