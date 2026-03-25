# MCP Client Support Design Spec

## Overview

为 AI Agent 添加 MCP (Model Context Protocol) 客户端支持，使 Agent 能够连接外部 MCP Server，自动发现并使用它们提供的工具。MCP 工具与现有本地工具统一管理，对模型透明。

## Goals

- Agent 作为 MCP Client，连接一个或多个 MCP Server
- 支持 stdio 和 SSE (HTTP) 两种传输方式
- MCP 工具自动发现后统一注册到现有工具系统
- 通过命名空间前缀 `mcp_{server}_{tool}` 避免与本地工具冲突
- 单个 Server 故障不影响整体可用性

## Non-Goals

- 不实现 MCP Server 端（不把本地工具暴露给外部）
- 不实现 MCP Resources 和 Prompts 功能（只实现 Tools）
- 不实现 Server 热重载（需重启 Agent 更新配置）

## Configuration

### 配置文件

`mcp_servers.json`（项目根目录），格式对齐 Claude Desktop：

```json
{
  "mcpServers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/dingdalong/documents"],
      "env": {}
    },
    "my-remote-api": {
      "transport": "sse",
      "url": "http://localhost:8080/sse"
    }
  }
}
```

### 配置数据结构

`src/mcp/config.py`:

```python
@dataclass
class MCPServerConfig:
    name: str               # Server 名称（配置文件的 key）
    transport: str          # "stdio" | "sse"
    command: str | None     # stdio 模式：启动命令
    args: list[str]         # stdio 模式：命令参数
    env: dict[str, str]     # 环境变量
    url: str | None         # sse 模式：Server URL
    enabled: bool = True    # 可通过配置禁用
```

`load_mcp_config(path)` 函数读取 JSON 文件，返回 `list[MCPServerConfig]`。配置文件不存在时返回空列表（不报错）。

## Architecture

### 新增文件

```
src/mcp/
├── __init__.py
├── config.py       # MCPServerConfig + load_mcp_config()
└── manager.py      # MCPManager
```

### MCPManager

`src/mcp/manager.py` — 管理所有 MCP Server 连接和工具路由。

```python
class MCPManager:
    async def connect_all(configs: list[MCPServerConfig])
        # 逐个连接 Server，失败跳过并记录警告

    async def disconnect_all()
        # 关闭所有 session，终止子进程/HTTP 连接

    def get_tools_schemas() -> list[dict]
        # 返回所有 MCP 工具的 OpenAI 格式 schema
        # 工具名格式：mcp_{server名}_{原工具名}

    async def call_tool(tool_name: str, arguments: dict) -> str
        # 解析 tool_name 前缀，路由到对应 Server 的 session
        # 调用 session.call_tool(原工具名, arguments)
        # 返回结果字符串
```

**内部状态**：
- `_sessions: dict[str, ClientSession]` — Server 名 → MCP session
- `_tool_map: dict[str, tuple[str, str]]` — 全名 → (server名, 原工具名)
- `_contexts: list` — 保持 stdio/sse 上下文管理器的引用，用于清理

### 工具命名规则

MCP 工具统一加前缀 `mcp_{server名}_{原工具名}`：
- `filesystem` Server 的 `read_file` → `mcp_filesystem_read_file`
- `my-remote-api` Server 的 `query` → `mcp_my_remote_api_query`

Server 名中的 `-` 转为 `_`，确保是合法的函数名。

### 工具 Schema 转换

MCP SDK 返回的工具定义格式：
```python
Tool(name="read_file", description="...", inputSchema={...})
```

转换为 OpenAI 格式：
```python
{
    "type": "function",
    "function": {
        "name": "mcp_filesystem_read_file",
        "description": "[filesystem] ...",
        "parameters": inputSchema
    }
}
```

description 前加 `[server名]` 标签，帮助模型理解工具来源。

## Integration with Existing System

### ToolExecutor 改动

`src/tools/tool_executor.py` 增加 MCP 路由：

```python
class ToolExecutor:
    def __init__(self, registry, mcp_manager=None):
        self.registry = registry
        self.mcp_manager = mcp_manager

    async def execute(self, tool_name, arguments, skip_confirm=False):
        if tool_name.startswith("mcp_") and self.mcp_manager:
            return await self.mcp_manager.call_tool(tool_name, arguments)
        # 原有本地工具执行逻辑...
```

### main.py 启动流程改动

```python
async def main():
    # 1. 初始化 MCP
    mcp_manager = MCPManager()
    await mcp_manager.connect_all(load_mcp_config("mcp_servers.json"))

    # 2. 合并工具列表
    all_tools = tools + mcp_manager.get_tools_schemas()

    # 3. 创建 ToolExecutor（传入 mcp_manager）
    tool_exec = ToolExecutor(_TOOL_REGISTRY, mcp_manager=mcp_manager)

    # 4. 对话循环（使用 all_tools 和 tool_exec）
    ...

    # 5. 退出时清理
    await mcp_manager.disconnect_all()
```

### config.py 改动

增加一个常量：
```python
MCP_CONFIG_PATH = os.getenv("MCP_CONFIG_PATH", "mcp_servers.json")
```

### 不改动的部分

- `@tool` 装饰器和本地工具注册逻辑
- ChatFlow / PlanningFlow（只依赖 tools schema 列表和 tool_executor 接口）
- 记忆系统
- FSM 框架
- 护栏系统

## Lifecycle

### 启动

```
main() 启动
  → load_mcp_config() 读取配置
  → MCPManager.connect_all()
    → 对每个 Server:
      → stdio: 启动子进程，建立 ClientSession
      → sse: 连接 HTTP endpoint，建立 ClientSession
      → session.initialize() 握手
      → session.list_tools() 发现工具
      → 注册到 _tool_map
    → 失败的 Server 跳过，打印警告
  → 合并工具列表
  → 进入对话循环
```

### 退出

```
用户输入 exit
  → MCPManager.disconnect_all()
    → 关闭所有 session
    → stdio: 终止子进程
    → sse: 关闭 HTTP 连接
  → 程序退出
```

## Error Handling

| 场景 | 处理方式 |
|------|----------|
| 配置文件不存在 | 返回空列表，只用本地工具 |
| Server 启动失败（命令不存在等） | 跳过该 Server，记录警告，其余正常 |
| Server 运行中崩溃 | 工具调用时捕获异常，返回错误字符串给模型 |
| 工具调用超时 | 30 秒超时，返回超时错误字符串 |
| 工具参数不合法 | 将 MCP Server 返回的错误直接传给模型 |

## Dependencies

- `mcp` — Anthropic 官方 MCP Python SDK（提供 ClientSession、stdio/sse transport）

## Testing

- 使用 `@modelcontextprotocol/server-filesystem` 作为测试 Server
- 验证：连接、工具发现、工具调用、错误处理、退出清理
