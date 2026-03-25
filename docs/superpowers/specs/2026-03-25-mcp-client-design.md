# MCP Client Support Design Spec

## Overview

为 AI Agent 添加 MCP (Model Context Protocol) 客户端支持，使 Agent 能够连接外部 MCP Server，自动发现并使用它们提供的工具。MCP 工具与现有本地工具统一管理，对模型透明。

## Goals

- Agent 作为 MCP Client，连接一个或多个 MCP Server
- 支持 stdio 和 Streamable HTTP 两种传输方式（SSE 已被 MCP 官方弃用）
- MCP 工具自动发现后统一注册到现有工具系统
- 通过命名空间前缀 `mcp_{server}_{tool}` 避免与本地工具冲突
- 单个 Server 故障不影响整体可用性

## Non-Goals

- 不实现 MCP Server 端（不把本地工具暴露给外部）
- 不实现 MCP Resources 和 Prompts 功能（只实现 Tools）
- 不实现 Server 热重载（需重启 Agent 更新配置）
- 不实现 Server 崩溃后自动重连

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
      "transport": "http",
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

### 配置数据结构

`src/mcp/config.py`:

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class MCPServerConfig:
    name: str                                       # Server 名称（配置文件的 key）
    transport: Literal["stdio", "http"]             # 传输方式
    command: str | None = None                      # stdio 模式：启动命令
    args: list[str] = field(default_factory=list)   # stdio 模式：命令参数
    env: dict[str, str] = field(default_factory=dict)  # 环境变量
    url: str | None = None                          # http 模式：Server URL
    enabled: bool = True                            # 可通过配置禁用
    timeout: float = 30.0                           # 工具调用超时（秒）
```

`load_mcp_config(path)` 函数读取 JSON 文件，返回 `list[MCPServerConfig]`。配置文件不存在时返回空列表（不报错）。加载时校验：stdio 必须有 `command`，http 必须有 `url`。

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
        # 通过 AsyncExitStack.aclose() 关闭所有连接

    def get_tools_schemas() -> list[dict]
        # 返回所有 MCP 工具的 OpenAI 格式 schema
        # 工具名格式：mcp_{server名}_{原工具名}

    async def call_tool(tool_name: str, arguments: dict) -> str
        # 解析 tool_name 前缀，路由到对应 Server 的 session
        # 调用 session.call_tool(原工具名, arguments)
        # 将 CallToolResult 转换为字符串返回
```

**内部状态**：
- `_exit_stack: AsyncExitStack` — 管理所有传输和 session 的生命周期
- `_sessions: dict[str, ClientSession]` — Server 名 → MCP session
- `_tool_map: dict[str, tuple[str, str]]` — 工具全名 → (server名, 原工具名)
- `_timeouts: dict[str, float]` — Server 名 → 超时时间

### 上下文管理器生命周期

MCP SDK 的 transport 和 ClientSession 都是 async context manager，不能直接存储。使用 `contextlib.AsyncExitStack` 管理：

```python
from contextlib import AsyncExitStack

class MCPManager:
    def __init__(self):
        self._exit_stack = AsyncExitStack()

    async def _connect_one(self, config: MCPServerConfig):
        if config.transport == "stdio":
            transport = stdio_client(server_params)
        else:
            transport = streamablehttp_client(url=config.url)

        read, write = await self._exit_stack.enter_async_context(transport)
        session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    async def disconnect_all(self):
        await self._exit_stack.aclose()
```

### 工具命名规则

MCP 工具统一加前缀 `mcp_{server名}_{原工具名}`：
- `filesystem` Server 的 `read_file` → `mcp_filesystem_read_file`
- `my-remote-api` Server 的 `query` → `mcp_my_remote_api_query`

Server 名中的 `-` 转为 `_`，确保是合法的函数名。

**冲突检测**：`connect_all` 中注册工具时，检查工具全名是否已存在于 `_tool_map`。如果两个 Server 产生同名工具，记录警告并跳过后注册的那个。

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

`list_tools()` 支持分页：循环调用直到 `nextCursor` 为 None，收集所有工具。

### CallToolResult 转换

`session.call_tool()` 返回 `CallToolResult`，转换规则：

```python
async def _convert_result(self, result: CallToolResult) -> str:
    # 1. 如果有错误标志
    if result.isError:
        texts = [c.text for c in result.content if hasattr(c, 'text')]
        return f"工具执行出错: {''.join(texts)}"

    # 2. 提取文本内容
    texts = []
    for item in result.content:
        if hasattr(item, 'text'):       # TextContent
            texts.append(item.text)
        elif hasattr(item, 'data'):     # ImageContent 等二进制内容
            texts.append(f"[{item.mimeType} content, {len(item.data)} bytes]")

    # 3. 截断过长结果
    output = '\n'.join(texts)
    if len(output) > 2000:
        output = output[:2000] + "...(结果已截断)"
    return output
```

## Integration with Existing System

### ToolExecutor 改动

`src/tools/tool_executor.py` 增加 `mcp_manager` 属性和 MCP 路由：

```python
class ToolExecutor:
    def __init__(self, registry, mcp_manager=None):
        self.registry = registry
        self.mcp_manager = mcp_manager    # 可在初始化后赋值

    async def execute(self, tool_name, arguments, skip_confirm=False):
        # MCP 工具路由
        if tool_name.startswith("mcp_") and self.mcp_manager:
            return await self.mcp_manager.call_tool(tool_name, arguments)
        # 原有本地工具执行逻辑（不变）...
```

MCP 工具不经过 Pydantic 验证和本地敏感工具确认流程（MCP Server 自身负责参数校验）。这是有意为之：MCP 协议的设计理念是 Server 自治。

### main.py 启动流程改动

保持 `tools` 和 `tool_executor` 为模块级单例，通过赋值方式注入 MCP：

```python
from src.tools import tools, tool_executor
from src.mcp.manager import MCPManager
from src.mcp.config import load_mcp_config
from config import MCP_CONFIG_PATH

async def main():
    # 1. 初始化 MCP
    mcp_manager = MCPManager()
    await mcp_manager.connect_all(load_mcp_config(MCP_CONFIG_PATH))

    # 2. 注入到现有 tool_executor
    tool_executor.mcp_manager = mcp_manager

    # 3. 合并工具列表（模块级 tools 列表 + MCP 工具）
    all_tools = tools + mcp_manager.get_tools_schemas()

    # 4. 对话循环
    print("Agent 已启动，输入 'exit' 退出。")
    while True:
        user_input = await agent_input("\n你: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        await handle_input(user_input, all_tools)
        await agent_output("\n")

    # 5. 退出时清理
    await mcp_manager.disconnect_all()
```

`handle_input` 需要接受 `all_tools` 参数（或通过模块级变量传递），以便 ChatFlow 和 PlanningFlow 使用合并后的工具列表。

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
    → 创建 AsyncExitStack
    → 对每个 Server:
      → stdio: 通过 exit_stack 进入 stdio_client context
      → http: 通过 exit_stack 进入 streamablehttp_client context
      → 通过 exit_stack 进入 ClientSession context
      → session.initialize() 握手
      → session.list_tools() 发现工具（分页）
      → 检查冲突后注册到 _tool_map
    → 失败的 Server 跳过，打印警告
  → tool_executor.mcp_manager = mcp_manager
  → 合并 tools + mcp tools
  → 进入对话循环
```

### 退出

```
用户输入 exit
  → MCPManager.disconnect_all()
    → AsyncExitStack.aclose()
    → 自动关闭所有 session 和 transport
  → 程序退出
```

## Error Handling

| 场景 | 处理方式 |
|------|----------|
| 配置文件不存在 | 返回空列表，只用本地工具 |
| 配置校验失败（如 stdio 缺 command） | 跳过该 Server，记录警告 |
| Server 启动失败（命令不存在等） | 跳过该 Server，记录警告，其余正常 |
| Server 运行中崩溃 | 工具调用时捕获异常，返回错误字符串给模型 |
| 工具调用超时 | 使用 Server 配置的 timeout（默认 30s），返回超时错误 |
| 工具参数不合法 | 将 MCP Server 返回的错误直接传给模型 |
| 工具名冲突 | 警告日志，后注册的工具被跳过 |
| 本地工具名以 `mcp_` 开头 | 启动时警告（但不阻止） |

## Dependencies

- `mcp` — Anthropic 官方 MCP Python SDK（提供 ClientSession、stdio/streamable_http transport）

## Testing

- 使用 `@modelcontextprotocol/server-filesystem` 作为 stdio 测试 Server
- 验证：连接、工具发现（含分页）、工具调用、CallToolResult 转换、错误处理、退出清理
- 验证：配置文件不存在时正常降级
- 验证：工具名冲突检测
