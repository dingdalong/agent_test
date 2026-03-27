# 配置详解

项目使用两个配置文件：`config.yaml`（用户配置）和 `.env`（密钥）。

## config.yaml

### LLM 配置

```yaml
llm:
  provider: deepseek          # 提供商标识（目前仅用于注释，实际通过 base_url 区分）
  model: deepseek-chat         # 模型名称
  base_url: https://api.deepseek.com/v1  # API 端点
  # concurrency: 5             # 并发请求上限（默认 5）
  # max_retries: 3             # 失败重试次数（默认 3）
```

### Embedding 配置

```yaml
embedding:
  provider: ollama
  model: qwen3-embedding:0.6b
  base_url: http://127.0.0.1:11434
```

Embedding 用于记忆系统的向量化。默认使用本地 Ollama 服务。

### 记忆配置

```yaml
memory:
  provider: chroma             # 存储后端（目前仅支持 chroma）
  path: ./chroma_data          # ChromaDB 持久化目录
```

### 用户配置

```yaml
user:
  id: user_001                 # 用户标识，用于记忆隔离
```

### 高级参数（可选）

```yaml
# 规划系统
plan:
  max_adjustments: 3           # 计划调整最大次数
  max_clarification_rounds: 3  # 澄清对话最大轮次

# 智能体
agents:
  max_handoffs: 3              # handoff 最大深度
  max_tool_rounds: 3           # 工具调用最大轮次

# 工具
tools:
  max_output_length: 2000      # 工具输出最大长度（字符）

# MCP
mcp:
  config_path: mcp_servers.json  # MCP 服务器配置文件路径

# 技能
skills:
  dirs:                        # 技能搜索目录
    - skills/
    - .agents/skills/
```

## .env 环境变量

| 变量 | 用途 | 示例 |
|------|------|------|
| `OPENAI_API_KEY` | LLM API 密钥 | `sk-...` |
| `OPENAI_BASE_URL` | LLM API 端点（备用，优先使用 config.yaml） | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | 模型名称（备用） | `gpt-4` |
| `OPENAI_MODEL_EMBEDDING` | Embedding 模型名称（备用） | `text-embedding-3-small` |
| `OPENAI_MODEL_EMBEDDING_URL` | Embedding API 端点（备用） | `https://api.openai.com/v1` |
| `USER_ID` | 用户标识（备用） | `user_001` |

`.env` 中的值仅在 `config.yaml` 对应字段为空时生效。

## 切换 LLM 提供商

项目使用 OpenAI 兼容接口，任何支持该接口的提供商都可以使用：

### DeepSeek（默认）

```yaml
llm:
  provider: deepseek
  model: deepseek-chat
  base_url: https://api.deepseek.com/v1
```

### OpenAI

```yaml
llm:
  provider: openai
  model: gpt-4o
  base_url: https://api.openai.com/v1
```

### 本地模型（通过 Ollama）

```yaml
llm:
  provider: ollama
  model: qwen2.5:7b
  base_url: http://127.0.0.1:11434/v1
```

注意：本地模型需要支持 function calling 才能使用规划系统和事实提取功能。
