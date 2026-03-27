# 快速入门

## 环境准备

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | 3.13+ | 运行环境 |
| [uv](https://docs.astral.sh/uv/) | 最新 | 包管理 |
| [Ollama](https://ollama.com/) | 最新 | 本地 embedding 服务 |

## 安装

### 1. 克隆项目

```bash
git clone <repo-url>
cd aitest
```

### 2. 安装依赖

```bash
uv sync
```

### 3. 启动 Ollama 并拉取 embedding 模型

```bash
# 启动 Ollama 服务（如果还没有运行）
ollama serve

# 拉取 embedding 模型
ollama pull qwen3-embedding:0.6b
```

### 4. 配置 API Key

创建 `.env` 文件：

```bash
OPENAI_API_KEY=your-api-key-here
```

项目默认使用 DeepSeek API。如需切换，参见 [配置详解](configuration.md)。

### 5. 运行

```bash
uv run python main.py
```

## 示例对话

```
Agent 已启动，输入 'exit' 退出。

你: 帮我算一下 123 * 456
助手: 123 × 456 = 56,088

你: 今天天气怎么样？
助手: [orchestrator 会将请求转交给 weather_agent 处理]

你: /plan 查询北京天气然后发邮件告诉同事
助手: [启动规划流程：澄清 → 生成计划 → 确认 → 执行]

你: exit
```

## 常见问题

### Ollama 连接失败

```
ConnectionError: Cannot connect to http://127.0.0.1:11434
```

确保 Ollama 服务已启动：`ollama serve`

### API Key 无效

```
AuthenticationError: Invalid API key
```

检查 `.env` 文件中的 `OPENAI_API_KEY` 是否正确。DeepSeek 的 API Key 从 [DeepSeek Platform](https://platform.deepseek.com/) 获取。

### 模型不支持

如果使用的 LLM 不支持 function calling（工具调用），规划系统和事实提取器将无法工作。推荐使用 DeepSeek-Chat 或 OpenAI GPT-4 系列。
