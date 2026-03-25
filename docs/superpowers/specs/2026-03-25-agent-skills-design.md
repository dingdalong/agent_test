# Agent Skills 支持设计文档

日期：2026-03-25

## 概述

为 AI Agent 实现 [Agent Skills 开放标准](https://agentskills.io/specification) 支持。Skill 是一个文件夹，包含 SKILL.md 元数据+指令文件，以及可选的脚本、参考文档和资源。Agent 在启动时发现所有可用 skill，通过渐进式披露（Progressive Disclosure）按需加载，通过 system prompt 注入引导 LLM 按特定流程工作。

### 与现有系统的关系

- **Tool**：单个函数调用能力。Skill 教 Agent *如何使用*这些工具完成特定任务。
- **Flow**：FSM 驱动的多步骤对话流程。Skill 更轻量，通过 system prompt 注入指令而非独立状态机。
- **Agent**：专家智能体（天气、日历等）。Skill 为 Agent 补充领域知识和工作流程。
- **MCP**：外部工具连接协议。Skill 与 MCP 互补：MCP 提供能力，Skill 教如何使用能力。

## Skill 格式

### 目录结构

遵循 Agent Skills 标准：

```
skill-name/
├── SKILL.md          # 必须：YAML frontmatter + Markdown 指令
├── scripts/          # 可选：可执行脚本
├── references/       # 可选：参考文档
├── assets/           # 可选：模板、资源文件
└── ...
```

### SKILL.md 格式

```markdown
---
name: code-review
description: 审查代码质量，检查 bug 和风格一致性。当用户要求代码审查、review 时使用。
---

## 使用场景
当用户要求审查代码时激活此 skill。

## 审查步骤
1. 阅读指定的代码文件
2. 检查常见 bug 模式
3. 检查代码风格一致性
4. 给出改进建议
```

### Frontmatter 字段

| 字段 | 必须 | 说明 |
|------|------|------|
| `name` | 是 | 1-64 字符，小写字母+数字+连字符，不可以连字符开头/结尾，不可包含连续连字符 |
| `description` | 是 | 1-1024 字符，描述 skill 做什么以及何时使用 |
| `license` | 否 | 许可证信息 |
| `compatibility` | 否 | 环境要求（1-500 字符） |
| `allowed-tools` | 否 | 空格分隔的预批准工具列表（实验性字段，本期仅解析存储，不强制执行） |
| `metadata` | 否 | 任意键值对 |

### 扫描位置

```
<project>/skills/          # 项目级 skill（主要）
<project>/.agents/skills/  # 跨客户端互操作（Agent Skills 标准约定）
```

> **V1 范围说明**：用户级目录（`~/.agents/skills/`）在本期暂不扫描，作为未来扩展项。

项目级 skill 优先于同名 skill。名称冲突时先扫描到的优先，记录警告日志。

### 扫描约束

- 跳过 `.git/`、`node_modules/`、`__pycache__/` 等目录
- 最大扫描深度：4 层
- 最大扫描目录数：2000

### 信任考量

> **V1 范围说明**：项目级 skill 来自当前仓库，可能不受信任。本期不实现信任检查，作为未来安全增强项记录。

## 核心模块

新增 `src/skills/` 包，包含三个模块。

### 1. `src/skills/models.py` — 数据模型

```python
@dataclass
class SkillInfo:
    """Skill 元数据，对应 SKILL.md frontmatter。"""
    name: str                          # kebab-case 名称（如 "code-review"）
    description: str                   # 描述（何时使用）
    location: Path                     # SKILL.md 文件的绝对路径
    body: str | None = None            # Markdown body（激活后加载）
    allowed_tools: str | None = None   # 预批准工具列表（实验性，仅存储）
```

> **`location` 语义**：始终指向 SKILL.md 文件本身。Skill 的基础目录通过 `location.parent` 派生。

### 2. `src/skills/parser.py` — SKILL.md 解析器

职责：查找和解析 SKILL.md 文件。

```python
def find_skill_md(skill_dir: Path) -> Path | None:
    """查找 SKILL.md 文件，优先大写，兼容小写。"""

def parse_frontmatter(content: str) -> tuple[dict, str]:
    """解析 YAML frontmatter，返回 (metadata_dict, body_str)。
    对异常 YAML（如未引用冒号）做宽容处理。"""

def read_skill_info(skill_dir: Path) -> SkillInfo:
    """解析 SKILL.md，返回 SkillInfo（仅元数据，body=None）。
    缺少必填字段时抛出 ValidationError。"""
```

宽容解析策略：
- name 不匹配目录名 → 警告但仍加载
- name 超过 64 字符 → 警告但仍加载
- description 为空 → 跳过该 skill
- YAML 完全无法解析 → 跳过该 skill

### 3. `src/skills/manager.py` — Skill 管理器

职责：发现、披露、激活 skill。核心类。

```python
class SkillManager:
    def __init__(self, skill_dirs: list[str]):
        self._skill_dirs = skill_dirs
        self._skills: dict[str, SkillInfo] = {}  # name -> SkillInfo
        self._activated: set[str] = set()         # 已激活 skill 名（去重）

    async def discover(self) -> None:
        """扫描所有 skill 目录，解析 frontmatter，构建 catalog。
        渐进式披露 Tier 1：仅加载 name + description (~50-100 tokens/skill)。
        跳过无效目录，记录警告日志。"""

    def get_catalog_prompt(self) -> str:
        """生成 <available_skills> XML + 行为指令，供注入 system prompt。
        每个 skill 包含 name、description、location（SKILL.md 路径）。
        无 skill 时返回空字符串。"""

    def activate(self, skill_name: str) -> str | None:
        """渐进式披露 Tier 2：读取完整 SKILL.md body。
        返回 <skill_content> 包裹的内容 + <skill_resources> 列表。
        记录已激活状态，防止重复注入。"""

    def list_resources(self, skill_name: str) -> list[str]:
        """渐进式披露 Tier 3：列出 skill 目录下的辅助文件路径。"""

    def build_activate_tool_schema(self) -> dict | None:
        """生成 activate_skill 工具的 function calling schema。
        name 参数使用 enum 限制为已发现的 skill 名。
        无 skill 时返回 None（不注册工具）。"""

    def is_slash_command(self, user_input: str) -> str | None:
        """检查输入是否为 /skill-name 格式。
        返回匹配的 skill name 或 None。"""

    def get_skill(self, name: str) -> SkillInfo | None:
        """根据名称获取 SkillInfo。"""
```

## 集成方式

### main.py 启动阶段

```python
async def main():
    # 初始化 MCP（已有）
    mcp_manager = MCPManager()
    await mcp_manager.connect_all(load_mcp_config(MCP_CONFIG_PATH))
    tool_executor.mcp_manager = mcp_manager

    # 初始化 Skills（新增）
    skill_manager = SkillManager(skill_dirs=["skills/", ".agents/skills/"])
    await skill_manager.discover()
    tool_executor.skill_manager = skill_manager  # 注入

    # 合并工具列表
    all_tools = tools + mcp_schemas
    activate_schema = skill_manager.build_activate_tool_schema()
    if activate_schema:
        all_tools.append(activate_schema)
```

### handle_input 路由阶段

在护栏检查之后、Flow 检测之前插入 skill 斜杠命令检测。通过 `tool_executor.skill_manager` 访问（与 MCP 一致的注入模式），不新增函数参数：

```python
async def handle_input(user_input: str, all_tools=None):
    effective_tools = all_tools or tools
    skill_manager = tool_executor.skill_manager  # 通过已注入的属性访问

    # 1. 护栏检查（已有）
    # 2. Skill 斜杠命令检测（新增）
    if skill_manager:
        skill_name = skill_manager.is_slash_command(user_input)
        if skill_name:
            skill_content = skill_manager.activate(skill_name)
            if skill_content:
                # 提取斜杠命令后的用户输入作为任务描述
                remaining_input = user_input[len(f"/{skill_name}"):].strip()
                # 注入 skill 后进入 MultiAgentFlow
                # skill_content 作为额外 system prompt 传入
                ...
    # 3. 关键词触发的特殊 Flow（已有，/plan 等保留命令优先）
    # 4. 复杂请求 → PlanningFlow（已有）
    # 5. 普通对话 → MultiAgentFlow（已有）
```

> **斜杠命令优先级**：`/plan` 等已有保留命令优先于 skill 斜杠命令。`is_slash_command()` 应排除保留前缀（`plan`、`book`）。

### MultiAgentFlow 集成

**System prompt 追加 skill catalog**：在 `on_enter_orchestrating` 构建 system prompt 时追加：

```python
# 在构建 system_prompt 之后
skill_manager = model.tool_executor.skill_manager
if skill_manager:
    catalog = skill_manager.get_catalog_prompt()
    if catalog:
        system_prompt += "\n\n" + catalog
```

**Orchestrator 工具列表追加 activate_skill**：当前 orchestrator 仅传 `[transfer_schema]` 给 `call_model`。需要追加 `activate_skill`：

```python
orchestrator_tools = [transfer_schema]
if skill_manager:
    activate_schema = skill_manager.build_activate_tool_schema()
    if activate_schema:
        orchestrator_tools.append(activate_schema)

content, tool_calls, _ = await call_model(model.messages, tools=orchestrator_tools)
```

**处理 activate_skill 调用**：在 `on_enter_orchestrating` 的 tool_calls 处理中，除了 `transfer_to_agent`，还需处理 `activate_skill`：

```python
for tc in tool_calls.values():
    if tc.get("name") == "activate_skill":
        args = json.loads(tc["arguments"])
        skill_content = skill_manager.activate(args["name"])
        # 将 skill 内容作为 tool result 返回给 LLM
        model.messages.append({"role": "assistant", ...})
        model.messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": skill_content or "Skill not found.",
        })
        # 继续在 orchestrating 状态中重新调用 LLM
```

### ToolExecutor 路由

在 `execute()` 中添加 skill 工具路由（类似 MCP 路由）：

```python
async def execute(self, tool_name, arguments, ...):
    if tool_name == "activate_skill" and self.skill_manager:
        return self.skill_manager.activate(arguments["name"])
    elif tool_name.startswith("mcp_") and self.mcp_manager:
        ...
    else:
        ...  # 本地工具
```

## 渐进式披露（Progressive Disclosure）

| 层级 | 加载内容 | 时机 | Token 开销 |
|------|---------|------|-----------|
| Tier 1 | name + description | 启动时全部加载 | ~50-100/skill |
| Tier 2 | SKILL.md 完整 body | 激活时加载 | <5000 推荐 |
| Tier 3 | scripts/、references/、assets/ | 按需加载 | 不定 |

### Tier 2 返回格式

```xml
<skill_content name="code-review">
## 使用场景
当用户要求审查代码时激活此 skill。

## 审查步骤
1. 阅读指定的代码文件
...

Skill 目录: /path/to/skills/code-review
此 skill 中的相对路径基于上述目录。

<skill_resources>
  <file>references/checklist.md</file>
  <file>scripts/lint.py</file>
</skill_resources>
</skill_content>
```

### Tier 3 资源访问

LLM 在 skill body 中看到相对路径引用时，使用现有的 `read_file` 工具读取。`activate_skill` 返回中列出 `<skill_resources>` 供参考，但不主动加载。

## 上下文管理

> **V1 范围说明**：当前 `ConversationBuffer` 的 `compress()` 方法不感知 skill 内容。如果 skill 指令被压缩丢失，Agent 行为会静默降级。本期不实现 skill 内容保护，作为未来增强项记录。后续可通过在消息中标记 `<skill_content>` 标签来在压缩时跳过这些内容。

## 文件变更清单

### 新增文件
- `src/skills/__init__.py` — 包初始化，导出 `SkillManager`、`SkillInfo`
- `src/skills/models.py` — SkillInfo 数据类
- `src/skills/parser.py` — SKILL.md 解析器
- `src/skills/manager.py` — SkillManager 核心类
- `tests/test_skills/test_parser.py` — 解析器测试
- `tests/test_skills/test_manager.py` — 管理器测试
- `tests/test_skills/test_integration.py` — 集成测试
- `skills/translate/SKILL.md` — 示例 skill：翻译
- `skills/code-review/SKILL.md` — 示例 skill：代码审查
- `skills/code-review/references/checklist.md` — 示例参考文档

### 修改文件
- `main.py` — 添加 SkillManager 初始化和斜杠命令路由
- `src/tools/tool_executor.py` — 添加 activate_skill 路由
- `src/agents/orchestrator.py` — system prompt 追加 skill catalog
- `config.py` — 添加 SKILLS_DIRS 配置

## 测试策略

### 单元测试

**test_parser.py**:
- 解析有效 SKILL.md → 正确提取 name、description、body
- 处理缺失 frontmatter → 抛出 ParseError
- 处理缺失必填字段 → 抛出 ValidationError
- 宽容处理异常 YAML（如未引用的冒号）
- name 包含连续连字符 → 警告
- find_skill_md 优先大写、兼容小写

**test_manager.py**:
- discover() 扫描目录 → 正确发现所有 skill
- 跳过无效 skill 目录
- get_catalog_prompt() → 生成正确 XML
- activate() → 返回包裹的 skill body + 资源列表
- activate() 去重 → 重复激活不重复注入
- is_slash_command() → 正确识别 /skill-name
- build_activate_tool_schema() → enum 限制正确
- 名称冲突优先级

### 集成测试

**test_integration.py**:
- 创建临时 skill 目录 → discover → activate → 验证返回内容
- 验证 activate_skill 工具路由端到端通路
- 斜杠命令 → skill 激活 → system prompt 注入

### 示例 Skill

提供两个示例 skill 供测试和参考使用：
- `skills/translate/` — 简单翻译 skill（仅 SKILL.md）
- `skills/code-review/` — 包含 references 的代码审查 skill
