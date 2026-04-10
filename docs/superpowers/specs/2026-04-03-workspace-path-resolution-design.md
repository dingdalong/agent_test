# Workspace Path Resolution Design

**Date:** 2026-04-03
**Status:** Approved

## Problem

框架中所有路径（config.yaml、chroma_data、mcp_servers.json、skills 目录、tool_categories.json 等）都是硬编码的 CWD 相对路径，导致 agent 必须从项目根目录启动。同时缺少两个关键概念：

1. **用户工作目录 (workspace)** — agent 当前操作的项目上下文目录
2. **系统数据目录 (data_dir)** — agent 自身的运行时数据（数据库、日志、缓存等）

## Design

### AppConfig Dataclass

在 `src/config.py` 中新增 `AppConfig`，替代当前 `load_config` 返回的 raw dict：

```python
@dataclass(frozen=True)
class AppConfig:
    workspace: Path       # 用户项目上下文目录（绝对路径）
    data_dir: Path        # 系统数据目录（绝对路径）
    raw: dict             # 原始 config dict

    def resolve(self, relative: str) -> Path:
        """相对于 workspace 解析路径，绝对路径原样返回。"""
        p = Path(relative)
        return p if p.is_absolute() else (self.workspace / p).resolve()

    def resolve_data(self, relative: str) -> Path:
        """相对于 data_dir 解析路径，绝对路径原样返回。"""
        p = Path(relative)
        return p if p.is_absolute() else (self.data_dir / p).resolve()
```

- `frozen=True`：配置加载后不可变
- `resolve()`：用户侧路径（skills、mcp 配置、工具发现等）
- `resolve_data()`：系统数据路径（chroma、日志等）
- `raw`：保留原始 dict，下游组件按需读取非路径配置

### config.yaml 新增字段

```yaml
workspace: .              # 用户项目目录，默认当前目录
data_dir: .agent_data     # 系统数据目录，相对于 workspace
```

### load_config 改造

`load_config` 返回类型从 `dict` 改为 `AppConfig`：

```python
def load_config(path: str = "config.yaml") -> AppConfig:
    load_dotenv()
    config_path = Path(path)
    raw = {}
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

    # .env secrets 合并（逻辑不变）
    ...

    # 路径解析：workspace 相对于 config.yaml 所在目录
    config_dir = config_path.resolve().parent
    workspace = (config_dir / raw.get("workspace", ".")).resolve()
    data_dir = (workspace / raw.get("data_dir", ".agent_data")).resolve()

    return AppConfig(workspace=workspace, data_dir=data_dir, raw=raw)
```

关键决策：**workspace 相对于 config.yaml 所在目录解析**（非 CWD），保证无论从哪里启动，指向同一 config.yaml 时解析结果一致。

### 下游改造

**bootstrap.py：** `create_app` 签名从 `config_path: str` 改为 `config: AppConfig | None`：

```python
async def create_app(config: AppConfig | None = None) -> AgentApp:
    if config is None:
        config = load_config()
    raw = config.raw
```

路径替换清单：

| 原代码 | 改为 | 方法 |
|--------|------|------|
| `Path("src/tools/builtin")` | `config.resolve("src/tools/builtin")` | `resolve` |
| `"mcp_servers.json"` fallback | `config.resolve(mcp_path)` | `resolve` |
| `["skills/", ".agents/skills/"]` | `[config.resolve(d) for d in dirs]` | `resolve` |
| `memory_cfg.get("path", "./chroma_data")` | `config.resolve_data(memory_path)` | `resolve_data` |
| `"tool_categories.json"` fallback | `config.resolve(categories_path)` | `resolve` |

**classify.py CLI：** 同样通过 `load_config()` 获取 `AppConfig`，使用 `resolve` 解析路径。

**下游组件不需要改动：** `SkillManager`、`ChromaMemoryStore`、`load_mcp_config` 等接收的已经是解析后的绝对路径。

**main.py 不需要改动：** `create_app()` 无参调用时内部自动 `load_config()`。

## Scope

- 修改：`src/config.py`、`src/app/bootstrap.py`、`src/tools/classify.py`、`config.yaml`
- 不修改：`main.py`、下游组件（SkillManager、ChromaMemoryStore、load_mcp_config 等）
- 新增：无新文件
