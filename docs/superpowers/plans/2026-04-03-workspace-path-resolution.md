# Workspace Path Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 引入 AppConfig dataclass 统一所有路径解析，支持 workspace（用户项目上下文）和 data_dir（系统数据）两个目录概念。

**Architecture:** 在 `src/config.py` 新增 frozen dataclass `AppConfig`，包含 `workspace`、`data_dir`、`raw` 三个字段和两个 resolve 方法。`load_config` 返回类型从 `dict` 改为 `AppConfig`。`bootstrap.py` 和 `classify.py` 中所有硬编码路径改为通过 `config.resolve()` / `config.resolve_data()` 获取绝对路径。

**Tech Stack:** Python 3.13, dataclasses, pathlib, pytest

---

## File Structure

| 文件 | 动作 | 职责 |
|------|------|------|
| `src/config.py` | 修改 | 新增 `AppConfig` dataclass，改造 `load_config` 返回类型 |
| `tests/test_config.py` | 新建 | `AppConfig` 和 `load_config` 的单元测试 |
| `src/app/bootstrap.py` | 修改 | `create_app` 签名改为接收 `AppConfig`，所有路径用 `config.resolve` |
| `src/tools/classify.py` | 修改 | `run_classify` 使用 `AppConfig`，路径用 `config.resolve` |
| `config.yaml` | 修改 | 新增 `workspace` 和 `data_dir` 字段（注释形式） |

---

### Task 1: AppConfig dataclass + load_config 改造

**Files:**
- Modify: `src/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: 写 AppConfig 和 load_config 的测试**

```python
# tests/test_config.py
"""AppConfig 和 load_config 测试。"""
import yaml
import pytest
from pathlib import Path

from src.config import AppConfig, load_config


class TestAppConfig:
    """AppConfig 路径解析测试。"""

    def test_resolve_relative(self, tmp_path: Path):
        config = AppConfig(workspace=tmp_path, data_dir=tmp_path / ".agent_data", raw={})
        result = config.resolve("skills/")
        assert result == (tmp_path / "skills").resolve()

    def test_resolve_absolute(self, tmp_path: Path):
        config = AppConfig(workspace=tmp_path, data_dir=tmp_path / ".agent_data", raw={})
        abs_path = "/usr/local/share"
        result = config.resolve(abs_path)
        assert result == Path(abs_path)

    def test_resolve_data_relative(self, tmp_path: Path):
        data_dir = tmp_path / ".agent_data"
        config = AppConfig(workspace=tmp_path, data_dir=data_dir, raw={})
        result = config.resolve_data("chroma")
        assert result == (data_dir / "chroma").resolve()

    def test_resolve_data_absolute(self, tmp_path: Path):
        config = AppConfig(workspace=tmp_path, data_dir=tmp_path / ".agent_data", raw={})
        abs_path = "/var/data/chroma"
        result = config.resolve_data(abs_path)
        assert result == Path(abs_path)


class TestLoadConfig:
    """load_config 集成测试。"""

    def test_default_workspace_is_config_dir(self, tmp_path: Path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"llm": {"model": "test"}}))
        config = load_config(str(config_file))
        assert config.workspace == tmp_path.resolve()

    def test_custom_workspace_relative_to_config_dir(self, tmp_path: Path):
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"workspace": "my_project"}))
        config = load_config(str(config_file))
        assert config.workspace == project_dir.resolve()

    def test_default_data_dir(self, tmp_path: Path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({}))
        config = load_config(str(config_file))
        assert config.data_dir == (tmp_path / ".agent_data").resolve()

    def test_custom_data_dir(self, tmp_path: Path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"data_dir": "my_data"}))
        config = load_config(str(config_file))
        assert config.data_dir == (tmp_path / "my_data").resolve()

    def test_missing_config_file(self, tmp_path: Path):
        config = load_config(str(tmp_path / "nonexistent.yaml"))
        # workspace fallback 到 config_path 所在目录（即 tmp_path）
        assert config.workspace == tmp_path.resolve()
        assert config.data_dir == (tmp_path / ".agent_data").resolve()

    def test_raw_dict_preserved(self, tmp_path: Path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"llm": {"model": "deepseek"}}))
        config = load_config(str(config_file))
        assert config.raw["llm"]["model"] == "deepseek"

    def test_env_secrets_merged(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({}))
        config = load_config(str(config_file))
        assert config.raw["llm"]["api_key"] == "sk-test-key"
```

- [ ] **Step 2: 运行测试，确认全部失败**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'AppConfig' from 'src.config'`

- [ ] **Step 3: 实现 AppConfig 和改造 load_config**

```python
# src/config.py
"""配置加载器 — 读 config.yaml + .env，返回 AppConfig。"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    """应用配置，持有解析后的绝对路径和原始配置字典。

    Attributes:
        workspace: 用户项目上下文目录（绝对路径）。
        data_dir: 系统数据目录（绝对路径）。
        raw: 原始 config dict，下游组件按需读取非路径配置。
    """

    workspace: Path
    data_dir: Path
    raw: dict = field(default_factory=dict)

    def resolve(self, relative: str) -> Path:
        """相对于 workspace 解析路径，绝对路径原样返回。"""
        p = Path(relative)
        return p if p.is_absolute() else (self.workspace / p).resolve()

    def resolve_data(self, relative: str) -> Path:
        """相对于 data_dir 解析路径，绝对路径原样返回。"""
        p = Path(relative)
        return p if p.is_absolute() else (self.data_dir / p).resolve()


def load_config(path: str = "config.yaml") -> AppConfig:
    """加载配置文件，返回 AppConfig。文件不存在时使用空配置。"""
    load_dotenv()
    config_path = Path(path)
    raw: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

    # .env 中的 secrets 合并到 raw
    if "llm" not in raw:
        raw["llm"] = {}
    if not raw["llm"].get("api_key"):
        raw["llm"]["api_key"] = os.getenv("OPENAI_API_KEY", "")
    if not raw["llm"].get("base_url"):
        raw["llm"]["base_url"] = os.getenv("OPENAI_BASE_URL", "")
    if not raw["llm"].get("model"):
        raw["llm"]["model"] = os.getenv("OPENAI_MODEL", "")

    if "embedding" not in raw:
        raw["embedding"] = {}
    if not raw["embedding"].get("model"):
        raw["embedding"]["model"] = os.getenv("OPENAI_MODEL_EMBEDDING", "")
    if not raw["embedding"].get("base_url"):
        raw["embedding"]["base_url"] = os.getenv("OPENAI_MODEL_EMBEDDING_URL", "")

    if "user" not in raw:
        raw["user"] = {}
    if not raw["user"].get("id"):
        raw["user"]["id"] = os.getenv("USER_ID", "default_user")

    # 路径解析：workspace 相对于 config.yaml 所在目录
    config_dir = config_path.resolve().parent
    workspace = (config_dir / raw.get("workspace", ".")).resolve()
    data_dir = (workspace / raw.get("data_dir", ".agent_data")).resolve()

    return AppConfig(workspace=workspace, data_dir=data_dir, raw=raw)
```

- [ ] **Step 4: 运行测试，确认全部通过**

Run: `uv run pytest tests/test_config.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat(config): add AppConfig dataclass with workspace/data_dir path resolution"
```

---

### Task 2: bootstrap.py 改造

**Files:**
- Modify: `src/app/bootstrap.py`

- [ ] **Step 1: 修改 create_app 签名和路径解析**

将 `src/app/bootstrap.py` 做以下改动：

1. 导入 `AppConfig`：

```python
from src.config import load_config, AppConfig
```

2. `create_app` 签名改为：

```python
async def create_app(config: AppConfig | None = None) -> AgentApp:
```

3. 函数开头改为：

```python
    if config is None:
        config = load_config()
    raw = config.raw
```

（删除原来的 `raw = load_config(config_path)` 行）

4. 替换所有硬编码路径（5 处）：

```python
# 第 75 行：工具发现
# 原：discover_tools("src.tools.builtin", Path("src/tools/builtin"))
discover_tools("src.tools.builtin", config.resolve("src/tools/builtin"))

# 第 88 行：MCP 配置路径
# 原：mcp_config_path = raw.get("mcp", {}).get("config_path", "mcp_servers.json")
mcp_config_path = config.resolve(raw.get("mcp", {}).get("config_path", "mcp_servers.json"))

# 第 95 行：Skills 目录
# 原：skill_dirs = raw.get("skills", {}).get("dirs", ["skills/", ".agents/skills/"])
# 原：skill_manager = SkillManager(skill_dirs=skill_dirs)
skill_dirs = raw.get("skills", {}).get("dirs", ["skills/", ".agents/skills/"])
skill_manager = SkillManager(skill_dirs=[str(config.resolve(d)) for d in skill_dirs])

# 第 121 行：Memory 存储路径
# 原：persist_dir=memory_cfg.get("path", "./chroma_data"),
persist_dir=str(config.resolve_data(memory_cfg.get("path", "chroma"))),

# 第 144-145 行：工具分类配置路径
# 原：categories_path = raw.get("tools", {}).get("categories_path", "tool_categories.json")
categories_path = str(config.resolve(raw.get("tools", {}).get("categories_path", "tool_categories.json")))
```

注意：`mcp_config_path` 传给 `load_mcp_config` 的需要是 `str`，加 `str()` 转换。同理 `categories_path` 传给 `load_categories` 也需要 `str`。

- [ ] **Step 2: 运行已有测试确认无回归**

Run: `uv run pytest tests/ -v --timeout=10`
Expected: 所有已有测试 PASS（bootstrap 不会被单元测试直接调用，主要确认导入无误）

- [ ] **Step 3: 提交**

```bash
git add src/app/bootstrap.py
git commit -m "refactor(bootstrap): use AppConfig for path resolution"
```

---

### Task 3: classify.py CLI 改造

**Files:**
- Modify: `src/tools/classify.py`

- [ ] **Step 1: 修改 classify.py 使用 AppConfig**

1. 修改导入：

```python
from src.config import load_config, AppConfig
```

2. 修改 `run_classify` 函数：

```python
async def run_classify(force: bool = False, output: str = DEFAULT_OUTPUT) -> None:
    """分类主流程。"""
    config = load_config()
    raw = config.raw

    # 1. Local tools
    discover_tools("src.tools.builtin", config.resolve("src/tools/builtin"))
    local_schemas = get_registry().get_schemas()

    # 2. MCP tools
    mcp_schemas: list[dict] = []
    mcp_manager = None
    try:
        from src.mcp.config import load_mcp_config
        from src.mcp.manager import MCPManager

        mcp_config_path = str(config.resolve(raw.get("mcp", {}).get("config_path", "mcp_servers.json")))
        mcp_manager = MCPManager(configs=load_mcp_config(mcp_config_path))
        await mcp_manager.connect_all()
        mcp_schemas = mcp_manager.get_tools_schemas()
    except Exception:
        logger.warning("MCP 连接失败，仅使用本地工具", exc_info=True)
```

其余代码不变（`output` 参数是 CLI 用户指定的输出路径，保持原样）。

- [ ] **Step 2: 运行 classify CLI 测试确认无回归**

Run: `uv run pytest tests/tools/test_classify_cli.py -v`
Expected: 全部 PASS（这些测试不涉及 `run_classify`，只测 `detect_changes` 和 `_build_output`）

- [ ] **Step 3: 提交**

```bash
git add src/tools/classify.py
git commit -m "refactor(classify): use AppConfig for path resolution"
```

---

### Task 4: config.yaml 更新 + 全量验证

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: 在 config.yaml 中添加 workspace 和 data_dir 注释**

在 `config.yaml` 文件顶部（`llm:` 之前）添加：

```yaml
# ===== 目录配置 =====
# workspace: .              # 用户项目目录（相对于本文件所在目录），默认 "."
# data_dir: .agent_data     # 系统数据目录（相对于 workspace），默认 ".agent_data"

# memory.path 现在相对于 data_dir 解析，不再相对于 CWD
```

并将 memory 配置中的 `path` 从 `./chroma_data` 改为 `chroma`（因为现在相对于 data_dir 解析）：

```yaml
memory:
  provider: chroma
  path: chroma
```

- [ ] **Step 2: 运行全量测试**

Run: `uv run pytest tests/ -v --timeout=30`
Expected: 全部 PASS

- [ ] **Step 3: 提交**

```bash
git add config.yaml
git commit -m "docs(config): add workspace/data_dir config comments, update memory path"
```
