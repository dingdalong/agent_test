# Agent Skills 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 AI Agent 实现 Agent Skills 开放标准支持，包括 SKILL.md 解析、渐进式披露、斜杠命令+自动识别双触发、以及与 MultiAgentFlow 的集成。

**Architecture:** 新增 `src/skills/` 包（models/parser/manager 三模块），通过 `SkillManager` 实现发现→披露→激活三层渐进式加载。通过 `tool_executor.skill_manager` 属性注入（与 MCP 一致的模式），在 `handle_input` 中检测斜杠命令，在 orchestrator 中通过 `activate_skill` 工具实现 LLM 自动触发。

**Tech Stack:** Python 3.13+, PyYAML, pytest, pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-03-25-agent-skills-design.md`

---

### Task 1: 数据模型 — SkillInfo

**Files:**
- Create: `src/skills/models.py`
- Create: `tests/test_skills/__init__.py`
- Create: `tests/test_skills/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_skills/test_models.py
from pathlib import Path
from src.skills.models import SkillInfo


def test_skill_info_required_fields():
    info = SkillInfo(
        name="code-review",
        description="Review code quality.",
        location=Path("/tmp/skills/code-review/SKILL.md"),
    )
    assert info.name == "code-review"
    assert info.description == "Review code quality."
    assert info.location == Path("/tmp/skills/code-review/SKILL.md")
    assert info.body is None
    assert info.allowed_tools is None


def test_skill_info_optional_fields():
    info = SkillInfo(
        name="translate",
        description="Translate text.",
        location=Path("/tmp/skills/translate/SKILL.md"),
        body="## Instructions\nTranslate text.",
        allowed_tools="Bash(git:*) Read",
    )
    assert info.body == "## Instructions\nTranslate text."
    assert info.allowed_tools == "Bash(git:*) Read"


def test_skill_info_base_dir():
    """location.parent gives the skill base directory."""
    info = SkillInfo(
        name="test",
        description="Test.",
        location=Path("/tmp/skills/test/SKILL.md"),
    )
    assert info.location.parent == Path("/tmp/skills/test")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_skills/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.skills'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/skills/__init__.py
# (empty for now, will add exports in Task 5)
```

```python
# src/skills/models.py
"""Agent Skills 数据模型。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SkillInfo:
    """Skill 元数据，对应 SKILL.md frontmatter。

    Attributes:
        name: kebab-case 名称（如 "code-review"）
        description: 描述（何时使用）
        location: SKILL.md 文件的绝对路径。基础目录通过 location.parent 派生。
        body: Markdown body（激活后加载，发现时为 None）
        allowed_tools: 预批准工具列表（实验性，仅存储不执行）
    """

    name: str
    description: str
    location: Path
    body: Optional[str] = None
    allowed_tools: Optional[str] = None
```

```python
# tests/test_skills/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_skills/test_models.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/skills/__init__.py src/skills/models.py tests/test_skills/__init__.py tests/test_skills/test_models.py
git commit -m "feat(skills): add SkillInfo data model"
```

---

### Task 2: SKILL.md 解析器 — parser.py

**Files:**
- Create: `src/skills/parser.py`
- Create: `tests/test_skills/test_parser.py`

**Dependencies:** PyYAML（已在 pydantic 依赖链中可用）

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_skills/test_parser.py
import pytest
from pathlib import Path
from src.skills.parser import find_skill_md, parse_frontmatter, read_skill_info


class TestFindSkillMd:
    def test_finds_uppercase(self, tmp_path):
        (tmp_path / "SKILL.md").write_text("---\nname: test\n---")
        assert find_skill_md(tmp_path) == tmp_path / "SKILL.md"

    def test_finds_lowercase(self, tmp_path):
        (tmp_path / "skill.md").write_text("---\nname: test\n---")
        assert find_skill_md(tmp_path) == tmp_path / "skill.md"

    def test_prefers_uppercase(self, tmp_path):
        (tmp_path / "SKILL.md").write_text("upper")
        (tmp_path / "skill.md").write_text("lower")
        assert find_skill_md(tmp_path) == tmp_path / "SKILL.md"

    def test_returns_none_if_missing(self, tmp_path):
        assert find_skill_md(tmp_path) is None


class TestParseFrontmatter:
    def test_valid_frontmatter(self):
        content = "---\nname: test-skill\ndescription: A test skill.\n---\n\n## Body\nHello"
        metadata, body = parse_frontmatter(content)
        assert metadata["name"] == "test-skill"
        assert metadata["description"] == "A test skill."
        assert body == "## Body\nHello"

    def test_missing_frontmatter_raises(self):
        with pytest.raises(ValueError, match="frontmatter"):
            parse_frontmatter("No frontmatter here")

    def test_unclosed_frontmatter_raises(self):
        with pytest.raises(ValueError, match="frontmatter"):
            parse_frontmatter("---\nname: test\n")

    def test_optional_fields_parsed(self):
        content = "---\nname: x\ndescription: y\nlicense: MIT\ncompatibility: Python 3.13+\nallowed-tools: Bash Read\nmetadata:\n  author: test\n  version: '1.0'\n---\nbody"
        metadata, body = parse_frontmatter(content)
        assert metadata["license"] == "MIT"
        assert metadata["compatibility"] == "Python 3.13+"
        assert metadata["allowed-tools"] == "Bash Read"
        assert metadata["metadata"] == {"author": "test", "version": "1.0"}

    def test_empty_body(self):
        content = "---\nname: x\ndescription: y\n---\n"
        metadata, body = parse_frontmatter(content)
        assert body == ""

    def test_lenient_yaml_colon_in_value(self):
        """YAML with unquoted colon in description should not crash."""
        content = '---\nname: test\ndescription: "Use when: user asks about PDFs"\n---\nbody'
        metadata, body = parse_frontmatter(content)
        assert "PDFs" in metadata["description"]


class TestReadSkillInfo:
    def test_valid_skill(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Does things.\nallowed-tools: Read\n---\n\n## Steps\n1. Do thing"
        )
        info = read_skill_info(skill_dir)
        assert info.name == "my-skill"
        assert info.description == "Does things."
        assert info.location == skill_dir / "SKILL.md"
        assert info.body is None  # body not loaded during discovery
        assert info.allowed_tools == "Read"

    def test_missing_skill_md_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_skill_info(tmp_path)

    def test_missing_name_raises(self, tmp_path):
        skill_dir = tmp_path / "bad"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\ndescription: no name\n---\n")
        with pytest.raises(ValueError, match="name"):
            read_skill_info(skill_dir)

    def test_missing_description_raises(self, tmp_path):
        skill_dir = tmp_path / "bad"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: bad\n---\n")
        with pytest.raises(ValueError, match="description"):
            read_skill_info(skill_dir)

    def test_empty_description_raises(self, tmp_path):
        skill_dir = tmp_path / "bad"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text('---\nname: bad\ndescription: ""\n---\n')
        with pytest.raises(ValueError, match="description"):
            read_skill_info(skill_dir)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_skills/test_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.skills.parser'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/skills/parser.py
"""SKILL.md 文件解析器。"""

import logging
from pathlib import Path
from typing import Optional

import yaml

from .models import SkillInfo

logger = logging.getLogger(__name__)


def find_skill_md(skill_dir: Path) -> Optional[Path]:
    """查找 SKILL.md 文件，优先大写，兼容小写。"""
    for name in ("SKILL.md", "skill.md"):
        path = skill_dir / name
        if path.exists():
            return path
    return None


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """解析 YAML frontmatter，返回 (metadata_dict, body_str)。

    Raises:
        ValueError: frontmatter 缺失或无法解析
    """
    if not content.startswith("---"):
        raise ValueError("SKILL.md must start with YAML frontmatter (---)")

    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError("SKILL.md frontmatter not properly closed with ---")

    frontmatter_str = parts[1]
    body = parts[2].strip()

    try:
        metadata = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in frontmatter: {e}")

    if not isinstance(metadata, dict):
        raise ValueError("SKILL.md frontmatter must be a YAML mapping")

    # 将 metadata 值统一转为字符串（Agent Skills 标准定义为 string->string 映射）
    # 注意：YAML 可能解析 version: 1.0 为 float，这里有意做字符串强制转换
    if "metadata" in metadata and isinstance(metadata["metadata"], dict):
        metadata["metadata"] = {str(k): str(v) for k, v in metadata["metadata"].items()}

    return metadata, body


def read_skill_info(skill_dir: Path) -> SkillInfo:
    """解析 SKILL.md，返回 SkillInfo（仅元数据，body=None）。

    Raises:
        FileNotFoundError: SKILL.md 不存在
        ValueError: 缺少必填字段
    """
    skill_dir = Path(skill_dir)
    skill_md = find_skill_md(skill_dir)

    if skill_md is None:
        raise FileNotFoundError(f"SKILL.md not found in {skill_dir}")

    content = skill_md.read_text(encoding="utf-8")
    metadata, _ = parse_frontmatter(content)

    # 验证必填字段
    if "name" not in metadata or not metadata["name"]:
        raise ValueError("Missing required field in frontmatter: name")

    description = metadata.get("description")
    if not description or (isinstance(description, str) and not description.strip()):
        raise ValueError("Missing required field in frontmatter: description")

    name = str(metadata["name"]).strip()
    description = str(description).strip()

    # 宽容警告
    if len(name) > 64:
        logger.warning(f"Skill name '{name}' exceeds 64 character limit")
    if "--" in name:
        logger.warning(f"Skill name '{name}' contains consecutive hyphens")
    if name != skill_dir.name:
        logger.warning(f"Skill name '{name}' does not match directory '{skill_dir.name}'")

    return SkillInfo(
        name=name,
        description=description,
        location=skill_md.resolve(),
        allowed_tools=metadata.get("allowed-tools"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_skills/test_parser.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/skills/parser.py tests/test_skills/test_parser.py
git commit -m "feat(skills): add SKILL.md parser with lenient validation"
```

---

### Task 3: Skill 管理器 — SkillManager

**Files:**
- Create: `src/skills/manager.py`
- Create: `tests/test_skills/test_manager.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_skills/test_manager.py
import pytest
from pathlib import Path
from src.skills.manager import SkillManager


def _make_skill(tmp_path, name, description="A skill.", body="## Instructions\nDo things."):
    """Helper: create a minimal skill directory with SKILL.md."""
    skill_dir = tmp_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}"
    )
    return skill_dir


def _make_skill_with_resources(tmp_path, name):
    """Helper: create a skill with references/ and scripts/."""
    skill_dir = _make_skill(tmp_path, name)
    refs = skill_dir / "references"
    refs.mkdir()
    (refs / "guide.md").write_text("# Guide")
    scripts = skill_dir / "scripts"
    scripts.mkdir()
    (scripts / "run.py").write_text("print('hi')")
    return skill_dir


# --- discover ---

@pytest.mark.asyncio
async def test_discover_finds_skills(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _make_skill(skills_dir, "translate", "Translate text.")
    _make_skill(skills_dir, "code-review", "Review code.")

    mgr = SkillManager(skill_dirs=[str(skills_dir)])
    await mgr.discover()

    assert mgr.get_skill("translate") is not None
    assert mgr.get_skill("code-review") is not None


@pytest.mark.asyncio
async def test_discover_skips_invalid(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _make_skill(skills_dir, "good", "Good skill.")
    # bad skill: no SKILL.md
    (skills_dir / "bad").mkdir()

    mgr = SkillManager(skill_dirs=[str(skills_dir)])
    await mgr.discover()

    assert mgr.get_skill("good") is not None
    assert mgr.get_skill("bad") is None


@pytest.mark.asyncio
async def test_discover_skips_dotdirs(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    git_dir = skills_dir / ".git"
    git_dir.mkdir()
    (git_dir / "SKILL.md").write_text("---\nname: git\ndescription: bad\n---\n")

    mgr = SkillManager(skill_dirs=[str(skills_dir)])
    await mgr.discover()

    assert mgr.get_skill("git") is None


@pytest.mark.asyncio
async def test_discover_multiple_dirs(tmp_path):
    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    _make_skill(dir1, "skill-a", "Skill A.")

    dir2 = tmp_path / "dir2"
    dir2.mkdir()
    _make_skill(dir2, "skill-b", "Skill B.")

    mgr = SkillManager(skill_dirs=[str(dir1), str(dir2)])
    await mgr.discover()

    assert mgr.get_skill("skill-a") is not None
    assert mgr.get_skill("skill-b") is not None


@pytest.mark.asyncio
async def test_discover_name_collision_first_wins(tmp_path):
    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    _make_skill(dir1, "dup", "First version.")

    dir2 = tmp_path / "dir2"
    dir2.mkdir()
    _make_skill(dir2, "dup", "Second version.")

    mgr = SkillManager(skill_dirs=[str(dir1), str(dir2)])
    await mgr.discover()

    assert mgr.get_skill("dup").description == "First version."


@pytest.mark.asyncio
async def test_discover_nonexistent_dir(tmp_path):
    mgr = SkillManager(skill_dirs=[str(tmp_path / "nonexistent")])
    await mgr.discover()  # should not raise
    assert mgr.get_skill("anything") is None


# --- get_catalog_prompt ---

@pytest.mark.asyncio
async def test_catalog_prompt_with_skills(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _make_skill(skills_dir, "translate", "Translate text.")

    mgr = SkillManager(skill_dirs=[str(skills_dir)])
    await mgr.discover()

    prompt = mgr.get_catalog_prompt()
    assert "<available_skills>" in prompt
    assert "<name>\ntranslate\n</name>" in prompt
    assert "<description>\nTranslate text.\n</description>" in prompt
    assert "<location>" in prompt
    assert "SKILL.md" in prompt
    assert "activate_skill" in prompt  # behavioral instructions


@pytest.mark.asyncio
async def test_catalog_prompt_empty():
    mgr = SkillManager(skill_dirs=[])
    await mgr.discover()
    assert mgr.get_catalog_prompt() == ""


# --- activate ---

@pytest.mark.asyncio
async def test_activate_returns_content(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _make_skill(skills_dir, "translate", "Translate text.", "## Steps\n1. Translate")

    mgr = SkillManager(skill_dirs=[str(skills_dir)])
    await mgr.discover()

    result = mgr.activate("translate")
    assert result is not None
    assert '<skill_content name="translate">' in result
    assert "## Steps" in result
    assert "1. Translate" in result
    assert "Skill 目录:" in result
    assert "</skill_content>" in result


@pytest.mark.asyncio
async def test_activate_includes_resources(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _make_skill_with_resources(skills_dir, "my-skill")

    mgr = SkillManager(skill_dirs=[str(skills_dir)])
    await mgr.discover()

    result = mgr.activate("my-skill")
    assert "<skill_resources>" in result
    assert "references/guide.md" in result
    assert "scripts/run.py" in result


@pytest.mark.asyncio
async def test_activate_dedup(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _make_skill(skills_dir, "x", "X skill.")

    mgr = SkillManager(skill_dirs=[str(skills_dir)])
    await mgr.discover()

    result1 = mgr.activate("x")
    result2 = mgr.activate("x")
    assert result1 is not None
    assert result2 is not None  # still returns content
    # But _activated set tracks it
    assert "x" in mgr._activated


@pytest.mark.asyncio
async def test_activate_unknown_returns_none(tmp_path):
    mgr = SkillManager(skill_dirs=[])
    await mgr.discover()
    assert mgr.activate("nonexistent") is None


# --- list_resources ---

@pytest.mark.asyncio
async def test_list_resources(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _make_skill_with_resources(skills_dir, "my-skill")

    mgr = SkillManager(skill_dirs=[str(skills_dir)])
    await mgr.discover()

    resources = mgr.list_resources("my-skill")
    rel_paths = [str(r) for r in resources]
    assert "references/guide.md" in rel_paths
    assert "scripts/run.py" in rel_paths


@pytest.mark.asyncio
async def test_list_resources_unknown():
    mgr = SkillManager(skill_dirs=[])
    await mgr.discover()
    assert mgr.list_resources("nonexistent") == []


# --- build_activate_tool_schema ---

@pytest.mark.asyncio
async def test_tool_schema_with_skills(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _make_skill(skills_dir, "translate", "Translate.")
    _make_skill(skills_dir, "review", "Review.")

    mgr = SkillManager(skill_dirs=[str(skills_dir)])
    await mgr.discover()

    schema = mgr.build_activate_tool_schema()
    assert schema is not None
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "activate_skill"
    enum = schema["function"]["parameters"]["properties"]["name"]["enum"]
    assert set(enum) == {"translate", "review"}


@pytest.mark.asyncio
async def test_tool_schema_no_skills():
    mgr = SkillManager(skill_dirs=[])
    await mgr.discover()
    assert mgr.build_activate_tool_schema() is None


# --- is_slash_command ---

@pytest.mark.asyncio
async def test_slash_command_match(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _make_skill(skills_dir, "translate", "Translate.")

    mgr = SkillManager(skill_dirs=[str(skills_dir)])
    await mgr.discover()

    assert mgr.is_slash_command("/translate some text") == "translate"
    assert mgr.is_slash_command("/translate") == "translate"


@pytest.mark.asyncio
async def test_slash_command_no_match(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _make_skill(skills_dir, "translate", "Translate.")

    mgr = SkillManager(skill_dirs=[str(skills_dir)])
    await mgr.discover()

    assert mgr.is_slash_command("/unknown something") is None
    assert mgr.is_slash_command("translate") is None  # no slash
    assert mgr.is_slash_command("hello world") is None


@pytest.mark.asyncio
async def test_slash_command_reserved_prefix(tmp_path):
    """Reserved commands like /plan and /book should not match skills."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _make_skill(skills_dir, "plan", "A plan skill.")

    mgr = SkillManager(skill_dirs=[str(skills_dir)])
    await mgr.discover()

    assert mgr.is_slash_command("/plan something") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_skills/test_manager.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.skills.manager'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/skills/manager.py
"""SkillManager: 发现、披露、激活 Agent Skills。"""

import html
import logging
from pathlib import Path
from typing import Optional

from .models import SkillInfo
from .parser import find_skill_md, parse_frontmatter, read_skill_info

logger = logging.getLogger(__name__)

# 保留的斜杠命令前缀，不可被 skill 覆盖
_RESERVED_COMMANDS = {"plan", "book"}

# 扫描时跳过的目录名
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv"}

# 扫描约束
_MAX_SCAN_DEPTH = 4
_MAX_SCAN_DIRS = 2000


class SkillManager:
    """Agent Skills 管理器：发现、披露、激活。

    遵循 Agent Skills 开放标准的渐进式披露架构：
    - Tier 1 (discover): 仅加载 name + description
    - Tier 2 (activate): 加载完整 SKILL.md body
    - Tier 3 (list_resources): 列出辅助文件
    """

    def __init__(self, skill_dirs: list[str]):
        self._skill_dirs = skill_dirs
        self._skills: dict[str, SkillInfo] = {}
        self._activated: set[str] = set()

    async def discover(self) -> None:
        """扫描所有 skill 目录，解析 frontmatter，构建 catalog。"""
        self._skills.clear()
        self._activated.clear()
        dirs_scanned = 0

        for skill_dir_path in self._skill_dirs:
            base = Path(skill_dir_path)
            if not base.exists() or not base.is_dir():
                logger.debug(f"Skill 目录不存在: {base}")
                continue

            for sub in sorted(base.iterdir()):
                if dirs_scanned >= _MAX_SCAN_DIRS:
                    logger.warning(f"已达到最大扫描目录数 {_MAX_SCAN_DIRS}，停止扫描")
                    break

                if not sub.is_dir():
                    continue
                if sub.name in _SKIP_DIRS or sub.name.startswith("."):
                    continue

                dirs_scanned += 1

                try:
                    info = read_skill_info(sub)
                except (FileNotFoundError, ValueError) as e:
                    logger.debug(f"跳过 {sub.name}: {e}")
                    continue

                if info.name in self._skills:
                    logger.warning(
                        f"Skill 名称冲突: '{info.name}' 已存在，跳过 {sub}"
                    )
                    continue

                self._skills[info.name] = info
                logger.info(f"发现 Skill: {info.name}")

    def get_skill(self, name: str) -> Optional[SkillInfo]:
        """根据名称获取 SkillInfo。"""
        return self._skills.get(name)

    def get_catalog_prompt(self) -> str:
        """生成 <available_skills> XML + 行为指令，供注入 system prompt。

        无 skill 时返回空字符串。
        """
        if not self._skills:
            return ""

        lines = [
            "以下 skills 提供了特定任务的专业指令。",
            "当任务匹配某个 skill 的描述时，调用 activate_skill 工具加载其完整指令。",
            "",
            "<available_skills>",
        ]

        for info in self._skills.values():
            lines.append("<skill>")
            lines.append("<name>")
            lines.append(html.escape(info.name))
            lines.append("</name>")
            lines.append("<description>")
            lines.append(html.escape(info.description))
            lines.append("</description>")
            lines.append("<location>")
            lines.append(str(info.location))
            lines.append("</location>")
            lines.append("</skill>")

        lines.append("</available_skills>")
        return "\n".join(lines)

    def activate(self, skill_name: str) -> Optional[str]:
        """激活 skill，返回 <skill_content> 包裹的内容。

        渐进式披露 Tier 2：读取完整 SKILL.md body。
        """
        info = self._skills.get(skill_name)
        if info is None:
            return None

        # 读取 body（如果还没加载）
        if info.body is None:
            content = info.location.read_text(encoding="utf-8")
            _, body = parse_frontmatter(content)
            info.body = body

        self._activated.add(skill_name)

        # 构建返回内容
        skill_dir = info.location.parent
        resources = self.list_resources(skill_name)

        parts = [f'<skill_content name="{html.escape(skill_name)}">']
        parts.append(info.body)
        parts.append("")
        parts.append(f"Skill 目录: {skill_dir}")
        parts.append("此 skill 中的相对路径基于上述目录。")

        if resources:
            parts.append("")
            parts.append("<skill_resources>")
            for res in resources:
                parts.append(f"  <file>{res}</file>")
            parts.append("</skill_resources>")

        parts.append("</skill_content>")
        return "\n".join(parts)

    def list_resources(self, skill_name: str) -> list[str]:
        """列出 skill 目录下的辅助文件（相对路径）。"""
        info = self._skills.get(skill_name)
        if info is None:
            return []

        skill_dir = info.location.parent
        resources = []

        for path in sorted(skill_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.name in ("SKILL.md", "skill.md"):
                continue
            rel = path.relative_to(skill_dir)
            resources.append(str(rel))

        return resources

    def build_activate_tool_schema(self) -> Optional[dict]:
        """生成 activate_skill 工具的 function calling schema。

        无 skill 时返回 None。
        """
        if not self._skills:
            return None

        names = sorted(self._skills.keys())
        return {
            "type": "function",
            "function": {
                "name": "activate_skill",
                "description": "激活指定的 Skill，加载其完整指令到当前对话中。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": names,
                            "description": "要激活的 Skill 名称",
                        },
                    },
                    "required": ["name"],
                },
            },
        }

    def is_slash_command(self, user_input: str) -> Optional[str]:
        """检查输入是否为 /skill-name 格式，返回 skill name 或 None。

        保留命令（如 /plan、/book）不匹配。
        """
        if not user_input.startswith("/"):
            return None

        # 提取命令名（第一个空格前的部分，去掉 /）
        parts = user_input[1:].split(None, 1)
        if not parts:
            return None

        cmd = parts[0]
        if cmd in _RESERVED_COMMANDS:
            return None

        if cmd in self._skills:
            return cmd

        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_skills/test_manager.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/skills/manager.py tests/test_skills/test_manager.py
git commit -m "feat(skills): add SkillManager with discover/activate/slash commands"
```

---

### Task 4: 包导出与配置

**Files:**
- Modify: `src/skills/__init__.py`
- Modify: `config.py:46` (after MCP_CONFIG_PATH)

- [ ] **Step 1: Update `src/skills/__init__.py`**

```python
# src/skills/__init__.py
from .models import SkillInfo
from .manager import SkillManager

__all__ = ["SkillInfo", "SkillManager"]
```

- [ ] **Step 2: Add SKILLS_DIRS to config.py**

在 `config.py` 的 MCP_CONFIG_PATH 行（第 46 行）之后添加：

```python
# Skills 配置
SKILLS_DIRS = ["skills/", ".agents/skills/"]
```

- [ ] **Step 3: Run existing tests to verify no regression**

Run: `python -m pytest tests/test_skills/ -v`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/skills/__init__.py config.py
git commit -m "feat(skills): add package exports and SKILLS_DIRS config"
```

---

### Task 5: ToolExecutor 集成 — activate_skill 路由

**Files:**
- Modify: `src/tools/tool_executor.py:8,14,27-29`
- Create: `tests/test_skills/test_tool_executor_skill.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_skills/test_tool_executor_skill.py
import pytest
from unittest.mock import MagicMock
from src.tools.tool_executor import ToolExecutor


@pytest.mark.asyncio
async def test_executor_routes_activate_skill():
    """activate_skill tool calls are routed to skill_manager."""
    mock_skill_mgr = MagicMock()
    mock_skill_mgr.activate.return_value = '<skill_content name="test">body</skill_content>'

    executor = ToolExecutor({})
    executor.skill_manager = mock_skill_mgr

    result = await executor.execute("activate_skill", {"name": "test"})

    mock_skill_mgr.activate.assert_called_once_with("test")
    assert "skill_content" in result


@pytest.mark.asyncio
async def test_executor_activate_skill_not_found():
    """activate_skill for unknown skill returns error."""
    mock_skill_mgr = MagicMock()
    mock_skill_mgr.activate.return_value = None

    executor = ToolExecutor({})
    executor.skill_manager = mock_skill_mgr

    result = await executor.execute("activate_skill", {"name": "nonexistent"})

    assert "未找到" in result or "not found" in result.lower()


@pytest.mark.asyncio
async def test_executor_no_skill_manager_for_activate():
    """Without skill_manager, activate_skill is treated as unknown tool."""
    executor = ToolExecutor({})
    result = await executor.execute("activate_skill", {"name": "test"})
    assert "未知工具" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_skills/test_tool_executor_skill.py -v`
Expected: FAIL (skill_manager attribute missing or no routing)

- [ ] **Step 3: Modify ToolExecutor**

在 `src/tools/tool_executor.py` 中：

1. 在 `__init__` 中添加 `skill_manager` 属性（第 14 行后）：
```python
self.skill_manager = None
```

2. 在 `execute` 方法中，MCP 路由之前（第 27 行前），添加 skill 路由：
```python
# Skill 工具路由
if tool_name == "activate_skill" and self.skill_manager:
    result = self.skill_manager.activate(arguments.get("name", ""))
    return result if result else "未找到指定的 Skill"
```

- [ ] **Step 4: Run all tests to verify**

Run: `python -m pytest tests/test_skills/test_tool_executor_skill.py tests/test_mcp_manager.py -v`
Expected: all tests PASS（包括现有 MCP 路由测试）

- [ ] **Step 5: Commit**

```bash
git add src/tools/tool_executor.py tests/test_skills/test_tool_executor_skill.py
git commit -m "feat(skills): add activate_skill routing to ToolExecutor"
```

---

### Task 6: Orchestrator 集成 — catalog 注入 + activate_skill 处理

**Files:**
- Modify: `src/agents/orchestrator.py:120-167`

- [ ] **Step 1: Modify on_enter_orchestrating**

在 `src/agents/orchestrator.py` 的 `on_enter_orchestrating` 方法中做三处修改：

**修改 1** — 第 152-154 行，构建 system_prompt 之后追加 skill catalog 和预激活的 skill 内容：

```python
            # 构建系统提示
            system_prompt = model.registry.build_orchestrator_system_prompt()
            if memory_sections:
                system_prompt += "\n\n" + "\n\n".join(memory_sections)

            # 追加 Skill catalog（新增）
            skill_manager = getattr(model.tool_executor, "skill_manager", None)
            if skill_manager:
                catalog = skill_manager.get_catalog_prompt()
                if catalog:
                    system_prompt += "\n\n" + catalog

            # 注入斜杠命令预激活的 skill 内容（新增）
            skill_content = model.data.get("skill_content")
            if skill_content:
                system_prompt += "\n\n" + skill_content
```

**修改 2** — 第 163-167 行，工具列表追加 activate_skill：

```python
        # 调用总控 LLM
        transfer_schema = model.registry.build_transfer_tool_schema()
        orchestrator_tools = [transfer_schema]

        # 追加 activate_skill 工具（新增）
        skill_manager = getattr(model.tool_executor, "skill_manager", None)
        if skill_manager:
            activate_schema = skill_manager.build_activate_tool_schema()
            if activate_schema:
                orchestrator_tools.append(activate_schema)

        content, tool_calls, _ = await call_model(
            model.messages,
            tools=orchestrator_tools,
        )
```

**修改 3** — 在 tool_calls 处理逻辑中（第 170-203 行），添加 activate_skill 处理。在 `if tool_calls:` 块内、现有 transfer_to_agent 检测之前：

```python
        if tool_calls:
            # 检查是否有 activate_skill 调用（新增）
            skill_activated = False
            for tc in tool_calls.values():
                if tc.get("name") == "activate_skill":
                    try:
                        args = json.loads(tc["arguments"])
                        skill_content = skill_manager.activate(args.get("name", "")) if skill_manager else None
                    except (json.JSONDecodeError, KeyError):
                        skill_content = None

                    model.messages.append({
                        "role": "assistant",
                        "content": content if content else None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {"name": tc["name"], "arguments": tc["arguments"]},
                            }
                        ],
                    })
                    model.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": skill_content or "Skill not found.",
                    })
                    skill_activated = True
                    break

            if skill_activated:
                # Skill 已激活，递归调用自身重新查询 LLM（skill 内容已在 messages 中）
                # 深度保护：最多允许 3 次 skill 激活，防止无限递归
                depth = model.data.get("_skill_activation_depth", 0)
                if depth < 3:
                    model.data["_skill_activation_depth"] = depth + 1
                    await self.on_enter_orchestrating()
                else:
                    logger.warning("Skill 激活深度超过限制，停止递归")
                return

            # 原有的 transfer_to_agent 处理逻辑...
```

- [ ] **Step 2: Run existing orchestrator tests to verify no regression**

Run: `python -m pytest tests/ -v -k "not slow"`
Expected: all existing tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/agents/orchestrator.py
git commit -m "feat(skills): integrate skill catalog and activate_skill into orchestrator"
```

---

### Task 7: main.py 集成 — 初始化 + 斜杠命令路由

**Files:**
- Modify: `main.py:9,18-19,56-71,100-119`

- [ ] **Step 1: Add imports and initialization**

在 `main.py` 中：

1. 添加 import（第 9 行附近）：
```python
from src.skills import SkillManager
from config import USER_ID, MCP_CONFIG_PATH, SKILLS_DIRS
```

2. 在 `handle_input` 中（第 56 行 `async def handle_input`），护栏检查之后、Flow 检测之前，添加斜杠命令检测：

```python
async def handle_input(user_input: str, all_tools=None):
    """统一入口：护栏 → Skill 斜杠命令 → Flow 路由 → 执行"""
    effective_tools = all_tools or tools

    # 1. 护栏检查
    passed, reason = input_guard.check(user_input)
    if not passed:
        await agent_output(f"\n[安全拦截] {reason}\n")
        return

    # 2. Skill 斜杠命令检测（新增）
    skill_manager = getattr(tool_executor, "skill_manager", None)
    if skill_manager:
        skill_name = skill_manager.is_slash_command(user_input)
        if skill_name:
            skill_content = skill_manager.activate(skill_name)
            if skill_content:
                remaining = user_input[len(f"/{skill_name}"):].strip()
                actual_input = remaining or f"已激活 {skill_name} skill，请按指令执行。"
                multi_agent_flow = MultiAgentFlow(
                    registry=agent_registry,
                    memory=memory,
                    user_facts=user_facts,
                    conversation_summaries=conversation_summaries,
                    all_tools=effective_tools,
                    tool_executor=tool_executor,
                )
                multi_agent_flow.model.data["user_input"] = actual_input
                multi_agent_flow.model.data["skill_content"] = skill_content
                runner = FSMRunner(multi_agent_flow)
                await runner.run()
                return

    # 3. 关键词触发的特殊 Flow（如 /book）
    # ... (existing code unchanged)
```

3. 在 `main()` 中（第 100 行），MCP 初始化之后添加 Skills 初始化：

```python
async def main():
    # 初始化 MCP
    mcp_manager = MCPManager()
    await mcp_manager.connect_all(load_mcp_config(MCP_CONFIG_PATH))
    tool_executor.mcp_manager = mcp_manager

    # 初始化 Skills（新增）
    skill_manager = SkillManager(skill_dirs=SKILLS_DIRS)
    await skill_manager.discover()
    tool_executor.skill_manager = skill_manager

    # 合并工具列表（本地 tools + MCP tools + Skill 工具）
    mcp_schemas = mcp_manager.get_tools_schemas()
    local_names = {t["function"]["name"] for t in tools}
    for schema in mcp_schemas:
        mcp_name = schema["function"]["name"]
        if mcp_name in local_names:
            print(f"[警告] MCP 工具 '{mcp_name}' 与本地工具同名，可能产生冲突")
    all_tools = tools + mcp_schemas

    # 添加 activate_skill 工具（新增）
    activate_schema = skill_manager.build_activate_tool_schema()
    if activate_schema:
        all_tools.append(activate_schema)

    skill_count = len(skill_manager._skills)
    print("Agent 已启动，输入 'exit' 退出。")
    if mcp_schemas:
        print(f"已加载 {len(mcp_schemas)} 个 MCP 工具")
    if skill_count:
        print(f"已发现 {skill_count} 个 Skill")

    try:
        while True:
            user_input = await agent_input("\n你: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            await handle_input(user_input, all_tools)
            await agent_output("\n")
    finally:
        await mcp_manager.disconnect_all()
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -v -k "not slow"`
Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat(skills): integrate SkillManager into main.py startup and routing"
```

---

### Task 8: 示例 Skills

**Files:**
- Create: `skills/translate/SKILL.md`
- Create: `skills/code-review/SKILL.md`
- Create: `skills/code-review/references/checklist.md`

- [ ] **Step 1: Create translate skill**

```markdown
<!-- skills/translate/SKILL.md -->
---
name: translate
description: 翻译文本到指定语言。当用户要求翻译、translation 时使用。
---

## 使用场景

当用户需要翻译文本时激活此 skill。

## 翻译指令

1. 确认目标语言（如果用户未指定，询问目标语言）
2. 保持原文的格式和结构
3. 对专业术语提供注释
4. 翻译完成后简要说明翻译策略
```

- [ ] **Step 2: Create code-review skill**

```markdown
<!-- skills/code-review/SKILL.md -->
---
name: code-review
description: 审查代码质量，检查 bug、安全问题和风格一致性。当用户要求代码审查、code review、review 时使用。
---

## 使用场景

当用户要求审查代码文件时激活此 skill。

## 审查流程

1. 使用 read_file 工具读取指定的代码文件
2. 按照 [审查清单](references/checklist.md) 逐项检查
3. 汇总发现的问题，按严重程度排序
4. 给出具体的改进建议和代码示例

## 输出格式

按以下结构输出审查结果：

### 概要
- 文件：<文件路径>
- 总体评价：<一句话总结>

### 问题列表
| 严重程度 | 行号 | 描述 | 建议 |
|---------|------|------|------|
| 高/中/低 | L42 | 问题描述 | 修复建议 |

### 改进建议
逐条列出可以优化的地方。
```

```markdown
<!-- skills/code-review/references/checklist.md -->
# 代码审查清单

## 正确性
- [ ] 逻辑是否正确，边界条件是否处理
- [ ] 异常处理是否完善
- [ ] 返回值是否符合预期

## 安全性
- [ ] 是否存在注入风险（SQL、命令、XSS）
- [ ] 敏感数据是否正确处理
- [ ] 输入是否经过验证

## 风格与可读性
- [ ] 命名是否清晰一致
- [ ] 函数是否过长（超过 50 行考虑拆分）
- [ ] 注释是否必要且准确

## 性能
- [ ] 是否有明显的性能问题
- [ ] 循环中是否有不必要的重复操作
- [ ] 数据结构选择是否合适
```

- [ ] **Step 3: Verify skills can be discovered**

Run: `python -c "import asyncio; from src.skills import SkillManager; mgr = SkillManager(['skills/']); asyncio.run(mgr.discover()); print([s.name for s in mgr._skills.values()])"`
Expected: `['code-review', 'translate']`

- [ ] **Step 4: Commit**

```bash
git add skills/
git commit -m "feat(skills): add example skills (translate, code-review)"
```

---

### Task 9: 集成测试

**Files:**
- Create: `tests/test_skills/test_integration.py`

- [ ] **Step 1: Write integration tests**

```python
# tests/test_skills/test_integration.py
"""端到端集成测试：Skill 发现 → 激活 → 工具路由。"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path
from src.skills import SkillManager
from src.tools.tool_executor import ToolExecutor


def _make_skill(base, name, description="A skill.", body="## Do things"):
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}"
    )
    return skill_dir


@pytest.mark.asyncio
async def test_discover_activate_end_to_end(tmp_path):
    """Full cycle: discover → catalog → activate → verify content."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _make_skill(skills_dir, "test-skill", "A test skill.", "## Steps\n1. Do it")

    mgr = SkillManager(skill_dirs=[str(skills_dir)])
    await mgr.discover()

    # Catalog contains the skill
    catalog = mgr.get_catalog_prompt()
    assert "test-skill" in catalog

    # Activate returns wrapped content
    result = mgr.activate("test-skill")
    assert '<skill_content name="test-skill">' in result
    assert "## Steps" in result
    assert "1. Do it" in result


@pytest.mark.asyncio
async def test_tool_executor_skill_routing(tmp_path):
    """ToolExecutor correctly routes activate_skill to SkillManager."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _make_skill(skills_dir, "my-skill", "My skill.", "## Body")

    mgr = SkillManager(skill_dirs=[str(skills_dir)])
    await mgr.discover()

    executor = ToolExecutor({})
    executor.skill_manager = mgr

    result = await executor.execute("activate_skill", {"name": "my-skill"})
    assert "skill_content" in result
    assert "## Body" in result


@pytest.mark.asyncio
async def test_slash_command_full_flow(tmp_path):
    """Slash command detection + activation flow."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _make_skill(skills_dir, "translate", "Translate text.", "## Translate\n1. Do it")

    mgr = SkillManager(skill_dirs=[str(skills_dir)])
    await mgr.discover()

    # Slash command detected
    skill_name = mgr.is_slash_command("/translate hello world")
    assert skill_name == "translate"

    # Activate returns content
    result = mgr.activate(skill_name)
    assert "## Translate" in result


@pytest.mark.asyncio
async def test_activate_tool_schema_used_by_executor(tmp_path):
    """activate_skill schema has correct enum matching discovered skills."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _make_skill(skills_dir, "a-skill", "Skill A.")
    _make_skill(skills_dir, "b-skill", "Skill B.")

    mgr = SkillManager(skill_dirs=[str(skills_dir)])
    await mgr.discover()

    schema = mgr.build_activate_tool_schema()
    enum = schema["function"]["parameters"]["properties"]["name"]["enum"]
    assert "a-skill" in enum
    assert "b-skill" in enum
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/test_skills/test_integration.py -v`
Expected: all tests PASS

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v -k "not slow"`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_skills/test_integration.py
git commit -m "test(skills): add integration tests for skill discovery and activation"
```

---

### Task 10: 最终验证与清理

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v -k "not slow"`
Expected: all tests PASS

- [ ] **Step 2: Verify example skills load correctly**

Run: `python -c "import asyncio; from src.skills import SkillManager; mgr = SkillManager(['skills/', '.agents/skills/']); asyncio.run(mgr.discover()); [print(f'  {s.name}: {s.description}') for s in mgr._skills.values()]"`
Expected: lists translate and code-review skills

- [ ] **Step 3: Verify activate output**

Run: `python -c "import asyncio; from src.skills import SkillManager; mgr = SkillManager(['skills/']); asyncio.run(mgr.discover()); print(mgr.activate('code-review')[:500])"`
Expected: shows `<skill_content name="code-review">` with checklist reference in resources

- [ ] **Step 4: Commit final state**

如果有任何遗留修改：
```bash
git add -A
git commit -m "feat(skills): complete Agent Skills protocol support"
```
