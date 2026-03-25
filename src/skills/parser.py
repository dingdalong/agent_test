"""SKILL.md 文件解析器。"""

import logging
from pathlib import Path
from typing import Optional

import yaml

from .models import SkillInfo

logger = logging.getLogger(__name__)


def find_skill_md(skill_dir: Path) -> Optional[Path]:
    """查找 SKILL.md 文件，优先大写，兼容小写。

    在大小写不敏感的文件系统（如 macOS HFS+）上，通过比较目录实际条目名称来
    区分 SKILL.md 和 skill.md，避免误匹配。
    """
    try:
        entries = {e.name: e for e in skill_dir.iterdir() if e.is_file()}
    except (OSError, NotADirectoryError):
        return None

    for name in ("SKILL.md", "skill.md"):
        if name in entries:
            return entries[name]
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
