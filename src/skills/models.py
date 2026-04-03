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
        license: 许可证信息
        compatibility: 环境要求（1-500 字符）
        metadata: 任意键值对（string→string 映射）
    """

    name: str
    description: str
    location: Path
    body: Optional[str] = None
    allowed_tools: Optional[str] = None
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: Optional[dict[str, str]] = None
