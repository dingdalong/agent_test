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
