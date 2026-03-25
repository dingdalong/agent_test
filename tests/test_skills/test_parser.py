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
        assert info.location == (skill_dir / "SKILL.md").resolve()
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
