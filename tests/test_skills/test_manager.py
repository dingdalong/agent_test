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
    (skills_dir / "bad").mkdir()  # no SKILL.md
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
    await mgr.discover()
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
    assert "activate_skill" in prompt


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
    assert result2 is not None
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
    assert mgr.is_slash_command("translate") is None
    assert mgr.is_slash_command("hello world") is None


@pytest.mark.asyncio
async def test_slash_command_reserved_prefix(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _make_skill(skills_dir, "plan", "A plan skill.")
    mgr = SkillManager(skill_dirs=[str(skills_dir)])
    await mgr.discover()
    assert mgr.is_slash_command("/plan something") is None
