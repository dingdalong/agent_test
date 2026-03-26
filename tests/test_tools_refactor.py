"""Tests for refactored tools foundation modules."""
import pytest
from pydantic import BaseModel, Field

from src.tools.schemas import ToolDict
from src.tools.registry import ToolEntry, ToolRegistry
from src.tools.decorator import tool, get_registry


# === Helpers ===

class DummyModel(BaseModel):
    value: str = Field(description="test value")


def _make_entry(name: str = "dummy", sensitive: bool = False) -> ToolEntry:
    async def dummy_func(value: str) -> str:
        return value
    return ToolEntry(
        name=name,
        func=dummy_func,
        model=DummyModel,
        description="A dummy tool",
        parameters_schema=DummyModel.model_json_schema(),
        sensitive=sensitive,
        confirm_template=None,
    )


# === schemas tests ===

def test_tool_dict_type():
    td: ToolDict = {
        "type": "function",
        "function": {
            "name": "test",
            "description": "a test tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    assert td["type"] == "function"
    assert td["function"]["name"] == "test"


# === registry tests ===

def test_registry_register_and_get():
    reg = ToolRegistry()
    entry = _make_entry("test_tool")
    reg.register(entry)
    assert reg.has("test_tool")
    assert reg.get("test_tool") is entry
    assert reg.get("nonexistent") is None


def test_registry_duplicate_skips():
    reg = ToolRegistry()
    entry1 = _make_entry("dup")
    entry2 = _make_entry("dup")
    reg.register(entry1)
    reg.register(entry2)
    assert reg.get("dup") is entry1
    assert len(reg.list_entries()) == 1


def test_registry_get_schemas():
    reg = ToolRegistry()
    reg.register(_make_entry("tool_a"))
    reg.register(_make_entry("tool_b"))
    schemas = reg.get_schemas()
    assert len(schemas) == 2
    names = {s["function"]["name"] for s in schemas}
    assert names == {"tool_a", "tool_b"}
    assert all(s["type"] == "function" for s in schemas)


# === decorator tests ===

def test_tool_decorator_registers():
    registry = get_registry()
    initial_count = len(registry.list_entries())

    class TestParams(BaseModel):
        x: int = Field(description="a number")

    @tool(model=TestParams, description="test decorator tool")
    async def _test_decorator_func(x: int) -> str:
        return str(x)

    assert registry.has("_test_decorator_func")
    entry = registry.get("_test_decorator_func")
    assert entry.description == "test decorator tool"
    assert entry.sensitive is False
    assert len(registry.list_entries()) == initial_count + 1


def test_tool_decorator_sensitive():
    class SensParams(BaseModel):
        target: str = Field(description="target")

    @tool(model=SensParams, description="sensitive tool", sensitive=True,
          confirm_template="操作 {target}")
    async def _test_sensitive_func(target: str) -> str:
        return target

    registry = get_registry()
    entry = registry.get("_test_sensitive_func")
    assert entry.sensitive is True
    assert entry.confirm_template == "操作 {target}"


def test_tool_decorator_custom_name():
    class NameParams(BaseModel):
        v: str = Field(description="value")

    @tool(model=NameParams, description="custom name", name="my_custom_tool")
    async def _some_internal_func(v: str) -> str:
        return v

    registry = get_registry()
    assert registry.has("my_custom_tool")
    assert not registry.has("_some_internal_func")


# === discovery tests ===

def test_discover_tools(tmp_path):
    import sys
    from src.tools.discovery import discover_tools

    pkg_dir = tmp_path / "fake_tools"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "sample.py").write_text("LOADED = True\n")

    sys.path.insert(0, str(tmp_path))
    try:
        discover_tools("fake_tools", pkg_dir)
        import fake_tools.sample
        assert fake_tools.sample.LOADED is True
    finally:
        sys.path.remove(str(tmp_path))


def test_discover_tools_skips_init(tmp_path):
    import sys
    from src.tools.discovery import discover_tools

    pkg_dir = tmp_path / "skip_test"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("INIT_LOADED = True\n")
    (pkg_dir / "real.py").write_text("REAL_LOADED = True\n")

    sys.path.insert(0, str(tmp_path))
    try:
        discover_tools("skip_test", pkg_dir)
        import skip_test.real
        assert skip_test.real.REAL_LOADED is True
    finally:
        sys.path.remove(str(tmp_path))
