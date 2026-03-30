"""工具分类流水线测试。"""
import json
import pytest
from unittest.mock import AsyncMock


def test_extract_category_hints():
    from src.tools.classifier import extract_category_hints
    schemas = [
        {"function": {"name": "t1", "description": "[Filesystem] Read a file", "parameters": {}}},
        {"function": {"name": "t2", "description": "[Terminal] Run command", "parameters": {}}},
        {"function": {"name": "t3", "description": "Calculate math", "parameters": {}}},
    ]
    hints = extract_category_hints(schemas)
    assert hints["t1"] == "Filesystem"
    assert hints["t2"] == "Terminal"
    assert "t3" not in hints


def test_build_classify_prompt():
    from src.tools.classifier import build_classify_prompt
    schemas = [
        {"function": {"name": "read_file", "description": "读取文件", "parameters": {}}},
        {"function": {"name": "calculate", "description": "数学计算", "parameters": {}}},
    ]
    hints = {"read_file": "Filesystem"}
    prompt = build_classify_prompt(schemas, hints, max_per_category=8)
    assert "read_file" in prompt
    assert "calculate" in prompt
    assert "Filesystem" in prompt
    assert "8" in prompt
    # dict format instruction present
    assert "tool_name" in prompt
    assert "tool_description" in prompt


def test_parse_classify_response_valid():
    from src.tools.classifier import parse_classify_response
    raw = json.dumps({
        "categories": [
            {"name": "filesystem", "description": "文件操作", "tools": {"read_file": "Read a file", "write_file": "Write a file"}},
            {"name": "calculation", "description": "计算", "tools": {"calculate": "Perform math"}},
        ]
    })
    result = parse_classify_response(raw)
    assert "tool_filesystem" in result
    assert result["tool_filesystem"]["tools"] == {"read_file": "Read a file", "write_file": "Write a file"}
    assert "tool_calculation" in result


def test_parse_classify_response_code_block():
    from src.tools.classifier import parse_classify_response
    raw = '```json\n{"categories": [{"name": "test", "description": "d", "tools": {"t1": "Tool 1"}}]}\n```'
    result = parse_classify_response(raw)
    assert "tool_test" in result
    assert result["tool_test"]["tools"] == {"t1": "Tool 1"}


def test_parse_classify_response_invalid_json():
    from src.tools.classifier import parse_classify_response
    with pytest.raises(ValueError, match="JSON"):
        parse_classify_response("not json")


def test_build_split_prompt():
    from src.tools.classifier import build_split_prompt
    category = {
        "description": "文件操作",
        "tools": {f"t{i}": f"Tool {i}" for i in range(10)},
    }
    prompt = build_split_prompt("filesystem", category, max_per_category=8)
    assert "filesystem" in prompt
    assert "t0" in prompt
    assert "8" in prompt
    # dict format instruction present
    assert "tool_name" in prompt
    assert "tool_description" in prompt


@pytest.mark.asyncio
async def test_classify_tools_pipeline():
    from src.tools.classifier import classify_tools
    from src.llm.types import LLMResponse

    schemas = [
        {"function": {"name": "read_file", "description": "读取文件", "parameters": {}}},
        {"function": {"name": "write_file", "description": "写入文件", "parameters": {}}},
        {"function": {"name": "calculate", "description": "数学计算", "parameters": {}}},
    ]
    llm_response = json.dumps({
        "categories": [
            {"name": "filesystem", "description": "文件操作", "tools": {"read_file": "Read a file", "write_file": "Write a file"}},
            {"name": "calculation", "description": "计算", "tools": {"calculate": "Perform math"}},
        ]
    })
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=LLMResponse(content=llm_response, finish_reason="stop"))
    result = await classify_tools(schemas, mock_llm, max_per_category=8)
    assert "tool_filesystem" in result
    assert "tool_calculation" in result
    assert result["tool_filesystem"]["tools"] == {"read_file": "Read a file", "write_file": "Write a file"}


@pytest.mark.asyncio
async def test_classify_tools_with_overflow_split():
    from src.tools.classifier import classify_tools
    from src.llm.types import LLMResponse

    schemas = [{"function": {"name": f"t{i}", "description": f"Tool {i}", "parameters": {}}} for i in range(10)]
    first_response = json.dumps({
        "categories": [
            {"name": "big_group", "description": "所有工具", "tools": {f"t{i}": f"Tool {i}" for i in range(10)}},
        ]
    })
    split_response = json.dumps({
        "subcategories": [
            {"name": "group_a", "description": "A 组", "tools": {f"t{i}": f"Tool {i}" for i in range(5)}},
            {"name": "group_b", "description": "B 组", "tools": {f"t{i}": f"Tool {i}" for i in range(5, 10)}},
        ]
    })
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(side_effect=[
        LLMResponse(content=first_response, finish_reason="stop"),
        LLMResponse(content=split_response, finish_reason="stop"),
    ])
    result = await classify_tools(schemas, mock_llm, max_per_category=5)
    assert "tool_big_group" not in result
    assert "tool_big_group_group_a" in result
    assert "tool_big_group_group_b" in result
    assert len(result["tool_big_group_group_a"]["tools"]) == 5


@pytest.mark.asyncio
async def test_classify_tools_empty_schemas():
    from src.tools.classifier import classify_tools
    mock_llm = AsyncMock()
    result = await classify_tools([], mock_llm)
    assert result == {}
    mock_llm.chat.assert_not_called()
