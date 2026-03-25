"""
structured_output — 结构化输出

利用 function calling 机制约束 LLM 按 Pydantic 模型输出结构化 JSON。

用法：
    1. 定义 Pydantic 输出模型
    2. 调用 build_output_schema() 生成工具 schema
    3. 将 schema 作为 tools 参数传给 call_model
    4. 调用 parse_output() 从 tool_calls 中解析并验证结果

示例：
    class MyResult(BaseModel):
        score: float
        label: str

    schema = build_output_schema("submit_result", "提交结果", MyResult)
    _, tool_calls, _ = await call_model(messages, tools=[schema])
    result = parse_output(tool_calls, "submit_result", MyResult)
    # result 是 MyResult 实例或 None
"""
import json
import logging
from typing import Dict, Optional, Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


def build_output_schema(name: str, description: str, model: Type[BaseModel]) -> dict:
    """从 Pydantic 模型构建结构化输出的 tool schema。

    将 Pydantic 模型转为 OpenAI function calling 格式的 schema，
    传给 call_model 的 tools 参数，约束 LLM 输出为模型定义的 JSON 结构。
    """
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema.pop("description", None)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        }
    }


def parse_output(
    tool_calls: Dict[int, Dict[str, str]],
    name: str,
    model: Type[BaseModel]
) -> Optional[BaseModel]:
    """从 LLM 的 tool_calls 中解析结构化输出，用 Pydantic 模型验证。

    返回模型实例或 None（未匹配到或验证失败）。
    """
    for tc in tool_calls.values():
        if tc.get("name") == name:
            try:
                data = json.loads(tc["arguments"])
                return model(**data)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"结构化输出 '{name}' 解析失败: {e}")
                return None
    return None
