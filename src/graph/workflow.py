"""Skill 工作流的中间表示模型。

SkillWorkflowParser 将 markdown 解析为 WorkflowPlan，
WorkflowCompiler 将 WorkflowPlan 编译为 CompiledGraph。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StepType(Enum):
    """工作流步骤类型，对应 dot graph 中的 shape。"""
    ACTION = "action"
    DECISION = "decision"
    TERMINAL = "terminal"
    SUBWORKFLOW = "subworkflow"


@dataclass
class WorkflowStep:
    """工作流的一个步骤。"""
    id: str
    name: str
    instructions: str
    step_type: StepType
    subworkflow_skill: str | None = None


@dataclass
class WorkflowTransition:
    """步骤间的转换，可带条件标签。"""
    from_step: str
    to_step: str
    condition: str | None = None


@dataclass
class WorkflowPlan:
    """从 skill markdown 解析出的完整工作流。"""
    name: str
    steps: list[WorkflowStep]
    transitions: list[WorkflowTransition]
    entry_step: str
    constraints: list[str] = field(default_factory=list)
    full_body: str = ""  # 完整 SKILL.md 正文（去掉 frontmatter）
