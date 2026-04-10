# Skill 步骤指令注入重构设计

## 问题

当前 skill 工作流执行链路中，每个步骤的 agent 只拿到从 SKILL.md 中提取的碎片化指令，
丢失了完整的技能上下文，导致 LLM 无法正确理解和执行步骤任务。

### 根因分析

1. **Parser 的 section→step 匹配失败**：dot 节点名（如 `"Explore project context"`）和
   prose section 名（如 `"Understanding the idea"`）无法匹配，导致详细指令丢失，退化为
   checklist 的一句话摘要（`"check files, docs, recent commits"`）。

2. **Agent 只拿到碎片化指令**：`agent_factory` 把 terse 摘要直接作为 system prompt，
   agent 不知道自己在执行什么技能、完整流程是什么、该步骤的详细指导在哪里。

3. **WorkflowStep.instructions 职责混乱**：同时承载"给 compiler 的元信息"和
   "给 agent 的执行指令"两种职责，但实际内容只够做前者。

### 根本矛盾

SKILL.md 是面向 LLM 阅读的自然语言文档，不是面向解析器的结构化 DSL。
用正则从中提取 per-step 指令注定丢失语义。

## 方案

**全文注入 + Parser 简化 + Prompt Cache 优化**

- 每个 ACTION 步骤的 agent 拿到完整 SKILL.md 内容
- Parser 只提取拓扑（dot graph）和 checklist 概要，不再做 section→step 匹配
- 利用 prompt caching：将不变的 SKILL.md 内容放在 system prompt，步骤信息放在 user message

## 设计

### 1. 消息结构（prompt cache 友好）

所有 ACTION 步骤共享相同的 system prompt，步骤差异通过 user message 区分：

```
system prompt（所有步骤完全相同，可被 prompt cache 命中）:
  ## 技能文档
  {完整 SKILL.md body}

  ## 用户需求
  {user_input}

  ## 约束
  {constraints}

conversation_history（自然累积）:
  [前序步骤的对话记录]

user message（每步不同，即 agent.task）:
  请执行步骤「Explore project context」：check files, docs, recent commits
```

Token 开销：SKILL.md 首次调用后走缓存，后续步骤只有 user message 的增量开销。

### 2. WorkflowPlan 扩展

`WorkflowPlan` 新增 `full_body` 字段，保存完整 SKILL.md 内容：

```python
# src/graph/workflow.py

@dataclass
class WorkflowPlan:
    name: str
    steps: list[WorkflowStep]
    transitions: list[WorkflowTransition]
    entry_step: str
    constraints: list[str] = field(default_factory=list)
    full_body: str = ""  # 新增：完整 SKILL.md 内容
```

`WorkflowStep.instructions` 职责明确为 **checklist 概要描述**，仅用于：
- 步骤定位（告诉 agent "你在哪"）
- 日志/调试输出
- DecisionNode 的 question 文本

### 3. Parser 简化

`SkillWorkflowParser` 不再需要 `_extract_sections()` 做 section→step 匹配。

改动：
- 删除 `_extract_sections()` 方法
- `_parse_dot()` 中 `instructions` 只从 `checklist_map` 获取概要
- `parse()` 返回的 `WorkflowPlan` 携带 `full_body`

```python
# src/skills/workflow_parser.py

class SkillWorkflowParser:
    def parse(self, content: str, skill_name: str) -> WorkflowPlan:
        dot_block = self._extract_dot(content)
        checklist = self._extract_checklist(content)
        constraints = self._extract_constraints(content)

        # 提取 SKILL.md body（去掉 frontmatter）
        body = self._extract_body(content)

        if dot_block:
            plan = self._parse_dot(dot_block, checklist, constraints, skill_name)
        elif checklist:
            plan = self._parse_checklist(checklist, constraints, skill_name)
        else:
            plan = self._parse_fallback(content, skill_name)

        plan.full_body = body
        return plan

    def _extract_body(self, content: str) -> str:
        """提取 SKILL.md 正文（去掉 YAML frontmatter）。"""
        if content.startswith("---"):
            parts = content.split("---", 2)
            return parts[2].strip() if len(parts) > 2 else content
        return content
```

`_parse_dot()` 简化——instructions 只取 checklist 概要：

```python
def _parse_dot(self, dot, checklist, constraints, skill_name):
    checklist_map = {name.strip(): desc.strip() for name, desc in checklist}
    steps = []
    # ...
    for match in _DOT_NODE_RE.finditer(dot):
        name = match.group(1)
        # ...
        step_id = _slugify(name)
        instructions = checklist_map.get(name, "")  # 只取 checklist 概要
        steps.append(WorkflowStep(
            id=step_id, name=name,
            instructions=instructions, step_type=step_type,
            subworkflow_skill=subworkflow_skill,
        ))
    # ...
```

### 4. Compiler 适配

`WorkflowCompiler.compile()` 的 `agent_factory` 签名扩展，传入 `full_body`：

```python
# src/skills/compiler.py

class WorkflowCompiler:
    def compile(
        self,
        plan: WorkflowPlan,
        agent_factory: Callable[[str, str, str], Agent],  # (step_id, step_name, checklist_desc)
        skill_manager: SkillManager | None = None,
    ) -> CompiledGraph:
        # ...
        for step in plan.steps:
            match step.step_type:
                case StepType.ACTION:
                    # decision hint 追加到 checklist_desc
                    checklist_desc = step.instructions
                    decision_hint = self._build_decision_hint(step.id, plan, step_map)
                    if decision_hint:
                        checklist_desc += decision_hint
                    agent = agent_factory(step.id, step.name, checklist_desc)
                    node = AgentNode(agent)
                    node.name = step.id
                    builder.add_node(node)
                # ... DECISION / TERMINAL / SUBWORKFLOW 不变
```

### 5. agent_factory 重写

`_handle_skill()` 中构建共享 system prompt，每步只变 task：

```python
# src/app/app.py

async def _handle_skill(self, user_input: str, skill_name: str) -> None:
    from src.skills.workflow_parser import SkillWorkflowParser
    from src.skills.compiler import WorkflowCompiler
    from src.agents.agent import Agent

    # 1. 激活 skill
    content = self.skill_manager.activate(skill_name)
    if not content:
        return
    remaining = user_input[len(f"/{skill_name}"):].strip()
    actual_input = remaining or f"已激活 {skill_name} skill，请按指令执行。"

    # 2. 解析 → WorkflowPlan（携带 full_body）
    parser = SkillWorkflowParser()
    workflow = parser.parse(content, skill_name)

    # 3. 构建共享 system prompt（所有步骤相同，prompt cache 友好）
    constraint_text = ""
    if workflow.constraints:
        lines = "\n".join(f"- {c}" for c in workflow.constraints)
        constraint_text = f"\n\n## 约束\n{lines}"

    shared_system_prompt = (
        f"## 技能文档\n{workflow.full_body}"
        f"\n\n## 用户需求\n{actual_input}"
        f"{constraint_text}"
    )

    # 4. agent_factory：共享 instructions，步骤信息在 task 中
    def make_step_agent(step_id: str, step_name: str, checklist_desc: str) -> Agent:
        return Agent(
            name=f"step_{step_id}",
            description=f"Workflow step: {step_id}",
            instructions=shared_system_prompt,
            task=f"请执行步骤「{step_name}」：{checklist_desc}",
            handoffs=[],
        )

    # 5. 编译 → 执行（不变）
    compiler = WorkflowCompiler()
    skill_graph = compiler.compile(workflow, make_step_agent, self.skill_manager)
    skill_engine = GraphEngine()
    ctx = RunContext(input=actual_input, state=DynamicState(), deps=self.deps)
    result = await skill_engine.run(skill_graph, ctx)

    # 6. 输出（不变）
    # ...
```

### 6. Compiler 中 constraint_prefix 移除

约束不再在 compiler 中注入（已移到 shared_system_prompt），compiler 的 ACTION 分支简化：

```python
case StepType.ACTION:
    checklist_desc = step.instructions
    decision_hint = self._build_decision_hint(step.id, plan, step_map)
    if decision_hint:
        checklist_desc += decision_hint
    agent = agent_factory(step.id, step.name, checklist_desc)
```

## 改动文件清单

| 文件 | 改动 | 类型 |
|------|------|------|
| `src/graph/workflow.py` | `WorkflowPlan` 新增 `full_body: str` | 扩展 |
| `src/skills/workflow_parser.py` | 删除 `_extract_sections()`，新增 `_extract_body()`，`_parse_dot` 简化 | 简化 |
| `src/skills/compiler.py` | `agent_factory` 签名改为 `(step_id, step_name, checklist_desc)`，移除 `constraint_prefix` | 重构 |
| `src/app/app.py` | `make_step_agent` 使用 `shared_system_prompt` + 步骤 task | 重构 |

## 不变的部分

- `src/graph/engine.py` — 图引擎拓扑执行逻辑
- `src/graph/nodes.py` — Decision/Terminal/Subworkflow 节点
- `src/agents/runner.py` — AgentRunner 工具循环
- `src/agents/agent.py` — Agent 数据模型
- `src/agents/delegate.py` — DelegateToolProvider
- `src/skills/manager.py` — SkillManager 发现/激活

## 验证标准

1. 使用 `/brainstorming 设计一个简单的aoi算法` 测试，第一步 `explore_project_context`
   的 agent 应收到完整 SKILL.md 内容，而非 `"check files, docs, recent commits"`
2. system prompt 在所有 ACTION 步骤间完全相同（可通过日志确认 prompt cache 可命中）
3. Decision 节点正常工作（用户交互、分支选择）
4. 现有测试通过
