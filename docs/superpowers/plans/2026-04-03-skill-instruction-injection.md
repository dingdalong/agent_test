# Skill 步骤指令注入重构 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让每个 skill 工作流步骤的 agent 拿到完整 SKILL.md 上下文，而非碎片化的 checklist 摘要。

**Architecture:** Parser 只提取拓扑和 checklist 概要，完整 SKILL.md body 通过 WorkflowPlan.full_body 传递。所有 ACTION 步骤共享相同的 system prompt（含完整 skill 文档），步骤差异通过 agent.task（user message）区分，利用 prompt cache。

**Tech Stack:** Python 3.13, dataclasses, regex

---

### Task 1: WorkflowPlan 新增 full_body 字段

**Files:**
- Modify: `src/graph/workflow.py:39-45`
- Test: `tests/skills/test_workflow_parser.py`

- [ ] **Step 1: 写失败测试**

在 `tests/skills/test_workflow_parser.py` 末尾新增测试类：

```python
class TestFullBodyExtraction:
    def test_dot_skill_has_full_body(self):
        parser = SkillWorkflowParser()
        plan = parser.parse(SAMPLE_SKILL_WITH_DOT, "test-skill")
        assert plan.full_body != ""
        assert "# Test Skill" in plan.full_body
        assert "Step one" in plan.full_body

    def test_checklist_only_skill_has_full_body(self):
        parser = SkillWorkflowParser()
        plan = parser.parse(SAMPLE_SKILL_CHECKLIST_ONLY, "simple")
        assert "# Simple Skill" in plan.full_body

    def test_fallback_skill_has_full_body(self):
        parser = SkillWorkflowParser()
        plan = parser.parse(SAMPLE_SKILL_NO_STRUCTURE, "plain")
        assert "Just do whatever" in plan.full_body

    def test_frontmatter_stripped_from_full_body(self):
        content = "---\nname: test\n---\n# Body\nContent here"
        parser = SkillWorkflowParser()
        plan = parser.parse(content, "test")
        assert "name: test" not in plan.full_body
        assert "# Body" in plan.full_body
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/skills/test_workflow_parser.py::TestFullBodyExtraction -v
```

预期：FAIL，`WorkflowPlan` 没有 `full_body` 属性。

- [ ] **Step 3: 实现 full_body 字段**

修改 `src/graph/workflow.py`，在 `WorkflowPlan` 中新增字段：

```python
@dataclass
class WorkflowPlan:
    """从 skill markdown 解析出的完整工作流。"""
    name: str
    steps: list[WorkflowStep]
    transitions: list[WorkflowTransition]
    entry_step: str
    constraints: list[str] = field(default_factory=list)
    full_body: str = ""  # 完整 SKILL.md 正文（去掉 frontmatter）
```

- [ ] **Step 4: 运行测试确认仍失败**

```bash
uv run pytest tests/skills/test_workflow_parser.py::TestFullBodyExtraction -v
```

预期：仍然 FAIL，因为 parser 还没有填充 `full_body`。（这验证了字段存在但 parser 未填充）

- [ ] **Step 5: 提交**

```bash
git add src/graph/workflow.py tests/skills/test_workflow_parser.py
git commit -m "feat(workflow): add full_body field to WorkflowPlan"
```

---

### Task 2: Parser 简化——删除 section 匹配，新增 full_body 提取

**Files:**
- Modify: `src/skills/workflow_parser.py:57-68` (parse 方法), `src/skills/workflow_parser.py:77-88` (_extract_sections 删除), `src/skills/workflow_parser.py:102-160` (_parse_dot 简化)
- Modify: `tests/skills/test_workflow_parser.py`

- [ ] **Step 1: 更新现有测试——移除 section 匹配相关断言**

`test_instructions_from_sections` 当前断言 `"first thing carefully" in by_name["Step one"].instructions`。
重构后 instructions 只保留 checklist 概要，不再包含 section 内容。
将此测试改为验证 instructions 是 checklist 概要：

```python
def test_instructions_from_checklist(self):
    """重构后 instructions 只包含 checklist 概要，不再匹配 section 内容。"""
    parser = SkillWorkflowParser()
    plan = parser.parse(SAMPLE_SKILL_WITH_DOT, "test-skill")
    by_name = {s.name: s for s in plan.steps}
    # instructions 来自 checklist 概要，不再来自 **Step one:** section
    assert by_name["Step one"].instructions == "do first thing"
    assert by_name["Step two"].instructions == "do second thing"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/skills/test_workflow_parser.py::TestDotGraphParsing::test_instructions_from_checklist -v
```

预期：FAIL，当前 parser 返回的 instructions 包含 section 内容。

- [ ] **Step 3: 实现 parser 改动**

修改 `src/skills/workflow_parser.py`：

**3a.** 新增 `_extract_body` 方法（在 `_extract_dot` 之后）：

```python
def _extract_body(self, content: str) -> str:
    """提取 SKILL.md 正文（去掉 YAML frontmatter）。"""
    if content.startswith("---"):
        parts = content.split("---", 2)
        return parts[2].strip() if len(parts) > 2 else content
    return content
```

**3b.** 修改 `parse()` 方法——删除 `_extract_sections` 调用，新增 `_extract_body`，设置 `full_body`：

```python
def parse(self, content: str, skill_name: str) -> WorkflowPlan:
    dot_block = self._extract_dot(content)
    checklist = self._extract_checklist(content)
    constraints = self._extract_constraints(content)
    body = self._extract_body(content)

    if dot_block:
        plan = self._parse_dot(dot_block, checklist, constraints, skill_name)
    elif checklist:
        plan = self._parse_checklist(checklist, constraints, skill_name)
    else:
        plan = self._parse_fallback(content, skill_name)

    plan.full_body = body
    return plan
```

**3c.** 简化 `_parse_dot()`——移除 `sections` 参数，instructions 只从 `checklist_map` 获取：

将方法签名从：
```python
def _parse_dot(self, dot, checklist, sections, constraints, skill_name):
```
改为：
```python
def _parse_dot(self, dot, checklist, constraints, skill_name):
```

内部 instructions 赋值从：
```python
instructions = sections.get(name, checklist_map.get(name, ""))
```
改为：
```python
instructions = checklist_map.get(name, "")
```

**3d.** 删除 `_extract_sections()` 方法（第 77-88 行整个方法）。

**3e.** 同样简化 `_parse_checklist()`——移除 `sections` 参数：

将方法签名从：
```python
def _parse_checklist(self, checklist, sections, constraints, skill_name):
```
改为：
```python
def _parse_checklist(self, checklist, constraints, skill_name):
```

内部 instructions 赋值从：
```python
instructions = sections.get(name, desc.strip())
```
改为：
```python
instructions = desc.strip()
```

- [ ] **Step 4: 运行全部 parser 测试**

```bash
uv run pytest tests/skills/test_workflow_parser.py -v
```

预期：全部 PASS（包括 Task 1 新增的 TestFullBodyExtraction 测试）。

- [ ] **Step 5: 提交**

```bash
git add src/skills/workflow_parser.py tests/skills/test_workflow_parser.py
git commit -m "refactor(parser): remove section matching, add full_body extraction"
```

---

### Task 3: Compiler 适配——新 agent_factory 签名，移除 constraint_prefix

**Files:**
- Modify: `src/skills/compiler.py:23-54` (compile 方法)
- Modify: `tests/skills/test_compiler.py`

- [ ] **Step 1: 更新测试的 agent_factory 签名和断言**

修改 `tests/skills/test_compiler.py`：

**1a.** 更新 factory 函数——从 `(name, instructions)` 改为 `(step_id, step_name, checklist_desc)`：

```python
def make_agent(step_id: str, step_name: str, checklist_desc: str):
    """创建简单 Agent 用于测试。"""
    from src.agents.agent import Agent
    return Agent(name=step_id, description="test", instructions=f"skill body here",
                 task=f"请执行步骤「{step_name}」：{checklist_desc}")


def make_prefixed_agent(step_id: str, step_name: str, checklist_desc: str):
    """模拟 app.py 的真实 factory：agent.name != step_id。"""
    from src.agents.agent import Agent
    return Agent(name=f"step_{step_id}", description="test", instructions="skill body here",
                 task=f"请执行步骤「{step_name}」：{checklist_desc}")
```

**1b.** 修改 `test_constraints_injected_into_instructions`——约束不再由 compiler 注入到 agent.instructions，改为验证 compiler 不再注入约束：

```python
def test_constraints_not_injected_by_compiler(self):
    """重构后约束由 app.py 的 shared_system_prompt 注入，compiler 不再处理。"""
    plan = WorkflowPlan(
        name="test",
        steps=[WorkflowStep(id="s1", name="S1", instructions="do it",
                            step_type=StepType.ACTION)],
        transitions=[],
        entry_step="s1",
        constraints=["Always be careful"],
    )
    compiler = WorkflowCompiler()
    graph = compiler.compile(plan, agent_factory=make_agent)
    agent_node = graph.nodes["s1"]
    # compiler 不再注入约束到 agent.instructions
    assert "Always be careful" not in agent_node.agent.instructions
```

**1c.** 修改 `test_action_before_decision_gets_hint`——decision hint 现在在 agent.task 中，不在 instructions 中：

```python
def test_action_before_decision_gets_hint_in_task(self):
    """ACTION 后继为 DECISION 时，decision hint 注入 agent.task。"""
    plan = WorkflowPlan(
        name="test",
        steps=[
            WorkflowStep(id="explore", name="Explore", instructions="look around",
                         step_type=StepType.ACTION),
            WorkflowStep(id="decide", name="Ready?", instructions="",
                         step_type=StepType.DECISION),
        ],
        transitions=[
            WorkflowTransition(from_step="explore", to_step="decide"),
            WorkflowTransition(from_step="decide", to_step="explore", condition="no"),
        ],
        entry_step="explore",
    )
    compiler = WorkflowCompiler()
    graph = compiler.compile(plan, agent_factory=make_agent)
    agent_node = graph.nodes["explore"]
    assert "严禁" in agent_node.agent.task
    assert "以陈述句结尾" in agent_node.agent.task
```

**1d.** 修改 `test_action_not_before_decision_no_hint`——同样检查 task 而非 instructions：

```python
def test_action_not_before_decision_no_hint(self):
    """ACTION 后继为非 DECISION 时，不注入决策提示。"""
    plan = WorkflowPlan(
        name="test",
        steps=[
            WorkflowStep(id="s1", name="S1", instructions="do",
                         step_type=StepType.ACTION),
            WorkflowStep(id="s2", name="S2", instructions="more",
                         step_type=StepType.ACTION),
        ],
        transitions=[WorkflowTransition(from_step="s1", to_step="s2")],
        entry_step="s1",
    )
    compiler = WorkflowCompiler()
    graph = compiler.compile(plan, agent_factory=make_agent)
    agent_node = graph.nodes["s1"]
    assert "严禁" not in agent_node.agent.task
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/skills/test_compiler.py -v
```

预期：多个 FAIL，因为 compiler 还是旧签名。

- [ ] **Step 3: 实现 compiler 改动**

修改 `src/skills/compiler.py`：

**3a.** 修改 `compile()` 方法签名和 ACTION 分支——移除 `constraint_prefix`，使用新 factory 签名：

```python
def compile(
    self,
    plan: WorkflowPlan,
    agent_factory: Callable[[str, str, str], Agent],  # (step_id, step_name, checklist_desc)
    skill_manager: SkillManager | None = None,
) -> CompiledGraph:
    builder = GraphBuilder()

    # 预建 step_id → step 映射
    step_map = {s.id: s for s in plan.steps}

    for step in plan.steps:
        match step.step_type:
            case StepType.ACTION:
                checklist_desc = step.instructions
                decision_hint = self._build_decision_hint(
                    step.id, plan, step_map,
                )
                if decision_hint:
                    checklist_desc += decision_hint
                agent = agent_factory(step.id, step.name, checklist_desc)
                node = AgentNode(agent)
                node.name = step.id
                builder.add_node(node)

            case StepType.DECISION:
                branches = [
                    t.condition
                    for t in plan.transitions
                    if t.from_step == step.id and t.condition
                ]
                node = DecisionNode(
                    name=step.id,
                    question=step.instructions or step.name,
                    branches=branches,
                )
                builder.add_node(node)

            case StepType.SUBWORKFLOW:
                sub_graph = self._compile_subworkflow(
                    step.subworkflow_skill or "",
                    skill_manager,
                    agent_factory,
                )
                builder.add_node(SubgraphNode(
                    name=step.id, sub_graph=sub_graph,
                ))

            case StepType.TERMINAL:
                builder.add_node(TerminalNode(name=step.id))

    for t in plan.transitions:
        builder.add_edge(t.from_step, t.to_step, condition=t.condition)

    builder.set_entry(plan.entry_step)
    return builder.compile()
```

关键变化：
- 删除 `constraint_prefix` 构建逻辑（第 31-35 行）
- ACTION 分支调用 `agent_factory(step.id, step.name, checklist_desc)` 而非 `agent_factory(step.id, instructions)`

- [ ] **Step 4: 运行全部 compiler 测试**

```bash
uv run pytest tests/skills/test_compiler.py -v
```

预期：全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add src/skills/compiler.py tests/skills/test_compiler.py
git commit -m "refactor(compiler): new agent_factory signature, remove constraint_prefix"
```

---

### Task 4: App 层——shared_system_prompt + 步骤 task

**Files:**
- Modify: `src/app/app.py:86-146` (_handle_skill 方法)

- [ ] **Step 1: 重写 `_handle_skill` 方法**

修改 `src/app/app.py` 的 `_handle_skill` 方法：

```python
async def _handle_skill(self, user_input: str, skill_name: str) -> None:
    """通过 SkillWorkflowParser + WorkflowCompiler 执行 skill 工作流。"""
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

    # 5. 编译 → CompiledGraph
    compiler = WorkflowCompiler()
    skill_graph = compiler.compile(
        workflow,
        agent_factory=make_step_agent,
        skill_manager=self.skill_manager,
    )

    # 6. 构建隔离的执行上下文
    skill_engine = GraphEngine()
    ctx = RunContext(
        input=actual_input,
        state=DynamicState(),
        deps=self.deps,
    )

    # 7. 执行
    result = await skill_engine.run(skill_graph, ctx)

    # 8. 输出
    output = result.output
    if isinstance(output, dict):
        text = output.get("text", str(output))
    elif hasattr(output, "text"):
        text = output.text
    else:
        text = str(output)

    await self.deps.ui.display(f"\n{text}\n")
```

- [ ] **Step 2: 运行全部测试确认无回归**

```bash
uv run pytest tests/ -v
```

预期：全部 PASS。

- [ ] **Step 3: 提交**

```bash
git add src/app/app.py
git commit -m "refactor(app): shared system prompt with full skill body for prompt cache"
```

---

### Task 5: 端到端验证

- [ ] **Step 1: 运行全部测试**

```bash
uv run pytest tests/ -v
```

预期：全部 PASS。

- [ ] **Step 2: 手动验证——检查 LLM 收到的消息**

在 `src/agents/runner.py` 的 `run()` 方法中临时加日志（验证后删除）：

```python
# 在 messages 构建完成后、llm.chat 调用前加入：
logger.info("=== LLM Messages ===")
for msg in messages:
    role = msg.get("role", "?")
    content = msg.get("content", "")[:200]
    logger.info(f"[{role}] {content}...")
```

启动 agent 并运行 `/brainstorming 设计一个简单的aoi算法`：

```bash
uv run python main.py
```

验证：
1. 第一步 `explore_project_context` 的 system prompt 包含完整 SKILL.md 内容（"Brainstorming Ideas Into Designs", "Understanding the idea" 等）
2. user message 为 `请执行步骤「Explore project context」：check files, docs, recent commits`
3. 第二步的 system prompt 与第一步完全相同（可被 prompt cache 命中）

- [ ] **Step 3: 删除临时日志，最终提交**

删除 Step 2 添加的临时日志代码，然后：

```bash
git add src/agents/runner.py
git commit -m "chore: clean up debug logging"
```
