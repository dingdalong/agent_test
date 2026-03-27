# 规划系统

## 职责

处理需要多步骤的复杂任务：通过 LLM 生成执行计划，编译为可执行的图，然后通过图引擎执行。

## 核心组件

### Step / Plan 模型（`src/plan/models.py`）

```python
@dataclass
class Step:
    id: str
    description: str
    tool_name: str | None = None    # 工具步骤
    tool_args: dict = field(...)
    agent_name: str | None = None   # 智能体步骤
    agent_prompt: str | None = None
    depends_on: list[str] = field(...)  # 依赖的步骤 ID

@dataclass
class Plan:
    goal: str
    steps: list[Step]
```

每个 Step 要么是工具步骤（`tool_name`），要么是智能体步骤（`agent_name`）。

### PlanFlow（`src/plan/flow.py`）

5 阶段编排流程：
1. **澄清** — `check_clarification_needed()` 判断是否需要补充信息，循环提问直到清晰
2. **生成** — `generate_plan()` 通过 LLM + `submit_plan` 工具生成 Plan
3. **确认** — 展示计划给用户，用户可以确认、调整或取消
4. **编译** — `PlanCompiler.compile()` 将 Plan 转为 CompiledGraph
5. **执行** — `GraphEngine.run()` 执行编译后的图

### PlanCompiler（`src/plan/compiler.py`）

将 Plan 编译为 CompiledGraph：
1. 验证（ID 唯一性、agent 存在性）
2. 分层拓扑排序 — 同层步骤无互相依赖，可并行
3. 每个 Step → FunctionNode（工具闭包或智能体闭包）
4. 单步骤层 → 顺序边；多步骤层 → ParallelGroup + merge 节点
5. 支持 `$step_id.field` 变量引用（步骤间传递数据）

### 变量引用

步骤间通过 `$step_id.field` 语法传递数据：

```yaml
- id: step1
  tool_name: get_weather
  tool_args: { city: "北京" }

- id: step2
  agent_name: email_agent
  agent_prompt: "发送天气信息: $step1.result"
  depends_on: [step1]
```

`resolve_variables()` 在执行时从 context.state 中解析变量值。

## 数据流

```
用户请求
  → PlanFlow.run()
    → 澄清循环（可选）
    → LLM 生成 Plan
    → 用户确认/调整循环
    → PlanCompiler.compile() → CompiledGraph
    → GraphEngine.run() → 结果
```
