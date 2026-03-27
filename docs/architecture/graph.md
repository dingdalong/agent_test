# 图执行引擎

## 职责

提供与智能体无关的异步图执行能力。将工作流定义（节点、边、并行组）编译为不可变的图结构，然后按拓扑顺序执行。

## 核心组件

### GraphNode Protocol（`src/graph/types.py`）

```python
class GraphNode(Protocol):
    name: str
    async def execute(self, context: Any) -> NodeResult: ...
```

`NodeResult` 包含：
- `output: Any` — 节点执行结果
- `next: str | list[str] | None` — 显式指定下一个节点
- `handoff: Any` — handoff 请求（交给其他智能体）

### FunctionNode（`src/graph/types.py`）

将普通 async 函数包装为 GraphNode。用于 PlanCompiler 将 Step 转为节点。

### GraphBuilder（`src/graph/builder.py`）

声明式、链式的图构建 API：
- `add_node(node)` — 添加 GraphNode
- `add_function(name, fn)` — 添加函数节点
- `add_edge(source, target)` — 添加边
- `add_parallel(nodes, then)` — 添加并行组
- `set_entry(name)` — 设置入口节点
- `compile()` → `CompiledGraph`（带验证）

### CompiledGraph（`src/graph/types.py`）

编译后的不可变图结构：`nodes`, `edges`, `entry`, `parallel_groups`。

### GraphEngine（`src/graph/engine.py`）

执行 CompiledGraph：
1. 从 `entry` 节点开始
2. 检查当前 pending 列表是否匹配 ParallelGroup → 并行执行
3. 否则顺序执行，将输出写入 context.state
4. 处理 handoff：更新 context.input，切换到目标节点
5. 处理显式 next 或按边路由
6. 支持 hooks（`on_graph_start/end`, `on_node_start/end`）
7. 深度限制防止无限 handoff

## 数据流

```
CompiledGraph + RunContext
  → GraphEngine.run()
    → pending = [entry]
    → while pending:
        → 并行组？ → asyncio.gather → 写入 state
        → 单节点？ → execute → 处理 handoff/next/edge
    → GraphResult(output, state, trace)
```
