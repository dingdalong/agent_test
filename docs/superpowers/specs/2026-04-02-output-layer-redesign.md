# 用户交互输出体系重整

## 背景

当前系统存在三个问题：
1. 事件级别分配不合理 — `TokenDelta` / `ThinkingDelta` 被归为 TRACE，默认配置下流式输出被静默过滤
2. `ThinkingDelta` 是空壳事件 — 定义了类型和 CLI 渲染，但 `openai.py` 从未 emit
3. 输出路径重复 — 流式 `TokenDelta` 已逐字输出回答，`_handle_normal` 结束后又整体打印一次

## 目标

- 三层输出模式，复用已有 `events.level` 配置，无新配置
- 默认模式下用户可见：流式回答 + 思考过程 + 错误
- 单一输出路径：所有用户可见内容走 EventBus，消除重复打印

## 设计

### 1. 事件级别重新分配

不改 `EventLevel` 枚举（PROGRESS=1, DETAIL=2, TRACE=3），只调整各事件的默认级别：

| 事件 | 原级别 | 新级别 | 变更 |
|------|--------|--------|------|
| `TokenDelta` | TRACE→PROGRESS(已改) | PROGRESS | - |
| `ThinkingDelta` | TRACE | **PROGRESS** | 改 |
| `ErrorOccurred` | PROGRESS | PROGRESS | - |
| `ToolCalled` | DETAIL | DETAIL | - |
| `ToolResult` | DETAIL | DETAIL | - |
| `Handoff` | DETAIL | DETAIL | - |
| `AgentStarted` | DETAIL | DETAIL | - |
| `AgentEnded` | DETAIL | DETAIL | - |
| `GraphStarted` | PROGRESS | **TRACE** | 改 |
| `GraphEnded` | PROGRESS | **TRACE** | 改 |
| `NodeStarted` | PROGRESS | **TRACE** | 改 |
| `NodeEnded` | PROGRESS | **TRACE** | 改 |

三层用户视角：

- **PROGRESS（默认）**：流式回答 + 思考过程 + 错误
- **DETAIL**：+ 工具调用/结果、Agent 启停、Handoff
- **TRACE**：+ Graph/Node 启停（调试信息）

### 2. 补充 ThinkingDelta 发射

在 `src/llm/openai.py` 的 `_parse_stream()` 中，识别 DeepSeek reasoning_content 并 emit `ThinkingDelta`。

DeepSeek 通过 OpenAI 兼容接口返回思考内容，存储在 `delta.reasoning_content` 字段中。在流式解析循环中，检测该字段并发射事件：

```python
# 在 chunk 解析循环中
reasoning = getattr(delta, "reasoning_content", None)
if reasoning and not silent and self._bus:
    await self._bus.emit(ThinkingDelta(
        timestamp=time.time(),
        source=self.model,
        content=reasoning,
    ))
```

### 3. 统一输出路径

`app.py` 的 `_handle_normal()` 修改：

```python
# 修改前（两条路径）：
if self.event_bus:
    await self.ui.display("\n")
else:
    await self.ui.display(f"\n{output}\n")

# 修改后（统一路径）：
if not self.event_bus:
    # 无 EventBus 时（测试场景）fallback 整体打印
    await self.ui.display(f"\n{output}\n")
else:
    # 流式输出已通过 TokenDelta 到达 CLI，只补换行
    await self.ui.display("\n")
```

逻辑不变，但语义更清晰：EventBus 是主路径，无 bus 是 fallback。

### 4. config.yaml 注释更新

```yaml
events:
    level: progress   # progress(流式回答+思考) | detail(+工具/Agent) | trace(+图/节点调试)
```

## 涉及文件

| 文件 | 改动 |
|------|------|
| `src/events/types.py` | `ThinkingDelta` TRACE→PROGRESS, `GraphStarted/Ended` PROGRESS→TRACE, `NodeStarted/Ended` PROGRESS→TRACE |
| `src/llm/openai.py` | `_parse_stream()` 增加 `ThinkingDelta` 发射 |
| `src/app/app.py` | `_handle_normal` 输出路径已是方案 A 逻辑（保持） |
| `config.yaml` | 更新 `events.level` 注释 |

## 不改动

- `EventLevel` 枚举本身
- `EventBus` 过滤逻辑
- `CLIInterface.on_event()` 渲染逻辑（已有 ThinkingDelta 处理）
- `UserInteractionService`（不相关）
- 运行时 `/mode` 切换命令（不在本次范围）
