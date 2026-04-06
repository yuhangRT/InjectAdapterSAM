# CODEX_AGENT_WORKFLOW.md

## 推荐目录放置
当前推荐布局如下：

- `.codex/agent/AGENTS.md`
- `.codex/agent/MASTER_ORCHESTRATOR_AGENT.md`
- `.codex/agent/REFACTOR_AGENT.md`
- `.codex/agent/REVIEW_AGENT.md`
- `WireCRHQInstSAM_Final_Execution_Plan_Checklist.md`

## 推荐运行方式
### 模式
采用“1 个主 agent + 2 个按需 subagent”的模式，避免一个 agent 一路写到底。

1. **主 agent 判步**
   - 读取计划
   - 找到当前步骤
   - 生成单步任务 prompt

2. **Refactor subagent**
   - 只执行当前步骤
   - 更新计划文件
   - 跑测试
   - 提交待审状态

3. **Review subagent**
   - 只审查当前步骤
   - 给出通过/退回结论

4. **主 agent 裁决**
   - 若通过，则进入下一步
   - 若退回，则重新派发修复任务

## 你如何安排
### 方案 A：一个主线程 + 两个 subagent（默认）
- 主线程固定扮演 `Master Orchestrator`
- 每一步按需创建两个 subagent：
  - `Refactor Agent`
  - `Review Agent`
- 两个 subagent 串行派发，不并发审批

优点：
- 最符合当前 checklist 审批流
- 最省界面管理成本
- 最容易阻止跳步与越界

### 方案 B：ChatGPT / Codex App 手工三线程
- 线程 1：Master Orchestrator
- 线程 2：Refactor Agent
- 线程 3：Review Agent

优点：
- 最直观
- 审查独立性最好

### subagent 生命周期
- `Refactor Agent`：按步骤创建，完成后即结束
- `Review Agent`：只在收到 refactor 结果后创建，完成后即结束
- 不采用“双子常驻”模式
- 不让 `Review Agent` 在没有 refactor 输出时空转等待

### 模型与类型默认
- 主 agent：保持当前主模型，不委托关键判步逻辑
- `Refactor Agent`
  - `S00-S02`：`worker + gpt-5.4-mini`
  - `S03+`：`worker + gpt-5.4`
- `Review Agent`
  - 固定 `explorer + gpt-5.4-mini`

## 推荐顺序
- `S00` 到 `S02`：建议强人工控制，逐步放行
- `S03` 以后：若前面流程稳定，可以继续同样模式，不建议合并角色

## 每次派发时的最短提示语
### 给主 agent
“读取 .codex/agent/AGENTS.md 和计划文件，判断当前应执行步骤，并决定派发给哪个 subagent。”

### 给 Refactor subagent
“只执行 `<STEP_ID>`。严格遵守 .codex/agent/AGENTS.md 和计划文件。不得跳步。完成后更新勾选、实施记录和测试结果。”

### 给 Review subagent
“只审查 `<STEP_ID>`。严格对照 .codex/agent/AGENTS.md 和计划文件。不得新增实现。输出通过或退回结论。”

## 何时不应进入下一步
- 任何测试失败
- 计划文件未更新
- 实施记录缺失
- 审查记录缺失
- 发现越界实现
- 发现实现与计划红线冲突

## 最佳实践
- 每一步都保留完整命令日志
- 每一步结束后都让 Review Agent 单独复核
- 不要一次给出“把整个计划做完”的任务
- 若计划需要调整，先更新计划文件，再继续执行
