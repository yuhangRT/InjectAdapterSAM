# AGENTS.md

本仓库采用 `1 个主 agent + 2 个按需 subagent` 的协作机制，所有 Codex 任务都必须遵守本文件以及仓库内引用的执行计划。

## 1. 单一事实来源
以下文件按优先级解释为本仓库的执行规则：

1. `WireCRHQInstSAM_Final_Execution_Plan_Checklist.md`
2. `.codex/agent/MASTER_ORCHESTRATOR_AGENT.md`
3. `.codex/agent/REFACTOR_AGENT.md`
4. `.codex/agent/REVIEW_AGENT.md`

若这些文件与当前聊天中的明确指令冲突，以当前聊天指令为准。

## 2. 总体约束
- 当前唯一正式主线为 `WireCRHQInstSAM`。
- 旧 `sam_wrapper / sam_fpn_segmentor / wirecr_instsam / train_instsam / eval_instsam / infer_instsam` 仅作历史备份，不得被新主线 import。
- `WireCRNet / WireCRAdapter` 必须保留，并作为核心创新点。
- 当前任务是两类实例分割：
  - `label_sleeve`
  - `empty_terminal`
- 示例标注验收标准：
  - `27 polygons`
  - 按 `(category, group)` 合并后得到 `8 label_sleeve + 8 empty_terminal`

## 3. 执行模式
- 默认运行形态为：
  - 当前主线程固定扮演 `Master Orchestrator`
  - `Refactor Agent` 与 `Review Agent` 作为按步骤创建的 subagent 运行
- subagent 必须串行派发：
  - 先 `Refactor Agent`
  - 后 `Review Agent`
  - 禁止并发审批、禁止让 `Review Agent` 空转等待
- 只允许按 `WireCRHQInstSAM_Final_Execution_Plan_Checklist.md` 中的步骤顺序推进。
- 一次只允许执行一个步骤（如 `S00`、`S01`）。
- 未经明确许可，不得跳到后续步骤。
- 每次任务结束后，必须：
  1. 更新计划文件中的勾选状态
  2. 填写该步骤的“实施记录”
  3. 填写该步骤的“审查记录”草稿或说明待审原因
  4. 输出本步修改文件清单
  5. 输出本步运行命令和测试结果

## 4. 角色分工
### Master Orchestrator
- 当前主线程固定扮演该角色
- 只负责读取计划、判断当前步骤、下发任务、阻止跳步
- 不负责直接实现功能代码
- 不负责批准审查

### Refactor Agent
- 作为按步骤创建的实现 subagent 运行，完成当前步骤后即结束
- 只负责当前步骤的实现
- 不得实现未授权的后续步骤
- 不得擅自更改计划文件的要求，只能更新执行状态与记录

### Review Agent
- 作为按步骤创建的审查 subagent 运行，只在收到 Refactor 结果后启动
- 只负责审查当前步骤
- 不负责新功能实现
- 负责判断当前步骤是否满足“审查通过”

### 默认 agent 类型
- `Master Orchestrator`：保持当前主模型，负责判步和派单
- `Refactor Agent`：优先使用 `worker`
  - `S00-S02`、文档与脚手架步骤：`gpt-5.4-mini`
  - `S03+`、模型与训练核心步骤：`gpt-5.4`
- `Review Agent`：固定使用 `explorer + gpt-5.4-mini`

## 5. 代码与训练红线
- 所有 trainable SAM 模块必须注册到主模型中
- `oriented_box` 只作为辅助几何元数据，不得直接替代 prompt box 输入
- `dense mask prompt` 必须是低分辨率 logits，不得直接用 full-res binary mask 直喂
- refine 只能使用 adapter/LoRA 后特征，禁止回退到 raw SAM image embedding
- 首版只保证 `vit_b + SAM1 checkpoint`
- 不做 DDP
- 不引入与当前步骤无关的大型外部训练框架

## 6. 数据与评估红线
- 正式实验默认使用 dedupe split；随机 split 仅用于 smoke test
- `coarse mask` 默认 stride=4，不得将 stride=8 设为首版正式默认路径
- 推理阈值与 NMS 参数必须通过 val 选择并写入 checkpoint metadata，不得在代码中永久硬编码为最终值
- 滑窗推理必须先做跨窗融合，再做全图 class-wise mask NMS

## 7. 输出格式要求
各角色固定输出以各自角色文件为准；若本节与角色文件存在轻微不一致，以角色文件为准。主 agent 在汇总时必须补齐以下栏目：

1. 当前步骤
2. 本步目标
3. 已修改文件
4. 已运行命令
5. 测试结果
6. 计划文件更新
7. 风险与阻塞
8. 是否允许进入下一步

## 8. 审查通过规则
只有在以下条件全部满足时，才允许将某步骤标记为“审查通过”：
- 实现范围没有越界
- 通过本步骤要求的测试
- 与计划文件要求一致
- 无未记录的阻塞项
- 输出完整的实施记录与审查记录
