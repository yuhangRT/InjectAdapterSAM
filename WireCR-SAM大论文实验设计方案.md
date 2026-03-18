# WireCR-SAM 大论文实验设计方案

## 1. 文档目的

本文档用于明确 WireCR-SAM 章节的完整实验设计，目标不是把所有参数做全排列组合，而是围绕论文主张建立一条清晰、可复现、可解释的实验论证链。所有设计均以当前仓库实现为准，训练入口为 [main.py](/home/zyh/InjectAdapterSAM/main.py) 或 [main_sam.py](/home/zyh/InjectAdapterSAM/main_sam.py)，自动化实验脚本为 [scripts/run_thesis_suite.py](/home/zyh/InjectAdapterSAM/scripts/run_thesis_suite.py)，表格导出脚本为 [scripts/export_thesis_tables.py](/home/zyh/InjectAdapterSAM/scripts/export_thesis_tables.py)。

## 2. 论文要证明的核心结论

本章实验不是简单证明“模型指标更高”，而是要系统回答下面 6 个问题：

1. WireCR-SAM 相比无适配器或简化适配器方案，是否能更好完成机床电路关键区域分割。
2. WireCR adapter 的关键结构设计是否真的有效，而不是偶然调参结果。
3. 类感知提示是否对 `wire` 和 `interface-hole` 两类语义解码有实质帮助。
4. 联合损失中的 Boundary、clDice 和 hole weight 是否分别对应了明确的性能收益。
5. 在少样本场景下，模型是否仍然具有工业应用价值。
6. 在精度之外，该方法是否具有合理的参数效率和工程代价。

只有把这 6 个问题分别回答清楚，本章才构成完整的论文实验闭环。

## 3. 命名与口径约定

当前代码中的规范类别名为：

- `background`
- `wire`
- `interface-hole`

其中，[dataset/sam_dataset.py](/home/zyh/InjectAdapterSAM/dataset/sam_dataset.py) 与 [models/sam_wrapper.py](/home/zyh/InjectAdapterSAM/models/sam_wrapper.py) 已统一为 `interface-hole`。  
为了便于论文表述，正文和表格中可以将 `interface-hole` 简写为 `hole`，但需在章节开头说明：

> 文中 `hole` 与代码实现中的 `interface-hole` 含义一致。

当前指标导出脚本 [scripts/export_thesis_tables.py](/home/zyh/InjectAdapterSAM/scripts/export_thesis_tables.py) 也已兼容 `hole_iou` 与 `interface_hole_iou` 两种字段。

## 4. 实验设计总原则

### 4.1 只做“单因素变化”，不做全因子组合爆炸

本章所有实验均围绕一个固定参考配置展开，每次只改变一个因素。这样做的原因有两个：

- 结果更容易归因，便于论文解释。
- 小样本工业数据对随机性敏感，全排列实验成本高且容易得出混乱结论。

### 4.2 结构问题与训练问题分开论证

本章必须明确区分两类消融：

- 结构消融：回答“模型为什么这样设计”
- 损失消融：回答“模型为什么这样训练”

`L1~L4` 不是 WireCR 结构消融，而是联合损失消融。  
WireCR 结构消融应放在 `adapter size / compression ratio / adapter_simple / class-aware prompts` 这一组里。

### 4.3 定量实验与定性实验都必须有

需要同时包含：

- 定量对比表
- 可视化结果图
- 失败案例图
- 工程效率统计

如果只有数值表，没有可视化和失败案例，论文说服力会弱很多。

## 5. 固定参考配置

除非某组实验明确声明修改某项变量，否则默认采用以下配置作为 Reference Config。

### 5.1 数据与训练基础设置

- 数据集：`wire_hole`
- 类别数：`3`
- 输入分辨率：`1024`
- 训练轮次：`100`
- 批大小：`1`
- 数据加载线程：`4`
- 优化器：Adam
- 学习率：`1e-4`
- 调度器：`const`
- 训练集采样比例：`1.0`

### 5.2 模型参考配置

- SAM backbone：`vit_b`
- adapter size：`medium`
- compression ratio：`8`
- use residual：`True`
- adapter simple：`False`
- disable adapter：`False`
- class-aware prompts：`True`
- freeze encoder：`True`
- freeze decoder：`False`
- freeze prompt encoder：`True`

### 5.3 损失参考配置

- BCE weight：`1.0`
- Dice weight：`1.0`
- Boundary loss weight：`0.1`
- clDice weight：`0.1`
- hole class weight：`2.0`

### 5.4 参考训练命令

```bash
python main.py \
  --mode sam \
  --gpu 0 \
  --data-dir ./samDataset_wire_hole \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --batch-size 1 \
  --workers 4 \
  --epochs 100 \
  --scheduler const \
  --adapter-size medium \
  --compression-ratio 8 \
  --class-aware-prompts \
  --boundary-loss-weight 0.1 \
  --cldice-weight 0.1 \
  --hole-class-weight 2.0 \
  --run-name reference_vitb_medium_cr8
```

## 6. 必做实验总览

本章建议的完整实验由 8 个部分组成：

1. 表 4-1 主实验训练配置表
2. 表 4-2 主对比实验
3. 表 4-3 结构消融实验
4. 表 4-4 损失消融实验
5. 表 4-5 少样本实验
6. 表 4-6 backbone 与效率分析实验
7. 图 4-4 典型样本可视化对比
8. 图 4-5 失败案例分析

其中，表 4-2 至表 4-5 是当前仓库已经支持自动化执行的核心实验；表 4-6、可视化和失败案例需要额外整理与统计，但仍属于本章完整实验设计中应做的内容。

## 7. 表 4-1 主实验训练配置表

### 7.1 目的

该表不是结果表，而是为整章实验建立统一参考配置，避免后续所有实验失去共同基准。

### 7.2 需要填写的字段

- 实现框架
- 数据集
- 类别定义
- 输入分辨率
- SAM backbone
- adapter size
- compression ratio
- 是否残差注入
- 是否简化适配器
- 类感知提示
- 冻结图像编码器
- 冻结提示编码器
- 冻结掩膜解码器
- 优化器
- 学习率
- 调度器
- 训练轮次
- 批大小
- Boundary loss weight
- clDice weight
- hole class weight
- 实验硬件

### 7.3 说明

这张表只保留最终论文采用的 Reference Config，不需要把所有变量都写进去。

## 8. 表 4-2 主对比实验

### 8.1 实验目的

主对比实验用于证明方法整体有效性，回答的问题是：

> WireCR-SAM 作为完整方法，是否比不使用 WireCR 适配器、或只使用简化适配器的方案更好。

### 8.2 必做对比模型

当前仓库中，以下 3 行是必须完成的：

1. Original SAM Transfer
2. Simple Adapter
3. WireCR-SAM

其对应定义已经写在 [scripts/run_thesis_suite.py](/home/zyh/InjectAdapterSAM/scripts/run_thesis_suite.py) 的 `table_4_2_specs()` 中。

### 8.3 三行模型的准确含义

#### M1. Original SAM Transfer

- `disable_adapter=True`
- `adapter_simple=False`
- `class_aware_prompts=True`

含义：不启用 WireCR adapter，但保留自动语义解码流程。  
这行用于证明“仅靠 SAM 原始图像特征迁移到本任务”能达到什么上限。

#### M2. Simple Adapter

- `disable_adapter=False`
- `adapter_simple=True`
- `class_aware_prompts=True`

含义：启用简化适配器，不使用完整的压缩-扩张瓶颈。  
这行用于证明“有适配器”和“完整 WireCR 结构”之间的差异。

#### M3. WireCR-SAM

- `disable_adapter=False`
- `adapter_simple=False`
- `class_aware_prompts=True`

含义：完整模型。  
这是本章的核心方法。

### 8.4 建议指标

- IoU
- Dice
- Precision
- Recall
- F1
- Boundary F1
- clDice
- Wire IoU
- Hole IoU
- Hole Recall

这与 [scripts/export_thesis_tables.py](/home/zyh/InjectAdapterSAM/scripts/export_thesis_tables.py) 中 `table4_2` 的导出字段一致。

### 8.5 论文分析重点

这张表不要只写“本文方法优于基线”。  
正确分析顺序应为：

1. 先看总体质量：`IoU / Dice / F1`
2. 再看细线结构：`Wire IoU / Boundary F1 / clDice`
3. 再看小目标：`Hole IoU / Hole Recall`

### 8.6 是否需要外部基线

如果时间充足，可以增加：

- U-Net
- DeepLabV3+
- SegFormer

但这些不属于当前仓库自动实验链的一部分，不应阻塞本章主实验完成。  
因此对当前项目而言，它们属于“可选扩展基线”，不是“必须完成”的基线。

### 8.7 自动执行命令

```bash
python scripts/run_thesis_suite.py \
  --table 4-2 \
  --output-root ./thesis_runs \
  --python python \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --batch-size 1 \
  --workers 4 \
  --epochs 100 \
  --scheduler const \
  --adapter-size medium \
  --compression-ratio 8 \
  --class-aware-prompts \
  --boundary-loss-weight 0.1 \
  --cldice-weight 0.1 \
  --hole-class-weight 2.0 \
  --data-dir ./samDataset_wire_hole
```

## 9. 表 4-3 结构消融实验

### 9.1 实验目的

结构消融不是再次证明“完整模型更强”，而是精确回答：

- adapter 宽度是否重要
- compression ratio 是否重要
- 完整瓶颈设计是否必要
- class-aware prompts 是否必要

### 9.2 必做变量

当前仓库里的结构消融定义位于 [scripts/run_thesis_suite.py](/home/zyh/InjectAdapterSAM/scripts/run_thesis_suite.py) 的 `table_4_3_specs()`，必须完成的实验如下。

#### A1. Reference

- `adapter_size=medium`
- `compression_ratio=8`
- `adapter_simple=False`
- `class_aware_prompts=True`

#### A2. Adapter Size = small

- 仅改 `adapter_size=small`

#### A3. Adapter Size = large

- 仅改 `adapter_size=large`

#### A4. Compression = 1/4

- 仅改 `compression_ratio=4`

#### A5. Compression = 1/16

- 仅改 `compression_ratio=16`

#### A6. Compression = 1/32

- 仅改 `compression_ratio=32`

#### A7. Compression = 1/64

- 仅改 `compression_ratio=64`

#### A8. Simple Adapter = On

- 仅改 `adapter_simple=True`

#### A9. Class-aware Prompts = Off

- 仅改 `class_aware_prompts=False`

### 9.3 为什么这 9 组足够

这 9 组已经能覆盖结构层面的四个关键问题：

- `small / medium / large`：适配器宽度影响
- `cr4 / cr8 / cr16 / cr32 / cr64`：瓶颈压缩程度影响
- `simple adapter`：完整瓶颈与门控是否必要
- `class-aware prompts off`：自动语义提示是否必要

因此不需要继续做 `size × cr × prompt × simple` 的全组合实验。

### 9.4 建议指标

- IoU
- Dice
- Boundary F1
- clDice
- Hole Recall

这是当前导表脚本 [scripts/export_thesis_tables.py](/home/zyh/InjectAdapterSAM/scripts/export_thesis_tables.py) `table4_3` 的默认字段。

### 9.5 论文分析重点

建议按以下顺序解释：

1. `Simple Adapter` 与 `Reference` 的差距，用于解释完整 WireCR 结构的必要性
2. `Class-aware Prompts Off` 与 `Reference` 的差距，用于解释提示机制的贡献
3. `small / medium / large` 的比较，用于说明规模与效果的平衡
4. `cr4 / cr8 / cr16 / cr32 / cr64` 的比较，用于说明瓶颈压缩比的最佳区间

### 9.6 自动执行命令

```bash
python scripts/run_thesis_suite.py \
  --table 4-3 \
  --output-root ./thesis_runs \
  --python python \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --batch-size 1 \
  --workers 4 \
  --epochs 100 \
  --scheduler const \
  --adapter-size medium \
  --compression-ratio 8 \
  --class-aware-prompts \
  --boundary-loss-weight 0.1 \
  --cldice-weight 0.1 \
  --hole-class-weight 2.0 \
  --data-dir ./samDataset_wire_hole
```

## 10. 表 4-4 损失消融实验

### 10.1 实验目的

损失消融的目标是回答：

- 边界约束是否真的改善边缘质量
- clDice 是否真的改善细线拓扑连续性
- hole class weight 是否真的改善孔洞检出

这部分不属于 WireCR 结构消融，而是训练目标设计实验。

### 10.2 必做 4 组

定义在 [scripts/run_thesis_suite.py](/home/zyh/InjectAdapterSAM/scripts/run_thesis_suite.py) 的 `table_4_4_specs()` 中：

#### L1. BCE + Dice

- `boundary_loss_weight=0.0`
- `cldice_weight=0.0`
- `hole_class_weight=1.0`

#### L2. + Boundary

- `boundary_loss_weight=0.1`
- `cldice_weight=0.0`
- `hole_class_weight=1.0`

#### L3. + clDice

- `boundary_loss_weight=0.1`
- `cldice_weight=0.1`
- `hole_class_weight=1.0`

#### L4. + Hole Weight

- `boundary_loss_weight=0.1`
- `cldice_weight=0.1`
- `hole_class_weight=2.0`

### 10.3 为什么按这种顺序设计

这种链式加法设计比随机开关更适合论文论证，因为它对应一条清晰的因果链：

1. 先建立基本分割能力
2. 再加入边界一致性约束
3. 再加入细线连通性约束
4. 最后专门加强孔洞类别召回

### 10.4 建议指标

- IoU
- Dice
- Boundary F1
- clDice
- Hole IoU
- Hole Recall

### 10.5 论文分析重点

每项损失都要绑定到最敏感指标来解释：

- `Boundary` 主要看 `Boundary F1`
- `clDice` 主要看 `clDice` 和 `Wire IoU`
- `hole weight` 主要看 `Hole IoU` 与 `Hole Recall`

不要只写“联合损失整体更优”。

### 10.6 自动执行命令

```bash
python scripts/run_thesis_suite.py \
  --table 4-4 \
  --output-root ./thesis_runs \
  --python python \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --batch-size 1 \
  --workers 4 \
  --epochs 100 \
  --scheduler const \
  --adapter-size medium \
  --compression-ratio 8 \
  --class-aware-prompts \
  --boundary-loss-weight 0.1 \
  --cldice-weight 0.1 \
  --hole-class-weight 2.0 \
  --data-dir ./samDataset_wire_hole
```

## 11. 表 4-5 少样本实验

### 11.1 实验目的

该实验回答：

> 当工业标注资源有限时，WireCR-SAM 是否仍保持可用性能。

### 11.2 必做 4 组

当前定义在 [scripts/run_thesis_suite.py](/home/zyh/InjectAdapterSAM/scripts/run_thesis_suite.py) 的 `table_4_5_specs()` 中：

- `subset_ratio=0.10`
- `subset_ratio=0.25`
- `subset_ratio=0.50`
- `subset_ratio=1.00`

### 11.3 建议指标

- IoU
- Dice
- Wire IoU
- Hole IoU
- Hole Recall
- Boundary F1
- clDice

### 11.4 论文分析重点

这组实验不要只说“样本少了性能下降”。  
需要重点分析：

1. 性能下降是否平缓
2. `Hole Recall` 是否在低样本下仍可接受
3. `clDice` 是否说明线束结构仍具有一定连通性

如果 `0.25` 或 `0.50` 样本比例下仍接近全量结果，这会显著增强论文的工业应用价值。

### 11.5 自动执行命令

```bash
python scripts/run_thesis_suite.py \
  --table 4-5 \
  --output-root ./thesis_runs \
  --python python \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --batch-size 1 \
  --workers 4 \
  --epochs 100 \
  --scheduler const \
  --adapter-size medium \
  --compression-ratio 8 \
  --class-aware-prompts \
  --boundary-loss-weight 0.1 \
  --cldice-weight 0.1 \
  --hole-class-weight 2.0 \
  --data-dir ./samDataset_wire_hole
```

## 12. 表 4-6 Backbone 与效率分析实验

### 12.1 为什么这张表必须补

如果没有 backbone 与效率分析，评审很容易提出两个问题：

1. 你的提升是否只是因为换了更大的 SAM
2. 你的方法在工程上是否值得使用

因此建议补一张“精度-代价平衡”表，作为主表之外的重要补充。

### 12.2 建议比较对象

在保持其余配置与 Reference Config 一致的前提下，仅比较：

- `vit_b`
- `vit_l`
- `vit_h`

如果 `vit_h` 在 24GB 4090 上训练无法稳定运行，可以在正文中说明工程约束，并至少保留：

- `vit_b`
- `vit_l`

### 12.3 除分割指标外，还应统计

- total params
- adapter params
- trainable params
- 单图推理时间
- 单 epoch 训练时间
- GPU 峰值显存

其中，参数相关字段已经由 [utils/experiment_io.py](/home/zyh/InjectAdapterSAM/utils/experiment_io.py) 输出到 `experiment_summary.json`。  
推理时间、单 epoch 时间和峰值显存需要额外手工记录，或后续补脚本统计。

### 12.4 建议表头

| Backbone | IoU | Dice | Wire IoU | Hole Recall | Total Params | Trainable Params | Inference Time | Peak VRAM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

### 12.5 建议结论方向

- `vit_b` 通常作为精度与代价平衡最好的主实验 backbone
- `vit_l` 用于证明方法在更大 backbone 上仍成立
- `vit_h` 只有在资源允许时才建议加入

### 12.6 手动执行命令模板

`vit_b`：

```bash
python main.py \
  --mode sam \
  --gpu 0 \
  --data-dir ./samDataset_wire_hole \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --batch-size 1 \
  --workers 4 \
  --epochs 100 \
  --adapter-size medium \
  --compression-ratio 8 \
  --class-aware-prompts \
  --boundary-loss-weight 0.1 \
  --cldice-weight 0.1 \
  --hole-class-weight 2.0 \
  --run-name backbone_vit_b
```

`vit_l`：

```bash
python main.py \
  --mode sam \
  --gpu 1 \
  --data-dir ./samDataset_wire_hole \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_l \
  --sam-checkpoint ./checkpoints/sam_vit_l_0b3195.pth \
  --batch-size 1 \
  --workers 4 \
  --epochs 100 \
  --adapter-size medium \
  --compression-ratio 8 \
  --class-aware-prompts \
  --boundary-loss-weight 0.1 \
  --cldice-weight 0.1 \
  --hole-class-weight 2.0 \
  --run-name backbone_vit_l
```

`vit_h` 仅在显存允许时执行：

```bash
python main.py \
  --mode sam \
  --gpu 2 \
  --data-dir ./samDataset_wire_hole \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_h \
  --sam-checkpoint ./checkpoints/sam_vit_h_4b8939.pth \
  --batch-size 1 \
  --workers 4 \
  --epochs 100 \
  --adapter-size medium \
  --compression-ratio 8 \
  --class-aware-prompts \
  --boundary-loss-weight 0.1 \
  --cldice-weight 0.1 \
  --hole-class-weight 2.0 \
  --run-name backbone_vit_h
```

## 13. 图 4-4 可视化分析

### 13.1 必须覆盖的场景类型

至少选取 4 组典型样本：

1. 背景复杂、纹理密集
2. 线束细长、弯折明显
3. 接口孔洞较小、边界模糊
4. 光照变化、反光或遮挡明显

### 13.2 每组图建议展示的内容

- 原图
- Ground Truth
- Original SAM Transfer
- Simple Adapter
- WireCR-SAM

### 13.3 论文分析重点

可视化分析应具体指出：

- 线束弯折处是否断裂
- 孔洞是否漏检
- 边界是否外扩
- 背景结构是否被误检为前景

不要写成笼统的“视觉上更清晰”。

## 14. 图 4-5 失败案例分析

### 14.1 必须覆盖的错误类型

建议至少各选 1 张，共 3 类：

1. 漏分割
2. 边界粘连
3. 孔洞误检

### 14.2 每类失败案例的分析方向

#### F1. 漏分割

重点分析：

- 是否因线束过细
- 是否因遮挡严重
- 是否因对比度过低

#### F2. 边界粘连

重点分析：

- 是否与端子、边框或标签相连
- 是否说明背景抑制不足

#### F3. 孔洞误检

重点分析：

- 是否与金属反光、圆形纹理混淆
- 是否说明 hard negative 建模不足

失败案例不是减分项，而是说明当前方法边界与改进空间的重要证据。

## 15. 统计学与复现实验要求

### 15.1 为什么要多随机种子

当前数据集规模较小，训练结果容易受到以下因素影响：

- 参数初始化
- subset 采样
- 数据划分中的偶然性

因此，核心表格不建议只报一次结果。

### 15.2 推荐执行策略

#### 理想方案

以下实验至少做 `3` 个随机种子，并报告 `mean ± std`：

- 表 4-2 主对比实验
- 表 4-5 少样本实验
- 表 4-6 backbone 实验

#### 资源受限时的折中方案

- 表 4-2：3 seeds
- 表 4-3：1 seed 初筛，关键结论行补 3 seeds
- 表 4-4：1 seed 初筛，Reference 与最终 L4 补 3 seeds
- 表 4-5：3 seeds
- 表 4-6：2 至 3 seeds

### 15.3 种子建议

建议使用固定种子组，例如：

- `42`
- `3407`
- `2025`

这样便于复现实验。

## 16. 建议执行顺序

### 16.1 第一步：烟测

先用 Reference Config 跑一个 5 epoch 烟测，确认：

- 数据加载正常
- 模型前向正常
- `experiment_summary.json` 正常生成

示例：

```bash
python main.py \
  --mode sam \
  --gpu 0 \
  --data-dir ./samDataset_wire_hole \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --batch-size 1 \
  --workers 4 \
  --epochs 5 \
  --adapter-size medium \
  --compression-ratio 8 \
  --class-aware-prompts \
  --boundary-loss-weight 0.1 \
  --cldice-weight 0.1 \
  --hole-class-weight 2.0 \
  --run-name smoke_test_reference
```

### 16.2 第二步：跑 Reference Config 正式基线

先得到一份完整 Reference 结果，后续所有实验都围绕它解释。

### 16.3 第三步：跑自动化实验套件

顺序建议如下：

1. 表 4-2 主对比
2. 表 4-3 结构消融
3. 表 4-4 损失消融
4. 表 4-5 少样本

### 16.4 第四步：补 backbone 与效率实验

这部分需要手动记录时间和显存，适合在自动化核心实验跑完后补充。

### 16.5 第五步：整理可视化与失败案例

最后从已完成实验中挑选代表性样本，不建议在实验开始前盲选图像。

## 17. 3 张 4090 的并行执行建议

你当前硬件是 3 张 4090，因此建议按以下方式并行调度：

- `GPU 0`：表 4-2 主对比实验
- `GPU 1`：表 4-3 结构消融实验
- `GPU 2`：表 4-4 损失消融实验

表 4-5 少样本实验可在其中一组结束后接续执行。  
如果要补 backbone 实验，建议：

- `vit_b` 放 `GPU 0`
- `vit_l` 放 `GPU 1`
- `vit_h` 放 `GPU 2`

## 18. 表格导出命令

### 18.1 表 4-2 导出

```bash
python scripts/export_thesis_tables.py \
  table4_2 \
  --manifest ./thesis_runs/table4_2/manifest_table4_2.json \
  --output ./thesis_tables/table4_2.csv
```

### 18.2 表 4-3 导出

```bash
python scripts/export_thesis_tables.py \
  table4_3 \
  --manifest ./thesis_runs/table4_3/manifest_table4_3.json \
  --output ./thesis_tables/table4_3.csv
```

### 18.3 表 4-4 导出

```bash
python scripts/export_thesis_tables.py \
  table4_4 \
  --manifest ./thesis_runs/table4_4/manifest_table4_4.json \
  --output ./thesis_tables/table4_4.csv
```

### 18.4 表 4-5 导出

```bash
python scripts/export_thesis_tables.py \
  table4_5 \
  --manifest ./thesis_runs/table4_5/manifest_table4_5.json \
  --output ./thesis_tables/table4_5.csv
```

## 19. 论文写作时的推荐章节结构

建议将第 4 章实验部分按以下顺序组织：

1. 实验设置
2. 主对比实验
3. 结构消融实验
4. 损失消融实验
5. 少样本实验
6. backbone 与效率分析
7. 可视化分析
8. 失败案例分析

这种组织方式的优点是：

- 先证明方法整体有效
- 再解释结构为什么有效
- 再解释训练为什么有效
- 再说明实际部署价值
- 最后补充直观案例与局限性

## 20. 最终建议

对于当前项目，真正“必须做且足以支撑大论文本章”的实验集合如下：

### 核心必须完成

- 表 4-1 主实验训练配置
- 表 4-2 主对比实验
- 表 4-3 结构消融实验
- 表 4-4 损失消融实验
- 表 4-5 少样本实验
- 可视化分析
- 失败案例分析

### 强烈建议补充

- 表 4-6 backbone 与效率分析

### 不建议作为本章主线强行加入

- 大量外部模型 baseline
- 所有参数的全排列组合
- 与当前仓库实现无关的额外模型分支

本章最重要的不是“实验数量多”，而是“实验之间逻辑严密，能够逐层证明方法有效、结构合理、训练设计有针对性、并具有工业应用价值”。
