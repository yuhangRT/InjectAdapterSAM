# WireCR-InstSAM 使用文档

本文档总结当前代码库中与 `WireCR-InstSAM` 实例分割主线相关的实际实现、数据格式、模型结构、训练评估流程与当前局限。目标不是描述未来计划，而是尽可能准确地说明：

- 当前代码库要解决什么问题
- 数据如何组织与转换
- 模型现在到底是怎么工作的
- 训练、评估、推理入口如何使用
- 当前版本已经做到了什么，还没做到什么

## 1. 代码库当前定位

当前仓库同时保留两条路线：

- 旧主线：`WireCR-SAM`，面向 `wire / hole / background` 的三类语义分割
- 新主线：`WireCR-InstSAM`，面向 `wire` 和 `interface-hole` 的两类实例分割

对应关系如下：

- [main_sam.py](/home/zyh/InjectAdapterSAM/main_sam.py)：旧语义分割训练/评估入口
- [models/sam_wrapper.py](/home/zyh/InjectAdapterSAM/models/sam_wrapper.py)：旧 prompt 语义分割主线
- [models/sam_fpn_segmentor.py](/home/zyh/InjectAdapterSAM/models/sam_fpn_segmentor.py)：旧 FPN 语义分割主线
- [train_instsam.py](/home/zyh/InjectAdapterSAM/train_instsam.py)：实例主线训练入口
- [eval_instsam.py](/home/zyh/InjectAdapterSAM/eval_instsam.py)：实例主线评估入口
- [infer_instsam.py](/home/zyh/InjectAdapterSAM/infer_instsam.py)：实例主线推理入口

一句话概括：

- 旧主线回答“哪里是 wire / hole”
- 新主线尝试回答“每个 wire / hole 分别是谁”

## 2. 要解决的问题

当前实例主线针对的任务是：

- 输入一张工业机床线路图像
- 输出每个独立的 `wire` 实例
- 输出每个独立的 `interface-hole` 实例

最终目标不是再生成一张三类语义图，而是生成实例列表。每个实例理论上包含：

- `instance_id`
- `category_id`
- `score`
- `bbox`
- `mask`
- `source_prompt`

不过需要注意，**当前推理脚本默认只导出 JSON 里的 `bbox / score / source_prompt`，不会自动把 mask 单独写成 PNG 或 RLE 文件**，这一点见后文“推理”。

## 3. 当前实例主线的真实状态

当前 `WireCR-InstSAM` 已经实现了：

- `ISAT JSON -> COCO instance` 的数据转换
- 基于 COCO instance 的 dataloader
- `SAM backend + WireCR multi-level backbone + CenterNet-lite proposal + SAM per-instance refine`
- proposal / refine / joint 三阶段训练框架
- COCO AP + 工业指标评估
- 单图 / 文件夹推理

当前尚未真正接通的部分：

- `SAM2.1` 官方 backend 只预留了抽象接口，没有真正接入 image encoder / prompt encoder / mask decoder
- 当前真正可运行的路径仍是 `SAM1` 集成

也就是说：

- [sam_backend.py](/home/zyh/InjectAdapterSAM/models/backbones/sam_backend.py) 中已经有 `SAMBackendBase`、`SAM1Backend`、`SAM21Backend`
- 但 `SAM21Backend` 当前会直接抛出 `ImportError`
- 所以训练与评估命令默认必须使用 `SAM1` 权重，例如 [sam_vit_b_01ec64.pth](/home/zyh/InjectAdapterSAM/checkpoints/sam_vit_b_01ec64.pth)

## 4. 目录结构与关键文件

实例主线最核心的文件如下：

- 数据转换
  - [convert_isat_to_coco_instance.py](/home/zyh/InjectAdapterSAM/scripts/convert_isat_to_coco_instance.py)
  - [check_group_consistency.py](/home/zyh/InjectAdapterSAM/scripts/check_group_consistency.py)
- 数据集
  - [wire_hole_instance_dataset.py](/home/zyh/InjectAdapterSAM/dataset/wire_hole_instance_dataset.py)
- 模型
  - [sam_backend.py](/home/zyh/InjectAdapterSAM/models/backbones/sam_backend.py)
  - [sam_wirecr_backbone.py](/home/zyh/InjectAdapterSAM/models/backbones/sam_wirecr_backbone.py)
  - [instance_proposal_head.py](/home/zyh/InjectAdapterSAM/models/heads/instance_proposal_head.py)
  - [instance_prompt_builder.py](/home/zyh/InjectAdapterSAM/models/prompts/instance_prompt_builder.py)
  - [sam_instance_refiner.py](/home/zyh/InjectAdapterSAM/models/refine/sam_instance_refiner.py)
  - [roi_boundary_refiner.py](/home/zyh/InjectAdapterSAM/models/refine/roi_boundary_refiner.py)
  - [wirecr_instsam.py](/home/zyh/InjectAdapterSAM/models/wirecr_instsam.py)
- 训练评估推理
  - [train_instsam.py](/home/zyh/InjectAdapterSAM/train_instsam.py)
  - [eval_instsam.py](/home/zyh/InjectAdapterSAM/eval_instsam.py)
  - [infer_instsam.py](/home/zyh/InjectAdapterSAM/infer_instsam.py)
- 指标与匹配
  - [instance_losses.py](/home/zyh/InjectAdapterSAM/utils/instance_losses.py)
  - [instance_matcher.py](/home/zyh/InjectAdapterSAM/utils/instance_matcher.py)
  - [instance_metrics.py](/home/zyh/InjectAdapterSAM/utils/instance_metrics.py)
  - [instsam_runtime.py](/home/zyh/InjectAdapterSAM/utils/instsam_runtime.py)

## 5. 数据集与格式

### 5.1 原始数据

当前原始标注是平铺的 ISAT 风格目录，例如：

```text
samDataset/
├── image_sam_002.jpg
├── image_sam_002.json
├── image_sam_003.jpg
├── image_sam_003.json
├── labeled_manifest.csv
├── labeled_manifest.txt
└── ...
```

JSON 中每个 object 通常至少包含：

- `category`
- `group`
- `segmentation`

### 5.2 实例定义

实例主线采用以下固定定义：

- 同一张图内，同类 `(category, group)` 视为同一实例
- 同组多个 polygon 合并为同一个实例 annotation
- 原始类别中的 `hole` 和 `interface-hole` 会统一映射到实例类别 `2`

类别定义如下：

- `1 = wire`
- `2 = interface-hole`

背景不再作为实例类别导出。

### 5.3 COCO instance 导出

数据转换脚本是 [convert_isat_to_coco_instance.py](/home/zyh/InjectAdapterSAM/scripts/convert_isat_to_coco_instance.py)。

它会完成以下工作：

- 扫描平铺的 ISAT 图片与 JSON
- 按 `(category, group)` 聚合同实例多 polygon
- 生成 COCO 风格的 `images / annotations / categories`
- 将 `hole` 与 `interface-hole` 合并到同一实例类别
- 基于感知哈希进行近重复聚类，再按组切分 train / val / test
- 导出人工复核表 `instance_review.csv`

示例命令：

```bash
python3 scripts/convert_isat_to_coco_instance.py \
  --src ./samDataset \
  --dst ./samDataset_instance_coco \
  --seed 42 \
  --overwrite
```

转换后目录结构大致为：

```text
samDataset_instance_coco/
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── reviews/
│   └── instance_review.csv
└── split_manifest.csv
```

### 5.4 近重复划分策略

划分逻辑不再是简单随机，而是基于 [coco_export.py](/home/zyh/InjectAdapterSAM/utils/coco_export.py) 中的感知哈希聚类：

- 对图片做 `dhash`
- 通过汉明距离将近似重复图聚成组
- 以组为单位切分 train / val / test

默认比例：

- `train = 0.8`
- `val = 0.1`
- `test = 0.1`

### 5.5 数据集类返回格式

实例 dataloader 在 [wire_hole_instance_dataset.py](/home/zyh/InjectAdapterSAM/dataset/wire_hole_instance_dataset.py) 中实现。

每个样本返回：

```python
{
  "image": Tensor[3, H, W],
  "original_size": (h, w),          # 当前 sample 尺寸，若做过 ROI，则是 crop 后尺寸
  "full_image_size": (H0, W0),      # 原始整图尺寸
  "processed_size": (image_size, image_size),
  "resize_scale": (scale_y, scale_x),
  "crop_box": Optional[List[float]],
  "image_id": int,
  "instances": {
      "boxes": Tensor[N, 4],        # xyxy，基于 processed_size
      "labels": Tensor[N],          # 1=wire, 2=interface-hole
      "masks": Tensor[N, H, W],     # uint8
      "areas": Tensor[N],
      "iscrowd": Tensor[N],
      "groups": List[int],
  },
  "image_path": str,
}
```

### 5.6 训练采样策略

训练集支持整图与 ROI 混合采样：

- `roi_prob = 0.4`
- `roi_focus_prob = 0.7`
- `roi_scale = 1.4`

含义是：

- 大约 `60%` 样本直接用整图
- 大约 `40%` 样本改为局部裁剪
- ROI 中优先采样：
  - 小 hole
  - 细长 wire
  - 密集区域

实现位置在 [wire_hole_instance_dataset.py](/home/zyh/InjectAdapterSAM/dataset/wire_hole_instance_dataset.py) 的 `_sample_crop_box()`。

## 6. 模型思路

### 6.1 整体结构

当前 `WireCR-InstSAM` 的实际结构是：

```text
Image
-> SAM preprocess
-> SAM image encoder + WireCR multi-level backbone
-> CenterNet-lite proposal head
-> box + 1 positive prompt builder
-> SAM prompt encoder + mask decoder
-> optional ROI boundary refiner
-> class-wise mask NMS
-> final instances
```

对应顶层模型文件是 [wirecr_instsam.py](/home/zyh/InjectAdapterSAM/models/wirecr_instsam.py)。

### 6.2 与旧语义主线的关系

旧语义主线仍然保留，但定位变了：

- [sam_wrapper.py](/home/zyh/InjectAdapterSAM/models/sam_wrapper.py)：按类别 prompt 做语义分割
- [sam_fpn_segmentor.py](/home/zyh/InjectAdapterSAM/models/sam_fpn_segmentor.py)：FPN 头输出语义图

它们的输出本质仍然是类别图，不是实例列表。  
新实例主线则尝试把“候选生成”和“逐实例 SAM 精分”拆开处理。

## 7. 当前实现的模型创新点

这里的“创新点”指的是当前代码层面已经落地的设计点，而不是论文结论。

### 7.1 从语义分割数据流改为实例分割数据流

当前代码已经不再把 ISAT JSON 直接画成一张语义 mask，而是：

- 保留实例级 polygon
- 合并同组多 polygon
- 导出 COCO instance 格式

这一步是从任务定义上彻底区别于旧主线的核心变化。

### 7.2 SAM backend 抽象

[sam_backend.py](/home/zyh/InjectAdapterSAM/models/backbones/sam_backend.py) 提供了统一接口：

- `preprocess`
- `postprocess_masks`
- `encode_prompts`
- `decode_masks`

这样上层实例模型只依赖 backend 抽象，不直接依赖 `SAM1` 或 `SAM2.1` 的具体实现。  
当前实际运行时仍是 `SAM1Backend`。

### 7.3 多层 WireCR backbone

[sam_wirecr_backbone.py](/home/zyh/InjectAdapterSAM/models/backbones/sam_wirecr_backbone.py) 从 SAM image encoder 中抽取多层 token 特征：

- `c2`
- `c3`
- `c4`
- `c5`

然后做：

- `proj_c2 / proj_c3 / proj_c4`
- `adapter_c3 / adapter_c4 / adapter_c5`

注意：

- `adapter_c2` 当前是 `Identity`
- `proj_c5` 当前也是 `Identity`
- `freeze_encoder=True` 时会冻结 SAM image encoder

### 7.4 类别相关的 proposal 中心定义

[instance_proposal_head.py](/home/zyh/InjectAdapterSAM/models/heads/instance_proposal_head.py) 当前对 `wire` 和 `hole` 使用不同的中心监督：

- `wire`：使用实例 mask 的 distance-transform 峰值点
- `hole`：使用 bbox 几何中心

这比“所有类别都用 bbox center”更符合当前工业目标几何形态。

### 7.5 proposal head 的当前实现要点

当前 proposal head 不是 DETR，而是 CenterNet-lite 风格的 dense head。  
它输出：

- `center_heatmap`
- `size_map`
- `offset_map`

当前代码里的实际实现细节包括：

- `stride = 8`
- 三路特征 `c3/c4/c5` 先各自投影
- 再 `concat + 1x1 merge_conv` 融合
- 再通过一层上采样和卷积生成 proposal 特征

这意味着当前代码已经不是最初的“简单相加融合”版本。

### 7.6 类别相关的 prompt 规则

[instance_prompt_builder.py](/home/zyh/InjectAdapterSAM/models/prompts/instance_prompt_builder.py) 当前实现是：

- box 先做 `1.10` 倍扩张
- 每个 proposal 只生成 `1` 个 positive point

并按类别分开：

- `hole`：直接用 box center
- `wire`：
  - 训练时若有 `gt_mask`，用 mask distance peak
  - 否则若有 proposal center，用 proposal center
  - 再不行才退化到 box center

### 7.7 SAM refine 的多候选筛选

[sam_instance_refiner.py](/home/zyh/InjectAdapterSAM/models/refine/sam_instance_refiner.py) 当前使用 `multimask_output=True`，即每个实例会生成 3 个候选 mask。

筛选逻辑是：

- 先计算 `mask-in-box recall`
- 过滤掉低于 `inside_box_recall_threshold=0.70` 的候选
- 如果全部被过滤，则回退为保留全部候选
- 再在剩余候选里选 decoder `iou_score` 最大者

这比只用 bbox IoU 过滤更适合细长实例。

### 7.8 工业指标补充

除了 COCO AP，当前还额外实现了工业指标：

- `wire_cldice`
- `instance_boundary_f1`
- `hole_recall`
- `count_mae`
- `merge_error_rate`
- `split_error_rate`

对应实现文件是 [instance_metrics.py](/home/zyh/InjectAdapterSAM/utils/instance_metrics.py)。

需要强调一点：

- `merge_error_rate` 和 `split_error_rate` 在当前代码里**名字叫 rate，但实现上是按图平均错误个数，不是归一化比例**

## 8. 模型各模块的当前实现

### 8.1 SAM backend

当前可运行的是 `SAM1Backend`：

- 支持 `vit_b / vit_l / vit_h`
- 当前实例主线默认实际使用 `vit_b`

`SAM2.1` 目前只保留接口，不可直接训练。

另一个重要实现细节是：

- dataloader 输出图像为 `[0,1]`
- SAM 原始预处理均值方差基于 `0-255`
- 所以 [sam_backend.py](/home/zyh/InjectAdapterSAM/models/backbones/sam_backend.py) 当前会在需要时自动先把图像乘回 `255.0`

这一点对 proposal 能否学起来影响很大。

### 8.2 backbone

backbone 文件是 [sam_wirecr_backbone.py](/home/zyh/InjectAdapterSAM/models/backbones/sam_wirecr_backbone.py)。

工作流程：

- 调用 `image_encoder.patch_embed`
- 依次经过 transformer blocks
- 在预设 block index 抽取中间层 token
- 转成 `BCHW`
- 投影到统一 `256` 通道
- 送入对应 WireCR adapter

### 8.3 proposal head

proposal head 文件是 [instance_proposal_head.py](/home/zyh/InjectAdapterSAM/models/heads/instance_proposal_head.py)。

当前实现的关键配置：

- `stride = 8`
- `feat_channels = 128`
- 输出类别数 `2`
- `center_head.bias = -2.19`

当前 target 构造还有两个额外细节：

- `hole_positive_weight` 会给 hole 正中心点更高权重
- 对 `hole` 类别，当前强制 `radius = max(radius, 2)`

这意味着 hole 周围会有更宽的高斯软区域。  
但要注意：由于当前 loss 把 `target_heatmap == 1` 才算正样本，所以这不会把 hole 变成更多正样本点，只是减弱中心附近的负样本压制。

### 8.4 proposal loss

proposal loss 在 [instance_losses.py](/home/zyh/InjectAdapterSAM/utils/instance_losses.py) 中实现。

当前默认：

- `alpha = 2.0`
- `beta = 4.0`
- `size_weight = 0.1`
- `offset_weight = 1.0`

总 loss：

```text
heatmap_loss + 0.1 * size_loss + 1.0 * offset_loss
```

这是当前 proposal 训练的重要默认设定。

### 8.5 refine

refine 模块调用流程：

- `prompt_builder` 生成 `boxes / point_coords / point_labels`
- backend `encode_prompts`
- backend `decode_masks`
- `postprocess_masks`
- 三候选中选择最佳 mask

### 8.6 ROI boundary refiner

[roi_boundary_refiner.py](/home/zyh/InjectAdapterSAM/models/refine/roi_boundary_refiner.py) 当前是一个可选模块，不是默认主路径。

结构非常轻：

- `feature_branch`
- `mask_branch`
- `refine`

输入是：

- 低层特征
- 当前 mask logits

默认不开，只有传 `--enable-roi-refiner` 才会参与 refine。

## 9. 训练

### 9.1 环境

推荐环境：

```bash
conda env create -f environment.yml
conda activate injectadaptersam
pip install -r requirements_sam.txt
pip install -e ./third_party/sam
```

至少需要：

- `torch`
- `torchvision`
- `numpy`
- `opencv-python`
- `pycocotools`
- `Pillow`

### 9.2 当前推荐 backend

当前实际推荐命令必须走 `SAM1`：

```bash
--sam-backend sam1
--sam-model-type vit_b
--sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth
```

`SAM2.1` 权重虽然已下载到 [sam2.1_hiera_small.pt](/home/zyh/InjectAdapterSAM/checkpoints/sam2.1_hiera_small.pt)，但当前代码还不能真正拿它完成训练。

### 9.3 训练阶段

当前 [train_instsam.py](/home/zyh/InjectAdapterSAM/train_instsam.py) 支持四种 `phase`：

- `oracle`
- `proposal`
- `refine`
- `joint`

其中：

- `oracle` 实际不训练，只打印提示，让你用 `eval_instsam.py --oracle`
- 真正训练阶段是 `proposal / refine / joint`

### 9.4 Proposal-only

proposal-only 阶段只优化 proposal loss，不训练最终 refine mask。

当前推荐命令：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate injectadaptersam

CUDA_VISIBLE_DEVICES=2 python3 train_instsam.py \
  --phase proposal \
  --data-dir ./samDataset_instance_coco \
  --sam-backend sam1 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --image-size 1024 \
  --batch-size 2 \
  --workers 8 \
  --persistent-workers \
  --prefetch-factor 4 \
  --amp \
  --amp-dtype bf16 \
  --channels-last \
  --epochs 150 \
  --lr 1e-4 \
  --seed 42 \
  --topk-per-class 64 \
  --proposal-box-nms-iou 0.5 \
  --hole-positive-weight 5.0 \
  --train-metrics-interval 5 \
  --run-name wirecr_instsam_v1_proposal_s8_sz01_hole5_gpu2_bs2_e150_ampbf16 \
  --save-dir ./checkpoints
```

日志中的两个关键指标：

- `proposal_recall@100`
- `hole_recall`

注意：

- 这里的 `hole_recall` 是 raw proposal 级别的 hole 召回，不是最终实例 AP

### 9.5 Refine-only

refine 阶段会：

- 前半段更多使用 GT proposals
- 后半段切换为 matched proposals
- 如果 matched proposal 为空，会回退到 GT proposals

当前代码里是否使用 GT proposal，由：

- `epoch < int(args.epochs * args.refine_gt_ratio)`

决定。

### 9.6 Joint

joint 阶段把：

- proposal loss
- refine loss

相加一起训练。

### 9.7 一个当前必须知道的默认行为

`--freeze-encoder` 当前只有启用参数，没有提供 `--no-freeze-encoder`。  
并且默认是 `True`，这意味着当前实际可用路径基本等价于：

- 冻结 SAM image encoder
- 只训练 WireCR backbone 里的 projection / adapter
- 再训练 proposal / refine 相关模块

### 9.8 checkpoint 保存逻辑

每个 run 目录都会保存：

- `last.pth`
- `best.pth`

但当前 `best.pth` 并不是按验证集 AP 选，而是按训练期指标选。  
在 `proposal` 阶段，打分逻辑是：

```text
avg_proposal_recall + avg_hole_recall
```

也就是说：

- `best.pth` 更像“训练期 raw proposal 指标最好”
- 不一定是“验证集最终实例 AP 最好”

## 10. 评估

评估脚本是 [eval_instsam.py](/home/zyh/InjectAdapterSAM/eval_instsam.py)。

### 10.1 标准评估流程

标准评估会依次做：

1. 读取数据集
2. 跑 `forward_backbone`
3. 跑 proposal head
4. decode proposal
5. 计算 raw proposal 指标
6. 再把 proposal 送进 refine
7. 根据 score threshold + mask NMS 过滤
8. 输出 COCO AP 与工业指标

所以要特别注意：

- `proposal_recall@100` 和 `hole_recall` 是 **refine 前** 的 raw proposal 指标
- `AP / AP50 / AP75` 是 **refine 后 + threshold + NMS 后** 的最终实例指标

### 10.2 threshold grid

如果不手动指定阈值，评估会自动搜索：

- `wire_score_thresh ∈ {0.10, 0.15, 0.20, 0.25}`
- `hole_score_thresh ∈ {0.10, 0.15, 0.20, 0.25}`

然后选择：

- `AP50` 更高者
- 若 `AP50` 相同，则 `hole_recall` 更高者

### 10.3 标准评估命令

```bash
CUDA_VISIBLE_DEVICES=2 python3 eval_instsam.py \
  --data-dir ./samDataset_instance_coco \
  --split val \
  --sam-backend sam1 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --checkpoint ./checkpoints/wirecr_instsam_v1_proposal_s8_sz01_hole5_gpu2_bs2_e150_ampbf16/best.pth \
  --workers 4 \
  --seed 42 \
  --topk-final 50 \
  --wire-mask-nms-iou 0.60 \
  --hole-mask-nms-iou 0.60
```

### 10.4 Oracle 评估

Oracle 模式会直接使用 GT proposals，而不是 learned proposals：

```bash
CUDA_VISIBLE_DEVICES=2 python3 eval_instsam.py \
  --oracle \
  --data-dir ./samDataset_instance_coco \
  --split val \
  --sam-backend sam1 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --checkpoint ./checkpoints/wirecr_instsam_v1_proposal_s8_sz01_hole5_gpu2_bs2_e150_ampbf16/best.pth
```

它主要用来回答：

- proposal 不拖后腿时，refine 路径本身上限有多高

### 10.5 当前工业指标的真实含义

[instance_metrics.py](/home/zyh/InjectAdapterSAM/utils/instance_metrics.py) 里：

- `wire_cldice`：仅对匹配到的 wire 实例统计
- `instance_boundary_f1`：仅对匹配到的实例统计
- `hole_recall`：匹配到的 GT hole 数 / GT hole 总数
- `count_mae`：每图总实例计数绝对误差平均值
- `merge_error_rate`：每图 `pred_count - matched_count` 的平均值下界为 0
- `split_error_rate`：每图 `gt_count - matched_count` 的平均值下界为 0

最后两个名字带 `rate`，但当前实现并不是百分比。

## 11. 推理

推理脚本是 [infer_instsam.py](/home/zyh/InjectAdapterSAM/infer_instsam.py)。

支持：

- `--image`
- `--image-dir`

### 11.1 推理流程

推理时会做：

1. 整图 resize 到 `image_size`
2. proposal decode
3. per-instance refine
4. score threshold 过滤
5. class-wise mask NMS
6. 导出 JSON

### 11.2 当前推理输出

当前默认输出目录例如：

```text
inference_instsam/
├── image_sam_002.json
├── image_sam_010.json
└── ...
```

每个 JSON 条目包含：

- `instance_id`
- `category_id`
- `score`
- `bbox`
- `source_prompt`

当前脚本**不会默认导出 mask 图或可视化图**。如果要把 mask 存成 PNG 或 RLE，需要在现有脚本上继续扩展。

### 11.3 推理命令示例

```bash
CUDA_VISIBLE_DEVICES=2 python3 infer_instsam.py \
  --image ./samDataset/image_sam_002.jpg \
  --data-dir ./samDataset_instance_coco \
  --sam-backend sam1 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --checkpoint ./checkpoints/wirecr_instsam_v1_proposal_s8_sz01_hole5_gpu2_bs2_e150_ampbf16/best.pth \
  --wire-score-thresh 0.20 \
  --hole-score-thresh 0.20 \
  --wire-mask-nms-iou 0.60 \
  --hole-mask-nms-iou 0.60 \
  --topk-final 50 \
  --output-dir ./inference_instsam
```

## 12. 当前实验现状

### 12.1 旧语义 FPN 仍然是最稳的工业域特征来源

当前旧语义主线中效果最好的一条是：

- [wirecr_fpn_c2c3_light_c4c5_full_2](/home/zyh/InjectAdapterSAM/checkpoints/wirecr_fpn_c2c3_light_c4c5_full_2)

它的语义分割结果大致是：

- `mIoU ≈ 0.6323`
- `wire_iou ≈ 0.5782`
- `hole_iou ≈ 0.3773`
- `hole_recall ≈ 0.5698`

这说明旧 FPN 线已经学到了一定的工业域特征，但它输出的仍然是类别图，不是实例。

### 12.2 当前实例 proposal-only 仍然偏弱

当前 `WireCR-InstSAM` 的 proposal-only 结果还不稳定。

例如：

- 经过归一化修复的一版 proposal run 曾出现过非零 `AP50`
- 后续几轮 proposal 训练中，raw `proposal_recall@100` 和 `hole_recall` 有时会抬起来
- 但最终 `AP / AP50` 仍然经常回到 `0`

这说明当前实例主线仍处在“proposal 开始有响应，但还没有稳定转化成最终实例结果”的阶段。

## 13. 当前已知局限

### 13.1 `SAM2.1` 尚未真正接通

目前只能实际跑 `SAM1`。

### 13.2 proposal-only 的标准评估会经过 refine

也就是说，即使 raw proposal 指标有提升，最终 AP 仍可能是 0。  
当前没有单独的“只评 raw proposal，不走 refine”的官方入口。

### 13.3 `best.pth` 不是按验证集 AP 选

当前 `best.pth` 更像训练期最优 proposal 指标，不是最终验证集最优实例模型。

### 13.4 `infer_instsam.py` 默认不导出 mask 文件

当前只导出 JSON。

### 13.5 CLI 中 encoder 基本默认冻结

当前没有显式的 `--no-freeze-encoder` 选项。

## 14. 建议的使用顺序

当前最稳的工作顺序是：

1. 检查原始标注

```bash
python3 scripts/check_group_consistency.py --src ./samDataset
```

2. 转 COCO instance

```bash
python3 scripts/convert_isat_to_coco_instance.py \
  --src ./samDataset \
  --dst ./samDataset_instance_coco \
  --seed 42 \
  --overwrite
```

3. 先跑 proposal-only

4. 再做标准评估与 Oracle 评估

5. proposal 有基础召回后，再进入 refine / joint

## 15. 一条当前最稳的训练命令

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate injectadaptersam

CUDA_VISIBLE_DEVICES=2 python3 train_instsam.py \
  --phase proposal \
  --data-dir ./samDataset_instance_coco \
  --sam-backend sam1 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --image-size 1024 \
  --batch-size 2 \
  --workers 8 \
  --persistent-workers \
  --prefetch-factor 4 \
  --amp \
  --amp-dtype bf16 \
  --channels-last \
  --epochs 150 \
  --lr 1e-4 \
  --seed 42 \
  --topk-per-class 64 \
  --proposal-box-nms-iou 0.5 \
  --hole-positive-weight 5.0 \
  --train-metrics-interval 5 \
  --run-name wirecr_instsam_v1_proposal_s8_sz01_hole5_gpu2_bs2_e150_ampbf16 \
  --save-dir ./checkpoints
```

如果你接下来要继续推进实例主线，最值得优先关注的不是最终 AP，而是：

- `proposal_recall@100`
- raw `hole_recall`

只有 proposal 先站住，后面的 refine 与 joint 才有意义。
