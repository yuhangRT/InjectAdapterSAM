# WireCR-InstSAM 使用文档

本文档对应仓库中的新实例分割主线 `WireCR-InstSAM`。这条主线的目标不是再输出一张 `background / wire / interface-hole` 语义图，而是输出每个 `wire` 和每个 `interface-hole` 的独立实例。

## 1. 当前实现状态

- 已实现的数据链路：`ISAT JSON -> COCO instance -> instance dataloader`
- 已实现的模型链路：`SAM backend + WireCR backbone + CenterNet-lite proposal + per-instance SAM refine`
- 已实现的入口脚本：
  - `scripts/convert_isat_to_coco_instance.py`
  - `scripts/check_group_consistency.py`
  - `train_instsam.py`
  - `eval_instsam.py`
  - `infer_instsam.py`
- 已下载的 `SAM2.1` 权重：
  - `./checkpoints/sam2.1_hiera_small.pt`

需要明确一点：

- 当前仓库已经预留了 `SAM2.1` backend 抽象。
- 但当前分支里 `SAM2.1` 后端还没有真正接通，`models/backbones/sam_backend.py` 里目前仍会回退到现有 `SAM1` 集成。
- 所以“当前可直接跑通的训练/评估命令”默认仍应使用 `SAM1` 权重，例如 `./checkpoints/sam_vit_b_01ec64.pth`。
- `./checkpoints/sam2.1_hiera_small.pt` 现在已经放好，后续把官方 `sam2` 后端接进来时可以直接复用。

## 2. 环境准备

### 2.1 推荐方式

推荐直接使用仓库现有环境文件：

```bash
conda env create -f environment.yml
conda activate injectadaptersam
```

然后补齐当前实例主线依赖：

```bash
pip install -r requirements_sam.txt
pip install -e ./third_party/sam
```

### 2.2 最少依赖

至少需要这些包：

- `torch`
- `torchvision`
- `numpy`
- `opencv-python` 或 `opencv-python-headless`
- `pycocotools`
- `Pillow`

如果缺少这些包，实例数据集和评估脚本会直接报错。

### 2.3 关于官方 SAM2

如果你只是想先跑通当前实例主线，这一步不是必须的，因为当前代码默认还会回退到 `SAM1`。

如果你后续要继续接通真正的 `SAM2.1` backend，可以按 Meta 官方 README 安装：

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

官方参考：

- `https://github.com/facebookresearch/sam2`

但要注意：本仓库当前分支还没有把 `SAM2.1` image/prompt/mask 接口真正接完。也就是说：

- `SAM2.1` 权重已经下载好了
- `SAM2.1` backend 抽象已经留好了
- 当前可运行主线仍是 `SAM1 fallback`

## 3. 权重文件

当前 `checkpoints/` 下建议至少保留两份权重：

```text
checkpoints/
├── sam_vit_b_01ec64.pth
└── sam2.1_hiera_small.pt
```

说明：

- `sam_vit_b_01ec64.pth`：当前代码实际可运行的 `SAM1` 路径
- `sam2.1_hiera_small.pt`：已经下载完成的 `SAM2.1` 权重，供后续接通官方 backend 后使用

## 4. 数据预处理

### 4.1 原始数据格式

当前原始标注仍然是平铺的 ISAT 结构，例如：

```text
samDataset/
├── image_sam_002.jpg
├── image_sam_002.json
├── image_sam_003.jpg
├── image_sam_003.json
└── ...
```

实例定义固定为：

- 同一张图内，同类 `(category, group)` 视为同一实例
- 同组多个 polygon 会在导出时合并成一个实例 annotation

支持的前景类别：

- `wire`
- `hole`
- `interface-hole`

其中 `hole` 和 `interface-hole` 会统一映射为 `category_id = 2`。

### 4.2 先检查 group 一致性

建议在转换前先跑一遍：

```bash
python3 scripts/check_group_consistency.py \
  --src ./samDataset
```

这个脚本会输出：

- 每个类别的 grouped instance 数量
- 每个类别的 polygon 数量
- 前若干张图的 grouped preview

它的作用是先检查 `(category, group)` 是否明显混乱。

### 4.3 转成 COCO instance

实例主线不再使用旧的语义 PNG mask，而是使用 COCO-style instance 标注。

执行：

```bash
python3 scripts/convert_isat_to_coco_instance.py \
  --src ./samDataset \
  --dst ./samDataset_instance_coco \
  --seed 42 \
  --overwrite
```

这个脚本会完成：

- 读取 ISAT JSON
- 按 `(category, group)` 合并同实例多 polygon
- 导出 `images/train|val|test`
- 导出 `annotations/instances_train.json`
- 额外导出人工复核表 `reviews/instance_review.csv`
- 按感知哈希近重复聚类后再 split，避免近似重复图跨集合泄漏

默认 split 比例：

- `train = 0.8`
- `val = 0.1`
- `test = 0.1`

你也可以显式改：

```bash
python3 scripts/convert_isat_to_coco_instance.py \
  --src ./samDataset \
  --dst ./samDataset_instance_coco \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --hash-threshold 6 \
  --seed 42 \
  --overwrite
```

### 4.4 转换后的目录结构

转换后目录大致如下：

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

### 4.5 当前数据集加载行为

`dataset/wire_hole_instance_dataset.py` 目前已经实现：

- 整图训练
- ROI crop 训练
- `60%` 整图 + `40%` ROI
- ROI 内部优先采样小 hole、细长 wire 和密集区

每个 sample 返回的核心字段是：

```python
{
  "image": Tensor[3, H, W],
  "original_size": (H0, W0),
  "processed_size": (H, W),
  "full_image_size": (H0, W0),
  "resize_scale": (scale_y, scale_x),
  "crop_box": Optional[List[float]],
  "image_id": int,
  "instances": {
      "boxes": Tensor[N, 4],
      "labels": Tensor[N],
      "masks": Tensor[N, H, W],
      "areas": Tensor[N],
      "iscrowd": Tensor[N],
      "groups": List[int],
  },
  "image_path": str,
}
```

## 5. 训练

### 5.1 先说明当前推荐运行路径

如果你现在就要真正训练，请先按“当前可运行路径”执行：

- `--sam-backend sam1`
- `--sam-model-type vit_b`
- `--sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth`

不要直接把 `./checkpoints/sam2.1_hiera_small.pt` 塞给当前训练脚本当作唯一 checkpoint，否则当前分支会因为 `SAM2.1` backend 尚未接通而回退失败。

如果你只是想保留未来兼容配置，可以另外记录：

- `--sam-backend auto`
- `--sam-model-type hiera_small`
- `--sam-checkpoint ./checkpoints/sam2.1_hiera_small.pt`
- `--fallback-sam-model-type vit_b`
- `--fallback-sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth`

但这套配置在当前代码里本质上仍会走 `SAM1 fallback`。

### 5.2 Phase 0：Oracle 上界验证

Oracle 不通过 `train_instsam.py` 训练，而是直接在评估脚本中启用 `--oracle`。

执行：

```bash
python3 eval_instsam.py \
  --oracle \
  --data-dir ./samDataset_instance_coco \
  --split val \
  --sam-backend sam1 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth
```

注意：

- `eval_instsam.py` 现在允许不传 `--checkpoint`
- 如果不传，评估将使用当前初始化权重
- 更稳妥的做法仍然是先完成 Phase 1 proposal 训练，再用该 checkpoint 做 Oracle 对比

Oracle 主要看：

- `AP50`
- `hole_recall`
- `merge_error_rate`

### 5.3 Phase 1：Proposal-only

只训练：

- WireCR adapters
- CenterNet-lite proposal head

示例命令：

```bash
CUDA_VISIBLE_DEVICES=2 python3 train_instsam.py \
  --phase proposal \
  --data-dir ./samDataset_instance_coco \
  --sam-backend sam1 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --image-size 1024 \
  --batch-size 4 \
  --workers 8 \
  --persistent-workers \
  --prefetch-factor 4 \
  --amp \
  --amp-dtype bf16 \
  --channels-last \
  --epochs 20 \
  --lr 1e-4 \
  --seed 42 \
  --topk-per-class 64 \
  --proposal-box-nms-iou 0.5 \
  --hole-positive-weight 1.5 \
  --train-metrics-interval 20 \
  --run-name wirecr_instsam_v1_proposal \
  --save-dir ./checkpoints

```

主要日志指标：

- `loss`
- `proposal_recall@100`
- `hole_recall`

输出目录：

```text
checkpoints/wirecr_instsam_v1_proposal/
├── best.pth
└── last.pth
```
这里最好一下评估，Proposal 模型评估

```bash
CUDA_VISIBLE_DEVICES=0 python3 eval_instsam.py \
  --data-dir ./samDataset_instance_coco \
  --split val \
  --sam-backend sam1 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --checkpoint ./checkpoints/wirecr_instsam_v1_proposal/best.pth \
  --seed 42 \
  --gpu 0 \
  --topk-final 50 \
  --wire-mask-nms-iou 0.60 \
  --hole-mask-nms-iou 0.60

```

用训练后 checkpoint 跑 Oracle
```bash
CUDA_VISIBLE_DEVICES=0 python3 eval_instsam.py \
  --oracle \
  --data-dir ./samDataset_instance_coco \
  --split val \
  --sam-backend sam1 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --checkpoint ./checkpoints/wirecr_instsam_v1_proposal/best.pth \
  --seed 42 \
  --gpu 0 \
  --topk-final 50 \
  --wire-mask-nms-iou 0.60 \
  --hole-mask-nms-iou 0.60

```

### 5.4 Phase 2：Refine-only

这一阶段固定 proposal，训练：

- prompt 构造后的 SAM refine
- 可选的 decoder 后段

示例命令：

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_instsam.py \
  --phase refine \
  --data-dir ./samDataset_instance_coco \
  --sam-backend sam1 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --image-size 1024 \
  --batch-size 1 \
  --workers 4 \
  --epochs 20 \
  --lr 1e-4 \
  --seed 42 \
  --gpu 0 \
  --topk-per-class 64 \
  --proposal-box-nms-iou 0.5 \
  --hole-positive-weight 1.5 \
  --refine-gt-ratio 0.5 \
  --resume ./checkpoints/wirecr_instsam_v1_proposal/best.pth \
  --run-name wirecr_instsam_v1_refine \
  --save-dir ./checkpoints

```

说明：

- 训练前半段默认用 GT proposals
- 后半段默认切换到 matched proposals

### 5.5 Phase 3：Joint

联合训练：

- adapters
- proposal head
- SAM refine 路径

示例命令：

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_instsam.py \
  --phase joint \
  --data-dir ./samDataset_instance_coco \
  --sam-backend sam1 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --image-size 1024 \
  --batch-size 1 \
  --workers 4 \
  --epochs 20 \
  --lr 1e-4 \
  --seed 42 \
  --gpu 0 \
  --topk-per-class 64 \
  --proposal-box-nms-iou 0.5 \
  --hole-positive-weight 1.5 \
  --refine-gt-ratio 0.5 \
  --resume ./checkpoints/wirecr_instsam_v1_refine/best.pth \
  --run-name wirecr_instsam_v1_joint \
  --save-dir ./checkpoints

```

### 5.6 Phase 4：可选 ROI Refiner

当前 V1 里，ROI refiner 不是默认主增益模块，只建议在主 pipeline 已经稳定后再打开。

示例命令：

```bash
python3 train_instsam.py \
  --phase joint \
  --enable-roi-refiner \
  --data-dir ./samDataset_instance_coco \
  --sam-backend sam1 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --image-size 1024 \
  --batch-size 1 \
  --workers 0 \
  --epochs 20 \
  --lr 1e-4 \
  --resume ./checkpoints/wirecr_instsam_v1_joint/best.pth \
  --run-name wirecr_instsam_v1_joint_roi \
  --save-dir ./checkpoints
```

## 6. 评估

### 6.1 标准评估

用验证集评估 `best.pth`：

```bash
python3 eval_instsam.py \
  --data-dir ./samDataset_instance_coco \
  --split val \
  --sam-backend sam1 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --checkpoint ./checkpoints/wirecr_instsam_v1_joint/best.pth
```

完整命令
```bash
CUDA_VISIBLE_DEVICES=0 python3 eval_instsam.py \
  --data-dir ./samDataset_instance_coco \
  --split val \
  --sam-backend sam1 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --checkpoint ./checkpoints/wirecr_instsam_v1_joint/best.pth \
  --seed 42 \
  --gpu 0 \
  --topk-final 50 \
  --wire-mask-nms-iou 0.60 \
  --hole-mask-nms-iou 0.60

```

如果你不手动指定阈值，脚本会自动搜索：

- `wire_score_thresh ∈ {0.10, 0.15, 0.20, 0.25}`
- `hole_score_thresh ∈ {0.10, 0.15, 0.20, 0.25}`

并输出最佳阈值组合。

### 6.2 手动指定阈值

```bash
python3 eval_instsam.py \
  --data-dir ./samDataset_instance_coco \
  --split val \
  --sam-backend sam1 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --checkpoint ./checkpoints/wirecr_instsam_v1_joint/best.pth \
  --wire-score-thresh 0.15 \
  --hole-score-thresh 0.20 \
  --wire-mask-nms-iou 0.60 \
  --hole-mask-nms-iou 0.60 \
  --topk-final 50
```

### 6.3 Oracle 评估

如果你想看 `GT bbox + GT positive -> SAM refine` 的上界：

```bash
python3 eval_instsam.py \
  --oracle \
  --data-dir ./samDataset_instance_coco \
  --split val \
  --sam-backend sam1 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --checkpoint ./checkpoints/wirecr_instsam_v1_joint/best.pth
```

### 6.4 当前评估输出

评估脚本当前会输出两类指标。

COCO 风格指标：

- `AP`
- `AP50`
- `AP75`
- `APS`
- `APM`
- `APL`

工业特化指标：

- `wire_cldice`
- `instance_boundary_f1`
- `hole_recall`
- `count_mae`
- `merge_error_rate`
- `split_error_rate`

同时还会输出：

- `proposal_recall@100`
- `hole_recall`

## 7. 推理

### 7.1 单张图推理

```bash
python3 infer_instsam.py \
  --image ./samDataset/image_sam_002.jpg \
  --data-dir ./samDataset_instance_coco \
  --sam-backend sam1 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --checkpoint ./checkpoints/wirecr_instsam_v1_joint/best.pth \
  --output-dir ./inference_instsam
```

### 7.2 整个目录推理

```bash
python3 infer_instsam.py \
  --image-dir ./samDataset \
  --data-dir ./samDataset_instance_coco \
  --sam-backend sam1 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --checkpoint ./checkpoints/wirecr_instsam_v1_joint/best.pth \
  --output-dir ./inference_instsam \
  --wire-score-thresh 0.15 \
  --hole-score-thresh 0.20
```

### 7.3 当前推理输出内容

当前 `infer_instsam.py` 默认会为每张图输出一个 JSON 文件，字段包括：

- `instance_id`
- `category_id`
- `score`
- `bbox`
- `source_prompt`

当前脚本还没有默认把最终 mask 直接另存成 PNG 或 RLE 文件。如果你需要 COCO-style mask 导出，请优先复用 `eval_instsam.py` 的评估路径或再扩展 `predictions_to_coco()`。

## 8. 这条实例主线和旧语义主线的关系

旧主线仍然保留，但角色已经变了：

- `models/sam_wrapper.py`：旧 prompt 语义主线
- `models/sam_fpn_segmentor.py`：旧 FPN 语义主线
- `scripts/convert_isat_to_wire_hole.py`：旧语义 mask 转换脚本

这些现在只应作为 baseline，不再作为最终主线继续修补。

实例主线应优先使用：

- `scripts/convert_isat_to_coco_instance.py`
- `dataset/wire_hole_instance_dataset.py`
- `train_instsam.py`
- `eval_instsam.py`
- `infer_instsam.py`

## 9. 常见问题

### 9.1 `pycocotools` / `cv2` / `torch` 导入失败

说明当前环境不完整。先补：

```bash
pip install -r requirements_sam.txt
pip install -e ./third_party/sam
```

如果你使用的是一个全新的环境，建议直接按 `environment.yml` 重建。

### 9.2 为什么已经下载了 `sam2.1_hiera_small.pt`，但训练还是不走 `SAM2.1`

因为当前仓库只是预留了 `SAM2.1` backend 抽象，还没有把官方 `sam2` 的 image encoder / prompt encoder / mask decoder 真正接进 `WireCR-InstSAM`。

也就是说：

- 权重已经准备好了
- 计划路径已经定了
- 当前真正可运行的还是 `SAM1 fallback`

### 9.3 当前最稳妥的命令应该怎么写

如果你现在只追求先把实例主线跑通，最稳妥的写法就是：

```bash
python3 train_instsam.py \
  --phase proposal \
  --data-dir ./samDataset_instance_coco \
  --sam-backend sam1 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --run-name wirecr_instsam_v1_proposal
```

先把 proposal-only 跑通，再进 refine-only 和 joint。
