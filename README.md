# InjectAdapterSAM

`InjectAdapterSAM` 将 CRNet 风格的多尺度卷积适配器接入 SAM（Segment Anything Model），形成当前仓库里的 `WireCR-SAM` 实验主线。

当前代码的定位已经固定为两类任务：

- `wire_hole`：主任务，工业机床电路线束与接口孔洞三类语义分割
- `coco`：辅任务，COCO 类别无关前景分割，仅用于辅助实验、迁移或烟测

完整中文说明见 [使用说明.md](./使用说明.md)。

## 当前特性

- SAM `vit_b` / `vit_l` / `vit_h` 三种主干
- `small` / `medium` / `large` 三种 WireCR 适配器
- `4 / 8 / 16 / 32 / 64` 五种压缩比
- 自动类感知 prompts，不再依赖旧版 `prompt-strategy` / `num-prompts`
- 多类 BCE + Dice + Boundary + clDice 损失
- `wire_hole` 主任务 + `coco` 辅助任务共用同一套训练入口

## 安装

```bash
git clone git@github.com:yuhangRT/InjectAdapterSAM.git
cd InjectAdapterSAM
git submodule update --init --recursive

pip install -r requirements_sam.txt
```

下载 SAM 权重：

```bash
mkdir -p checkpoints

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/
# 或
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -P checkpoints/
# 或
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P checkpoints/
```

## 数据集

### 1. `wire_hole`（主任务）

标签定义：

- `0`: background
- `1`: wire
- `2`: interface-hole

支持以下任一目录结构：

```text
<data-root>/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── masks/
    ├── train/
    ├── val/
    └── test/
```

或：

```text
<data-root>/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

`labels/` 也可以替代 `masks/`。

如果你的原始数据是当前仓库里的 `samDataset/` 这种平铺 ISAT 结构：

```text
samDataset/
├── image_sam_002.jpg
├── image_sam_002.json
├── ...
└── isat.yaml
```

请先离线转换成 `wire_hole` 目录，而不是直接把 JSON 喂给训练脚本：

```bash
python3 scripts/convert_isat_to_wire_hole.py \
  --src ./samDataset \
  --dst ./samDataset_wire_hole \
  --seed 42
```

转换脚本会：

- 按 `0=background, 1=wire, 2=interface-hole` 导出单通道 PNG mask
- 自动生成 `images/train|val|test` 和 `masks/train|val|test`
- 按 `int(0.8N)` / `int(0.9N)` 切点生成 `94/12/12` 的当前 split

说明：

- 不需要在 JSON 里单独标注 `background`
- 背景来自未被 `wire` 或 `interface-hole` 多边形覆盖的像素
- 若保留彩色 mask，固定颜色必须是黑=`background`、绿=`wire`、红=`interface-hole`

### 2. `coco`（辅助任务）

当前实现不是 COCO 多类实例分割，而是“把同一张图里的所有实例合并成前景”的二类语义分割：

- `0`: background
- `1`: foreground

必须使用：

```bash
--dataset coco --num-classes 2
```

支持以下任一结构：

```text
<data-root>/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

或：

```text
<data-root>/
├── images/
│   ├── train2017/
│   └── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

## 训练

### `wire_hole` 主任务

```bash
python main_sam.py \
  --mode sam \
  --data-dir /path/to/wire_hole \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --batch-size 1 \
  --workers 0 \
  --epochs 100 \
  --adapter-size small \
  --compression-ratio 16 \
  --class-aware-prompts \
  --boundary-loss-weight 0.1 \
  --cldice-weight 0.1 \
  --hole-class-weight 2.0
```

### `coco` 辅助任务

```bash
python main_sam.py \
  --mode sam \
  --data-dir /path/to/coco \
  --dataset coco \
  --num-classes 2 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --batch-size 1 \
  --workers 0 \
  --epochs 10 \
  --adapter-size small \
  --compression-ratio 16 \
  --class-aware-prompts \
  --boundary-loss-weight 0.1 \
  --cldice-weight 0.0
```

## 评估

```bash
python main_sam.py \
  --mode sam \
  --evaluate \
  --data-dir /path/to/wire_hole \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --pretrained ./checkpoints/wirecrsam_wire_hole_small_cr16/best_iou.pth \
  --batch-size 1 \
  --workers 0 \
  --adapter-size small \
  --compression-ratio 16
```

每次训练或评估结束后，程序会在对应运行目录下自动写出 `experiment_summary.json`。如需显式指定输出位置，可额外传入：

```bash
--run-name your_run_name \
--save-dir ./checkpoints \
--results-json ./checkpoints/your_run_name/experiment_summary.json
```

## 论文实验脚本

### 1. 运行表 4-2 到表 4-5 的实验套件

下面的脚本会直接调用当前 `main_sam.py`，并为每个表生成一份 manifest：

```bash
python scripts/run_thesis_suite.py \
  --table 4-2 \
  --output-root ./thesis_runs \
  --data-dir /path/to/wire_hole \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --batch-size 1 \
  --workers 0 \
  --epochs 100 \
  --adapter-size medium \
  --compression-ratio 8
```

支持：

- `--table 4-2`：主对比实验
- `--table 4-3`：结构消融
- `--table 4-4`：损失消融
- `--table 4-5`：少样本实验
- `--table all`：依次运行全部套件

只想先看将要执行的命令时，可加 `--dry-run`。

### 2. 直接导出 CSV

表 4-1 导出主实验训练配置：

```bash
python scripts/export_thesis_tables.py \
  table4_1 \
  --reference ./thesis_runs/table4_2/table4_2_wirecr_sam \
  --output ./thesis_tables/table4_1.csv
```

表 4-2 导出主对比实验：

```bash
python scripts/export_thesis_tables.py \
  table4_2 \
  --manifest ./thesis_runs/table4_2/manifest_table4_2.json \
  --output ./thesis_tables/table4_2.csv
```

表 4-3 导出结构消融：

```bash
python scripts/export_thesis_tables.py \
  table4_3 \
  --manifest ./thesis_runs/table4_3/manifest_table4_3.json \
  --output ./thesis_tables/table4_3.csv
```

表 4-4 导出损失消融：

```bash
python scripts/export_thesis_tables.py \
  table4_4 \
  --manifest ./thesis_runs/table4_4/manifest_table4_4.json \
  --output ./thesis_tables/table4_4.csv
```

表 4-5 导出少样本实验：

```bash
python scripts/export_thesis_tables.py \
  table4_5 \
  --manifest ./thesis_runs/table4_5/manifest_table4_5.json \
  --output ./thesis_tables/table4_5.csv
```

## 关键参数

| 参数 | 说明 |
|------|------|
| `--dataset` | `wire_hole` 或 `coco` |
| `--num-classes` | `wire_hole=3`, `coco=2` |
| `--sam-model-type` | `vit_b`, `vit_l`, `vit_h` |
| `--adapter-size` | `small`, `medium`, `large` |
| `--compression-ratio` | `4`, `8`, `16`, `32`, `64` |
| `--class-aware-prompts` | 是否启用自动类感知 prompts |
| `--freeze-encoder` | 是否冻结 SAM image encoder |
| `--freeze-decoder` | 是否冻结 SAM mask decoder |
| `--subset-ratio` | 训练集采样比例，支持少样本实验 |
| `--boundary-loss-weight` | 边界损失权重 |
| `--cldice-weight` | 细线结构 clDice 权重 |
| `--hole-class-weight` | `wire_hole` 中 hole 类正样本权重 |
| `--disable-adapter` | 关闭 WireCR adapter，作为无适配器对比基线 |
| `--run-name` | 显式指定当前实验名 |
| `--save-dir` | 指定实验输出根目录 |
| `--results-json` | 指定结构化结果 JSON 输出路径 |

## 说明

- 论文主线请使用 `wire_hole`，`coco` 只建议作为辅助实验。
- 当前推荐使用 `scripts/run_thesis_suite.py` 和 `scripts/export_thesis_tables.py` 组织论文实验。
- `scripts/grid_search.sh` 和 `scripts/aggregate_results.py` 仍然是旧版 COCO/prompt 流程脚本，未同步到当前类感知 prompt 接口。
- 如果你只是在 8GB 左右显存上验证可行性，优先使用 `vit_b + batch-size 1 + adapter-size small`。
