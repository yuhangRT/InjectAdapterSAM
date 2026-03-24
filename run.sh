# 数据增强
python3 scripts/augment_wire_hole_dataset.py \
  --src ./samDataset_wire_hole \
  --dst ./samDataset_wire_hole_aug_strong \
  --train-copies 2 \
  --augment-strength strong \
  --seed 42 \
  --overwrite

# 默认 prompt head 弃用，使用 FPN 结构
python main.py \
  --mode sam \
  --gpu 2 \
  --data-dir ./samDataset_wire_hole \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --head-type prompt \
  --batch-size 1 \
  --workers 4 \
  --epochs 100 \
  --adapter-size medium \
  --compression-ratio 8 \
  --class-aware-prompts \
  --boundary-loss-weight 0.1 \
  --cldice-weight 0.1 \
  --hole-class-weight 2.0 \
  --train-augment none \
  --run-name wirecrsam_prompt_vitb_med_cr8

# fpn head c4 c5 oldloss
python main.py \
  --mode sam \
  --gpu 2 \
  --data-dir ./samDataset_wire_hole \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --head-type fpn \
  --batch-size 1 \
  --workers 4 \
  --epochs 100 \
  --adapter-size medium \
  --compression-ratio 8 \
  --boundary-loss-weight 0.1 \
  --cldice-weight 0.1 \
  --hole-class-weight 2.0 \
  --main-class-weights 1.0 1.5 4.0 \
  --hole-aux-weight 0.3 \
  --train-augment none \
  --run-name wirecrsam_fpn_vitb_med_cr8

# 默认配置，newloss，fpn c4c5全量，c2c3仅映射
# 和下面的c2c3small做一个对比
python main.py \
  --mode sam \
  --gpu 2 \
  --data-dir ./samDataset_wire_hole \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --head-type fpn \
  --batch-size 1 \
  --workers 4 \
  --epochs 100 \
  --adapter-kind wirecr \
  --adapter-size medium \
  --compression-ratio 8 \
  --main-class-weights 1.0 1.5 4.0 \
  --hole-aux-weight 0.3 \
  --boundary-loss-weight 0.1 \
  --cldice-weight 0.1 \
  --hole-class-weight 2.0 \
  --fpn-adapter-levels c4,c5 \
  --train-augment none \
  --run-name wirecr_fpn_c2c3_project_c4c5_full_1


# 显示配置，c2c3轻量，c4c5全量
python main.py \
  --mode sam \
  --gpu 2 \
  --data-dir ./samDataset_wire_hole \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --head-type fpn \
  --batch-size 1 \
  --workers 4 \
  --epochs 100 \
  --adapter-kind wirecr \
  --adapter-size medium \
  --compression-ratio 8 \
  --boundary-loss-weight 0.1 \
  --cldice-weight 0.1 \
  --hole-class-weight 2.0 \
  --main-class-weights 1.0 1.5 4.0 \
  --hole-aux-weight 0.3 \
  --fpn-adapter-levels c2,c3,c4,c5 \
  --fpn-adapter-size-map c2=small,c3=small,c4=medium,c5=medium \
  --fpn-compression-map c2=16,c3=16,c4=8,c5=8 \
  --fpn-simple-map c2=1,c3=1,c4=0,c5=0 \
  --train-augment none \
  --run-name wirecr_fpn_c2c3_light_c4c5_full_1

# fpn + vanilla adapter baseline
python main.py \
  --mode sam \
  --gpu 2 \
  --data-dir ./samDataset_wire_hole \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --head-type fpn \
  --batch-size 1 \
  --workers 4 \
  --epochs 100 \
  --adapter-kind vanilla \
  --adapter-size medium \
  --compression-ratio 8 \
  --boundary-loss-weight 0.1 \
  --cldice-weight 0.1 \
  --hole-class-weight 2.0 \
  --main-class-weights 1.0 1.5 4.0 \
  --hole-aux-weight 0.3 \
  --fpn-adapter-levels c4,c5 \
  --train-augment none \
  --run-name vanilla_adapter_fpn_vitb_med_cr8


# 正式评估
python main.py \
  --mode sam \
  --evaluate \
  --gpu 2 \
  --data-dir ./samDataset_wire_hole \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --pretrained ./checkpoints/wirecrsam_4090_vitb_med_cr8/best_iou.pth \
  --batch-size 1 \
  --workers 4 \
  --adapter-size medium \
  --compression-ratio 8 \
  --class-aware-prompts
# fpn 评估
python main.py \
  --mode sam \
  --evaluate \
  --gpu 2 \
  --data-dir ./samDataset_wire_hole \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --head-type fpn \
  --pretrained ./checkpoints/wirecrsam_fpn_vitb_med_cr8_320_1/best_iou.pth \
  --batch-size 1 \
  --workers 4 \
  --adapter-size medium \
  --compression-ratio 8 \
  --main-class-weights 1.0 1.5 4.0 \
  --hole-aux-weight 0.3 \
# 完整评估
python main.py \
  --mode sam \
  --evaluate \
  --gpu 2 \
  --data-dir ./samDataset_wire_hole \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --head-type fpn \
  --pretrained ./checkpoints/wirecr_fpn_c2c3_light_c4c5_full_1/best_iou.pth \
  --batch-size 1 \
  --workers 4 \
  --adapter-kind wirecr \
  --adapter-size medium \
  --compression-ratio 8 \
  --main-class-weights 1.0 1.5 4.0 \
  --hole-aux-weight 0.3 \
  --fpn-adapter-levels c2,c3,c4,c5 \
  --fpn-adapter-size-map c2=small,c3=small,c4=medium,c5=medium \
  --fpn-compression-map c2=16,c3=16,c4=8,c5=8 \
  --fpn-simple-map c2=1,c3=1,c4=0,c5=0


# 单张图片推理
python3 scripts/infer_image.py \
  --image /path/to/test.jpg \
  --pretrained ./checkpoints/wirecrsam_fpn_vitb_med_cr8_320_1/best_iou.pth

# 批量推理
python3 scripts/infer_image.py \
  --image-dir samDataset_wire_hole/images/test \
  --pretrained ./checkpoints/fpn_baseline/best_iou.pth

# FPN 结构消融
python scripts/run_thesis_suite.py \
  --table 4-3 \
  --python python \
  --head-type fpn \
  --data-dir ./samDataset_wire_hole \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --batch-size 1 \
  --workers 4 \
  --epochs 100 \
  --adapter-size medium \
  --adapter-kind wirecr \
  --compression-ratio 8 \
  --fpn-adapter-levels c4,c5

# 主对比实验
python scripts/run_thesis_suite.py \
  --table 4-2 \
  --python python \
  --head-type fpn \
  --data-dir ./samDataset_wire_hole \
  --dataset wire_hole \
  --num-classes 3 \
  --sam-model-type vit_b \
  --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --batch-size 1 \
  --workers 4 \
  --epochs 100 \
  --adapter-size medium \
  --compression-ratio 8
