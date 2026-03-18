# 训练
python main.py \
  --mode sam \
  --gpu 2 \ 
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
  --run-name wirecrsam_4090_vitb_med_cr8

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


# 单张图片推理
python3 scripts/infer_image.py \
  --image /path/to/test.jpg \
  --pretrained ./checkpoints/wirecrsam_4090_vitb_med_cr8_1/best_iou.pth \
  --output-dir ./inference_outputs \
  --gpu 2
