#!/bin/bash
set -euo pipefail

now=$(date +"%Y%m%d_%H%M%S")

epoch=120
bs=1
lr=0.000005
encoder=vitl
dataset=xray
img_size=518
min_depth=0.0
max_depth=1000.0
pretrained_from=../checkpoints/depth_anything_v2_${encoder}.pth
save_path=exp/xray

mkdir -p "$save_path"

# Optional: force using only one GPU (GPU 0). Change to 1,2,... if desired.
export CUDA_VISIBLE_DEVICES=0

python3 train.py \
  --epochs "$epoch" \
  --encoder "$encoder" \
  --bs "$bs" \
  --lr "$lr" \
  --save-path "$save_path" \
  --dataset "$dataset" \
  --img-size "$img_size" \
  --min-depth "$min_depth" \
  --max-depth "$max_depth" \
  --pretrained-from "$pretrained_from" \
  2>&1 | tee -a "$save_path/$now.log"
