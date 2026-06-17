#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/TreeScanPL10K}"
WORK_DIR="${2:-work_dirs/treescan_finetune_voxel_0_1_decoder}"

cd /workspace

cd "${DATA_ROOT}"
python /workspace/data/ForAINetV2/batch_load_ForAINetV2_data.py \
  --train_scan_names_file meta_data/train_list.txt \
  --val_scan_names_file meta_data/val_list.txt \
  --test_scan_names_file meta_data/test_list.txt \
  --train_forainetv2_dir train_val_data \
  --test_forainetv2_dir test_data \
  --output_folder forainetv2_instance_data

cd /workspace
python tools/create_data_forainetv2.py forainetv2 \
  --root-path "${DATA_ROOT}" \
  --out-dir "${DATA_ROOT}"

python tools/fix_treescan_info_labels.py \
  --data-root "${DATA_ROOT}" \
  --backup

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_treescan_finetune.py \
  --work-dir "${WORK_DIR}"
