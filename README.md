# ForestFormer3D

ForestFormer3D is a Python research codebase based on OpenMMLab/MMDetection3D conventions for 3D semantic and instance segmentation of forest point clouds. The current project branch focuses on adapting the existing ForAINetV2-trained OneFormer3D model to TreeScanPL10K, then using the segmented tree instances as the input to a later classification/species-assignment stage.

## Repository Layout

- `oneformer3d/`: model, losses, metrics, datasets, transforms, and full-scene prediction code.
- `configs/`: OpenMMLab experiment configs. The active TreeScan config is `configs/oneformer3d_treescan_finetune.py`.
- `tools/`: training, testing, data conversion, checkpoint repair, and TreeScan preparation utilities.
- `laz_inference/`: LAZ/LAS conversion and zero-shot batch inference path.
- `replace_mmdetection_files/`: patched OpenMMLab files used by the Docker environment.
- `work_dirs/`: local training/evaluation outputs. This is intentionally ignored by Git.
- `results/`: local archived experiment artifacts. Heavy files are ignored; commit only small summaries such as README files.

## Environment

Build and enter the expected Docker environment from the repository root:

```bash
sudo docker build -t forestformer3d-image .
sudo docker run --gpus all --shm-size=128g -d -v "$PWD":/workspace --name forestformer3d-container forestformer3d-image
sudo docker exec -it forestformer3d-container /bin/bash
cd /workspace
```

The tested environment for the accepted TreeScan baseline used Python 3.10, PyTorch 1.13.1, CUDA 11.6, MMEngine 0.7.3, and one A100 40 GB GPU.

## Data Layout

Keep large data outside Git. In this workspace the TreeScanPL10K data is exposed through a local symlink:

```text
TreeScanPL10k -> /home/ubuntu/IDEAS_code/TreeScanPL10k
```

The fine-tuning pipeline writes converted data to:

```text
data/TreeScanPL10K/
  train_val_data/
  test_data/
  meta_data/
  forainetv2_instance_data/
  forainetv2_oneformer3d_infos_train.pkl
  forainetv2_oneformer3d_infos_val.pkl
  forainetv2_oneformer3d_infos_test.pkl
```

## Zero-Shot LAZ Inference

Run the existing ForAINetV2 checkpoint on raw TreeScan LAZ/LAS files:

```bash
bash laz_inference/infer.sh \
  --input-dir /data/TreeScanPL10k/batch_01/ \
  --voxel-size 0.5 \
  --gpu 0 \
  --force
```

Predictions are written under `work_dirs/oneformer3d_qs_radius16_qp300_2many/`. Converted intermediate inputs are regenerated under `data-test/`.

## TreeScanPL10K Fine-Tuning

The accepted fine-tuning setup uses voxel size `0.1`. Voxel size `0.2` is a lower-memory fallback, but the current baseline was reported with `0.1`.

Prepare the converted TreeScan dataset:

```bash
python tools/prepare_treescan_finetune.py \
  --input-dir /data/TreeScanPL10k/batch_01 \
  --data-root data/TreeScanPL10K \
  --voxel-size 0.1
```

Generate ForAINetV2-compatible metadata:

```bash
cd data/TreeScanPL10K
python /workspace/data/ForAINetV2/batch_load_ForAINetV2_data.py \
  --train_scan_names_file meta_data/train_list.txt \
  --val_scan_names_file meta_data/val_list.txt \
  --test_scan_names_file meta_data/test_list.txt \
  --train_forainetv2_dir train_val_data \
  --test_forainetv2_dir test_data \
  --output_folder forainetv2_instance_data

cd /workspace
python tools/create_data_forainetv2.py forainetv2 \
  --root-path data/TreeScanPL10K \
  --out-dir data/TreeScanPL10K
```

Patch existing info PKLs so TreeScan tree instances are foreground class `1`, not background/stuff class `0`:

```bash
python tools/fix_treescan_info_labels.py \
  --data-root data/TreeScanPL10K \
  --backup
```

Check the dataset summary in the next training log. The expected instance counts are `ground=0`, `tree/wood=<tree count>`, and `leaf=0`. If all instances appear under `ground`, decoder fine-tuning targets are wrong.

Run a smoke test before the long job:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_treescan_finetune_smoke.py \
  --work-dir work_dirs/treescan_finetune_voxel_0_1_smoke
```

Run the main fine-tuning job:

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_treescan_finetune.py \
  --work-dir work_dirs/treescan_finetune_voxel_0_1_decoder
```

The TreeScan config keeps the model head ForAINetV2-compatible for checkpoint loading. TreeScan labels use class `0` for background and class `1` for tree; class `2` remains unused. Keep `model.prepare_epoch=-1` because the base ForAINetV2 value delays decoder losses beyond this fine-tuning schedule.

## Checkpoint Repair and Evaluation

Repair the best checkpoint before final full-scene evaluation:

```bash
python tools/fix_spconv_checkpoint.py \
  --in-path work_dirs/treescan_finetune_voxel_0_1_decoder/best_mRQ_epoch_45.pth \
  --out-path work_dirs/treescan_finetune_voxel_0_1_decoder/best_mRQ_epoch_45_spconv_fix.pth
```

Use the repaired checkpoint for reporting. The unrepaired checkpoint loads with many current-`spconv` 5D weight shape mismatches and gives misleading full-scene metrics.

Evaluate on the validation split as a full-scene test run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/oneformer3d_treescan_finetune.py \
  work_dirs/treescan_finetune_voxel_0_1_decoder/best_mRQ_epoch_45_spconv_fix.pth \
  --work-dir work_dirs/treescan_finetune_voxel_0_1_decoder_fullval_spconv_fix \
  --cfg-options \
  test_dataloader.dataset.ann_file=forainetv2_oneformer3d_infos_val.pkl \
  model.test_cfg.output_dir=work_dirs/treescan_finetune_voxel_0_1_decoder_fullval_spconv_fix
```

Evaluate on the held-out test split:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/oneformer3d_treescan_finetune.py \
  work_dirs/treescan_finetune_voxel_0_1_decoder/best_mRQ_epoch_45_spconv_fix.pth \
  --work-dir work_dirs/treescan_finetune_voxel_0_1_decoder_test_spconv_fix \
  --cfg-options \
  model.test_cfg.output_dir=work_dirs/treescan_finetune_voxel_0_1_decoder_test_spconv_fix
```

## Accepted TreeScan Baseline

Accepted run:

```text
work_dirs/treescan_finetune_voxel_0_1_decoder/20260617_084113
```

Accepted repaired checkpoint:

```text
work_dirs/treescan_finetune_voxel_0_1_decoder/best_mRQ_epoch_45_spconv_fix.pth
```

Full-scene validation result using the repaired checkpoint:

```text
mIoU=0.7226
mPQ=0.6705
mSQ=0.8315
mRQ/F1=0.8064
mPrecision=0.7958
mRecall=0.8173
mMUCov=0.7038
mMWCov=0.7580
```

Held-out test result using the repaired checkpoint:

```text
mIoU=0.7011
mPQ=0.6351
mSQ=0.8297
mRQ/F1=0.7654
mPrecision=0.7485
mRecall=0.7831
mMUCov=0.7272
mMWCov=0.7691
```

Local archived artifacts are under:

```text
results/treescan_finetune_voxel_0_1_baseline_spconv_fix/
```

That archive contains the repaired checkpoint, final config, training log/scalars, full-scene validation predictions, held-out predictions, JSON metrics, and logs. Heavy artifacts are intentionally ignored by Git.

## Next Project Stage

The segmentation baseline is now strong enough to use as the input to classification/species assignment. The next stage should consume predicted tree instances from the repaired-checkpoint full-scene outputs, join them with TreeScan metadata such as `individual_tree_summary.csv` and `species_id_names.csv`, and train/evaluate a classifier with a fixed split that does not leak plots or sites across train/validation/test.

## Git Hygiene

Do not commit datasets, checkpoints, point clouds, Docker-generated `work_dirs`, or editor state. Commit configs, scripts, model-code changes, and concise experiment summaries that make the local artifacts reproducible.
