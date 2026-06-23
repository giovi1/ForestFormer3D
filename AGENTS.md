# Repository Guidelines

## Project Structure & Module Organization

ForestFormer3D is a Python research codebase built around OpenMMLab/MMDetection3D conventions. Core model, dataset, loss, metric, and transform code lives in `oneformer3d/`. Experiment definitions are in `configs/`, especially `oneformer3d_qs_radius16_qp300_2many.py`. Operational scripts are under `tools/` for training, testing, data conversion, checkpoint repair, merging predictions, and evaluation. `laz_inference/` contains the LAZ/LAS batch inference path used for TreeScanPL10K. Replacement files for patched OpenMMLab packages are in `replace_mmdetection_files/`. Sample/intermediate data lives in `data-test/`, `data/`, `work_dirs/`, and `results/`; avoid committing new large outputs unless they are intentional fixtures.

## Build, Test, and Development Commands

Build and enter the expected container environment:

```bash
sudo docker build -t forestformer3d-image .
sudo docker run --gpus all --shm-size=128g -d -v "$PWD":/workspace --name forestformer3d-container forestformer3d-image
sudo docker exec -it forestformer3d-container /bin/bash
```

Prepare ForAINetV2 metadata after placing raw data:

```bash
python tools/create_data_forainetv2.py forainetv2
```

Train or evaluate with a config/checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/oneformer3d_qs_radius16_qp300_2many.py --work-dir work_dirs/<run_name>
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/oneformer3d_qs_radius16_qp300_2many.py work_dirs/clean_forestformer/epoch_3000_fix.pth
```

Repair checkpoints for current `spconv` layout:

```bash
python tools/fix_spconv_checkpoint.py --in-path <input.pth> --out-path <output_fix.pth>
```

Run TreeScanPL10K batch inference inside Docker from `/workspace`:

```bash
bash laz_inference/infer.sh --input-dir /data/TreeScanPL10k/batch_01/ --voxel-size 0.7 --gpu 0 --force
```

Predictions are written to `work_dirs/oneformer3d_qs_radius16_qp300_2many/Rem_Gorlice_*.ply`. Logs and JSON metrics are in timestamped subfolders such as `work_dirs/oneformer3d_qs_radius16_qp300_2many/20260610_084821/`. Converted inputs and masks are regenerated under `data-test/`.

TreeScanPL10K fine-tuning is staged on branch `ft/treescan-finetuning`. The current useful fine-tuning run uses voxel `0.1`; voxel `0.2` remains a lower-memory fallback. First prepare the training root inside Docker from `/workspace`:

```bash
python tools/prepare_treescan_finetune.py \
  --input-dir /data/TreeScanPL10k/batch_01 \
  --data-root data/TreeScanPL10K \
  --voxel-size 0.1
```

Then generate metadata for the existing ForAINetV2-compatible loader:

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

Patch existing TreeScan info PKLs before training if they were generated before the TreeScan label fix. The expected instance table is `ground=0`, `wood=<tree count>`, `leaf=0`; if all instances appear under `ground`, decoder training targets are wrong.

```bash
python tools/fix_treescan_info_labels.py \
  --data-root data/TreeScanPL10K \
  --backup
```

Always run the smoke config before a long fine-tuning run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_treescan_finetune_smoke.py \
  --work-dir work_dirs/treescan_finetune_voxel_0_1_smoke
```

If the smoke run passes data loading, checkpoint loading, loss, validation, and output writing, run the main fine-tune:

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_treescan_finetune.py \
  --work-dir work_dirs/treescan_finetune_voxel_0_1_decoder
```

The TreeScan fine-tuning config keeps the model head ForAINetV2-compatible for checkpoint loading. Labels use class `0` for background and class `1` for tree; class `2` remains unused unless the dataset is later expanded. Keep `model.prepare_epoch=-1` for TreeScan decoder fine-tuning; the inherited `prepare_epoch=1000` prevents instance decoder losses from running in short fine-tunes.

Accepted TreeScanPL10K fine-tuning baseline on voxel `0.1`: train with
`configs/oneformer3d_treescan_finetune.py`, then repair the best checkpoint for
the current `spconv` layout before final evaluation:

```bash
python tools/fix_spconv_checkpoint.py \
  --in-path work_dirs/treescan_finetune_voxel_0_1_decoder/best_mRQ_epoch_45.pth \
  --out-path work_dirs/treescan_finetune_voxel_0_1_decoder/best_mRQ_epoch_45_spconv_fix.pth
```

Use `work_dirs/treescan_finetune_voxel_0_1_decoder/best_mRQ_epoch_45_spconv_fix.pth`
for reporting. The unrepaired checkpoint loads with many 5D `spconv` weight
shape mismatches in the current container and produces misleading full-scene
metrics. The repaired-checkpoint full-scene validation result is `mIoU=0.7226`,
`mPQ=0.6705`, `mSQ=0.8315`, `mRQ/F1=0.8064`, `mPrecision=0.7958`,
`mRecall=0.8173`, `mMUCov=0.7038`, and `mMWCov=0.7580`. The held-out test
result is `mIoU=0.7011`, `mPQ=0.6351`, `mSQ=0.8297`, `mRQ/F1=0.7654`,
`mPrecision=0.7485`, `mRecall=0.7831`, `mMUCov=0.7272`, and `mMWCov=0.7691`.

The accepted baseline artifacts are archived under
`results/treescan_finetune_voxel_0_1_baseline_spconv_fix/`, including the
repaired checkpoint, final config, training log/scalars, full-scene validation
predictions, held-out predictions, JSON metrics, and logs. Treat this as the
starting point for classification/species assignment.

Completed TreeScanPL10K ablations are under `work_dirs/ablation/`. The
zero-shot comparison uses the original ForestFormer3D checkpoint
`work_dirs/clean_forestformer/epoch_3000_fix.pth` with the TreeScan fine-tune
config so the dataset, split, voxel size, and evaluator match the fine-tuned
run. On held-out test, zero-shot gives `mIoU=0.2552`, `mPQ=0.3485`,
`mSQ=0.7367`, `mRQ/F1=0.4730`, `mPrecision=0.3914`, `mRecall=0.5976`,
`mMUCov=0.5613`, and `mMWCov=0.5946`. The fine-tuned repaired checkpoint with
the baseline threshold gives `mIoU=0.7016`, `mPQ=0.6386`, `mSQ=0.8285`,
`mRQ/F1=0.7708`, `mPrecision=0.7595`, `mRecall=0.7824`, `mMUCov=0.7260`, and
`mMWCov=0.7672`.

The validation-selected threshold setting is T2:

```bash
model.score_th=0.35
model.test_cfg.sp_score_thr=0.1
model.test_cfg.npoint_thr=10
model.test_cfg.min_instance_points=10
```

T2 was selected on validation because it had the best `mRQ/F1` and `mPQ`.
Run it on test only after selection:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/oneformer3d_treescan_finetune.py \
  work_dirs/treescan_finetune_voxel_0_1_decoder/best_mRQ_epoch_45_spconv_fix.pth \
  --work-dir work_dirs/ablation/T2_threshold_precision_test \
  --cfg-options \
    model.score_th=0.35 \
    model.test_cfg.sp_score_thr=0.1 \
    model.test_cfg.npoint_thr=10 \
    model.test_cfg.min_instance_points=10
```

The T2 held-out test result is `mIoU=0.7010`, `mPQ=0.6453`, `mSQ=0.8337`,
`mRQ/F1=0.7741`, `mPrecision=0.7979`, `mRecall=0.7516`, `mMUCov=0.7022`, and
`mMWCov=0.7548`. Use this as the validation-tuned threshold result when
reporting `mPQ`/`mRQ`; use the baseline-threshold result when comparing only the
effect of fine-tuning without threshold changes.

The decoder fine-tuning ablation trains the same config with
`model.prepare_epoch=1000`, which effectively prevents instance decoder losses
from running during the short TreeScan fine-tune:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_treescan_finetune.py \
  --work-dir work_dirs/ablation/A4_no_decoder_prepare1000 \
  --cfg-options model.prepare_epoch=1000
```

Repair the selected checkpoint before evaluation:

```bash
python tools/fix_spconv_checkpoint.py \
  --in-path work_dirs/ablation/A4_no_decoder_prepare1000/best_mRQ_epoch_35.pth \
  --out-path work_dirs/ablation/A4_no_decoder_prepare1000/best_mRQ_epoch_35_spconv_fix.pth
```

The no-decoder held-out test result is `mIoU=0.6699`, `mPQ=0.4364`,
`mSQ=0.7468`, `mRQ/F1=0.5844`, `mPrecision=0.6614`, `mRecall=0.5234`,
`mMUCov=0.5158`, and `mMWCov=0.5759`. This ablation supports the main
TreeScan result: fine-tuning substantially improves over zero-shot
ForestFormer3D, and enabling decoder losses with `prepare_epoch=-1` accounts for
most of the gain.

## Coding Style & Naming Conventions

Use Python 3.10-compatible code and follow the existing PEP 8 style: 4-space indentation, `snake_case` functions/variables, `PascalCase` classes, and descriptive config keys. Keep MMEngine registries and config dictionaries consistent with nearby modules. Prefer small, local changes in `oneformer3d/` and mirror existing data sample structures instead of adding parallel abstractions.

## Testing Guidelines

There is no dedicated unit-test suite in this repository. Validate changes with the smallest relevant smoke test: import touched modules, run `tools/test.py` on the sample/pretrained checkpoint when GPU access is available, or run focused scripts such as `tools/eval_instance_seg.py` for metric changes. For data pipeline edits, test conversion on one file from `data-test/` before running full preprocessing.

For TreeScanPL10K experiments, preserve each run before changing thresholds or voxel size:

```bash
cp -r work_dirs/oneformer3d_qs_radius16_qp300_2many results/batch_01_voxel_0_7
```

Compare `--voxel-size` values such as `0.7`, `0.5`, and `0.3`; coarse downsampling is fast but can reduce recall.

## Commit & Pull Request Guidelines

Recent history uses short imperative commit subjects, for example `Fix double spconv weight permutation...` and `Add laz inference pipeline...`. Keep commits focused and mention affected pipeline areas. Pull requests should include a concise summary, config/checkpoint/data assumptions, commands run, key metrics or logs, and any large files intentionally added. Link related issues or experiments when applicable.

## Security & Configuration Tips

Do not hard-code local absolute paths, credentials, or machine-specific CUDA settings. Keep large datasets and checkpoints in documented locations (`data/ForAINetV2/`, `work_dirs/clean_forestformer/`) and prefer README-documented Docker setup for reproducible dependency versions.

## Project Notes

Current TreeScanPL10K work is zero-shot ULS-to-TLS inference before fine-tuning. The active config still assumes ForAINetV2 semantic classes (`ground`, `wood`, `leaf`), while TreeScanPL10K is effectively tree/background plus `treeID`; treat semantic cleanup and `score_th` as likely recall bottlenecks during ablations.
