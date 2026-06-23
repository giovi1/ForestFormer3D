# TreeScanPL10K Extension of ForestFormer3D

This repository is a fork of ForestFormer3D adapted for TreeScanPL10K
inference, fine-tuning, and ablation experiments.

For the original ForestFormer3D installation instructions, base dataset setup,
model details, license, and citation, refer to the upstream repository:

https://github.com/SmartForest-no/ForestFormer3D

Original project resources:

- Project page: https://bxiang233.github.io/FF3D/
- Paper: https://www.arxiv.org/abs/2506.16991
- Dataset and pretrained model: https://zenodo.org/records/16742708

This README documents only the changes made in this fork.

## What This Fork Adds

- LAZ/LAS batch inference for TreeScanPL10K-style point clouds through
  `laz_inference/convert.py` and `laz_inference/infer.sh`.
- Safer full-scene inference output logic for arbitrary file names.
- Fixes around full-cloud prediction/evaluation so metrics are computed on the
  complete scene rather than only the last processed crop.
- A corrected `spconv` checkpoint workflow: convert checkpoints once with
  `tools/fix_spconv_checkpoint.py`, then load the repaired `_fix.pth`
  checkpoint directly during testing.
- TreeScanPL10K fine-tuning workflow and ablations, including zero-shot
  comparison, validation threshold selection, and decoder-loss ablation.
- A machine-readable metrics summary in
  `results/treescan_ablation_summary.json`.

## TreeScanPL10K Zero-Shot Inference

Run from inside the Docker container at `/workspace`:

```bash
bash laz_inference/infer.sh \
  --input-dir /data/TreeScanPL10k/batch_01/ \
  --voxel-size 0.7 \
  --gpu 0 \
  --force
```

The script converts `.laz`/`.las` inputs into the existing
ForAINetV2-compatible format, regenerates metadata, runs ForestFormer3D, and
writes predicted `.ply` files under
`work_dirs/oneformer3d_qs_radius16_qp300_2many/`.

## TreeScanPL10K Fine-Tuning

The fine-tuning implementation was developed on the `ft/treescan-finetuning`
branch. Use that branch, or merge these files into the branch you are using:

- `configs/oneformer3d_treescan_finetune.py`
- `configs/oneformer3d_treescan_finetune_smoke.py`
- `tools/prepare_treescan_finetune.py`
- `tools/fix_treescan_info_labels.py`
- `tools/run_treescan_finetune_after_prepare.sh`

The fine-tuning setup keeps the ForestFormer3D head compatible with the
original ForAINetV2 checkpoint. TreeScanPL10K is treated as background plus tree
instances; the unused ForAINetV2 class slot is kept for checkpoint
compatibility.

Prepare the TreeScan training root:

```bash
python tools/prepare_treescan_finetune.py \
  --input-dir /data/TreeScanPL10k/batch_01 \
  --data-root data/TreeScanPL10K \
  --voxel-size 0.1
```

Generate metadata:

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

python tools/fix_treescan_info_labels.py \
  --data-root data/TreeScanPL10K \
  --backup
```

Run the smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_treescan_finetune_smoke.py \
  --work-dir work_dirs/treescan_finetune_voxel_0_1_smoke
```

Run the main fine-tune:

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_treescan_finetune.py \
  --work-dir work_dirs/treescan_finetune_voxel_0_1_decoder
```

For TreeScanPL10K, keep:

```bash
model.prepare_epoch=-1
```

The inherited `prepare_epoch=1000` delays decoder losses beyond the short
TreeScan fine-tune and gives much weaker instance segmentation.

Repair the selected checkpoint before final evaluation:

```bash
python tools/fix_spconv_checkpoint.py \
  --in-path work_dirs/treescan_finetune_voxel_0_1_decoder/best_mRQ_epoch_45.pth \
  --out-path work_dirs/treescan_finetune_voxel_0_1_decoder/best_mRQ_epoch_45_spconv_fix.pth
```

## Ablation Results

All results below use the same TreeScanPL10K split and evaluator so the
original ForestFormer3D checkpoint can be compared directly with the fine-tuned
model. Exact values and source JSON paths are stored in
`results/treescan_ablation_summary.json`.

| Run | Checkpoint/settings | mIoU | mPQ | mRQ/F1 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Zero-shot ForestFormer3D | `epoch_3000_fix.pth` | 0.2552 | 0.3485 | 0.4730 | 0.3914 | 0.5976 |
| TreeScan fine-tuned baseline | `best_mRQ_epoch_45_spconv_fix.pth` | 0.7016 | 0.6386 | 0.7708 | 0.7595 | 0.7824 |
| Validation-selected threshold T2 | fine-tuned + threshold tuning | 0.7010 | 0.6453 | 0.7741 | 0.7979 | 0.7516 |
| No decoder-loss ablation | `model.prepare_epoch=1000` | 0.6699 | 0.4364 | 0.5844 | 0.6614 | 0.5234 |

The validation-selected threshold is:

```bash
model.score_th=0.35
model.test_cfg.sp_score_thr=0.1
model.test_cfg.npoint_thr=10
model.test_cfg.min_instance_points=10
```

These are inference/post-processing thresholds from the original
ForestFormer3D/ForAINet-style instance prediction path. They control confidence,
superpoint mask score, and minimum instance size filtering. They are not labels
or training losses.

Run the selected threshold once on held-out test:

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

## Interpretation

Fine-tuning is necessary for TreeScanPL10K: the original ForestFormer3D
checkpoint transfers poorly in zero-shot mode on this dataset. The
validation-selected threshold slightly improves `mPQ` and `mRQ/F1` mainly by
increasing precision. The decoder ablation shows that enabling decoder losses
with `model.prepare_epoch=-1` is responsible for most of the instance
segmentation gain.

## Citation

Please cite the original ForestFormer3D paper when using this codebase:

```bibtex
@inproceedings{xiang2025forestformer3d,
  title     = {ForestFormer3D: A Unified Framework for End-to-End Segmentation of Forest LiDAR 3D Point Clouds},
  author    = {Binbin Xiang and Maciej Wielgosz and Stefano Puliti and Kamil Kral and Martin Krucek and Azim Missarov and Rasmus Astrup},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```
