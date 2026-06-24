# MS-DISA Query Selection

MS-DISA is an experimental query-selection option for ForestFormer3D:
Multi-Scale Density-Aware ISA Query Selection. It changes only how instance
query voxels are selected before the existing transformer decoder. The sparse
3D U-Net backbone, decoder, one-to-many matching/losses, and score-based block
merging remain unchanged.

The baseline behavior is still available with:

```python
model.query_selection.mode = "isa"
```

No performance improvement is assumed by this implementation. Train and
evaluate the ablations before reporting results.

## Modes

- `isa`: original behavior. Predicted tree voxels are sampled with FPS in ISA
  embedding space.
- `multiscale`: splits queries between original ISA-FPS and XYZ-space FPS to
  improve spatial coverage.
- `density`: splits queries between original ISA-FPS and density-aware weighted
  sampling.
- `ms_disa`: combines ISA-FPS, XYZ-space FPS, and density-aware sampling.

## Config Fields

The base config exposes:

```python
query_selection=dict(
    mode="isa",
    num_queries=300,
    isa_ratio=0.34,
    multiscale_ratio=0.33,
    density_ratio=0.33,
    density_sampling_mode="balanced",
    density_deterministic=True,
    density_k=16,
    density_eps=1e-6,
    deduplicate_queries=True,
    fallback_to_isa=True,
    log_interval=100)
```

`density_sampling_mode` can be:

- `low_density`: prioritize sparse candidate regions.
- `high_density`: prioritize dense candidate regions.
- `balanced`: keep a uniform component while giving sparse regions extra
  probability.

The density estimate is a lightweight grid-occupancy approximation over
candidate query voxels. It avoids point-wise O(N^2) neighbor searches.
When `density_deterministic=True`, density-aware candidates are selected by
weight order instead of random multinomial sampling, which is recommended for
validation and test runs.

## Ablation Configs

Baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_qs_radius16_qp300_2many.py \
  --work-dir work_dirs/msdisa_baseline
```

Multi-scale only:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_qs_radius16_qp300_2many_msdisa_multiscale.py \
  --work-dir work_dirs/msdisa_multiscale
```

Density-aware only:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_qs_radius16_qp300_2many_msdisa_density.py \
  --work-dir work_dirs/msdisa_density
```

Combined MS-DISA:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_qs_radius16_qp300_2many_msdisa.py \
  --work-dir work_dirs/msdisa_combined
```

For TreeScanPL10K fine-tuning, use the same data preparation, checkpoint repair,
and evaluation flow as the accepted baseline. TreeScan-specific configs are
available for the baseline and each query-selection ablation:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_treescan_finetune.py \
  --work-dir work_dirs/treescan_finetune_voxel_0_1_decoder

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_treescan_finetune_msdisa_multiscale.py \
  --work-dir work_dirs/treescan_finetune_voxel_0_1_msdisa_multiscale

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_treescan_finetune_msdisa_density.py \
  --work-dir work_dirs/treescan_finetune_voxel_0_1_msdisa_density

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_treescan_finetune_msdisa.py \
  --work-dir work_dirs/treescan_finetune_voxel_0_1_msdisa
```

To smoke-test TreeScan data loading and decoder losses before a long run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_treescan_finetune_smoke.py \
  --work-dir work_dirs/treescan_finetune_voxel_0_1_smoke
```

The same settings can also be applied with CLI overrides:

```bash
--cfg-options \
  model.query_selection.mode=ms_disa \
  model.query_selection.num_queries=128 \
  model.query_selection.isa_ratio=0.34 \
  model.query_selection.multiscale_ratio=0.33 \
  model.query_selection.density_ratio=0.33 \
  model.query_selection.density_deterministic=True
```

## Evaluation Table

Use the existing `tools/test.py` and TreeScan evaluator outputs. The relevant
metrics for the ablation table are:

```text
Model | Precision | Recall | F1/mRQ | mPQ | mIoU
```

Keep each run in a separate `work_dirs/` or `results/` folder so the saved config
and JSON/log metrics can be traced back to the query-selection mode.

## Current TreeScan Run

The validation-selected combined MS-DISA checkpoint from the voxel `0.1`
fine-tune is:

```text
work_dirs/treescan_finetune_voxel_0_1_msdisa/best_mRQ_epoch_50_spconv_fix.pth
```

Repaired-checkpoint validation artifacts:

```text
work_dirs/treescan_finetune_voxel_0_1_msdisa_fullval_spconv_fix/
```

Validation metrics from that run:

```text
mIoU=0.7387, mPQ=0.7004, mSQ=0.8508, mRQ/F1=0.8232,
mPrecision=0.8286, mRecall=0.8179
```

Held-out test artifacts:

```text
work_dirs/treescan_finetune_voxel_0_1_msdisa_test_spconv_fix/
```

Held-out test metrics from that run:

```text
mIoU=0.7141, mPQ=0.6603, mSQ=0.8475, mRQ/F1=0.7791,
mPrecision=0.7759, mRecall=0.7824
```

These are measured results for this implementation and checkpoint only; they
should not be treated as a general performance claim without repeated runs.
