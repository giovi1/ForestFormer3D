# TreeScanPL10K Voxel 0.1 Fine-Tuning Baseline

Baseline date: 2026-06-17

## Summary

This is the accepted TreeScanPL10K instance segmentation baseline after repairing
the fine-tuned checkpoint for the current `spconv` weight layout.

## Checkpoint

Use:

```bash
work_dirs/treescan_finetune_voxel_0_1_decoder/best_mRQ_epoch_45_spconv_fix.pth
```

Archived copy:

```bash
results/treescan_finetune_voxel_0_1_baseline_spconv_fix/best_mRQ_epoch_45_spconv_fix.pth
```

The unrepaired `best_mRQ_epoch_45.pth` loads with 5D `spconv` weight shape
mismatches in the current container and must not be used for final metrics.

## Metrics

Full-scene validation:

```text
mIoU: 0.7226
mPQ: 0.6705
mSQ: 0.8315
mRQ/F1: 0.8064
mPrecision: 0.7958
mRecall: 0.8173
mMUCov: 0.7038
mMWCov: 0.7580
```

Held-out test:

```text
mIoU: 0.7011
mPQ: 0.6351
mSQ: 0.8297
mRQ/F1: 0.7654
mPrecision: 0.7485
mRecall: 0.7831
mMUCov: 0.7272
mMWCov: 0.7691
```

## Archived Artifacts

```text
oneformer3d_treescan_finetune.py
best_mRQ_epoch_45_spconv_fix.pth
training_run_20260617_084113/
treescan_finetune_voxel_0_1_decoder_fullval_spconv_fix/
treescan_finetune_voxel_0_1_decoder_test_spconv_fix/
```

The validation and held-out folders include their logs, JSON metrics, saved
config, and predicted `.ply` files.
