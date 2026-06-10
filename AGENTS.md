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
