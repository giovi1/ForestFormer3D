# Fix: Double spconv weight permutation in `tools/test.py`

## Symptom

Running inference with the pre-trained checkpoint `epoch_3000_fix.pth` produced:

- All instance predictions = `-1` (zero instances detected across all test plots)
- Semantic accuracy statistically indistinguishable from random:
  - Per-class recall ≈ per-class prediction rate for every class
  - Example on NIBIO plot 1: ground recall 9.5% ≈ ground prediction rate 7.8%
- Overall mIoU ≈ 0.27, driven entirely by the leaf class being the dominant class (87% of points)

No runtime error was raised. The model ran to completion and produced output files.

## Root cause

`tools/test.py` had been modified to apply the spconv weight permutation in-memory before loading the model:

```python
for layer in list(checkpoint_to_fix.keys()):
    if (layer.startswith('unet') or layer.startswith('input_conv')) \
        and layer.endswith('weight') \
        and len(checkpoint_to_fix[layer].shape) == 5:
        checkpoint_to_fix[layer] = checkpoint_to_fix[layer].permute(1, 2, 3, 4, 0)
```

This is the same transformation performed by `tools/fix_spconv_checkpoint.py`. The official pre-trained checkpoint `epoch_3000_fix.pth` is the output of that script — the `_fix` suffix indicates the permutation has already been applied.

Applying `permute(1, 2, 3, 4, 0)` twice to a 5D tensor does **not** produce the original tensor. It is a cyclic dimension shift and results in a corrupted layout:

```
Original shape:          (A, B, C, D, E)
After fix_spconv (once): (B, C, D, E, A)  ← correct for spconv
After test.py (twice):   (C, D, E, A, B)  ← corrupted
```

The convolutional filters in the backbone (`SpConvUNet`) and `input_conv` were therefore completely scrambled. The model still executed without errors because the tensor shapes remained valid, but the learned weights were meaningless.

## Fix

Removed the in-memory permutation block from `tools/test.py`. The checkpoint path is now passed directly to the mmengine runner via `cfg.load_from`:

```python
# Before
print('Applying spconv checkpoint fix in-memory...')
checkpoint = torch.load(args.checkpoint, map_location='cpu')
...
for layer in list(checkpoint_to_fix.keys()):
    if ...:
        checkpoint_to_fix[layer] = checkpoint_to_fix[layer].permute(1, 2, 3, 4, 0)
import tempfile
with tempfile.NamedTemporaryFile(...) as tmp:
    torch.save(checkpoint, tmp.name)
    cfg.load_from = tmp.name

# After
cfg.load_from = args.checkpoint
```

## Correct workflow

| Checkpoint source | Needs `fix_spconv_checkpoint.py`? | Use directly with `tools/test.py`? |
|---|---|---|
| Official `epoch_3000_fix.pth` | No (already fixed) | Yes |
| Self-trained `trained.pth` | Yes → produces `trained_fix.pth` | Use the `_fix.pth` output |

Any checkpoint whose filename ends in `_fix.pth` should be passed to `tools/test.py` as-is. Do **not** run `fix_spconv_checkpoint.py` on a checkpoint that already carries the `_fix` suffix.
