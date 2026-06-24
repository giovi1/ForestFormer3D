#!/usr/bin/env python3
"""Patch TreeScanPL10K info PKLs to use foreground tree instance labels.

The TreeScan conversion stores point semantics as 0=background and 1=tree.
Older info PKLs stored every tree bbox as class 0, which the inherited
ForAINetV2 config treats as stuff/background. This script remaps those box
labels to class 1 so decoder fine-tuning receives foreground instance targets.
"""

import argparse
import pickle
import shutil
from pathlib import Path


def patch_file(path: Path, backup: bool) -> None:
    with path.open("rb") as f:
        data = pickle.load(f)

    before: dict[int, int] = {}
    after: dict[int, int] = {}
    changed = 0

    for item in data.get("data_list", []):
        for inst in item.get("instances", []):
            label = inst.get("bbox_label_3d")
            before[label] = before.get(label, 0) + 1
            if label == 0:
                inst["bbox_label_3d"] = 1
                changed += 1
            new_label = inst.get("bbox_label_3d")
            after[new_label] = after.get(new_label, 0) + 1

    metainfo = data.setdefault("metainfo", {})
    metainfo["categories"] = {"ground": 0, "tree": 1}
    metainfo["dataset"] = "treescanpl10k"

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"{path}: changed={changed}, before={before}, after={after}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap TreeScan info PKL bbox labels from 0 to 1.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/TreeScanPL10K"),
        help="TreeScanPL10K converted dataset root.")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create .bak files before overwriting PKLs.")
    args = parser.parse_args()

    files = sorted(args.data_root.glob("forainetv2_oneformer3d_infos_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No info PKLs found under {args.data_root}")

    for path in files:
        patch_file(path, args.backup)


if __name__ == "__main__":
    main()
