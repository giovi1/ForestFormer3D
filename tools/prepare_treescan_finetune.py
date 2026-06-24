#!/usr/bin/env python3
"""Prepare TreeScanPL10K data for ForestFormer3D fine-tuning.

This creates the directory layout consumed by the existing ForAINetV2 data
loader:

    data/TreeScanPL10K/
      train_val_data/*.ply
      test_data/*.ply
      meta_data/train_list.txt
      meta_data/val_list.txt
      meta_data/test_list.txt

Run the OpenMMLab metadata steps after this script finishes; the exact commands
are printed at the end.
"""
import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from laz_inference.convert import convert_one  # noqa: E402


def find_clouds(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.rglob("*.laz")) + sorted(input_dir.rglob("*.las"))
    if not files:
        raise FileNotFoundError(f"No .laz/.las files found in {input_dir}")
    return files


def get_site(path: Path) -> str:
    parts = path.stem.split("_")
    return parts[1] if len(parts) > 1 else "unknown"


def split_files(
    files: list[Path],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    rng = random.Random(seed)
    train_files: list[Path] = []
    val_files: list[Path] = []
    test_files: list[Path] = []

    by_site: dict[str, list[Path]] = {}
    for path in files:
        by_site.setdefault(get_site(path), []).append(path)

    for site_files in by_site.values():
        site_files = site_files.copy()
        rng.shuffle(site_files)
        n_total = len(site_files)
        n_test = (
            max(1, round(n_total * test_ratio))
            if test_ratio > 0 and n_total >= 3 else 0
        )
        n_val = (
            max(1, round(n_total * val_ratio))
            if val_ratio > 0 and n_total - n_test >= 2 else 0
        )

        test_files.extend(site_files[:n_test])
        val_files.extend(site_files[n_test:n_test + n_val])
        train_files.extend(site_files[n_test + n_val:])

    train_files.sort()
    val_files.sort()
    test_files.sort()
    if not train_files:
        raise ValueError("Split produced no training files; provide more plots.")
    return train_files, val_files, test_files


def write_list(path: Path, scan_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(scan_names) + ("\n" if scan_names else ""))


def convert_many(files: list[Path], out_dir: Path,
                 voxel_size: float | None) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [convert_one(path, out_dir, voxel_size) for path in files]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert and split TreeScanPL10K LAZ/LAS files for fine-tuning.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Single directory to split into train/val/test.")
    parser.add_argument(
        "--train-val-dir",
        type=Path,
        help="Directory to split into train/val. Use with --test-dir.")
    parser.add_argument(
        "--test-dir",
        type=Path,
        help="Held-out test directory. Use with --train-val-dir.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/TreeScanPL10K"),
        help="Output dataset root.")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.1,
        help="Voxel size in metres for conversion. Use 0 to disable downsampling.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.input_dir and (args.train_val_dir or args.test_dir):
        parser.error("Use either --input-dir or --train-val-dir/--test-dir, not both.")
    if not args.input_dir and not (args.train_val_dir and args.test_dir):
        parser.error("Provide --input-dir or both --train-val-dir and --test-dir.")

    voxel_size = args.voxel_size if args.voxel_size > 0 else None
    train_val_out = args.data_root / "train_val_data"
    test_out = args.data_root / "test_data"
    meta_dir = args.data_root / "meta_data"

    if args.input_dir:
        all_files = find_clouds(args.input_dir)
        train_files, val_files, test_files = split_files(
            all_files, args.val_ratio, args.test_ratio, args.seed)
    else:
        train_val_files = find_clouds(args.train_val_dir)
        train_files, val_files, _ = split_files(
            train_val_files, args.val_ratio, 0.0, args.seed)
        test_files = find_clouds(args.test_dir)

    train_names = convert_many(train_files, train_val_out, voxel_size)
    val_names = convert_many(val_files, train_val_out, voxel_size)
    test_names = convert_many(test_files, test_out, voxel_size)

    write_list(meta_dir / "train_list.txt", train_names)
    write_list(meta_dir / "val_list.txt", val_names)
    write_list(meta_dir / "test_list.txt", test_names)

    print("\nPrepared TreeScanPL10K fine-tuning split:")
    print(f"  train: {len(train_names)}")
    print(f"  val:   {len(val_names)}")
    print(f"  test:  {len(test_names)}")
    print(f"  root:  {args.data_root}")
    print("\nNext metadata commands:")
    print(f"  cd {args.data_root}")
    print("  python /workspace/data/ForAINetV2/batch_load_ForAINetV2_data.py \\")
    print("    --train_scan_names_file meta_data/train_list.txt \\")
    print("    --val_scan_names_file meta_data/val_list.txt \\")
    print("    --test_scan_names_file meta_data/test_list.txt \\")
    print("    --train_forainetv2_dir train_val_data \\")
    print("    --test_forainetv2_dir test_data \\")
    print("    --output_folder forainetv2_instance_data")
    print("  cd /workspace")
    print("  python tools/create_data_forainetv2.py forainetv2 \\")
    print(f"    --root-path {args.data_root} --out-dir {args.data_root}")


if __name__ == "__main__":
    main()
