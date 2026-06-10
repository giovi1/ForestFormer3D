#!/usr/bin/env python3
"""
Convert .laz / .las point cloud files to the binary .ply format expected
by the ForestFormer3D pipeline, and update data/ForAINetV2/meta_data/test_list.txt.

Usage (from /workspace inside the container):
    python laz_inference/convert.py cloud.laz
    python laz_inference/convert.py cloud1.laz cloud2.laz cloud3.laz
    python laz_inference/convert.py --input-dir /path/to/laz_dir/
    python laz_inference/convert.py cloud.laz --out-dir /custom/out --test-list /custom/list.txt
    python laz_inference/convert.py cloud.laz --voxel-size 0.1   # subsample at 0.1m voxels
    python laz_inference/convert.py cloud.laz --voxel-size 0     # disable subsampling
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data" / "ForAINetV2"))

from plyutils import write_ply  # noqa: E402


def voxel_downsample(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """Return sorted indices keeping one point per voxel (first encountered)."""
    vox = np.floor(xyz / voxel_size).astype(np.int64)
    _, first_idx = np.unique(vox, axis=0, return_index=True)
    return np.sort(first_idx)


def convert_one(laz_path: Path, out_dir: Path, voxel_size: float | None) -> str:
    try:
        import laspy
    except ImportError:
        sys.exit(
            "laspy is not installed.\n"
            "Run inside the container:  pip install laspy 'laspy[lazrs]'"
        )

    scan_name = laz_path.stem
    out_ply = out_dir / f"{scan_name}.ply"

    print(f"  {laz_path}  →  {out_ply}")
    las = laspy.read(str(laz_path))

    x = np.asarray(las.x, dtype=np.float32)
    y = np.asarray(las.y, dtype=np.float32)
    z = np.asarray(las.z, dtype=np.float32)

    # Ground-truth instance labels from treeID (0 = non-tree).
    if hasattr(las, "treeID"):
        tree_id = np.asarray(las.treeID, dtype=np.int32)
    else:
        tree_id = np.zeros(len(x), dtype=np.int32)
        print("  WARNING: no treeID field — ground-truth labels unavailable")

    # Semantic class encoding expected by load_forainetv2_data.py:
    #   label_ids = semantic_seg - 1,  bg_sem = [0]
    # → semantic_seg=2 → label_ids=1 → tree (counted in evaluation)
    # → semantic_seg=1 → label_ids=0 → background (ignored)
    semantic_seg = np.where(tree_id > 0,
                            np.int32(2), np.int32(1)).astype(np.int32)

    xyz = np.column_stack([x, y, z])
    n_orig = len(x)

    if voxel_size is not None:
        idx = voxel_downsample(xyz, voxel_size)
        xyz        = xyz[idx]
        semantic_seg = semantic_seg[idx]
        tree_id    = tree_id[idx]
        print(f"  Subsampled {n_orig:,} → {len(idx):,} pts  (voxel_size={voxel_size} m)")
    else:
        print(f"  {n_orig:,} points (no subsampling)")

    write_ply(str(out_ply),
              [xyz, semantic_seg.reshape(-1, 1), tree_id.reshape(-1, 1)],
              ["x", "y", "z", "semantic_seg", "treeID"])
    print(f"  Saved {len(xyz):,} points → {out_ply}")
    return scan_name


def main():
    default_out = str(REPO_ROOT / "data" / "ForAINetV2" / "test_data")
    default_list = str(REPO_ROOT / "data" / "ForAINetV2" / "meta_data" / "test_list.txt")

    parser = argparse.ArgumentParser(
        description="Convert .laz/.las files to ForestFormer3D binary PLY format"
    )
    parser.add_argument(
        "files", nargs="*", metavar="FILE",
        help=".laz / .las file(s) to convert"
    )
    parser.add_argument(
        "--input-dir", metavar="DIR",
        help="Directory containing .laz / .las files (processes all of them)"
    )
    parser.add_argument(
        "--out-dir", metavar="DIR", default=default_out,
        help=f"Output directory for converted .ply files (default: {default_out})"
    )
    parser.add_argument(
        "--test-list", metavar="FILE", default=default_list,
        help=f"Path to write scan-name list (default: {default_list})"
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.1, metavar="M",
        help=(
            "Voxel grid size in metres for subsampling (default: 0.1). "
            "TLS plots are typically 50-100M pts; 0.1 m reduces to ~1-3M. "
            "Use 0 to disable subsampling."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    voxel_size = args.voxel_size if args.voxel_size > 0 else None

    input_files: list[Path] = [Path(f) for f in args.files]
    if args.input_dir:
        d = Path(args.input_dir)
        input_files += sorted(d.glob("*.laz")) + sorted(d.glob("*.las"))

    if not input_files:
        parser.error("No input files. Provide FILE arguments or --input-dir DIR.")

    print(f"Converting {len(input_files)} file(s) ...")
    scan_names = []
    for f in input_files:
        if not f.exists():
            print(f"WARNING: {f} not found — skipping.")
            continue
        scan_names.append(convert_one(f, out_dir, voxel_size))

    test_list_path = Path(args.test_list)
    test_list_path.parent.mkdir(parents=True, exist_ok=True)
    test_list_path.write_text("\n".join(scan_names) + "\n")

    print(f"\ntest_list.txt written → {test_list_path}")
    print("Scan names:")
    for name in scan_names:
        print(f"  {name}")


if __name__ == "__main__":
    main()
