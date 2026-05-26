#!/usr/bin/env python3
"""
Convert .laz / .las point cloud files to the binary .ply format expected
by the ForestFormer3D pipeline, and update data/ForAINetV2/meta_data/test_list.txt.

Usage (from /workspace inside the container):
    python laz_inference/convert.py cloud.laz
    python laz_inference/convert.py cloud1.laz cloud2.laz cloud3.laz
    python laz_inference/convert.py --input-dir /path/to/laz_dir/
    python laz_inference/convert.py cloud.laz --out-dir /custom/out --test-list /custom/list.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data" / "ForAINetV2"))

from plyutils import write_ply  # noqa: E402


def convert_one(laz_path: Path, out_dir: Path) -> str:
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
    n = len(x)

    xyz = np.column_stack([x, y, z])
    # Dummy labels: semantic_seg=1 (tree class), treeID=0 (no ground-truth instance).
    # These are only used to satisfy the pipeline's file format; the model
    # predicts instance segmentation from raw XYZ coordinates during inference.
    semantic_seg = np.ones(n, dtype=np.int32).reshape(-1, 1)
    tree_id = np.zeros(n, dtype=np.int32).reshape(-1, 1)

    write_ply(str(out_ply), [xyz, semantic_seg, tree_id],
              ["x", "y", "z", "semantic_seg", "treeID"])
    print(f"  Saved {n:,} points.")
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
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        scan_names.append(convert_one(f, out_dir))

    test_list_path = Path(args.test_list)
    test_list_path.parent.mkdir(parents=True, exist_ok=True)
    test_list_path.write_text("\n".join(scan_names) + "\n")

    print(f"\ntest_list.txt written → {test_list_path}")
    print("Scan names:")
    for name in scan_names:
        print(f"  {name}")


if __name__ == "__main__":
    main()
