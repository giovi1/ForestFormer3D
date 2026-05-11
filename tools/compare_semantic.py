"""
Reads a result PLY (with semantic_pred and semantic_gt fields) and writes a new
PLY adding two extra scalar fields:
  - semantic_error : 0 = correct, 1 = wrong prediction
  - semantic_diff  : semantic_pred - semantic_gt (signed difference, useful to
                     see which class was confused with which)

Usage:
    python tools/compare_semantic.py <input.ply> [output.ply]

If output.ply is omitted, the file is saved as <input>_compared.ply.

Semantic class mapping:  0=ground  1=wood  2=leaf
"""
import sys
import os
import numpy as np


def parse_ply_header(f):
    """Return (num_vertices, properties, header_byte_length)."""
    properties = []
    num_vertices = 0
    header_lines = []
    while True:
        line = f.readline().decode("ascii").strip()
        header_lines.append(line)
        if line.startswith("element vertex"):
            num_vertices = int(line.split()[-1])
        elif line.startswith("property"):
            parts = line.split()
            properties.append((parts[1], parts[2]))  # (type, name)
        elif line == "end_header":
            break
    header_bytes = b"\n".join(l.encode("ascii") for l in header_lines) + b"\n"
    return num_vertices, properties, header_bytes


def dtype_for(type_str):
    mapping = {
        "float": np.float32, "double": np.float64,
        "int": np.int32, "uint": np.uint32,
        "short": np.int16, "ushort": np.uint16,
        "char": np.int8, "uchar": np.uint8,
    }
    return mapping[type_str]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else \
        in_path.replace(".ply", "_compared.ply")

    print(f"Reading {in_path} ...")
    with open(in_path, "rb") as f:
        num_verts, props, _ = parse_ply_header(f)
        # Read the rest as text (ascii PLY)
        data_lines = f.read().decode("ascii").strip().split("\n")

    prop_names = [p[1] for p in props]
    prop_types = [dtype_for(p[0]) for p in props]

    print(f"  {num_verts} points, fields: {prop_names}")

    # Parse into structured array
    rows = np.array([list(map(float, l.split())) for l in data_lines],
                    dtype=np.float64)

    col = {name: i for i, name in enumerate(prop_names)}

    sem_pred = rows[:, col["semantic_pred"]].astype(np.int32)
    sem_gt   = rows[:, col["semantic_gt"]].astype(np.int32)

    error = (sem_pred != sem_gt).astype(np.int32)   # 0=correct, 1=wrong
    diff  = (sem_pred - sem_gt).astype(np.int32)    # signed difference

    # Build output header
    new_header_lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_verts}",
    ]
    for ptype, pname in props:
        new_header_lines.append(f"property {ptype} {pname}")
    new_header_lines += [
        "property int semantic_error",
        "property int semantic_diff",
        "end_header",
    ]
    header_str = "\n".join(new_header_lines) + "\n"

    print(f"Writing {out_path} ...")
    class_names = {0: "ground", 1: "wood", 2: "leaf"}
    with open(out_path, "w") as f:
        f.write(header_str)
        for i, row in enumerate(rows):
            vals = " ".join(
                str(int(v)) if prop_types[j] in (np.int32, np.uint32) else
                f"{v:.10g}"
                for j, v in enumerate(row)
            )
            f.write(f"{vals} {error[i]} {diff[i]}\n")

    # Print confusion matrix
    classes = sorted(set(sem_gt.tolist()) | set(sem_pred.tolist()))
    print("\n--- Per-class accuracy ---")
    for c in classes:
        mask = sem_gt == c
        if mask.sum() == 0:
            continue
        acc = (sem_pred[mask] == sem_gt[mask]).mean() * 100
        print(f"  {class_names.get(c, c):8s} (gt={c}): {acc:.1f}% correct  "
              f"({mask.sum()} points)")

    total_acc = (error == 0).mean() * 100
    print(f"\nOverall accuracy: {total_acc:.2f}%")
    wrong = int(error.sum())
    print(f"Wrong points: {wrong} / {num_verts}  ({100*wrong/num_verts:.2f}%)")

    print(f"\nDone. Open {out_path} in CloudCompare and use scalar field "
          "'semantic_error' (0=correct, 1=wrong) or 'semantic_diff' to inspect errors.")


if __name__ == "__main__":
    main()
