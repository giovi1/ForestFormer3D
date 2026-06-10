#!/usr/bin/env python3
"""
Instance-segmentation evaluation for ForestFormer3D predictions against
ground-truth tree instances stored in the original .laz/.las files.

Metric definitions follow the ForestFormer3D paper (arXiv 2506.16991) and the
repo's tools/final_eval.py, specialised to the single "tree" instance class:

  * A predicted instance is a True Positive (TP) if its best IoU with any GT
    tree is >= the IoU threshold (default 0.5), otherwise a False Positive (FP).
  * Precision = TP / (#predicted instances)
  * Recall    = TP / (#GT instances)
  * F1        = 2 * P * R / (P + R)
  * Coverage (MUCov) = mean over GT trees of (best IoU with any prediction)
  * Weighted coverage (MWCov) = the same, weighted by GT tree point count

The FF3D output .ply carries `instance_pred` (>= 0 per predicted tree, -1 for
background) per point, in the SAME point order as the input .laz. The GT comes
from the `treeID` field of the .laz (0 = non-tree). Points are aligned by index
after verifying the prediction/GT clouds differ only by a constant translation;
if that check fails the script falls back to nearest-neighbour matching.

Usage:
  python tools/eval_instance_seg.py \
      --pred work_dirs/.../Rem_Gorlice_2015_0101703.ply \
      --laz  data-test/Rem_Gorlice_2015_0101703.laz \
      --out  results/zero_shot_metrics.json

  # batch: match every *.ply in a dir to the .laz of the same stem
  python tools/eval_instance_seg.py --pred work_dirs/<run>/ --laz data-test/ \
      --out results/zero_shot_metrics.json

Run inside an env with laspy, pandas, numpy, scipy (e.g. the TreeLearn conda env).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# .ply reading (ascii + binary), returns only the requested vertex properties
# ---------------------------------------------------------------------------
_PLY_TO_NP = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _parse_ply_header(path):
    """Return (fmt, prop_names, prop_np_types, n_vertices, header_nbytes,
    header_nlines)."""
    fmt = None
    props = []          # list of (name, np_type)
    n_vertices = None
    in_vertex = False
    header_nlines = 0
    with open(path, "rb") as f:
        while True:
            raw = f.readline()
            header_nlines += 1
            line = raw.decode("ascii", "replace").strip()
            toks = line.split()
            if not toks:
                pass
            elif toks[0] == "format":
                fmt = toks[1]
            elif toks[0] == "element":
                in_vertex = toks[1] == "vertex"
                if in_vertex:
                    n_vertices = int(toks[2])
            elif toks[0] == "property" and in_vertex:
                # property <type> <name>  (list properties not supported here)
                if toks[1] == "list":
                    raise ValueError(f"list property unsupported: {line}")
                props.append((toks[2], _PLY_TO_NP[toks[1]]))
            elif toks[0] == "end_header":
                header_nbytes = f.tell()
                break
    if fmt is None or n_vertices is None:
        raise ValueError(f"Malformed PLY header in {path}")
    names = [p[0] for p in props]
    types = [p[1] for p in props]
    return fmt, names, types, n_vertices, header_nbytes, header_nlines


def read_ply(path, want):
    """Read selected vertex properties from a PLY file.

    Returns a dict {name: np.ndarray}. Supports 'ascii' and
    'binary_little_endian' / 'binary_big_endian'.
    """
    fmt, names, types, n, hbytes, hlines = _parse_ply_header(path)
    missing = [w for w in want if w not in names]
    if missing:
        raise ValueError(f"{path}: missing PLY properties {missing}; have {names}")

    if fmt == "ascii":
        import pandas as pd
        dtype = {nm: np.dtype(tp) for nm, tp in zip(names, types) if nm in want}
        df = pd.read_csv(
            path, sep=r"\s+", header=None, names=names, usecols=list(want),
            skiprows=hlines, dtype=dtype, engine="c",
        )
        return {w: df[w].to_numpy() for w in want}

    byteorder = "<" if fmt == "binary_little_endian" else ">"
    rec = np.dtype([(nm, byteorder + tp) for nm, tp in zip(names, types)])
    arr = np.fromfile(path, dtype=rec, count=n, offset=hbytes)
    return {w: np.asarray(arr[w]) for w in want}


# ---------------------------------------------------------------------------
# .laz / .las ground-truth reading
# ---------------------------------------------------------------------------
def read_laz(path, gt_field):
    import laspy
    las = laspy.read(str(path))
    dims = list(las.point_format.dimension_names)
    if gt_field not in dims:
        raise ValueError(f"{path}: field '{gt_field}' not found; have {dims}")
    xyz = np.column_stack(
        [np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]
    ).astype(np.float64)
    gt = np.asarray(getattr(las, gt_field)).astype(np.int64)
    return xyz, gt


# ---------------------------------------------------------------------------
# Align GT (laz) to predictions (ply) by point
# ---------------------------------------------------------------------------
def align_gt_to_pred(pred_xyz, laz_xyz, laz_gt, tol=0.05, n_sample=100000,
                     seed=0):
    """Return (gt_aligned, info). Prefer direct index alignment after checking
    the two clouds differ only by a constant translation; otherwise fall back to
    nearest-neighbour matching in the translated frame."""
    info = {}
    if len(pred_xyz) == len(laz_xyz):
        rng = np.random.default_rng(seed)
        k = min(n_sample, len(pred_xyz))
        idx = rng.choice(len(pred_xyz), size=k, replace=False)
        diff = pred_xyz[idx].astype(np.float64) - laz_xyz[idx]
        offset = np.median(diff, axis=0)
        resid = np.abs(diff - offset).max()
        info.update(method="index", offset=offset.tolist(),
                    max_residual=float(resid), n_points=int(len(pred_xyz)))
        if resid <= tol:
            return laz_gt, info
        info["index_align_failed"] = True  # fall through to NN

    # Fallback: nearest neighbour in a translation-corrected frame.
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    k = min(n_sample, len(pred_xyz), len(laz_xyz))
    pi = rng.choice(len(pred_xyz), size=k, replace=False)
    # rough offset = difference of centroids
    offset = pred_xyz.mean(0) - laz_xyz.mean(0)
    tree = cKDTree(laz_xyz + offset)
    dist, nn = tree.query(pred_xyz, k=1, workers=-1)
    info.update(method="nearest_neighbour", offset=offset.tolist(),
                nn_max_dist=float(dist.max()), nn_mean_dist=float(dist.mean()),
                n_points=int(len(pred_xyz)))
    return laz_gt[nn], info


# ---------------------------------------------------------------------------
# Metric computation (FF3D / final_eval definitions, single tree class)
# ---------------------------------------------------------------------------
def compute_metrics(pred_inst, gt_inst, iou_thr=0.5):
    """pred_inst: per-point predicted instance id (-1 = background/non-tree).
    gt_inst: per-point GT instance id (0 = non-tree).
    Computes IoU between every GT tree and every predicted tree via a contingency
    table, then the FF3D instance metrics."""
    # Compact, contiguous ids for GT trees (id > 0) and predicted trees (id >= 0)
    gt_labels = np.unique(gt_inst[gt_inst > 0])
    pred_labels = np.unique(pred_inst[pred_inst >= 0])
    n_gt, n_pred = len(gt_labels), len(pred_labels)

    if n_gt == 0:
        return dict(precision=0.0, recall=0.0, f1=0.0, coverage=0.0,
                    weighted_coverage=0.0, mean_iou_matched=0.0,
                    n_gt=0, n_pred=int(n_pred), tp=0, fp=int(n_pred))

    gt_remap = {l: i for i, l in enumerate(gt_labels)}
    pred_remap = {l: i for i, l in enumerate(pred_labels)}
    g_idx = np.full(len(gt_inst), -1, dtype=np.int64)
    for l, i in gt_remap.items():
        g_idx[gt_inst == l] = i
    p_idx = np.full(len(pred_inst), -1, dtype=np.int64)
    for l, i in pred_remap.items():
        p_idx[pred_inst == l] = i

    gt_size = np.bincount(g_idx[g_idx >= 0], minlength=n_gt).astype(np.float64)
    pred_size = np.bincount(p_idx[p_idx >= 0], minlength=n_pred).astype(np.float64)

    # Intersection counts only over points assigned to both a GT and a pred tree.
    both = (g_idx >= 0) & (p_idx >= 0)
    inter = np.zeros((n_gt, n_pred), dtype=np.float64)
    if both.any() and n_pred > 0:
        flat = np.bincount(g_idx[both] * n_pred + p_idx[both],
                           minlength=n_gt * n_pred)
        inter = flat.reshape(n_gt, n_pred).astype(np.float64)

    if n_pred > 0:
        union = gt_size[:, None] + pred_size[None, :] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter),
                        where=union > 0)
    else:
        iou = np.zeros((n_gt, 0))

    # Coverage: best IoU per GT tree.
    best_iou_gt = iou.max(axis=1) if n_pred > 0 else np.zeros(n_gt)
    coverage = float(best_iou_gt.mean())
    weighted_coverage = float(np.average(best_iou_gt, weights=gt_size))

    # TP/FP: per predicted tree, best IoU over GT trees (matches final_eval).
    if n_pred > 0:
        best_iou_pred = iou.max(axis=0)
        is_tp = best_iou_pred >= iou_thr
        tp = int(is_tp.sum())
        fp = int(n_pred - tp)
        mean_iou_matched = float(best_iou_pred[is_tp].mean()) if tp else 0.0
    else:
        tp, fp, mean_iou_matched = 0, 0, 0.0

    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    return dict(precision=precision, recall=recall, f1=f1, coverage=coverage,
                weighted_coverage=weighted_coverage,
                mean_iou_matched=mean_iou_matched,
                n_gt=int(n_gt), n_pred=int(n_pred), tp=tp, fp=fp)


# ---------------------------------------------------------------------------
# Pairing prediction files with GT files
# ---------------------------------------------------------------------------
def collect_pairs(pred_arg, laz_arg):
    pred_path, laz_path = Path(pred_arg), Path(laz_arg)
    preds = sorted(pred_path.glob("*.ply")) if pred_path.is_dir() else [pred_path]
    if laz_path.is_dir():
        laz_map = {}
        for ext in ("*.laz", "*.las"):
            for p in laz_path.glob(ext):
                laz_map[p.stem] = p
        pairs = []
        for pp in preds:
            gt = laz_map.get(pp.stem)
            if gt is None:
                print(f"WARNING: no .laz/.las for '{pp.stem}' — skipping.")
                continue
            pairs.append((pp.stem, pp, gt))
        return pairs
    # single laz file
    if len(preds) == 1:
        return [(preds[0].stem, preds[0], laz_path)]
    # match by stem against the single laz only if it matches
    return [(pp.stem, pp, laz_path) for pp in preds if pp.stem == laz_path.stem]


def fmt_row(name, m, width=42):
    return (f"{name[:width]:<{width}} {m['precision']:>9.4f} {m['recall']:>9.4f} "
            f"{m['f1']:>9.4f} {m['coverage']:>9.4f} {m['n_gt']:>6d} "
            f"{m['n_pred']:>7d} {m['tp']:>5d} {m['fp']:>5d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True,
                    help="FF3D prediction .ply file or directory of .ply files")
    ap.add_argument("--laz", required=True,
                    help="Ground-truth .laz/.las file or directory")
    ap.add_argument("--gt-field", default="treeID",
                    help="LAZ field holding GT instance ids (default: treeID)")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="IoU threshold for a TP (default: 0.5)")
    ap.add_argument("--out", default=None, help="Path to write results JSON")
    ap.add_argument("--align-tol", type=float, default=0.05,
                    help="Max per-axis residual (m) to accept index alignment")
    args = ap.parse_args()

    pairs = collect_pairs(args.pred, args.laz)
    if not pairs:
        sys.exit("No (prediction, ground-truth) pairs found.")

    header = (f"{'plot':<42} {'Prec':>9} {'Recall':>9} {'F1':>9} "
              f"{'Cover':>9} {'#GT':>6} {'#Pred':>7} {'TP':>5} {'FP':>5}")
    print(f"\nIoU threshold = {args.iou}   GT field = {args.gt_field}\n")
    print(header)
    print("-" * len(header))

    per_plot = {}
    tot_tp = tot_fp = tot_gt = tot_pred = 0
    cover_list, wcover_list = [], []

    for name, pp, gt in pairs:
        ply = read_ply(pp, want=("x", "y", "z", "instance_pred"))
        pred_xyz = np.column_stack([ply["x"], ply["y"], ply["z"]]).astype(np.float64)
        pred_inst = ply["instance_pred"].astype(np.int64)

        laz_xyz, laz_gt = read_laz(gt, args.gt_field)
        gt_inst, align_info = align_gt_to_pred(
            pred_xyz, laz_xyz, laz_gt, tol=args.align_tol)

        m = compute_metrics(pred_inst, gt_inst, iou_thr=args.iou)
        m["alignment"] = align_info
        per_plot[name] = m
        print(fmt_row(name, m))

        tot_tp += m["tp"]; tot_fp += m["fp"]
        tot_gt += m["n_gt"]; tot_pred += m["n_pred"]
        cover_list.append(m["coverage"]); wcover_list.append(m["weighted_coverage"])

    # Aggregated: pooled TP/FP/GT for P/R/F1 (micro); mean coverage over plots.
    agg_p = tot_tp / tot_pred if tot_pred else 0.0
    agg_r = tot_tp / tot_gt if tot_gt else 0.0
    agg_f1 = (2 * agg_p * agg_r / (agg_p + agg_r)) if (agg_p + agg_r) else 0.0
    agg = dict(precision=agg_p, recall=agg_r, f1=agg_f1,
               coverage=float(np.mean(cover_list)),
               weighted_coverage=float(np.mean(wcover_list)),
               n_gt=tot_gt, n_pred=tot_pred, tp=tot_tp, fp=tot_fp,
               n_plots=len(per_plot))

    print("-" * len(header))
    print(fmt_row("AGGREGATED (pooled TP/FP/GT)", agg))
    print(f"\nAggregated over {agg['n_plots']} plot(s):")
    print(f"  Precision : {agg['precision']:.4f}")
    print(f"  Recall    : {agg['recall']:.4f}")
    print(f"  F1        : {agg['f1']:.4f}")
    print(f"  Coverage  : {agg['coverage']:.4f}  (mean over plots; "
          f"weighted {agg['weighted_coverage']:.4f})")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"config": {"iou_threshold": args.iou,
                                  "gt_field": args.gt_field},
                       "per_plot": per_plot,
                       "aggregated": agg}, f, indent=2)
        print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
