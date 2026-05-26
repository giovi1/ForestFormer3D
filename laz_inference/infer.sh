#!/usr/bin/env bash
# Run the full ForestFormer3D inference pipeline on .laz / .las files.
#
# Usage (from /workspace inside the container):
#   bash laz_inference/infer.sh cloud.laz
#   bash laz_inference/infer.sh cloud1.laz cloud2.laz
#   bash laz_inference/infer.sh --input-dir /path/to/laz_dir/
#
# Options:
#   --gpu N        CUDA device index to use (default: 0)
#   --force        Delete existing intermediate .npy files and reprocess
#   --input-dir D  Convert all .laz/.las files found in directory D
#   --config F     Path to the config file
#                  (default: configs/oneformer3d_qs_radius16_qp300_2many.py)
#   --model F      Path to the checkpoint file
#                  (default: work_dirs/clean_forestformer/epoch_3000_fix.pth)

set -euo pipefail

WORK_DIR="/workspace"
CONFIG="$WORK_DIR/configs/oneformer3d_qs_radius16_qp300_2many.py"
MODEL="$WORK_DIR/work_dirs/clean_forestformer/epoch_3000_fix.pth"
DATA_DIR="$WORK_DIR/data-test"
TEST_LIST="$DATA_DIR/meta_data/test_list.txt"
GPU=0
FORCE=0
CONVERT_ARGS=()

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)    GPU="$2";    shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --model)  MODEL="$2";  shift 2 ;;
        --force)  FORCE=1;     shift   ;;
        *)        CONVERT_ARGS+=("$1"); shift ;;
    esac
done

if [ ${#CONVERT_ARGS[@]} -eq 0 ]; then
    echo "Usage: bash laz_inference/infer.sh FILE.laz [FILE2.laz ...] [--gpu N] [--force]"
    echo "       bash laz_inference/infer.sh --input-dir DIR/ [--gpu N] [--force]"
    exit 1
fi

cd "$WORK_DIR"

# ── 1. Convert .laz → .ply and write test_list.txt ───────────────────────────
echo "=== [1/4] Converting .laz files ==="
python laz_inference/convert.py "${CONVERT_ARGS[@]}" \
    --out-dir  "$DATA_DIR/test_data" \
    --test-list "$TEST_LIST"

# ── 2. Optionally clear stale intermediate files ──────────────────────────────
if [ "$FORCE" -eq 1 ]; then
    echo ""
    echo "=== [--force] Removing existing intermediate files ==="
    while IFS= read -r scan || [ -n "$scan" ]; do
        [ -z "$scan" ] && continue
        target="$DATA_DIR/forainetv2_instance_data/${scan}_vert.npy"
        if [ -f "$target" ]; then
            rm -f "$DATA_DIR/forainetv2_instance_data/${scan}"_*.npy
            echo "  Cleared: $scan"
        fi
    done < "$TEST_LIST"
fi

# ── 3. Preprocess point clouds ────────────────────────────────────────────────
echo ""
echo "=== [2/4] Preprocessing point clouds ==="
cd "$DATA_DIR"
python "$WORK_DIR/data/ForAINetV2/batch_load_ForAINetV2_data.py" \
    --test_scan_names_file  meta_data/test_list.txt \
    --train_scan_names_file /dev/null \
    --val_scan_names_file   /dev/null
cd "$WORK_DIR"

# ── 4. Build .pkl info files ──────────────────────────────────────────────────
echo ""
echo "=== [3/4] Building .pkl info files ==="
# create_data_forainetv2.py requires train/val list files to exist even when empty
touch "$DATA_DIR/meta_data/train_list.txt"
touch "$DATA_DIR/meta_data/val_list.txt"
python tools/create_data_forainetv2.py forainetv2 \
    --root-path "$DATA_DIR" \
    --out-dir   "$DATA_DIR"

# ── 5. Run inference ──────────────────────────────────────────────────────────
echo ""
echo "=== [4/4] Running inference (GPU $GPU) ==="
CUDA_VISIBLE_DEVICES="$GPU" python tools/test.py "$CONFIG" "$MODEL" \
    --cfg-options test_dataloader.dataset.data_root="$DATA_DIR/"

echo ""
echo "Done. Results are in work_dirs/ under the latest timestamped folder."
