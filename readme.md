# ForestFormer3D: A Unified Framework for End-to-End Segmentation of Forest LiDAR 3D Point Clouds

This is the official implementation of the paper:

**"ForestFormer3D: A Unified Framework for End-to-End Segmentation of Forest LiDAR 3D Point Clouds"**

(*Accepted as Oral at ICCV 2025 –  🏝️ Honolulu!* 🎉)

- 🌐 [Project page](https://bxiang233.github.io/FF3D/)
- 📄 [Paper on arXiv](https://www.arxiv.org/abs/2506.16991)
- 📦 [Dataset & pre-trained model on zenodo](https://zenodo.org/records/16742708)

---

## Changes in this fork

This fork extends the original ForestFormer3D codebase for TreeScanPL10K
experiments. The original model and ForAINetV2 pipeline are kept as the base,
while the added work focuses on making ForestFormer3D usable for LAZ/LAS forest
scans, comparing zero-shot inference against TreeScan fine-tuning, and
documenting the ablations needed to justify the final configuration.

### Codebase changes

- Fixed the `spconv` checkpoint workflow so `_fix.pth` checkpoints are not
  permuted twice during testing. Use `tools/fix_spconv_checkpoint.py` once when
  converting a trained checkpoint, then load the repaired checkpoint directly
  with `tools/test.py`.
- Fixed full-scene inference/evaluation so metrics receive full-cloud
  predictions instead of only the final processed cylinder.
- Made inference output less dependent on local hard-coded paths by adding
  configurable output directories and safer test-mode logic.
- Added LAZ/LAS inference support through `laz_inference/convert.py` and
  `laz_inference/infer.sh`. This path converts TreeScan-style `.laz`/`.las`
  files into the existing ForAINetV2-compatible format, regenerates metadata,
  runs ForestFormer3D, and writes predicted `.ply` files.
- Added quick inspection/evaluation helpers such as `tools/eval_instance_seg.py`
  and `tools/compare_semantic.py`.
- Developed TreeScanPL10K fine-tuning utilities on the
  `ft/treescan-finetuning` branch:
  `configs/oneformer3d_treescan_finetune.py`,
  `configs/oneformer3d_treescan_finetune_smoke.py`,
  `tools/prepare_treescan_finetune.py`,
  `tools/fix_treescan_info_labels.py`, and
  `tools/run_treescan_finetune_after_prepare.sh`.

### TreeScanPL10K inference

Run zero-shot TreeScan inference from inside the Docker container at
`/workspace`:

```bash
bash laz_inference/infer.sh \
  --input-dir /data/TreeScanPL10k/batch_01/ \
  --voxel-size 0.7 \
  --gpu 0 \
  --force
```

The script writes converted inputs under `data-test/`, updates the test
metadata, runs `tools/test.py`, and saves predictions in
`work_dirs/oneformer3d_qs_radius16_qp300_2many/`.

### TreeScanPL10K fine-tuning workflow

Run this workflow from the `ft/treescan-finetuning` branch, or after merging
the listed fine-tuning files into the branch you are using.

The TreeScan fine-tuning setup keeps the ForestFormer3D head compatible with
the original ForAINetV2 checkpoint. TreeScan labels are mapped as background
and tree instances; the unused ForAINetV2 class slot remains only to preserve
checkpoint compatibility.

Prepare the TreeScan training root:

```bash
python tools/prepare_treescan_finetune.py \
  --input-dir /data/TreeScanPL10k/batch_01 \
  --data-root data/TreeScanPL10K \
  --voxel-size 0.1
```

Generate metadata through the existing ForAINetV2-compatible loader:

```bash
cd data/TreeScanPL10K
python /workspace/data/ForAINetV2/batch_load_ForAINetV2_data.py \
  --train_scan_names_file meta_data/train_list.txt \
  --val_scan_names_file meta_data/val_list.txt \
  --test_scan_names_file meta_data/test_list.txt \
  --train_forainetv2_dir train_val_data \
  --test_forainetv2_dir test_data \
  --output_folder forainetv2_instance_data

cd /workspace
python tools/create_data_forainetv2.py forainetv2 \
  --root-path data/TreeScanPL10K \
  --out-dir data/TreeScanPL10K

python tools/fix_treescan_info_labels.py \
  --data-root data/TreeScanPL10K \
  --backup
```

Run a smoke test before the full fine-tune:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_treescan_finetune_smoke.py \
  --work-dir work_dirs/treescan_finetune_voxel_0_1_smoke
```

Then run the main fine-tune:

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_treescan_finetune.py \
  --work-dir work_dirs/treescan_finetune_voxel_0_1_decoder
```

For TreeScan, keep `model.prepare_epoch=-1`. The upstream-style
`prepare_epoch=1000` delays decoder losses beyond the short TreeScan fine-tune
and substantially reduces instance segmentation quality.

Repair the selected checkpoint before final evaluation:

```bash
python tools/fix_spconv_checkpoint.py \
  --in-path work_dirs/treescan_finetune_voxel_0_1_decoder/best_mRQ_epoch_45.pth \
  --out-path work_dirs/treescan_finetune_voxel_0_1_decoder/best_mRQ_epoch_45_spconv_fix.pth
```

### Ablation results

All TreeScan results below use the same TreeScan dataset split and evaluator so
the original ForestFormer3D checkpoint can be compared directly with the
fine-tuned model. A machine-readable summary is stored in
`results/treescan_ablation_summary.json`.

| Run | Checkpoint/settings | mIoU | mPQ | mRQ/F1 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Zero-shot ForestFormer3D | `epoch_3000_fix.pth` | 0.2552 | 0.3485 | 0.4730 | 0.3914 | 0.5976 |
| TreeScan fine-tuned baseline | `best_mRQ_epoch_45_spconv_fix.pth` | 0.7016 | 0.6386 | 0.7708 | 0.7595 | 0.7824 |
| Validation-selected threshold T2 | fine-tuned + threshold tuning | 0.7010 | 0.6453 | 0.7741 | 0.7979 | 0.7516 |
| No decoder-loss ablation | `model.prepare_epoch=1000` | 0.6699 | 0.4364 | 0.5844 | 0.6614 | 0.5234 |

The threshold setting selected on validation is:

```bash
model.score_th=0.35
model.test_cfg.sp_score_thr=0.1
model.test_cfg.npoint_thr=10
model.test_cfg.min_instance_points=10
```

These are inference/post-processing thresholds inherited from the original
ForestFormer3D/ForAINet-style instance prediction path. They control which
predicted masks survive confidence, superpoint score, and minimum-size
filtering. They are not new labels or training losses.

Run the selected threshold once on held-out test:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/oneformer3d_treescan_finetune.py \
  work_dirs/treescan_finetune_voxel_0_1_decoder/best_mRQ_epoch_45_spconv_fix.pth \
  --work-dir work_dirs/ablation/T2_threshold_precision_test \
  --cfg-options \
    model.score_th=0.35 \
    model.test_cfg.sp_score_thr=0.1 \
    model.test_cfg.npoint_thr=10 \
    model.test_cfg.min_instance_points=10
```

The ablation supports two conclusions. First, fine-tuning is necessary for
TreeScanPL10K because zero-shot transfer from the original ForestFormer3D
checkpoint has much lower panoptic quality and recognition quality. Second,
decoder fine-tuning is the main contributor to the gain: disabling decoder
losses keeps semantic/coverage behavior partially useful but sharply reduces
instance recognition and panoptic quality.

---

## 📚 Citation

If you find this project helpful, please cite our paper:

```bibtex
@inproceedings{xiang2025forestformer3d,
  title     = {ForestFormer3D: A Unified Framework for End-to-End Segmentation of Forest LiDAR 3D Point Clouds},
  author    = {Binbin Xiang and Maciej Wielgosz and Stefano Puliti and Kamil Král and Martin Krůček and Azim Missarov and Rasmus Astrup},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```

---

## 🆕 📢  For a faster way to run ForestFormer3D inference on your own test data, please use the following instruction:
[FF3D_inference – ff3d_forestsens](https://github.com/bxiang233/FF3D_inference/tree/main/ff3d_forestsens)

This version uses 2 inference iterations by default. If your trees are not extremely densely distributed, you can set the number of iterations to 1 instead.

# ForestFormer3D environment setup
This guide provides step-by-step instructions to build and configure the Docker environment for ForestFormer3D, set up debugging in Visual Studio Code, and resolve common issues.

At first, please download the dataset and pretrained model from Zenodo, and unzip and place them in the correct locations. Make sure the directory structure looks like:

```bash
ForestFormer3D/
├── data/
│   └── ForAINetV2/
│       ├── train_val_data/
│       └── test_data/
├── work_dirs/
│   └── clean_forestformer/
│       └── epoch_3000_fix.pth
```
---

## Steps to build and configure the environment

### **1. Build Docker image**

```bash
# Navigate to the project directory
cd #locationoftheproject#

# Build the Docker image
sudo docker build -t forestformer3d-image .

# Run the Docker container with GPU support, shared memory allocation, and port mapping
sudo docker run --gpus all --shm-size=128g -d -p 127.0.0.1:49211:22 \
  -v #locationofproject#:/workspace \
  -v segmentator:segmentator \
  --name forestformer3d-container forestformer3d-image

# Enter the running container
sudo docker exec -it forestformer3d-container /bin/bash

# Verify required files exist in your container
# ls
# Expected output:
# Dockerfile  configs  data  oneformer3d  readme  replace_mmdetection_files  segmentator  tools  work_dirs
```

### **2. Resolve Torch-Points-Kernels import error**
 
```bash
#test whether you successfully installed torch_points_kernels
python -c "from torch_points_kernels import instance_iou; print('torch-points-kernels loaded successfully')"
```
and if you encounter the following error:

```bash
ModuleNotFoundError: No module named 'torch_points_kernels.points_cuda'
```

please try to fix it by:
```bash
# Uninstall the existing torch-points-kernels version
pip uninstall torch-points-kernels -y

# Reinstall the specific compatible version
pip install --no-deps --no-cache-dir torch-points-kernels==0.7.0
```

### **3. Reinstall torch-cluster**
```bash
pip uninstall torch-cluster
pip install torch-cluster --no-cache-dir --no-deps
```

### **4. Replace required files**

```bash
# Find the mmengine package path
pip show mmengine

# Replace the following files with updated versions:
cp replace_mmdetection_files/loops.py /opt/conda/lib/python3.10/site-packages/mmengine/runner/
cp replace_mmdetection_files/base_model.py /opt/conda/lib/python3.10/site-packages/mmengine/model/base_model/
cp replace_mmdetection_files/transforms_3d.py /opt/conda/lib/python3.10/site-packages/mmdet3d/datasets/transforms/
```

### **5. Run the program**

#### **Data preparation**

Ensure the following three folders are set up in your workspace:

- `data/ForAINetV2/meta_data`
- `data/ForAINetV2/test_data`
- `data/ForAINetV2/train_val_data`

- Place all `.ply` files for training and validation in the `train_val_data` folder.
- Place all `.ply` files for testing in the `test_data` folder.

#### **Data preprocessing steps**

```bash
# Step 1: Navigate to the data folder
cd data/ForAINetV2

pip install laspy
pip install "laspy[lazrs]"

# Step 2: Run the data loader script
python batch_load_ForAINetV2_data.py
# After this you will have folder data/ForAINetV2/forainetv2_instance_data

# Step 3: Navigate back to the main directory
cd ../..

# Step 4: Create data for training
python tools/create_data_forainetv2.py forainetv2
```

#### **Start training**
```bash
export PYTHONPATH=/workspace
# Run the training script with the specified configuration and work directory
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/oneformer3d_qs_radius16_qp300_2many.py \
  --work-dir work_dirs/<output_folder_name>
```

#### **Run testing**
##### Use your own trained Checkpoint
```bash
#1. Fix the checkpoint file:
python tools/fix_spconv_checkpoint.py \
  --in-path work_dirs/oneformer3d_1xb4_forainetv2/trained.pth \
  --out-path work_dirs/oneformer3d_1xb4_forainetv2/trained_fix.pth

#2. Modify the output_path in function "predict" in class ForAINetV2OneFormer3D_XAwarequery in file oneformer3d/oneformer3d.py

#3. Run the test script:
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/oneformer3d_qs_radius16_qp300_2many.py \
  work_dirs/oneformer3d_1xb4_forainetv2/trained_fix.pth

```
##### Load pre-trained model
```bash
# If you want to use the official pre-trained model, run:
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/oneformer3d_qs_radius16_qp300_2many.py work_dirs/clean_forestformer/epoch_3000_fix.pth

```

---

## Test your own test files

To evaluate your own test files, follow these steps:

### 1. Copy test files

Place your test files under the following directory:

```
data/ForAINetV2/test_data
```

### 2. Update the test list

Edit the following file:

```
data/ForAINetV2/meta_data/test_list.txt
```

Append the base names (without extension) of your test files. For example:

```
your_custom_test_file_name  # <-- add your file name here
```


### 3. Re-run data preprocessing

```bash
# Step 1: Navigate to the data folder
cd data/ForAINetV2

# Step 2: Install required libraries (if not already installed)
pip install laspy
pip install "laspy[lazrs]"

# Step 3: Run the data loader script
python batch_load_ForAINetV2_data.py
# This will regenerate data/ForAINetV2/forainetv2_instance_data

# Step 4: Navigate back to the main directory
cd ../..

# Step 5: Create data for training/testing
python tools/create_data_forainetv2.py forainetv2
```

### 4. Run testing script again

Once preprocessing is complete, you can run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/oneformer3d_qs_radius16_qp300_2many.py work_dirs/clean_forestformer/epoch_3000_fix.pth
```


---

## ⚠️ Using non-ply test files

If your test files are **not in `.ply` format**, you need to modify the data loading logic.

### 1. Update `batch_load_ForAINetV2_data.py`

File path:
```
data/ForAINetV2/batch_load_ForAINetV2_data.py
```

Find the function `export_one_scan()` and modify:

```python
ply_file = osp.join(forainetv2_dir, scan_name + '.ply')
```

to match your test file format, for example:

```python
pc_file = osp.join(forainetv2_dir, scan_name + '.laz')  # or other formats
```

---

### 2. Update `load_forainetv2_data.py` to support new format

File path:
```
data/ForAINetV2/load_forainetv2_data.py
```

Find the function `export()` and modify:

```python
pcd = read_ply(ply_file)
```

If you're using `.laz`, replace with something like:

```python
import laspy

def read_laz(filename):
    las = laspy.read(filename)
    return {
        "x": las.x,
        "y": las.y,
        "z": las.z,
        # Add more fields as needed
    }

pcd = read_laz(ply_file)
```

---

### 3. If test files do not have ground truth labels

Still in `load_forainetv2_data.py`, locate the following lines:

```python
semantic_seg = pcd["semantic_seg"].astype(np.int64)
treeID = pcd["treeID"].astype(np.int64)
```

If the test file lacks these labels, replace them with:

```python
semantic_seg = np.ones((points.shape[0],), dtype=np.int64)
treeID = np.zeros((points.shape[0],), dtype=np.int64)
# semantic_seg = pcd["semantic_seg"].astype(np.int64)
# treeID = pcd["treeID"].astype(np.int64)
```

This will prevent errors when labels are missing in test data.


**Recommendation**: The **easiest solution** is to convert your test files to `.ply` format in advance. This avoids having to change the code and ensures full compatibility with the pipeline.

---
## **Optional Tips**

#### **Tensorboard Visualization**
```bash
tensorboard --logdir=work_dirs/YOUR_OUTPUT_FOLDER/vis_data/ --host=0.0.0.0 --port=6006
```

#### **Configure SSH for Debugging in Visual Studio Code**
```bash
# Install and start OpenSSH server
apt-get install -y openssh-server
service ssh start

# Set a password for the root user
passwd root

# Modify SSH configuration to enable root login and password authentication
echo -e "PermitRootLogin yes\nPasswordAuthentication yes" >> /etc/ssh/sshd_config

# Restart the SSH service
service ssh restart
```

To connect via SSH in VS Code, ensure you forward port 22 of the container to a host port during docker run. For example, include -p 127.0.0.1:49211:22 in your docker run command.


## 🌲 Handling missed detections in dense test data

In extremely dense test plots, the initial inference run may miss some trees. To address this, we apply ForestFormer3D a second time **only on the remaining points** that were not segmented in the first round.

You can perform this secondary inference by running:

```bash
bash tools/inference_bluepoint.sh
```

This script re-runs inference on remaining "blue points" after the first round.

### 🛠️ How to use

1. Prepare your test data as usual (see earlier sections).
2. Instead of running `tools/test.py`, execute:

```bash
bash tools/inference_bluepoint.sh
```

3. Make the following adjustments before running:

- Put all your test file names in:

```
data/ForAINetV2/meta_data/test_list_initial.txt
```

instead of the default `test_list.txt`.

- Modify `BLUEPOINTS_DIR` in the script to match your output directory (the output_path in function "predict" in class ForAINetV2OneFormer3D_XAwarequery in file workspace/oneformer3d/oneformer3d.py), for example:

```bash
BLUEPOINTS_DIR="$WORK_DIR/work_dirs/YOUROUTPUTPATH"
```

- In the file:
```
oneformer3d/oneformer3d.py
```
Inside the function `predict` of class `ForAINetV2OneFormer3D_XAwarequery`, change:

```python
self.save_ply_withscore(...)
# self.save_bluepoints(...)
```

to:

```python
# self.save_ply_withscore(...)
self.save_bluepoints(...)
```

- Also replace:

```python
# is_test = True
# if is_test:
if 'test' in lidar_path:
```
with the appropriate logic to ensure test mode is active when needed:

```python
is_test = True
if is_test:
#if 'test' in lidar_path:
```

This two-step (or multiple-step) inference improves robustness in challenging, highly dense forests.

---
## 🙋‍♀️ Questions & suggestions

Welcome to ask questions via Issues! This helps more people see the discussion and avoid duplicated questions.

🔍 Before opening a new issue, please check if someone has already asked the same question.  
Thank you for your cooperation, and we’re looking forward to your suggestions and ideas! 🌟

---
## 💡Note on training GPU requirements

The training was run on a single A100 GPU. If you're using a GPU with less memory, try reducing the cylinder radius in the config to prevent OOM.

For inference, batch_size is not used, because each cylinder is processed sequentially. If you encounter CUDA OOM issues during inference, try:

1. Lowering the chunk value in the config

2. Reducing `num_points` in the code ([see this line](https://github.com/SmartForest-no/ForestFormer3D/blob/8ca0f45196ce0cc8a656d046b3f935cbf34f315b/oneformer3d/oneformer3d.py#L2273))

3. Reducing the cylinder radius, which is also configurable in the config file.

---
## 📝 License
ForestFormer3D is based on the OneFormer3D codebase by Danila Rukhovich (https://github.com/filaPro/oneformer3d),  
which is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

This repository is therefore also released under the same license.  
Please cite appropriately if you use or modify this code. Thank you.
