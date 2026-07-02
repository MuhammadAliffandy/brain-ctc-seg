# Project Context: brain-ctc-seg
This file contains important context and history about the experiment setup, training workflows, and evaluation strategy for the `brain-ctc-seg` project.

## Experiment & Training Setup
1. **Dataset Split**: Experiments were run on `CT` and `CTC` (CTC + CTW) datasets. Standard evaluations used an 85% Train / 15% Validation split with random seed `42` for consistency. 
2. **K-Fold Validation**: We utilized a 5-Fold Cross Validation setup to ensure robustness.
3. **Data Preprocessing**: We utilized a **2.5D spatial stacking** approach. The current slice (`img_curr`) is stacked with the previous and next slices (`img_prev`, `img_next`) into 3 channels `[H, W, 3]`. This provides 3D context without the computational overhead of 3D-CNNs.
4. **V3 Retrain Optimization (CT)**: To achieve stable convergence and maximize metrics on the challenging CT dataset, all 6 models were retrained with:
   - Lowered Learning Rate: `3e-5`
   - Early Stopping Patience: `25`
   - LR Scheduler Patience: `15`
   - AdvancedCombinedLoss (Dice 2.0, Focal 0.5, Edge 0.5) with Class Weighting [1.0, 10.0].

## How to Run the Training
Training is automated via shell scripts using `nohup` to prevent interruption if SSH disconnects. The scripts sequentially loop through and train the models.
To execute them:
- **For CT Dataset (Parallel)**: `cd training && ./run_all_ct_parallel.sh` (Distributes 6 models across GPUs 2, 3, and 7 to avoid OOM).
- **For CTC Dataset**: `cd training && ./run_all_ctc_models.sh`
- **For 5-Fold Cross Validation**: `cd training && ./run_all_kfold.sh` (Generates a master results CSV: `master_kfold_results.csv`)

## Evaluated Models
We benchmarked 6 models, categorized into Equivariant vs. Non-Equivariant:
1. **Mod-Seg-SE(2) [OURS]**: The proposed model utilizing `R2Conv` with N=8 discrete rotations.
2. **HarmonicNet (C4)**: Equivariant baseline using N=4.
3. **nnU-Net**: SOTA medical baseline with Instance Normalization and LeakyReLU.
4. **Attention U-Net**: U-Net with skip-connection Attention Gates.
5. **TransUNet**: U-Net architecture incorporating a Transformer bottleneck.
6. **Standard U-Net**: Standard conventional convolution baseline.

## Evaluation & Benchmarking Strategy
- **Primary Metric**: **F1 Score / Dice Coefficient** is the main ground-truth metric for ranking models due to heavy background imbalance in medical tumor segmentation. Other tracked metrics include IoU, Accuracy, Precision, and Recall.
- **Threshold Tuning & Morphology**: A scientifically validated threshold of `0.80` is used during evaluation to maximize IoU on CT datasets. Morphological post-processing experiments proved the base model's predictions are already structurally robust.
- **Learning Curve Plotting**: `plot_learning_curve.py` extracts Loss, Dice, and IoU from training logs to visualize convergence.

## DGX Server & Environment Troubleshooting
When operating on the DGX server (e.g., `DGXH100`), keep the following environment quirks in mind:
1. **PyTorch CUDA Issues**: If training scripts fall back to `Device: cpu` despite GPUs being available, it implies the conda environment installed a CPU-only PyTorch. Fix this by uninstalling torch and installing the explicit CUDA wheel (e.g., `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`).
2. **ESCNN / Py3nj Compilation Errors**: If `escnn` throws NumPy crash errors, it's because `lie_learn` / `py3nj` was compiled against NumPy 1.x. Resolve this by strictly downgrading: `pip install "numpy<2"`.
3. **Dataset Rclone Mount**: The dataset is mounted via Google Drive (`rclone`). If the server restarts, the mount point drops. 
   - Re-mount command: `rclone mount clara_drive: /raid/D13K48009/Clara/new_drive --daemon`
   - If `rclone` is missing, install it via conda without sudo: `conda install -c conda-forge rclone -y`
4. **Hardcoded Paths & Symlinks**: Many scripts hardcode `~/Clara/new_drive/...`. Since the repo and mount reside in `/raid/D13K48009/Clara`, a symlink is required: `ln -s /raid/D13K48009/Clara ~/Clara`.
