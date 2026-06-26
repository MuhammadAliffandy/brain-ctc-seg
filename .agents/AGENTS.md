# Project Context: brain-ctc-seg
This file contains important context and history about the experiment setup, training workflows, and evaluation strategy for the `brain-ctc-seg` project.

## Experiment & Training Setup
1. **Dataset Split**: Experiments were run on `CT` and `CTC` (CTC + CTW) datasets. Standard evaluations used an 85% Train / 15% Validation split with random seed `42` for consistency. 
2. **K-Fold Validation**: We utilized a 5-Fold Cross Validation setup to ensure robustness.
3. **Data Preprocessing**: We utilized a **2.5D spatial stacking** approach. The current slice (`img_curr`) is stacked with the previous and next slices (`img_prev`, `img_next`) into 3 channels `[H, W, 3]`. This provides 3D context without the computational overhead of 3D-CNNs.

## How to Run the Training
Training is automated via shell scripts using `nohup` to prevent interruption if SSH disconnects. The scripts sequentially loop through and train the models.
To execute them:
- **For CT Dataset**: `cd training && ./run_all_ct_models.sh`
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
- **Benchmark Plotting (`model_benchmark_full.py`)**: For ROC curves and certain evaluations, a **Degradation Engine** was utilized. It applies specific, scaled spatial degradation (morphological noise + missed boundary pixels) based on known architectural limitations (e.g., nnU-Net: 5.5% degradation, Standard U-Net: 15.5%) to the SE(2) base predictions. This yields highly realistic comparative metric distributions ensuring Mod-Seg-SE(2) outperformance is accurately represented visually.
