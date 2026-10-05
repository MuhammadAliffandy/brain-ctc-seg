import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import components
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from evaluate_trained_models import (
    SE2_CNNET, HarmonicNet, nnUNet, AttentionUNet, TransUNet, StandardUNet,
    CTBrain25DDatasetNoResize, CTBrain25DDataset, filter_df_by_dataset, load_se2_weights
)

def get_valid_path(rel_path):
    candidates = [
        os.path.expanduser(f"~/Clara/{rel_path}"),
        f"/raid/D13K48009/Clara/{rel_path}",
    ]
    for c in candidates:
        if os.path.exists(c): return c
    return candidates[0]

def main():
    parser = argparse.ArgumentParser(description="Fix ROC Curve without subsampling")
    parser.add_argument('--dataset', type=str, choices=['ct', 'ctc', 'kaggle', 'kaggle_hemorrhage'], required=True)
    args = parser.parse_args()
    ds = args.dataset

    DATA_PATH = get_valid_path("local_ct_workspace_full")
    SAVE_DIR  = get_valid_path("brain-ctc-seg/training/saved_models_25D")
    OUT_FILE  = get_valid_path(f"brain-ctc-seg/training/Journal_Figures/FIXED_ROC_Curve_{ds.upper()}.png")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"📈 GENERATING TRUE PIXEL-WISE ROC CURVE FOR: {ds.upper()}")
    print(f"   (No subsampling. Full probability map resolution)")
    print(f"{'='*70}\n")

    # ─── 1. Prepare Validation Data ───
    if ds in ['ct', 'ctc']:
        CSV_REPORT = get_valid_path("new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
        df = pd.read_csv(CSV_REPORT)
        pc = 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'
        df = filter_df_by_dataset(df, ds, pc)
        
        train_df = df.sample(frac=0.85, random_state=42)
        val_df   = df.drop(train_df.index)
        
        val_dataset_native = CTBrain25DDatasetNoResize(val_df, DATA_PATH)
        val_dataset_256    = CTBrain25DDataset(val_df, DATA_PATH)
        
        val_loader_native = DataLoader(val_dataset_native, batch_size=8, shuffle=False, num_workers=2)
        val_loader_256    = DataLoader(val_dataset_256, batch_size=8, shuffle=False, num_workers=2)
    else:
        # For Kaggle datasets
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "public_dataset"))
        import kagglehub
        from torch.utils.data import Dataset
        import cv2
        class KaggleDataset(Dataset):
            def __init__(self, samples): self.samples = samples
            def __len__(self): return len(self.samples)
            def __getitem__(self, idx):
                img_path, mask_path = self.samples[idx]
                img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (256, 256))
                mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (256, 256), interpolation=cv2.INTER_NEAREST)
                img = (img.astype(np.float32) - img.min()) / (img.max() - img.min() + 1e-7)
                img_3c = np.stack([img, img, img], axis=0)
                mask = (mask > 127).astype(np.int64)
                return torch.from_numpy(img_3c).float(), torch.from_numpy(mask).long()
        
        if ds == 'kaggle':
            from train_all_intra import get_kaggle_splits
            root_dir = kagglehub.dataset_download("ozguraslank/brain-stroke-ct-dataset")
            _, test_samples = get_kaggle_splits(root_dir)
        else:
            from train_all_intra_hemorrhage import get_kaggle_hemorrhage_splits
            _, test_samples = get_kaggle_hemorrhage_splits()
            
        test_dataset = KaggleDataset(test_samples)
        val_loader_native = val_loader_256 = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
        SAVE_DIR = get_valid_path("brain-ctc-seg/public_dataset/saved_models")

    # ─── 2. Model Registry ───
    if ds in ['ct', 'ctc']:
        MODELS = [
            ("CT-SE(2) [Proposed]", SE2_CNNET, f"se2_unet_{ds}_best.pth", True, 'red', 'solid', 2.5),
            ("HarmonicNet", HarmonicNet, f"harmonic_net_{ds}_best.pth", False, 'orange', 'dashed', 2.0),
            ("nnU-Net", nnUNet, f"nn_unet_{ds}_best.pth", False, 'blue', 'dashdot', 2.0),
            ("Attention U-Net", AttentionUNet, f"attention_unet_{ds}_best.pth", False, 'purple', 'dotted', 2.0),
            ("TransUNet", TransUNet, f"trans_unet_{ds}_best.pth", False, 'green', 'dashed', 2.0),
            ("Standard U-Net", StandardUNet, f"standard_unet_{ds}_best.pth", False, 'gray', 'solid', 1.5),
        ]
    else:
        MODELS = [
            ("CT-SE(2) [Proposed]", SE2_CNNET, f"Mod-Seg-SE2_{ds}_best.pth", True, 'red', 'solid', 2.5),
            ("HarmonicNet", HarmonicNet, f"HarmonicNet_{ds}_best.pth", True, 'orange', 'dashed', 2.0),
            ("nnU-Net", nnUNet, f"nnU-Net_{ds}_best.pth", False, 'blue', 'dashdot', 2.0),
            ("Attention U-Net", AttentionUNet, f"Attention_U-Net_{ds}_best.pth", False, 'purple', 'dotted', 2.0),
            ("TransUNet", TransUNet, f"TransUNet_{ds}_best.pth", False, 'green', 'dashed', 2.0),
            ("Standard U-Net", StandardUNet, f"Standard_U-Net_{ds}_best.pth", False, 'gray', 'solid', 1.5),
        ]

    plt.figure(figsize=(10, 8), facecolor='white')

    for name, ModelClass, weight_file, use_se2_loader, color, ls, lw in MODELS:
        weight_path = os.path.join(SAVE_DIR, weight_file)
        if not os.path.exists(weight_path):
            print(f"⚠️  Skipping {name}: Weight not found ({weight_path})")
            continue
            
        print(f"Inference: {name}...")
        loader = val_loader_native if use_se2_loader else val_loader_256
        model = ModelClass(n_channels=3, n_classes=2)
        
        # Load weights
        if name.startswith("CT-SE(2)"):
            if ds in ['ct', 'ctc']:
                model = load_se2_weights(model, weight_path, device)
            else:
                model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        else:
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
            
        model.to(device)
        model.eval()

        # Allocate memory for full probability map arrays
        all_y_true = []
        all_y_scores = []

        with torch.no_grad():
            for imgs, masks in tqdm(loader, desc=f"  Evaluating", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    logits = model(imgs)
                
                # Get exact probabilities for positive class (tumor)
                probs = F.softmax(logits, dim=1)[:, 1, :, :]
                
                # We NO LONGER subsample. We take all pixels.
                y_true = masks.reshape(-1).cpu().numpy().astype(np.int8)
                y_scores = probs.reshape(-1).cpu().numpy().astype(np.float16) # use float16 to save RAM
                
                all_y_true.append(y_true)
                all_y_scores.append(y_scores)

        # Concatenate full arrays
        all_y_true = np.concatenate(all_y_true)
        all_y_scores = np.concatenate(all_y_scores)

        # Compute ROC exactly
        print(f"  -> Computing exact ROC for {len(all_y_true):,} pixels...")
        fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw, label=f'{name} (AUC = {roc_auc:.4f})')
        
        del all_y_true, all_y_scores, model
        torch.cuda.empty_cache()

    # ─── 3. Finalize Plot Formatting ───
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
    
    # Reviewer's point: "At FPR=0.0004, TPR=0.99... The curve must pass through (0.0004, 0.99)"
    # A log-scale X-axis (FPR) plot is often used in medical imaging to reveal exactly this behavior
    # But for standard ROC, we just plot it linearly, it should spike immediately.
    
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
    plt.title(f'Exact ROC Curve (No Subsampling) - {ds.upper()}', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=12, frameon=True, shadow=True, edgecolor='black')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=300, facecolor='white', bbox_inches='tight')
    
    # Also save a zoomed-in version (FPR 0 to 0.05) to prove to the reviewer the curve passes through the expected point
    plt.xlim([-0.001, 0.05])
    zoom_file = OUT_FILE.replace(".png", "_zoomed.png")
    plt.title(f'Zoomed ROC Curve (FPR 0-5%) - {ds.upper()}', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(zoom_file, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Exact ROC Curve for {ds.upper()} saved successfully at:")
    print(f"   {OUT_FILE}")
    print(f"   {zoom_file}")

if __name__ == "__main__":
    main()
