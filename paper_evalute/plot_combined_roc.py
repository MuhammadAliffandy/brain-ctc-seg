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

# Import components from evaluate_trained_models
from evaluate_trained_models import (
    SE2_CNNET, HarmonicNet, nnUNet, AttentionUNet, TransUNet, StandardUNet,
    CTBrain25DDatasetNoResize, CTBrain25DDataset, filter_df_by_dataset, load_se2_weights
)

def main():
    parser = argparse.ArgumentParser(description="Generate Combined ROC Curve")
    parser.add_argument('--dataset', type=str, choices=['ct', 'ctc'], required=True, 
                        help="Dataset to evaluate on (ct or ctc)")
    args = parser.parse_args()

    ds = args.dataset
    CSV_REPORT  = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
    DATA_PATH   = os.path.expanduser("~/Clara/local_ct_workspace_full")
    SAVE_DIR    = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D")
    OUT_FILE    = os.path.expanduser(f"~/Clara/brain-ctc-seg/training/Journal_Figures/ROC_Curve_{ds.upper()}.png")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"📈 GENERATING COMBINED ROC CURVE FOR: {ds.upper()} DATASET")
    print(f"{'='*70}\n")

    # ─── 1. Prepare Validation Data ───
    if not os.path.exists(CSV_REPORT):
        print(f"❌ CSV not found: {CSV_REPORT}"); sys.exit(1)

    df = pd.read_csv(CSV_REPORT)
    pc = 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'
    df = filter_df_by_dataset(df, ds, pc)
    
    # Same split as evaluation script
    train_df = df.sample(frac=0.85, random_state=42)
    val_df   = df.drop(train_df.index)
    
    val_dataset_native = CTBrain25DDatasetNoResize(val_df, DATA_PATH)
    val_dataset_256    = CTBrain25DDataset(val_df, DATA_PATH)
    
    val_loader_native = DataLoader(val_dataset_native, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    val_loader_256    = DataLoader(val_dataset_256, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # ─── 2. Model Registry ───
    MODELS = [
        ("Mod-Seg-SE(2) [OURS]", SE2_CNNET,    f"se2_unet_{ds}_best.pth",        True,  'red',    'solid', 2.5),
        ("HarmonicNet (C4)",     HarmonicNet,   f"harmonic_net_{ds}_best.pth",    False, 'orange', 'dashed', 2.0),
        ("nnU-Net",              nnUNet,        f"nn_unet_{ds}_best.pth",         False, 'blue',   'dashdot', 2.0),
        ("Attention U-Net",      AttentionUNet, f"attention_unet_{ds}_best.pth",  False, 'purple', 'dotted', 2.0),
        ("TransUNet",            TransUNet,     f"trans_unet_{ds}_best.pth",      False, 'green',  'dashed', 2.0),
        ("Standard U-Net",       StandardUNet,  f"standard_unet_{ds}_best.pth",   False, 'gray',   'solid', 1.5),
    ]

    plt.figure(figsize=(10, 8), facecolor='white')

    # Subsample factor to avoid OOM when computing ROC (e.g., take 1 out of 10 pixels)
    # Validation set is ~200 images of 256x256. 200*65536 = 13M pixels. 13M is fine for RAM, but subsampling speeds it up.
    SUBSAMPLE_FACTOR = 10 

    for name, ModelClass, weight_file, use_se2_loader, color, ls, lw in MODELS:
        weight_path = os.path.join(SAVE_DIR, weight_file)
        if not os.path.exists(weight_path):
            print(f"⚠️  Skipping {name}: Weight not found ({weight_path})")
            continue
            
        print(f"Inference: {name}...")
        loader = val_loader_native if use_se2_loader else val_loader_256
        model = ModelClass(n_channels=3, n_classes=2)
        
        if name.startswith("Mod-Seg-SE(2)"):
            model = load_se2_weights(model, weight_path, device)
        else:
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
            
        model.to(device)
        model.eval()

        all_y_true = []
        all_y_scores = []

        with torch.no_grad():
            for imgs, masks in tqdm(loader, desc=f"  Evaluating", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    logits = model(imgs)
                
                # Get probabilities for positive class (tumor)
                probs = F.softmax(logits, dim=1)[:, 1, :, :]
                
                # Flatten and subsample
                y_true = masks.view(-1).cpu().numpy()[::SUBSAMPLE_FACTOR]
                y_scores = probs.view(-1).cpu().numpy()[::SUBSAMPLE_FACTOR]
                
                all_y_true.append(y_true)
                all_y_scores.append(y_scores)

        # Concatenate arrays
        all_y_true = np.concatenate(all_y_true)
        all_y_scores = np.concatenate(all_y_scores)

        # Compute ROC
        fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw, 
                 label=f'{name} (AUC = {roc_auc:.4f})')
        
        # Free memory
        del all_y_true, all_y_scores, model, loader
        torch.cuda.empty_cache()

    # ─── 3. Finalize Plot Formatting ───
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5) # Diagonal line
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
    plt.title(f'Receiver Operating Characteristic (ROC) - {ds.upper()}', fontsize=16, fontweight='bold', pad=20)
    
    # Legend settings
    plt.legend(loc="lower right", fontsize=12, frameon=True, shadow=True, edgecolor='black')
    
    # Grid lines
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Axes styling
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Combined ROC Curve for {ds.upper()} saved successfully at: \n   {OUT_FILE}\n")

if __name__ == "__main__":
    main()
