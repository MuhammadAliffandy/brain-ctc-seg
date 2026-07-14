import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import albumentations as A
from torch.utils.data import DataLoader

# ==========================================
# IMPORT MODELS & DATASET
# ==========================================
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_comparison_models import (
    CTBrain25DDataset, HarmonicNet, StandardUNet, nnUNet, 
    AttentionUNet, TransUNet, filter_df_by_dataset
)
from train_se2_by_dataset import SE2_CNNET

# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_models_25D")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Inference_Figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_REPORT = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
LOCAL_DATA = os.path.expanduser("~/Clara/local_ct_workspace_full")

MODELS_CONFIG = {
    "Mod-Seg-SE(2)":   {"class": lambda: SE2_CNNET(3, 2, 8, 32), "color": "#D32F2F", "prefix_ct": "se2_unet_ct_best.pth", "prefix_ctc": "se2_unet_ctc_best.pth"},
    "HarmonicNet":     {"class": lambda: HarmonicNet(3, 2),      "color": "#FBC02D", "prefix_ct": "harmonic_net_ct_best.pth", "prefix_ctc": "harmonic_net_ctc_best.pth"},
    "nnU-Net":         {"class": lambda: nnUNet(3, 2),           "color": "#1976D2", "prefix_ct": "nn_unet_ct_best.pth", "prefix_ctc": "nn_unet_ctc_best.pth"},
    "Standard U-Net":  {"class": lambda: StandardUNet(3, 2),     "color": "#388E3C", "prefix_ct": "standard_unet_ct_best.pth", "prefix_ctc": "standard_unet_ctc_best.pth"},
    "Attention U-Net": {"class": lambda: AttentionUNet(3, 2),    "color": "#7B1FA2", "prefix_ct": "attention_unet_ct_best.pth", "prefix_ctc": "attention_unet_ctc_best.pth"},
    "TransUNet":       {"class": lambda: TransUNet(3, 2),        "color": "#00796B", "prefix_ct": "trans_unet_ct_best.pth", "prefix_ctc": "trans_unet_ctc_best.pth"},
}

# ==========================================
# INFERENCE FUNCTION
# ==========================================
def run_inference(model, dataloader):
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Inference", leave=False):
            images = images.to(DEVICE)
            labels = labels.cpu().numpy().flatten()
            
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1, :, :]
                probs = probs.cpu().numpy().flatten()
                
            all_probs.append(probs)
            all_targets.append(labels)
            
    return np.concatenate(all_targets), np.concatenate(all_probs)

# ==========================================
# MAIN PLOTTING PIPELINE
# ==========================================
def generate_multipanel_roc():
    print(f"🚀 Starting Multi-Panel ROC Generation...")
    
    # 1. Load Data
    df = pd.read_csv(CSV_REPORT)
    pc = 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'
    
    # Validation splits (using the same seed=42 to match training)
    df_ct = filter_df_by_dataset(df, 'ct', pc)
    df_ct_train = df_ct.sample(frac=0.85, random_state=42)
    df_ct_val = df_ct.drop(df_ct_train.index)
    
    df_ctc = filter_df_by_dataset(df, 'ctc', pc)
    df_ctc_train = df_ctc.sample(frac=0.85, random_state=42)
    df_ctc_val = df_ctc.drop(df_ctc_train.index)
    
    print(f"📊 Validation Data - CT: {len(df_ct_val)} patients | CTC: {len(df_ctc_val)} patients")
    
    # Only load 20 patients from validation for speed during inference, to avoid memory issues and long waits
    df_ct_val = df_ct_val.head(20)
    df_ctc_val = df_ctc_val.head(20)
    
    ds_ct = CTBrain25DDataset(df_ct_val, LOCAL_DATA, transform=None)
    ds_ctc = CTBrain25DDataset(df_ctc_val, LOCAL_DATA, transform=None)
    
    nw = min(os.cpu_count() or 4, 8)
    dl_ct = DataLoader(ds_ct, batch_size=8, shuffle=False, num_workers=nw)
    dl_ctc = DataLoader(ds_ctc, batch_size=8, shuffle=False, num_workers=nw)
    
    # 2. Setup Figure (1x2 Panel)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    titles = [
        ("a) NTUH Cohort CT Non Contrast Dataset", dl_ct, "ct"),
        ("b) NTUH Cohort CT with Contrast Dataset", dl_ctc, "ctc")
    ]
    
    # 3. Iterate over Datasets and Models
    for ax, (title, dl, dkey) in zip(axes, titles):
        print(f"\nEvaluating for panel: {title}")
        
        for m_name, m_info in MODELS_CONFIG.items():
            pth_file = os.path.join(MODEL_DIR, m_info[f'prefix_{dkey}'])
            
            if not os.path.exists(pth_file):
                print(f"⚠️ Missing {pth_file}, skipping {m_name}")
                continue
                
            print(f"  -> Inferencing {m_name}...")
            model = m_info["class"]().to(DEVICE)
            
            try:
                state_dict = torch.load(pth_file, map_location=DEVICE)
                if 'model_state_dict' in state_dict:
                    model.load_state_dict(state_dict['model_state_dict'])
                else:
                    model.load_state_dict(state_dict)
            except Exception as e:
                print(f"     ❌ Failed to load weights: {e}")
                continue
                
            # Get predictions
            y_true, y_probs = run_inference(model, dl)
            
            # Compute ROC
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            roc_auc = auc(fpr, tpr)
            
            # Plot line
            ax.plot(fpr, tpr, color=m_info["color"], lw=2.5, 
                    label=f'{m_name} (AUC = {roc_auc:.3f})')
            
        # Standardize plot styling matching the reference
        ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.500)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.grid(True, linestyle='-', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(loc="lower right", fontsize=11, framealpha=1.0, edgecolor='black')
        
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "MultiPanel_ROC_Curve.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n🎉 DONE! Multi-Panel ROC saved to: {out_path}")

if __name__ == "__main__":
    generate_multipanel_roc()
