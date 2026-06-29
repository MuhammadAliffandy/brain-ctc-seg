"""
generate_secondary_analysis_table.py
======================================
Script to generate the secondary analysis table for the journal paper.
Metrics included:
- Parameters (M)
- FLOPs / MACs (G)
- Dice Score (Mean ± 95% CI)
- Hausdorff95 / HD95 (Mean ± 95% CI)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from thop import profile
except ImportError:
    print("Please pip install thop")
    sys.exit(1)

try:
    from medpy.metric.binary import hd95
except ImportError:
    print("Please pip install medpy")
    sys.exit(1)

# Import models and dataset from evaluate_trained_models
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import argparse
from evaluate_trained_models import (
    SE2_CNNET, HarmonicNet, StandardUNet, nnUNet, AttentionUNet, TransUNet, 
    CTBrain25DDatasetNoResize, load_se2_weights, filter_df_by_dataset
)

# Paths
CSV_REPORT = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
DATA_PATH  = os.path.expanduser("~/Clara/local_ct_workspace_full")
SAVE_DIR   = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D")

def compute_ci95(data):
    if len(data) == 0: return 0.0, 0.0, 0.0
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    # Using Z=1.96 for 95% Confidence Interval
    margin = 1.96 * std / np.sqrt(len(data))
    return mean, mean - margin, mean + margin

def calculate_hd95(pred, target):
    p = pred.astype(bool)
    t = target.astype(bool)
    if p.sum() == 0 and t.sum() == 0:
        return 0.0
    elif p.sum() == 0 or t.sum() == 0:
        return np.nan 
    
    try:
        return hd95(p, t)
    except Exception:
        return np.nan

def get_macs_params(model, device):
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    model.eval()
    try:
        # Prevent escnn from printing excessive warnings during profiling
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        return macs / 1e9, params / 1e6  # GMACs, MParams
    except Exception as e:
        print(f"    [Warning] thop profile failed: {e}")
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return np.nan, params / 1e6

def evaluate_model(model, loader, device, name):
    model.eval()
    dice_scores = []
    hd95_scores = []
    
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=f"  {name}", ncols=80):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).cpu().numpy()
            
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
            
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().numpy()
            
            for i in range(len(preds)):
                p = preds[i]
                t = masks[i]
                
                # Dice
                tp = np.sum((p == 1) & (t == 1))
                fp = np.sum((p == 1) & (t == 0))
                fn = np.sum((p == 0) & (t == 1))
                eps = 1e-7
                dice = (2 * tp) / (2 * tp + fp + fn + eps)
                dice_scores.append(dice)
                
                # HD95
                h = calculate_hd95(p, t)
                if not np.isnan(h):
                    hd95_scores.append(h)
                    
    d_mean, d_low, d_high = compute_ci95(dice_scores)
    h_mean, h_low, h_high = compute_ci95(hd95_scores)
    
    return {
        "Dice_Mean": d_mean, "Dice_L": d_low, "Dice_H": d_high,
        "HD95_Mean": h_mean, "HD95_L": h_low, "HD95_H": h_high
    }

def main(dataset_key):
    print("="*90)
    print(f"📊 GENERATING SECONDARY ANALYSIS TABLE FOR {dataset_key.upper()} (Params, FLOPs, HD95, CI 95%)".center(90))
    print("="*90)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    if not os.path.exists(CSV_REPORT):
        print(f"❌ CSV not found: {CSV_REPORT}")
        sys.exit(1)
        
    df = pd.read_csv(CSV_REPORT)
    pc = 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'
    df = filter_df_by_dataset(df, dataset_key, pc)
    print(f"  Dataset '{dataset_key}': {len(df)} patients total")
    if len(df) == 0:
        print("❌ No patients match this dataset type.")
        sys.exit(1)
        
    train_df = df.sample(frac=0.85, random_state=42)
    val_df   = df.drop(train_df.index)
    
    val_set = CTBrain25DDatasetNoResize(val_df, DATA_PATH)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    print(f"  Validation samples: {len(val_set)}\n")

    MODELS = [
        ("Mod-Seg-SE(2) [OURS]", SE2_CNNET, f"se2_unet_{dataset_key}_best.pth", True),
        ("HarmonicNet (C4)", HarmonicNet, f"harmonic_net_{dataset_key}_best.pth", False),
        ("nnU-Net", nnUNet, f"nn_unet_{dataset_key}_best.pth", False),
        ("Attention U-Net", AttentionUNet, f"attention_unet_{dataset_key}_best.pth", False),
        ("TransUNet", TransUNet, f"trans_unet_{dataset_key}_best.pth", False),
        ("Standard U-Net", StandardUNet, f"standard_unet_{dataset_key}_best.pth", False),
    ]
    
    FALLBACK = {
        "Mod-Seg-SE(2) [OURS]": ["se2_unet_epoch_100.pth", "se2_unet_all_best.pth", "se2_unet_ct_best.pth", "se2_unet_ctc_best.pth"],
        "HarmonicNet (C4)": ["harmonic_net_epoch_100.pth", "harmonic_net_all_best.pth"],
        "nnU-Net": ["nn_unet_epoch_100.pth", "nn_unet_all_best.pth"],
        "Attention U-Net": ["attention_unet_epoch_100.pth", "attention_unet_all_best.pth"],
        "TransUNet": ["trans_unet_epoch_100.pth", "trans_unet_all_best.pth"],
        "Standard U-Net": ["standard_unet_epoch_100.pth", "standard_unet_all_best.pth"],
    }

    results = []
    
    for display_name, ModelClass, weight_file, is_se2 in MODELS:
        print(f"─"*90)
        print(f"  Model: {display_name}")
        
        candidates = [weight_file] + FALLBACK.get(display_name, [])
        weight_path = None
        for wf in candidates:
            p = os.path.join(SAVE_DIR, wf)
            if os.path.exists(p): 
                weight_path = p
                break
            
        if not weight_path:
            print(f"  ⚠️ Weights not found. Calculating Params/MACs with random init.")
            if is_se2:
                model = ModelClass(n_channels=3, n_classes=2, base_channels=24).to(device)
            else:
                model = ModelClass(n_channels=3, n_classes=2).to(device)
            do_eval = False
        else:
            if is_se2:
                model = load_se2_weights(ModelClass, weight_path, device)
            else:
                model = ModelClass(n_channels=3, n_classes=2).to(device)
                model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True), strict=False)
                print(f"  ✅ Weights loaded: {weight_path}")
            do_eval = True

        macs, params = get_macs_params(model, device)
        print(f"  ⚙️ Params: {params:.2f}M | MACs/FLOPs: {macs:.2f}G")
        
        if do_eval:
            metrics = evaluate_model(model, val_loader, device, display_name)
        else:
            metrics = {"Dice_Mean": 0, "Dice_L": 0, "Dice_H": 0, "HD95_Mean": 0, "HD95_L": 0, "HD95_H": 0}
            
        results.append({
            "Model": display_name,
            "Params (M)": f"{params:.2f}",
            "MACs (G)": f"{macs:.2f}" if not np.isnan(macs) else "N/A",
            "Dice (95% CI)": f"{metrics['Dice_Mean']:.3f} ({metrics['Dice_L']:.3f} - {metrics['Dice_H']:.3f})",
            "HD95 (95% CI)": f"{metrics['HD95_Mean']:.2f} ({metrics['HD95_L']:.2f} - {metrics['HD95_H']:.2f})"
        })
        
        del model
        torch.cuda.empty_cache()

    print("\n" + "="*90)
    print("SECONDARY ANALYSIS RESULTS (JOURNAL FORMAT)".center(90))
    print("="*90)
    
    df_res = pd.DataFrame(results)
    
    # Custom format with tabulate for nice console output
    try:
        from tabulate import tabulate
        print(tabulate(df_res, headers='keys', tablefmt='grid', showindex=False))
    except ImportError:
        print(df_res.to_string(index=False))
    print("="*90)
    
    out_csv = os.path.expanduser(f"~/Clara/secondary_analysis_table_{dataset_key}.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\n💾 Saved to CSV: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate secondary analysis table")
    parser.add_argument('--dataset', default='ct', choices=['ct', 'ctc', 'all'],
                        help="Dataset type to evaluate on: 'ct', 'ctc', or 'all'")
    args = parser.parse_args()
    main(args.dataset)
