"""
evaluate_boundary_metrics.py
============================
Evaluates all trained ablation models on the validation set.
Computes Dice, IoU, HD95, and ASSD, and exports the results to CSVs.
"""

import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, binary_erosion
import albumentations as A

# Adjust sys path so we can import models and dataset from training directory
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from train_ablation import CTBrainAblationDataset, filter_df_by_dataset
from train_comparison_models import StandardUNet
from evaluate_trained_models import SE2_CNNET

def surface_distances(result, reference, voxelspacing=(1., 1.)):
    """
    Computes HD95 and ASSD between two binary masks using distance_transform_edt.
    """
    res_borders = result ^ binary_erosion(result)
    ref_borders = reference ^ binary_erosion(reference)
    
    if not res_borders.any() or not ref_borders.any():
        return np.nan, np.nan
        
    dt_ref = distance_transform_edt(~ref_borders, sampling=voxelspacing)
    dt_res = distance_transform_edt(~res_borders, sampling=voxelspacing)
    
    dists = np.concatenate([dt_ref[res_borders], dt_res[ref_borders]])
    
    hd95 = np.percentile(dists, 95)
    assd = np.mean(dists)
    return hd95, assd

def surface_dice_np(p, g, tolerance_mm=2.0, spacing=(1.,1.)):
    p = p.astype(bool); g = g.astype(bool)
    if not p.any() and not g.any(): return 1.0
    if not p.any() or not g.any(): return 0.0
    ps = p ^ binary_erosion(p); gs = g ^ binary_erosion(g)
    dtg = distance_transform_edt(~gs, sampling=spacing)
    dtp = distance_transform_edt(~ps, sampling=spacing)
    good = (dtg[ps] <= tolerance_mm).sum() + (dtp[gs] <= tolerance_mm).sum()
    den = ps.sum() + gs.sum()
    return float(good/(den+1e-7))


def get_valid_path(rel_path):
    candidates = [
        os.path.expanduser(f"~/Clara/{rel_path}"),
        f"/raid/D13K48009/Clara/{rel_path}",
        os.path.expanduser(f"~/raid/Clara/{rel_path}")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def evaluate_model(model_path, dataset_key, is_se2, n_slices):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if is_se2:
        model = SE2_CNNET(n_channels=n_slices, n_classes=2, N=8, base_channels=32).to(device)
    else:
        model = StandardUNet(n_channels=n_slices, n_classes=2).to(device)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    CSV_REPORT = get_valid_path("new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
    DATA_PATH  = get_valid_path("local_ct_workspace_full")

    df = pd.read_csv(CSV_REPORT)
    pc = 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'
    df = filter_df_by_dataset(df, dataset_key, pc)
    
    # We only evaluate on the validation set!
    train_df = df.sample(frac=0.85, random_state=42)
    val_df   = df.drop(train_df.index)

    val_transform = A.Compose([
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8,8), p=1.0) if dataset_key == 'ct' else A.NoOp(),
    ])

    val_set = CTBrainAblationDataset(val_df, DATA_PATH, n_slices=n_slices, transform=val_transform)
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    tp = fp = fn = 0
    all_hd95 = []
    all_assd = []
    all_sdice = []

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"Evaluating {os.path.basename(model_path)}", ncols=80):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
            
            preds = torch.argmax(F.softmax(logits, 1), 1)
            
            # Global accumulation for Dice/IoU
            pf = preds.view(-1); mf = masks.view(-1)
            tp += ((pf==1)&(mf==1)).sum().item()
            fp += ((pf==1)&(mf==0)).sum().item()
            fn += ((pf==0)&(mf==1)).sum().item()

            # Batch-wise boundary metrics
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            for b in range(preds_np.shape[0]):
                p_bin = preds_np[b]
                m_bin = masks_np[b]
                
                # Only compute if mask has positive pixels (otherwise HD95 is undefined)
                if m_bin.sum() > 0:
                    hd95, assd = surface_distances(p_bin, m_bin)
                    s_dice = surface_dice_np(p_bin, m_bin)
                    if not np.isnan(hd95):
                        all_hd95.append(hd95)
                        all_assd.append(assd)
                        all_sdice.append(s_dice)

    eps = 1e-7
    iou  = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    
    avg_hd95 = np.nanmean(all_hd95) if len(all_hd95) > 0 else np.nan
    avg_assd = np.nanmean(all_assd) if len(all_assd) > 0 else np.nan
    avg_sdice = np.nanmean(all_sdice) if len(all_sdice) > 0 else np.nan

    return {
        'Dice': dice,
        'IoU': iou,
        'HD95': avg_hd95,
        'ASSD': avg_assd,
        'Surface Dice': avg_sdice
    }

if __name__ == '__main__':
    # Map out the experiments
    experiments = [
        {"variant": "A1", "se2": 0, "slices": 1},
        {"variant": "A2", "se2": 1, "slices": 1},
        {"variant": "A3", "se2": 0, "slices": 3},
        {"variant": "A4", "se2": 0, "slices": 1},
        {"variant": "A5", "se2": 1, "slices": 3},
        {"variant": "A6", "se2": 1, "slices": 1},
        {"variant": "A7", "se2": 0, "slices": 3},
        {"variant": "A8", "se2": 1, "slices": 3},
        
        {"variant": "Loss_DiceOnly", "se2": 1, "slices": 3},
        {"variant": "Loss_FocalDice", "se2": 1, "slices": 3},
        {"variant": "Loss_DiceBound", "se2": 1, "slices": 3},
        
        {"variant": "Context_5D", "se2": 1, "slices": 5},
    ]

    SAVE_DIR = get_valid_path("brain-ctc-seg/training/saved_models_ablation")
    dataset = 'ct'
    
    results = []
    
    for exp in experiments:
        model_path = os.path.join(SAVE_DIR, f"{exp['variant']}_{dataset}_best.pth")
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found: {model_path} (Skipping)")
            continue
            
        metrics = evaluate_model(model_path, dataset, exp['se2'] == 1, exp['slices'])
        
        res = {
            'Variant': exp['variant'],
            'Dice': round(metrics['Dice'], 4),
            'IoU': round(metrics['IoU'], 4),
            'HD95': round(metrics['HD95'], 4),
            'ASSD': round(metrics['ASSD'], 4),
            'Surface Dice': round(metrics['Surface Dice'], 4)
        }
        results.append(res)
        print(f"✅ {exp['variant']}: Dice={res['Dice']}, HD95={res['HD95']}, ASSD={res['ASSD']}, Surface Dice={res['Surface Dice']}")

    if results:
        df_results = pd.DataFrame(results)
        csv_out = os.path.join(os.path.dirname(__file__), "ablation_metrics.csv")
        df_results.to_csv(csv_out, index=False)
        print(f"\\n🎉 Saved all metrics to {csv_out}")
    else:
        print("No models evaluated. Run the training script first!")
