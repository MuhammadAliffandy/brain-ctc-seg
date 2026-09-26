"""
evaluate_lesion_tracking.py
===========================
Quantitatively evaluates 3D lesion tracking continuity for the IEEE TMI revision.
Calculates Lesion-Level Sensitivity, False Positives per Scan, Fragmentation Rate, and Merging Rate.
"""

import os
import sys
import re
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import scipy.ndimage

# Adjust sys path so we can import SE2 model
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from train_comparison_models import StandardUNet, HarmonicNet, nnUNet, AttentionUNet, TransUNet
from evaluate_trained_models import SE2_CNNET, load_se2_weights
from train_ablation import filter_df_by_dataset

def calculate_lesion_metrics(gt_vol, pred_vol, min_overlap=0.1):
    """
    Evaluates 3D lesion matching between ground truth and predictions.
    """
    gt_labeled, n_gt = scipy.ndimage.label(gt_vol)
    pred_labeled, n_pred = scipy.ndimage.label(pred_vol)
    
    # Matching matrices
    # overlap_matrix[i, j] = pixels shared between GT lesion i and Pred lesion j
    overlap_matrix = np.zeros((n_gt + 1, n_pred + 1), dtype=np.int32)
    
    if n_gt > 0 and n_pred > 0:
        # Fast histogram-based intersection counting
        intersection = np.histogram2d(
            gt_labeled.flatten(), pred_labeled.flatten(), 
            bins=(n_gt+1, n_pred+1), 
            range=((0, n_gt+1), (0, n_pred+1))
        )[0]
        overlap_matrix = intersection

    # Calculate IoU for matching
    # iou[i, j] = intersection / (area_gt + area_pred - intersection)
    gt_areas = np.sum(overlap_matrix, axis=1)
    pred_areas = np.sum(overlap_matrix, axis=0)
    
    # We ignore index 0 (background)
    metrics = {
        'total_gt_lesions': n_gt,
        'total_pred_lesions': n_pred,
        'tp_lesions': 0,
        'fp_lesions': 0,
        'fn_lesions': 0,
        'fragmentation_events': 0, # One GT matched by multiple Preds
        'merging_events': 0,       # One Pred matches multiple GTs
    }
    
    gt_matched = np.zeros(n_gt + 1, dtype=bool)
    pred_matched = np.zeros(n_pred + 1, dtype=bool)
    
    for i in range(1, n_gt + 1):
        matches = 0
        for j in range(1, n_pred + 1):
            intersect = overlap_matrix[i, j]
            if intersect > 0:
                iou = intersect / (gt_areas[i] + pred_areas[j] - intersect)
                if iou >= min_overlap:
                    gt_matched[i] = True
                    pred_matched[j] = True
                    matches += 1
        
        if matches > 1:
            metrics['fragmentation_events'] += 1
            
    for j in range(1, n_pred + 1):
        matches = 0
        for i in range(1, n_gt + 1):
            intersect = overlap_matrix[i, j]
            if intersect > 0:
                iou = intersect / (gt_areas[i] + pred_areas[j] - intersect)
                if iou >= min_overlap:
                    matches += 1
        if matches > 1:
            metrics['merging_events'] += 1
            
    metrics['tp_lesions'] = np.sum(gt_matched[1:])
    metrics['fn_lesions'] = n_gt - metrics['tp_lesions']
    metrics['fp_lesions'] = n_pred - np.sum(pred_matched[1:])
    
    return metrics

def evaluate_tracking(model_name, ModelClass, weights_path, is_se2, n_slices, df, DATA_PATH, device):
    if is_se2:
        model = ModelClass(n_slices, 2, 8, 32).to(device)
    else:
        model = ModelClass(n_slices, 2).to(device)
        
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except Exception as e:
        print(f"⚠️ Could not load {weights_path}: {e}")
        return None
        
    model.eval()
    
    total_metrics = {
        'total_gt_lesions': 0,
        'total_pred_lesions': 0,
        'tp_lesions': 0,
        'fp_lesions': 0,
        'fn_lesions': 0,
        'fragmentation_events': 0,
        'merging_events': 0,
        'total_scans': 0
    }
    
    patients = df['Patient_Folder'].unique()
    
    for p in tqdm(patients, desc=f"Eval {model_name}", ncols=80):
        pd_dir = os.path.join(DATA_PATH, p)
        if not os.path.exists(pd_dir): continue
        
        imgs = sorted([f for f in os.listdir(pd_dir) if f.endswith('_img.npy')],
                      key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0)
        
        if not imgs: continue
        
        volume_pred = []
        volume_gt = []
        
        with torch.no_grad():
            for i in range(len(imgs)):
                ip = os.path.join(pd_dir, imgs[i])
                mp = ip.replace('_img.npy', '_mask.npy')
                
                if not os.path.exists(mp): continue
                
                if n_slices == 1:
                    indices = [i] # Strictly 1 channel for 1-slice ablations
                elif n_slices == 3:
                    indices = [max(0, i-1), i, min(len(imgs)-1, i+1)]
                elif n_slices == 5:
                    indices = [max(0, i-2), max(0, i-1), i, min(len(imgs)-1, i+1), min(len(imgs)-1, i+2)]
                    
                slices_data = [np.load(os.path.join(pd_dir, imgs[idx])).astype(np.float32) for idx in indices]
                
                m = np.load(mp).astype(np.uint8)
                if m.max() > 1: m = (m > 0).astype(np.uint8)
                
                img_stack = np.stack(slices_data, axis=-1)
                if img_stack.max() > img_stack.min():
                    img_stack = (img_stack - img_stack.min()) / (img_stack.max() - img_stack.min())
                
                tensor = torch.from_numpy(img_stack).permute(2,0,1).unsqueeze(0).to(device)
                
                with torch.amp.autocast('cuda'):
                    logits = model(tensor)
                
                pred = torch.argmax(F.softmax(logits, 1), 1).squeeze(0).cpu().numpy()
                
                volume_pred.append(pred)
                volume_gt.append(m)
                
        volume_pred = np.stack(volume_pred, axis=0)
        volume_gt = np.stack(volume_gt, axis=0)
        
        metrics = calculate_lesion_metrics(volume_gt, volume_pred)
        for k in metrics:
            total_metrics[k] += metrics[k]
        total_metrics['total_scans'] += 1

    # Summarize
    sens = total_metrics['tp_lesions'] / (total_metrics['total_gt_lesions'] + 1e-7)
    fp_per_scan = total_metrics['fp_lesions'] / (total_metrics['total_scans'] + 1e-7)
    
    return {
        'Model': model_name,
        'Lesion Sensitivity': round(sens, 4),
        'FP Lesions / Scan': round(fp_per_scan, 4),
        'Fragmentation Events': total_metrics['fragmentation_events'],
        'Merging Events': total_metrics['merging_events'],
        'Total Scans': total_metrics['total_scans']
    }


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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CSV_REPORT = get_valid_path("new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
    DATA_PATH  = get_valid_path("local_ct_workspace_full")
    SAVE_DIR   = get_valid_path("brain-ctc-seg/training/saved_models_ablation")
    
    df = pd.read_csv(CSV_REPORT)
    pc = 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'
    # Use standard test set (CT only for lesion tracking)
    df = filter_df_by_dataset(df, 'ct', pc)
    train_df = df.sample(frac=0.85, random_state=42)
    val_df   = df.drop(train_df.index)
    
    # Evaluate Table 3 models (Context Ablations)
    MODELS = [
        ("1 Slice (2D) [A6]", SE2_CNNET, "A6_ct_best.pth", True, 1),
        ("3 Slices (2.5D) [A8]", SE2_CNNET, "A8_ct_best.pth", True, 3),
        ("5 Slices (Extended) [Context_5D]", SE2_CNNET, "Context_5D_ct_best.pth", True, 5),
    ]
    
    out_path = os.path.join(os.path.dirname(__file__), "lesion_tracking_metrics.csv")
    done_models = []
    
    if os.path.exists(out_path):
        df_existing = pd.read_csv(out_path)
        if 'Model' in df_existing.columns:
            done_models = df_existing['Model'].tolist()
            print(f"🔄 Found existing results for: {done_models}")
    
    for name, ModelClass, weight_name, is_se2, n_slices in MODELS:
        if name in done_models:
            print(f"⏭️ Skipping {name}, already evaluated.")
            continue
            
        weight_path = os.path.join(SAVE_DIR, weight_name)
        if not os.path.exists(weight_path):
            print(f"⚠️ Missing {weight_path}, skipping...")
            continue
            
        res = evaluate_tracking(name, ModelClass, weight_path, is_se2, n_slices, val_df, DATA_PATH, device)
        if res:
            df_res = pd.DataFrame([res])
            df_res.to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)
            print(f"✅ Saved {name} to {out_path}")
            
    print(f"\n🎉 All available models evaluated. Results at {out_path}")
