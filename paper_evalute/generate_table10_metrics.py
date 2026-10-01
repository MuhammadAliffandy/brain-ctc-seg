import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import scipy.ndimage
from pingouin import intraclass_corr

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from train_ablation import filter_df_by_dataset
from evaluate_trained_models import SE2_CNNET
from torch.utils.data import Dataset, DataLoader
import re

def get_valid_path(rel_path):
    candidates = [
        os.path.expanduser(f"~/Clara/{rel_path}"),
        f"/raid/D13K48009/Clara/{rel_path}",
    ]
    for c in candidates:
        if os.path.exists(c): return c
    return candidates[0]

class PatientVolumeDataset(Dataset):
    def __init__(self, patient_id, root_dir, n_slices=3):
        self.patient_dir = os.path.join(root_dir, patient_id)
        self.n_slices = n_slices
        self.slices = []
        if os.path.exists(self.patient_dir):
            img_files = sorted(
                [f for f in os.listdir(self.patient_dir) if f.endswith('_img.npy')],
                key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0
            )
            for img_name in img_files:
                img_path = os.path.join(self.patient_dir, img_name)
                mask_path = img_path.replace('_img.npy', '_mask.npy')
                if os.path.exists(mask_path):
                    self.slices.append((img_path, mask_path))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img_path, mask_path = self.slices[idx]
        img = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        if self.n_slices > 1:
            stack = []
            half = self.n_slices // 2
            for offset in range(-half, half + 1):
                neighbor_idx = max(0, min(idx + offset, len(self.slices) - 1))
                n_img_path, _ = self.slices[neighbor_idx]
                n_img = np.load(n_img_path).astype(np.float32)
                stack.append(n_img)
            img = np.stack(stack, axis=0)
        else:
            img = np.expand_dims(img, axis=0)

        mask = np.expand_dims(mask, axis=0)
        return torch.from_numpy(img), torch.from_numpy(mask)

def calculate_lesion_metrics_full(gt_vol, pred_vol, voxel_vol_ml):
    gt_labeled, n_gt = scipy.ndimage.label(gt_vol)
    pred_labeled, n_pred = scipy.ndimage.label(pred_vol)
    
    overlap_matrix = np.zeros((n_gt + 1, n_pred + 1), dtype=np.int32)
    if n_gt > 0 and n_pred > 0:
        intersection = np.histogram2d(
            gt_labeled.flatten(), pred_labeled.flatten(), 
            bins=(n_gt+1, n_pred+1), range=((0, n_gt+1), (0, n_pred+1))
        )[0]
        overlap_matrix = intersection

    gt_areas = np.sum(overlap_matrix, axis=1)
    pred_areas = np.sum(overlap_matrix, axis=0)
    
    metrics = {
        'total_gt_lesions': n_gt,
        'total_pred_lesions': n_pred,
        'tp_lesions': 0,
        'fp_lesions': 0,
        'fn_lesions': 0,
        'fragmented_tracks': 0,
        'merged_tracks': 0,
        'volumes_gt': [],
        'volumes_pred': [],
        'abs_errors': []
    }
    
    gt_matched = np.zeros(n_gt + 1, dtype=bool)
    pred_matched = np.zeros(n_pred + 1, dtype=bool)
    
    for i in range(1, n_gt + 1):
        matches = 0
        best_iou = 0
        best_pred_idx = -1
        for j in range(1, n_pred + 1):
            intersect = overlap_matrix[i, j]
            if intersect > 0:
                iou = intersect / (gt_areas[i] + pred_areas[j] - intersect)
                if iou >= 0.1:
                    gt_matched[i] = True
                    pred_matched[j] = True
                    matches += 1
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = j
        
        if matches > 1:
            metrics['fragmented_tracks'] += 1
            
        vol_gt = gt_areas[i] * voxel_vol_ml
        metrics['volumes_gt'].append(vol_gt)
        if best_pred_idx != -1:
            vol_pred = pred_areas[best_pred_idx] * voxel_vol_ml
            metrics['volumes_pred'].append(vol_pred)
            metrics['abs_errors'].append(abs(vol_gt - vol_pred))
        else:
            metrics['volumes_pred'].append(0.0)
            metrics['abs_errors'].append(vol_gt)
            
    for j in range(1, n_pred + 1):
        matches = 0
        for i in range(1, n_gt + 1):
            intersect = overlap_matrix[i, j]
            if intersect > 0:
                iou = intersect / (gt_areas[i] + pred_areas[j] - intersect)
                if iou >= 0.1:
                    matches += 1
        if matches > 1:
            metrics['merged_tracks'] += 1
            
    metrics['tp_lesions'] = np.sum(gt_matched[1:])
    metrics['fn_lesions'] = n_gt - metrics['tp_lesions']
    metrics['fp_lesions'] = n_pred - np.sum(pred_matched[1:])
    
    return metrics

def run_table10_evaluation():
    print("="*60)
    print(" 📊 GENERATING TABLE 10: FULL LESION TRACKING & ICC")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CSV_REPORT = get_valid_path("new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
    DATA_PATH = get_valid_path("local_ct_workspace_full")
    MODEL_PATH = get_valid_path("brain-ctc-seg/training/saved_models_ablation/A8_ct_best.pth")

    model = SE2_CNNET(n_channels=3, n_classes=2, N=8, base_channels=32).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    df = pd.read_csv(CSV_REPORT)
    pc = 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'
    df = filter_df_by_dataset(df, 'ct', pc)
    
    train_df = df.sample(frac=0.85, random_state=42)
    val_df = df.drop(train_df.index)
    val_patients = val_df[pc].unique()

    # Voxel volume in mL (cm^3). Assuming 0.5mm x 0.5mm x 2.5mm = 0.625 mm^3 = 0.000625 mL
    voxel_vol_ml = 0.000625 
    
    total_metrics = {
        'total_gt': 0, 'total_pred': 0, 'tp': 0, 'fp': 0, 'fn': 0,
        'frag': 0, 'merge': 0, 'scans': 0
    }
    all_vol_gt = []
    all_vol_pred = []
    all_abs_err = []

    with torch.no_grad():
        for patient in tqdm(val_patients, desc="Processing Scans"):
            dataset = PatientVolumeDataset(patient, DATA_PATH, n_slices=3)
            if len(dataset) == 0: continue
            loader = DataLoader(dataset, batch_size=8, shuffle=False)

            vol_preds = []
            vol_masks = []
            for imgs, masks in loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                preds = torch.argmax(F.softmax(logits, 1), 1)
                vol_preds.append(preds.cpu().numpy())
                vol_masks.append(masks.squeeze(1).cpu().numpy())

            vol_preds = np.concatenate(vol_preds, axis=0)
            vol_masks = np.concatenate(vol_masks, axis=0)

            m = calculate_lesion_metrics_full(vol_masks, vol_preds, voxel_vol_ml)
            total_metrics['total_gt'] += m['total_gt_lesions']
            total_metrics['total_pred'] += m['total_pred_lesions']
            total_metrics['tp'] += m['tp_lesions']
            total_metrics['fp'] += m['fp_lesions']
            total_metrics['fn'] += m['fn_lesions']
            total_metrics['frag'] += m['fragmented_tracks']
            total_metrics['merge'] += m['merged_tracks']
            total_metrics['scans'] += 1
            
            all_vol_gt.extend(m['volumes_gt'])
            all_vol_pred.extend(m['volumes_pred'])
            all_abs_err.extend(m['abs_errors'])

    # Calculations
    eps = 1e-7
    sens = total_metrics['tp'] / (total_metrics['total_gt'] + eps)
    prec = total_metrics['tp'] / (total_metrics['total_pred'] + eps)
    f1 = 2 * (prec * sens) / (prec + sens + eps)
    fp_per_scan = total_metrics['fp'] / (total_metrics['scans'] + eps)
    
    frag_pct = total_metrics['frag'] / (total_metrics['total_gt'] + eps)
    merge_pct = total_metrics['merge'] / (total_metrics['total_gt'] + eps)
    one_to_one_pct = (total_metrics['tp'] - total_metrics['frag'] - total_metrics['merge']) / (total_metrics['total_gt'] + eps)
    
    # ICC Calculation using Pingouin
    icc_val = "N/A"
    try:
        df_icc = pd.DataFrame({
            'Target': np.tile(np.arange(len(all_vol_gt)), 2),
            'Rater': np.repeat(['GT', 'Pred'], len(all_vol_gt)),
            'Score': np.concatenate([all_vol_gt, all_vol_pred])
        })
        icc_res = intraclass_corr(data=df_icc, targets='Target', raters='Rater', ratings='Score')
        icc_val = icc_res.set_index('Type').loc['ICC2', 'ICC']
    except Exception as e:
        icc_val = f"Failed (pip install pingouin required)"

    print("\n" + "="*50)
    print(" 📈 TABLE 10: MOD-SE(2) LESION TRACKING METRICS")
    print("="*50)
    print(f" Total Scans                   : {total_metrics['scans']}")
    print(f" Total Reference Lesions       : {total_metrics['total_gt']}")
    print(f" Lesion-wise Sensitivity       : {sens*100:.2f}%")
    print(f" Lesion-wise Precision         : {prec*100:.2f}%")
    print(f" Lesion-wise F1 Score          : {f1:.4f}")
    print(f" False-positive lesions/scan   : {fp_per_scan:.2f}")
    print(f" Ref tracks recovered as 1     : {max(0, one_to_one_pct)*100:.2f}%")
    print(f" Fragmented tracks             : {frag_pct*100:.2f}%")
    print(f" Merged tracks                 : {merge_pct*100:.2f}%")
    print(f" Volume ICC(2,1)               : {icc_val}")
    print("="*50)
    print("Run this to complete Table 10!")

if __name__ == "__main__":
    run_table10_evaluation()
