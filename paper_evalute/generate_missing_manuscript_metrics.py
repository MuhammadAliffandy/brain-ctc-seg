import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, binary_erosion
import albumentations as A

# Adjust sys path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from train_ablation import filter_df_by_dataset
from evaluate_trained_models import SE2_CNNET
from torch.utils.data import Dataset, DataLoader
import re

# ==========================================
# 1. 3D HD95 & VOLUMETRIC METRICS
# ==========================================
def surface_distances_3d(result, reference, voxelspacing=(1., 1., 1.)):
    """Computes HD95 on the full 3D patient volume."""
    res_borders = result ^ binary_erosion(result)
    ref_borders = reference ^ binary_erosion(reference)
    
    if not res_borders.any() or not ref_borders.any():
        return np.nan
        
    dt_ref = distance_transform_edt(~ref_borders, sampling=voxelspacing)
    dt_res = distance_transform_edt(~res_borders, sampling=voxelspacing)
    
    dists = np.concatenate([dt_ref[res_borders], dt_res[ref_borders]])
    if len(dists) == 0:
        return np.nan
    return np.percentile(dists, 95)

# ==========================================
# 2. DATASET LOADER (PATIENT-LEVEL 3D)
# ==========================================
def get_valid_path(rel_path):
    candidates = [
        os.path.expanduser(f"~/Clara/{rel_path}"),
        f"/raid/D13K48009/Clara/{rel_path}",
    ]
    for c in candidates:
        if os.path.exists(c): return c
    return candidates[0]

class PatientVolumeDataset(Dataset):
    """Loads all slices for a single patient to reconstruct the 3D volume."""
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

        # Handle 2.5D stacking
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


# ==========================================
# 3. EVALUATION RUNNER
# ==========================================
def evaluate_manuscript_metrics():
    print("="*60)
    print(" 🚀 RUNNING MASTER MANUSCRIPT METRICS GENERATOR")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CSV_REPORT = get_valid_path("new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
    DATA_PATH = get_valid_path("local_ct_workspace_full")
    MODEL_PATH = get_valid_path("brain-ctc-seg/training/saved_models_ablation/A8_ct_best.pth")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Cannot find A8 model at {MODEL_PATH}")
        return

    # 1. Load A8 Model (Mod-SE(2))
    print("Loading Mod-SE(2) / A8 model...")
    model = SE2_CNNET(n_channels=3, n_classes=2, N=8, base_channels=32).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # 2. Prepare Validation Data (85/15 split)
    df = pd.read_csv(CSV_REPORT)
    pc = 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'
    df = filter_df_by_dataset(df, 'ct', pc)
    
    train_df = df.sample(frac=0.85, random_state=42)
    val_df = df.drop(train_df.index)
    val_patients = val_df[pc].unique()

    print(f"Evaluating on {len(val_patients)} NCCT Validation Patients...")

    # Metrics
    tp = fp = fn = tn = 0
    patient_hd95_list = []

    with torch.no_grad():
        for patient in tqdm(val_patients, desc="Processing 3D Patient Volumes"):
            dataset = PatientVolumeDataset(patient, DATA_PATH, n_slices=3)
            if len(dataset) == 0: continue
            loader = DataLoader(dataset, batch_size=8, shuffle=False)

            vol_preds = []
            vol_masks = []

            for imgs, masks in loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                preds = torch.argmax(F.softmax(logits, 1), 1)
                
                # Classification Metrics (Table 3)
                p_flat = preds.cpu().view(-1).numpy()
                m_flat = masks.cpu().view(-1).numpy()
                
                tp += ((p_flat == 1) & (m_flat == 1)).sum()
                fp += ((p_flat == 1) & (m_flat == 0)).sum()
                fn += ((p_flat == 0) & (m_flat == 1)).sum()
                tn += ((p_flat == 0) & (m_flat == 0)).sum()

                vol_preds.append(preds.cpu().numpy())
                vol_masks.append(masks.squeeze(1).cpu().numpy())

            # Reconstruct 3D Volume for HD95 (Table 10 style)
            vol_preds = np.concatenate(vol_preds, axis=0)
            vol_masks = np.concatenate(vol_masks, axis=0)

            # Assuming Z spacing = 2.5mm, X/Y = 0.5mm
            hd95 = surface_distances_3d(vol_preds, vol_masks, voxelspacing=(2.5, 0.5, 0.5))
            if not np.isnan(hd95):
                patient_hd95_list.append(hd95)

    # Calculate Final Metrics
    eps = 1e-7
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)

    median_hd95 = np.median(patient_hd95_list)
    q1_hd95 = np.percentile(patient_hd95_list, 25)
    q3_hd95 = np.percentile(patient_hd95_list, 75)

    print("\n" + "="*50)
    print(" 🎯 TABLE 3: MOD-SE(2) MISSING CLASSIFICATION METRICS")
    print("="*50)
    print(f" Accuracy  : {accuracy:.4f}")
    print(f" Precision : {precision:.4f}")
    print(f" Recall    : {recall:.4f}")
    print(f" Dice      : {dice:.4f}")
    print(f" IoU       : {iou:.4f}")
    print("\n" + "="*50)
    print(" 📏 3D VOLUMETRIC HD95 (Median [IQR])")
    print("="*50)
    print(f" HD95 per scan : {median_hd95:.2f} mm [{q1_hd95:.2f} - {q3_hd95:.2f}]")
    print("="*50)
    print("Run this on DGX to get your missing Table 3 values and accurate 3D HD95!")

if __name__ == "__main__":
    evaluate_manuscript_metrics()
