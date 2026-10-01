import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, binary_erosion
import re
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from train_ablation import filter_df_by_dataset
from train_comparison_models import StandardUNet
from evaluate_trained_models import SE2_CNNET

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

def surface_distances_3d(result, reference, voxelspacing=(2.5, 0.45, 0.45)):
    result = result.astype(bool)
    reference = reference.astype(bool)
    
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

ABLATION_MODELS = [
    # Table 7: Components
    ("A1 (Base, 2D, No Bound)", "A1_ct_best.pth", False, 1),
    ("A2 (SE2, 2D, No Bound)", "A2_ct_best.pth", True, 1),
    ("A3 (Base, 2.5D, No Bound)", "A3_ct_best.pth", False, 3),
    ("A4 (Base, 2D, Bound)", "A4_ct_best.pth", False, 1),
    ("A5 (SE2, 2.5D, No Bound)", "A5_ct_best.pth", True, 3),
    ("A6 (SE2, 2D, Bound)", "A6_ct_best.pth", True, 1),
    ("A7 (Base, 2.5D, Bound)", "A7_ct_best.pth", False, 3),
    ("A8 (CT-SE2, 2.5D, Bound)", "A8_ct_best.pth", True, 3),
    
    # Table 8: Loss
    ("Loss: Dice Only", "Loss_DiceOnly_ct_best.pth", True, 3),
    ("Loss: Focal+Dice", "Loss_FocalDice_ct_best.pth", True, 3),
    ("Loss: Dice+Bound", "Loss_DiceBound_ct_best.pth", True, 3),
    
    # Table 9: Context
    ("Context: 5 Slices", "Context_5D_ct_best.pth", True, 5),
]

def run_ablation_hd95():
    print("="*60)
    print(" 🔬 GENERATING HD95 FOR ABLATION MODELS (TABLES 7, 8, 9)")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CSV_REPORT = get_valid_path("new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
    DATA_PATH = get_valid_path("local_ct_workspace_full")
    SAVE_DIR = get_valid_path("brain-ctc-seg/training/saved_models_ablation")

    df = pd.read_csv(CSV_REPORT)
    pc = 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'
    df = filter_df_by_dataset(df, 'ct', pc)
    
    train_df = df.sample(frac=0.85, random_state=42)
    val_df = df.drop(train_df.index)
    val_patients = val_df[pc].unique()

    results = []

    for name, weight_name, is_se2, n_slices in ABLATION_MODELS:
        weight_path = os.path.join(SAVE_DIR, weight_name)
        if not os.path.exists(weight_path):
            print(f"⚠️ Missing {weight_name}. Skipping...")
            continue

        if is_se2:
            model = SE2_CNNET(n_channels=n_slices, n_classes=2, N=8, base_channels=32).to(device)
        else:
            model = StandardUNet(n_channels=n_slices, n_classes=2).to(device)

        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.eval()

        hd95_list = []
        with torch.no_grad():
            for patient in tqdm(val_patients, desc=f"Evaluating {name}"):
                dataset = PatientVolumeDataset(patient, DATA_PATH, n_slices=n_slices)
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

                if np.any(vol_masks) and np.any(vol_preds):
                    val = surface_distances_3d(vol_preds, vol_masks)
                    if not np.isnan(val):
                        hd95_list.append(val)

        if len(hd95_list) > 0:
            median_hd95 = np.median(hd95_list)
            print(f"✅ {name} -> HD95: {median_hd95:.2f} mm")
            results.append((name, median_hd95))
        else:
            print(f"❌ {name} -> Failed (NaN)")
            results.append((name, np.nan))

    print("\n" + "="*60)
    print(" 📊 FINAL ABLATION HD95 RESULTS")
    print("="*60)
    for name, val in results:
        print(f"{name:30s} : {val:.2f} mm")

if __name__ == "__main__":
    run_ablation_hd95()
