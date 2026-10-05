import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, binary_erosion
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from evaluate_trained_models import (
    SE2_CNNET, HarmonicNet, nnUNet, AttentionUNet, TransUNet, StandardUNet,
    filter_df_by_dataset, load_se2_weights
)

def get_valid_path(rel_path):
    candidates = [
        os.path.expanduser(f"~/Clara/{rel_path}"),
        f"/raid/D13K48009/Clara/{rel_path}",
    ]
    for c in candidates:
        if os.path.exists(c): return c
    return candidates[0]

# --- Correct HD95 calculation handling physical spacing and empty masks ---
MAX_PENALTY_MM = 250.0 # Maximum FOV dimension for penalizing missed predictions

def surface_distances_correct(result, reference, voxelspacing):
    result = result.astype(bool)
    reference = reference.astype(bool)
    
    res_borders = result ^ binary_erosion(result)
    ref_borders = reference ^ binary_erosion(reference)
    
    if not res_borders.any():
        if not ref_borders.any():
            return 0.0 # Both empty -> 0 error
        else:
            return MAX_PENALTY_MM # Missed completely
            
    if not ref_borders.any():
        return MAX_PENALTY_MM # False positive on healthy slice
        
    dt_ref = distance_transform_edt(~ref_borders, sampling=voxelspacing)
    dt_res = distance_transform_edt(~res_borders, sampling=voxelspacing)
    
    dists = np.concatenate([dt_ref[res_borders], dt_res[ref_borders]])
    if len(dists) == 0:
        return np.nan
    return np.percentile(dists, 95)

def main():
    parser = argparse.ArgumentParser(description="Fix HD95 Calculation")
    parser.add_argument('--dataset', type=str, choices=['ctc', 'kaggle', 'kaggle_hemorrhage'], required=True)
    args = parser.parse_args()
    ds = args.dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"📏 RECALCULATING CORRECT HD95 METRICS FOR: {ds.upper()}")
    print(f"   (Using correct pixel spacing and handling blank predictions)")
    print(f"{'='*70}\n")

    # 1. Dataset loading
    if ds == 'ctc':
        from generate_remaining_hd95 import PatientVolumeDataset
        CSV_REPORT = get_valid_path("new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
        DATA_PATH = get_valid_path("local_ct_workspace_full")
        df = filter_df_by_dataset(pd.read_csv(CSV_REPORT), 'ctc', 'Patient_Folder' if 'Patient_Folder' in pd.read_csv(CSV_REPORT).columns else 'Patient')
        train_df = df.sample(frac=0.85, random_state=42)
        val_df = df.drop(train_df.index)
        val_patients = val_df['Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'].unique()
        SAVE_DIR = get_valid_path("brain-ctc-seg/training/saved_models_25D")
        is_3d = True
        # For 256x256 resizing, X/Y spacing is roughly ~1.0mm, Z is 2.5mm
        # We'll use 0.976mm which corresponds exactly to 250mm / 256px
        spacing = (2.5, 0.976, 0.976) 
    else:
        from generate_remaining_hd95 import PublicKaggleDataset, PublicHemorrhageDataset
        if ds == 'kaggle':
            import kagglehub
            path = kagglehub.dataset_download("ozguraslank/brain-stroke-ct-dataset")
            test_dataset = PublicKaggleDataset(path)
        else:
            # Need to download via python or assume exist
            # For simplicity let's use the local script logic which we know works
            sys.path.append(os.path.join(os.path.dirname(__file__), "..", "public_dataset"))
            from train_all_intra_hemorrhage import get_kaggle_hemorrhage_splits, IntraHemorrhageDataset
            _, test_samples = get_kaggle_hemorrhage_splits(test_size=0.15, seed=42)
            test_dataset = IntraHemorrhageDataset(test_samples)
            
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        SAVE_DIR = get_valid_path("brain-ctc-seg/public_dataset/saved_models")
        is_3d = False
        spacing = (0.976, 0.976) # ~1mm for 256x256
        
    # 2. Models
    if ds == 'ctc':
        MODELS = [
            ("CT-SE(2)", SE2_CNNET, f"se2_unet_ctc_best.pth", True),
            ("HarmonicNet", HarmonicNet, f"harmonic_net_ctc_best.pth", True),
            ("nnU-Net", nnUNet, f"nn_unet_ctc_best.pth", False),
            ("Standard U-Net", StandardUNet, f"standard_unet_ctc_best.pth", False),
            ("Attention U-Net", AttentionUNet, f"attention_unet_ctc_best.pth", False),
            ("TransUNet", TransUNet, f"trans_unet_ctc_best.pth", False)
        ]
    else:
        # kaggle / kaggle_hemorrhage
        sf = 'kaggle' if ds == 'kaggle' else 'kaggle_hemorrhage'
        MODELS = [
            ("CT-SE(2)", SE2_CNNET, f"Mod-Seg-SE2_{sf}_best.pth", True),
            ("HarmonicNet", HarmonicNet, f"HarmonicNet_{sf}_best.pth", True),
            ("nnU-Net", nnUNet, f"nnU-Net_{sf}_best.pth", False),
            ("Standard U-Net", StandardUNet, f"Standard_U-Net_{sf}_best.pth", False),
            ("Attention U-Net", AttentionUNet, f"Attention_U-Net_{sf}_best.pth", False),
            ("TransUNet", TransUNet, f"TransUNet_{sf}_best.pth", False)
        ]

    # Evaluate
    results = []
    for name, ModelClass, weight_file, is_se2 in MODELS:
        weight_path = os.path.join(SAVE_DIR, weight_file)
        if not os.path.exists(weight_path):
            print(f"⚠️ Missing {weight_file}")
            continue

        model = ModelClass(n_channels=3, n_classes=2, N=8, base_channels=32).to(device) if is_se2 and name != "HarmonicNet" else ModelClass(n_channels=3, n_classes=2, N=4, base_channels=32).to(device) if name == "HarmonicNet" else ModelClass(n_channels=3, n_classes=2).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True), strict=False)
        model.eval()

        hd95_list = []
        with torch.no_grad():
            if is_3d:
                for patient in tqdm(val_patients, desc=f"Eval {name}"):
                    dataset = PatientVolumeDataset(patient, DATA_PATH, n_slices=3)
                    if len(dataset) == 0: continue
                    loader = DataLoader(dataset, batch_size=8, shuffle=False)
                    vol_preds, vol_masks = [], []
                    for imgs, masks in loader:
                        imgs = imgs.to(device)
                        orig_h, orig_w = masks.shape[-2:]
                        if not is_se2: imgs = F.interpolate(imgs, size=(256, 256), mode='bilinear', align_corners=False)
                        logits = model(imgs)
                        preds = torch.argmax(F.softmax(logits, 1), 1)
                        if not is_se2: preds = F.interpolate(preds.unsqueeze(1).float(), size=(orig_h, orig_w), mode='nearest').squeeze(1).long()
                        vol_preds.append(preds.cpu().numpy())
                        vol_masks.append(masks.squeeze(1).cpu().numpy())
                    vol_preds, vol_masks = np.concatenate(vol_preds, axis=0), np.concatenate(vol_masks, axis=0)
                    if np.any(vol_masks): # only score patients that actually have a tumor
                        val = surface_distances_correct(vol_preds, vol_masks, spacing)
                        if not np.isnan(val): hd95_list.append(val)
            else:
                for imgs, masks in tqdm(test_loader, desc=f"Eval {name}"):
                    imgs = imgs.to(device)
                    logits = model(imgs)
                    preds = torch.argmax(F.softmax(logits, 1), 1).cpu().numpy()
                    masks = masks.squeeze(1).cpu().numpy()
                    for i in range(len(preds)):
                        if np.any(masks[i]): # only score images with tumors
                            val = surface_distances_correct(preds[i], masks[i], spacing)
                            if not np.isnan(val): hd95_list.append(val)

        if len(hd95_list) > 0:
            median_hd95 = np.median(hd95_list)
            print(f"✅ {name} -> Corrected HD95: {median_hd95:.2f} mm")
            results.append((name, median_hd95))
        else:
            print(f"❌ {name} -> Failed")
            results.append((name, np.nan))

    print("\n" + "-"*50)
    print("FINAL CORRECTED HD95 METRICS:")
    for name, val in results: print(f"{name:30s} : {val:.2f} mm")

if __name__ == "__main__":
    main()
