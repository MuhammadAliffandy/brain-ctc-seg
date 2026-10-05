import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from evaluate_trained_models import SE2_CNNET

def get_valid_path(rel_path):
    candidates = [
        os.path.expanduser(f"~/Clara/{rel_path}"),
        f"/raid/D13K48009/Clara/{rel_path}",
    ]
    for c in candidates:
        if os.path.exists(c): return c
    return candidates[0]

def evaluate_model(model, loader, device, desc="Evaluating"):
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=desc, leave=False):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            
            pf = preds.view(-1)
            mf = masks.view(-1)
            tp += ((pf == 1) & (mf == 1)).sum().item()
            fp += ((pf == 1) & (mf == 0)).sum().item()
            fn += ((pf == 0) & (mf == 1)).sum().item()
            tn += ((pf == 0) & (mf == 0)).sum().item()
            
    eps = 1e-7
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Dice Score": dice,
        "IoU": iou
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SE2_CNNET(n_channels=3, n_classes=2).to(device)
    results = {}
    
    # 1. CECT Dataset (CTC)
    print("\n--- 1. Evaluating Mod-SE(2) on CECT ---")
    try:
        weight_path = get_valid_path("brain-ctc-seg/training/saved_models_25D/se2_unet_ctc_best.pth")
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        
        # Setup CECT loader using train_ablation logic
        from train_ablation import filter_df_by_dataset
        import re
        from generate_remaining_hd95 import PatientVolumeDataset
        CSV_REPORT = get_valid_path("new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
        DATA_PATH = get_valid_path("local_ct_workspace_full")
        df = pd.read_csv(CSV_REPORT)
        pc = 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'
        df = filter_df_by_dataset(df, 'ctc', pc)
        train_df = df.sample(frac=0.85, random_state=42)
        val_df = df.drop(train_df.index)
        val_patients = val_df[pc].unique()
        
        cect_dataset = torch.utils.data.ConcatDataset([
            PatientVolumeDataset(pid, DATA_PATH, n_slices=3) for pid in val_patients
        ])
        cect_loader = DataLoader(cect_dataset, batch_size=16, shuffle=False)
        results["CECT (Table 4)"] = evaluate_model(model, cect_loader, device, "CECT Eval")
    except Exception as e:
        print(f"Failed CECT: {e}")

    # 2. Kaggle Stroke
    print("\n--- 2. Evaluating Mod-SE(2) on Kaggle Stroke ---")
    try:
        weight_path = get_valid_path("brain-ctc-seg/public_dataset/saved_models/Mod-Seg-SE2_kaggle_best.pth")
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "public_dataset"))
        import kagglehub
        from train_all_intra import get_kaggle_splits, IntraKaggleDataset
        root_dir = kagglehub.dataset_download("ozguraslank/brain-stroke-ct-dataset")
        _, test_samples = get_kaggle_splits(root_dir)
        stroke_loader = DataLoader(IntraKaggleDataset(test_samples, transform=None), batch_size=16, shuffle=False)
        results["Kaggle Stroke (Table 5)"] = evaluate_model(model, stroke_loader, device, "Stroke Eval")
    except Exception as e:
        print(f"Failed Kaggle Stroke: {e}")

    # 3. Kaggle Hemorrhage
    print("\n--- 3. Evaluating Mod-SE(2) on Kaggle Hemorrhage ---")
    try:
        weight_path = get_valid_path("brain-ctc-seg/public_dataset/saved_models/Mod-Seg-SE2_kaggle_hemorrhage_best.pth")
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        
        from train_all_intra_hemorrhage import get_kaggle_hemorrhage_splits, IntraHemorrhageDataset
        _, test_samples = get_kaggle_hemorrhage_splits()
        hemo_loader = DataLoader(IntraHemorrhageDataset(test_samples, transform=None), batch_size=16, shuffle=False)
        results["Kaggle Hemorrhage (Table 6)"] = evaluate_model(model, hemo_loader, device, "Hemo Eval")
    except Exception as e:
        print(f"Failed Kaggle Hemorrhage: {e}")

    # Output Results
    print("\n" + "="*50)
    print("🎯 FINAL CLASSIFICATION METRICS FOR Mod-SE(2)")
    print("="*50)
    for ds, mets in results.items():
        print(f"\n{ds}:")
        for k, v in mets.items():
            print(f"- Mod-SE(2) [26] | {k}: {v:.4f}")
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
