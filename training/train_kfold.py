import os
import argparse
import random
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import albumentations as A
from tqdm import tqdm

from train_comparison_models import (
    HarmonicNet, StandardUNet, nnUNet, AttentionUNet, TransUNet, 
    filter_df_by_dataset, CTBrain25DDataset, DiceLoss, FocalLoss, CombinedLoss
)
from train_se2_by_dataset import SE2_CNNET

# ================================================================
# MODEL REGISTRY
# ================================================================
MODEL_REGISTRY = {
    'se2':        (SE2_CNNET,      'Proposed SE(2) Equivariant', 'se2_cnnet'),
    'harmonic':   (HarmonicNet,    'Group-equivariant (C4)', 'harmonic_net'),
    'unet':       (StandardUNet,   'Non group-equivariant',  'standard_unet'),
    'nnunet':     (nnUNet,         'Non group-equivariant',  'nn_unet'),
    'attention':  (AttentionUNet,  'Non group-equivariant',  'attention_unet'),
    'transunet':  (TransUNet,      'Non group-equivariant',  'trans_unet'),
}

# ================================================================
# KFOLD EVALUATION FUNCTION
# ================================================================
def evaluate_fold(model, val_loader, device):
    model.eval()
    tp = fp = fn = tn = 0
    eps = 1e-7
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                logits = model(images)
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            
            pf = preds.view(-1)
            mf = masks.view(-1)
            
            tp += ((pf == 1) & (mf == 1)).sum().item()
            fp += ((pf == 1) & (mf == 0)).sum().item()
            fn += ((pf == 0) & (mf == 1)).sum().item()
            tn += ((pf == 0) & (mf == 0)).sum().item()
            
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Dice": dice,
        "IoU": iou
    }

# ================================================================
# KFOLD MAIN PIPELINE
# ================================================================
def run_kfold(model_key: str, dataset_key: str, k_folds: int = 5):
    ModelClass, model_type, save_name = MODEL_REGISTRY[model_key]
    
    CSV_REPORT  = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
    DATA_PATH   = os.path.expanduser("~/Clara/local_ct_workspace_full")
    SAVE_DIR    = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_kfold")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    use_standard_pipeline = (model_key != 'se2')
    
    if use_standard_pipeline:
        LR=1e-4; BATCH=8; ACCUM=4; EPOCHS=100
    else:
        LR=1e-4; BATCH=8; ACCUM=4; EPOCHS=150
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*65}")
    print(f"🚀 K-FOLD CROSS VALIDATION: {ModelClass.__name__} ({model_type})")
    print(f"Dataset: {dataset_key.upper()} | Folds: {k_folds} | Epochs: {EPOCHS} | Device: {device}")
    print(f"{'='*65}\n")
    
    # 1. Load Data
    df = pd.read_csv(CSV_REPORT)
    pc = 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'
    df = filter_df_by_dataset(df, dataset_key, pc)
    print(f"📊 Dataset '{dataset_key}': {len(df)} patients found.")
    
    if len(df) < k_folds:
        print(f"❌ Jumlah pasien ({len(df)}) lebih sedikit dari K-Folds ({k_folds})!"); return
        
    patients = df[pc].values
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    aug_train = A.Compose([
        A.Affine(scale=(0.9,1.1), translate_percent=(-0.06,0.06), rotate=(-15,15), p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
    ])
    
    fold_results = []
    
    # 2. Iterate Over Folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(patients)):
        print(f"\n" + "-"*50)
        print(f"🔁 FOLD {fold+1}/{k_folds}")
        print("-" * 50)
        
        train_patients = patients[train_idx]
        val_patients = patients[val_idx]
        
        train_df = df[df[pc].isin(train_patients)]
        val_df = df[df[pc].isin(val_patients)]
        
        train_dataset = CTBrain25DDataset(DATA_PATH, train_df, transform=aug_train)
        val_dataset = CTBrain25DDataset(DATA_PATH, val_df, transform=None)
        
        nw = min(os.cpu_count() or 4, 8)
        train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=nw, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=nw, pin_memory=True)
        
        # Inisialisasi Model Baru per Fold
        if model_key == 'se2':
            model = ModelClass(n_channels=3, n_classes=2).to(device)
            criterion = CombinedLoss(class_weights=torch.tensor([1.0, 10.0], device=device)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        else:
            model = ModelClass(n_channels=3, n_classes=2).to(device)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0], device=device)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            scheduler = None
            
        scaler = torch.amp.GradScaler('cuda')
        best_dice = 0.0
        fold_save_path = os.path.join(SAVE_DIR, f"{save_name}_{dataset_key}_fold{fold+1}.pth")
        
        # 3. Training Loop
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Fold {fold+1}]", leave=False)
            for i, (imgs, masks) in enumerate(pbar):
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    logits = model(imgs)
                    loss = criterion(logits, masks) / ACCUM
                    
                scaler.scale(loss).backward()
                
                if (i + 1) % ACCUM == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                running_loss += loss.item() * ACCUM
                pbar.set_postfix({'loss': f"{loss.item() * ACCUM:.4f}"})
                
            # Validation at end of epoch
            val_metrics = evaluate_fold(model, val_loader, device)
            epoch_dice = val_metrics["Dice"]
            
            if epoch_dice > best_dice:
                best_dice = epoch_dice
                torch.save(model.state_dict(), fold_save_path)
                
            if scheduler:
                scheduler.step(epoch_dice)
                
        # 4. Final Evaluation of Best Model for this Fold
        print(f"✅ Fold {fold+1} selesai. Best Val Dice: {best_dice:.4f}")
        model.load_state_dict(torch.load(fold_save_path))
        final_fold_metrics = evaluate_fold(model, val_loader, device)
        final_fold_metrics["Fold"] = fold + 1
        final_fold_metrics["Model"] = model_key
        final_fold_metrics["Dataset"] = dataset_key
        fold_results.append(final_fold_metrics)
        
    # 5. Summarize Results across Folds
    df_results = pd.DataFrame(fold_results)
    mean_metrics = df_results.mean(numeric_only=True)
    std_metrics = df_results.std(numeric_only=True)
    
    print("\n" + "="*60)
    print(f"📊 FINAL K-FOLD RESULTS: {ModelClass.__name__} di {dataset_key.upper()}")
    print("="*60)
    for col in ['Accuracy', 'Precision', 'Recall', 'Dice', 'IoU']:
        print(f"{col:>10}: {mean_metrics[col]:.4f} ± {std_metrics[col]:.4f}")
        
    # Append to Master CSV
    MASTER_CSV = os.path.join(SAVE_DIR, "master_kfold_results.csv")
    
    summary_dict = {
        "Model": model_key,
        "Dataset": dataset_key,
        "Folds": k_folds
    }
    for col in ['Accuracy', 'Precision', 'Recall', 'Dice', 'IoU']:
        summary_dict[f"{col}_Mean"] = mean_metrics[col]
        summary_dict[f"{col}_Std"] = std_metrics[col]
        
    summary_df = pd.DataFrame([summary_dict])
    
    if os.path.exists(MASTER_CSV):
        summary_df.to_csv(MASTER_CSV, mode='a', header=False, index=False)
    else:
        summary_df.to_csv(MASTER_CSV, index=False)
        
    print(f"\n✅ Hasil direkam di: {MASTER_CSV}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="K-Fold Cross Validation Pipeline")
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_REGISTRY.keys()), help="Model to train")
    parser.add_argument('--dataset', type=str, required=True, choices=['ct', 'ctc', 'all'], help="Dataset to use")
    parser.add_argument('--folds', type=int, default=5, help="Number of Folds")
    
    args = parser.parse_args()
    run_kfold(args.model, args.dataset, args.folds)
