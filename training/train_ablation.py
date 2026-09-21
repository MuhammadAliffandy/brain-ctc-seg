"""
train_ablation.py
=================
Unified script to run all component ablations requested by the IEEE TMI reviewer.
Supports switching SE(2) on/off, changing 2.5D slices (1, 3, 5), and switching loss functions.

Usage example:
  python train_ablation.py --variant A1 --dataset ct --se2 0 --slices 1 --loss focal_dice
"""

import os
import sys
import numpy as np
import random
import argparse
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A

# Adjust sys path so we can import SE2 model from src if needed
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from train_comparison_models import StandardUNet, FocalLoss, DiceLoss, EdgeBoundaryLoss

try:
    from train_se2_by_dataset import SE2_CNNET
except ImportError:
    # Fallback import if it's in another file
    from evaluate_trained_models import SE2_CNNET


class CombinedAblationLoss(nn.Module):
    def __init__(self, loss_type='focal_dice_boundary', class_weights=None):
        super().__init__()
        self.loss_type = loss_type
        self.f = FocalLoss(alpha=0.75, gamma=3.0)
        self.d = DiceLoss()
        self.e = EdgeBoundaryLoss(class_weights=class_weights)
        
    def forward(self, l, t):
        if self.loss_type == 'dice':
            return self.d(l, t)
        elif self.loss_type == 'focal_dice':
            return 0.5 * self.f(l, t) + 2.0 * self.d(l, t)
        elif self.loss_type == 'dice_boundary':
            return 2.0 * self.d(l, t) + 0.5 * self.e(l, t)
        elif self.loss_type == 'focal_dice_boundary':
            return 0.5 * self.f(l, t) + 2.0 * self.d(l, t) + 0.5 * self.e(l, t)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class CTBrainAblationDataset(Dataset):
    def __init__(self, dataframe, root_dir, n_slices=3, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.n_slices = n_slices
        self.patient_slices = {}
        self.all_samples = []
        
        pc = 'Patient_Folder' if 'Patient_Folder' in dataframe.columns else 'Patient'
        for p in dataframe[pc].unique():
            pd_ = os.path.join(root_dir, p)
            if not os.path.exists(pd_): continue
            imgs = sorted([f for f in os.listdir(pd_) if f.endswith('_img.npy')],
                          key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0)
            pairs = []
            for n in imgs:
                ip = os.path.join(pd_, n)
                mp = ip.replace('_img.npy', '_mask.npy')
                if os.path.exists(mp): pairs.append((ip, mp))
            if pairs:
                self.patient_slices[p] = pairs
                for i in range(len(pairs)): self.all_samples.append((p, i))

    def __len__(self): return len(self.all_samples)

    def __getitem__(self, idx):
        p, si = self.all_samples[idx]
        sl = self.patient_slices[p]
        
        if self.n_slices == 1:
            indices = [si, si, si] # Duplicate to keep 3 channels for standard RGB-like models
        elif self.n_slices == 3:
            indices = [max(0, si-1), si, min(len(sl)-1, si+1)]
        elif self.n_slices == 5:
            # If model strictly expects 3 channels, 5 slices will crash unless we change model input channels.
            # Wait, SE2_CNNET and StandardUNet are initialized with n_channels! 
            # We must return exactly self.n_slices channels.
            indices = [max(0, si-2), max(0, si-1), si, min(len(sl)-1, si+1), min(len(sl)-1, si+2)]
            
        try:
            slices_data = [np.load(sl[i][0]).astype(np.float32) for i in indices]
            m = np.load(sl[si][1]).astype(np.uint8)
            if m.max() > 1: m = (m > 0).astype(np.uint8)
            img = np.stack(slices_data, axis=-1)
            
            # NORMALIZATION FIX: Min-Max scale to [0, 1]
            if img.max() > img.min():
                img = (img - img.min()) / (img.max() - img.min())
                
            if self.transform:
                aug = self.transform(image=img, mask=m)
                img = aug['image']
                m = aug['mask']
                
            return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(m).long()
        except Exception as e:
            print(f"Error loading {p} slice {si}: {e}")
            return self.__getitem__(random.randint(0, len(self.all_samples)-1))


def filter_df_by_dataset(df, dataset_key, patient_col='Patient_Folder'):
    if dataset_key == 'ct':
        mask = df[patient_col].str.startswith('CT_')
    elif dataset_key == 'ctc':
        mask = df[patient_col].str.startswith('CTC_') | df[patient_col].str.startswith('CTW_')
    else:
        mask = pd.Series([True] * len(df), index=df.index)
    return df[mask]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, help='Variant name (e.g. A1, A2)')
    parser.add_argument('--dataset', type=str, default='ct', choices=['ct', 'ctc'])
    parser.add_argument('--se2', type=int, default=1, help='1 for SE2, 0 for Standard UNet')
    parser.add_argument('--slices', type=int, default=3, help='Number of input slices (1, 3, 5)')
    parser.add_argument('--loss', type=str, default='focal_dice_boundary', 
                        choices=['dice', 'focal_dice', 'dice_boundary', 'focal_dice_boundary'])
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    CSV_REPORT = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
    DATA_PATH  = os.path.expanduser("~/Clara/local_ct_workspace_full")
    SAVE_DIR   = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_ablation")
    os.makedirs(SAVE_DIR, exist_ok=True)
    SAVE_PATH  = os.path.join(SAVE_DIR, f"{args.variant}_{args.dataset}_best.pth")

    df = pd.read_csv(CSV_REPORT)
    pc = 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'
    df = filter_df_by_dataset(df, args.dataset, pc)
    
    train_df = df.sample(frac=0.85, random_state=42)
    val_df   = df.drop(train_df.index)

    # Calculate class weights for boundary loss
    neg_px = train_df['0s'].sum()
    pos_px = train_df['1s'].sum()
    total_px = neg_px + pos_px
    weight_0 = total_px / (2 * neg_px)
    weight_1 = total_px / (2 * pos_px)
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(device)

    # Transforms (same as proposed model)
    train_transform = A.Compose([
        A.Rotate(limit=25, p=0.8),
        A.HorizontalFlip(p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8,8), p=0.5) if args.dataset == 'ct' else A.NoOp(),
    ])
    val_transform = A.Compose([
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8,8), p=1.0) if args.dataset == 'ct' else A.NoOp(),
    ])

    train_set = CTBrainAblationDataset(train_df, DATA_PATH, n_slices=args.slices, transform=train_transform)
    val_set   = CTBrainAblationDataset(val_df,   DATA_PATH, n_slices=args.slices, transform=val_transform)

    nw = 4 if torch.cuda.is_available() else 0
    BATCH = 8
    train_loader = DataLoader(train_set, BATCH, shuffle=True,  pin_memory=True, num_workers=nw)
    val_loader   = DataLoader(val_set,   BATCH, shuffle=False, pin_memory=True, num_workers=nw)

    if args.se2 == 1:
        # SE2 model needs specific initialization depending on slices
        model = SE2_CNNET(n_channels=args.slices, n_classes=2, N=8, base_channels=32).to(device)
    else:
        model = StandardUNet(n_channels=args.slices, n_classes=2).to(device)

    criterion = CombinedAblationLoss(loss_type=args.loss, class_weights=class_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True, min_lr=1e-7)
    scaler = torch.amp.GradScaler('cuda')
    
    EPOCHS = 150
    ACCUM = 4
    best_iou = 0.0
    early_stop_counter = 0

    print(f"\\n{'='*65}")
    print(f"  Training Variant: {args.variant} | SE2: {bool(args.se2)} | Slices: {args.slices} | Loss: {args.loss}")
    print(f"  Dataset: {args.dataset.upper()} | Device: {device}")
    print(f"{'='*65}\\n")

    for epoch in range(1, EPOCHS+1):
        model.train(); optimizer.zero_grad(); train_loss=0.0
        for i, (imgs, masks) in enumerate(tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS} [Train]", ncols=80)):
            imgs = imgs.to(device, non_blocking=True); masks = masks.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                loss = criterion(model(imgs), masks) / ACCUM
            scaler.scale(loss).backward()
            if (i+1) % ACCUM == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            train_loss += loss.item() * ACCUM

        model.eval(); tp=fp=fn=0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Ep {epoch}/{EPOCHS} [Val]", ncols=80):
                imgs = imgs.to(device, non_blocking=True); masks = masks.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'): logits = model(imgs)
                preds = torch.argmax(F.softmax(logits, 1), 1)
                pf = preds.view(-1); mf = masks.view(-1)
                tp += ((pf==1)&(mf==1)).sum().item()
                fp += ((pf==1)&(mf==0)).sum().item()
                fn += ((pf==0)&(mf==1)).sum().item()

        eps = 1e-7
        iou  = tp/(tp+fp+fn+eps)
        dice = (2*tp)/(2*tp+fp+fn+eps)
        
        print(f"  ➜ Val IoU: {iou:.4f} | Val Dice: {dice:.4f} | Train Loss: {train_loss/len(train_loader):.4f}")
        
        scheduler.step(iou)
        
        if iou > best_iou:
            best_iou = iou
            early_stop_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  🌟 New Best Model! Saved to {SAVE_PATH}")
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= 20:
            print(f"  🛑 Early stopping triggered at epoch {epoch}")
            break
