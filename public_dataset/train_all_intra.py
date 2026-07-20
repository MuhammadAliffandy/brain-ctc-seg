import os
import sys
import os
import random
import signal

# --- ANTI-TERMINATE SHIELD ---
# Mengabaikan sinyal SIGTERM dari DGX server watchdog
signal.signal(signal.SIGTERM, signal.SIG_IGN)
# -----------------------------

import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import albumentations as A
import kagglehub

cv2.setNumThreads(0)
os.environ["OMP_NUM_THREADS"] = "1"

plt.switch_backend('agg')

# Import Models & Loss dari pipeline kita
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from train_comparison_models import HarmonicNet, StandardUNet, nnUNet, AttentionUNet, TransUNet, CombinedLoss
from train_se2_by_dataset import SE2_CNNET

# ==========================================
# 1. DATASET LOADER (INTRA-DOMAIN KAGGLE)
# ==========================================
class IntraKaggleDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: list of tuples (img_path, mask_path)
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            img = np.zeros((256, 256), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            
        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)
        
        if self.transform:
            # Albumentations expects (H, W, C) or (H, W)
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
        img_float = img.astype(np.float32)
        if img_float.max() > img_float.min():
            img_norm = (img_float - img_float.min()) / (img_float.max() - img_float.min())
        else:
            img_norm = img_float
            
        # Karena arsitektur CT kita inputnya 3 channel (2.5D), 
        # kita duplikasi slice 2D Kaggle ini menjadi 3 channel agar shape cocok.
        img_3c = np.stack([img_norm, img_norm, img_norm], axis=0)
        
        return torch.from_numpy(img_3c), torch.from_numpy(mask).long()

def get_kaggle_splits(root_dir, test_size=0.15, seed=42):
    external_test_dir = None
    for r, d, f in os.walk(root_dir):
        if "External_Test" in d:
            external_test_dir = os.path.join(r, "External_Test")
            break
            
    if not external_test_dir:
        # Some kagglehub downloads flatten the structure or just have it directly
        if os.path.exists(os.path.join(root_dir, "External_Test")):
            external_test_dir = os.path.join(root_dir, "External_Test")
        elif os.path.exists(os.path.join(root_dir, "brain-stroke-ct-dataset", "External_Test")):
            external_test_dir = os.path.join(root_dir, "brain-stroke-ct-dataset", "External_Test")
        else:
            raise ValueError("Folder External_Test Kaggle tidak ditemukan!")
        
    png_dir = os.path.join(external_test_dir, "PNG")
    mask_dir = os.path.join(external_test_dir, "MASKS")
    
    inputs = sorted(glob.glob(os.path.join(png_dir, "*.png")))
    valid_samples = []
    
    for img_path in inputs:
        base_name = os.path.basename(img_path)
        mask_path_exact = os.path.join(mask_dir, base_name)
        if os.path.exists(mask_path_exact):
            valid_samples.append((img_path, mask_path_exact))
        else:
            name_without_ext = os.path.splitext(base_name)[0]
            possible_masks = glob.glob(os.path.join(mask_dir, f"*{name_without_ext}*.png"))
            if possible_masks:
                valid_samples.append((img_path, possible_masks[0]))
                
    # Deterministic Split
    random.seed(seed)
    random.shuffle(valid_samples)
    
    split_idx = int(len(valid_samples) * (1 - test_size))
    train_samples = valid_samples[:split_idx]
    test_samples = valid_samples[split_idx:]
    
    return train_samples, test_samples

# ==========================================
# 2. TRAINING LOOP
# ==========================================
def train_and_eval_model(model_name, ModelClass, is_se2, train_loader, test_loader, device, save_dir):
    print(f"\n" + "="*50)
    print(f"🚀 TRAINING INTRA-DOMAIN: {model_name}")
    print("="*50)
    
    model = ModelClass(n_channels=3, n_classes=2).to(device)
    EPOCHS = 100
    LR = 1e-4
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    class_weights = torch.tensor([1.0, 10.0], device=device)
    
    if is_se2:
        criterion = CombinedLoss(class_weights=class_weights).to(device)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True, min_lr=1e-7
        )
        print("  📌 Menggunakan Pipeline Proposed (EdgeBoundaryLoss + Scheduler)")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        scheduler = None
        print("  📌 Menggunakan Pipeline Standar (CE Loss, No Scheduler)")
        
    scaler = torch.amp.GradScaler('cuda')
    best_iou = 0.0
    save_path = os.path.join(save_dir, f"{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_kaggle_best.pth")
    
    # --- TRAINING ---
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        for imgs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Validasi ringan setiap epoch
        model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for imgs, masks in test_loader: # Pakai test loader sbg validasi intra-domain
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
                
        eps = 1e-7
        epoch_iou = tp / (tp + fp + fn + eps)
        epoch_dice = (2 * tp) / (2 * tp + fp + fn + eps)
        epoch_prec = tp / (tp + fp + eps)
        epoch_rec = tp / (tp + fn + eps)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_loss = running_loss / len(train_loader)
        
        print(f"  Ep {epoch+1:>3} | Loss {epoch_loss:.4f} | Dice {epoch_dice:.4f} | IoU {epoch_iou:.4f} | Prec {epoch_prec:.4f} | Rec {epoch_rec:.4f} | LR {current_lr:.2e}")
        
        if epoch_iou > best_iou:
            best_iou = epoch_iou
            torch.save(model.state_dict(), save_path)
            
        if scheduler:
            scheduler.step(epoch_dice)
            
    print(f"✅ Training selesai. Best IoU: {best_iou:.4f} -> {save_path}")
    
    # --- FINAL EVALUATION PADA BEST MODEL ---
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    tp = fp = fn = tn = 0
    print(f"📊 Mengeksekusi Evaluasi Final untuk {model_name}...")
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc=f"Testing {model_name}", leave=False):
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
            
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sens.)": recall,
        "F1 (Dice)": dice,
        "IoU": iou
    }

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def main():
    print("Mendownload / Memuat cache dataset Kaggle Stroke...")
    root_dir = kagglehub.dataset_download("ozguraslank/brain-stroke-ct-dataset")
    save_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Mempersiapkan Dataset Intra-Domain Kaggle (85% Train, 15% Test)...")
    train_samples, test_samples = get_kaggle_splits(root_dir)
    print(f"Total Train: {len(train_samples)} slices | Total Test: {len(test_samples)} slices")
    
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])
    
    train_dataset = IntraKaggleDataset(train_samples, transform=train_transform)
    test_dataset = IntraKaggleDataset(test_samples, transform=None)
    
    nw = 0 # Set ke 0 untuk mencegah deadlock PyTorch Dataloader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=nw, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=nw, pin_memory=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models_to_train = {
        "Mod-Seg-SE(2)": (SE2_CNNET, True),
        "HarmonicNet": (HarmonicNet, False),
        "nnU-Net": (nnUNet, False),
        "Attention U-Net": (AttentionUNet, False),
        "TransUNet": (TransUNet, False),
        "Standard U-Net": (StandardUNet, False),
    }
    
    results = []
    
    for name, (ModelClass, is_se2) in models_to_train.items():
        res = train_and_eval_model(name, ModelClass, is_se2, train_loader, test_loader, device, save_dir)
        results.append(res)
        
        # Simpan CSV setiap kali satu model selesai (safety)
        df = pd.DataFrame(results)
        csv_path = os.path.join(os.path.dirname(__file__), "public_intra_eval_metrics.csv")
        df.to_csv(csv_path, index=False)
        
    print("\n" + "="*60)
    print("📊 FINAL RESULTS ON KAGGLE DATASET (INTRA-DOMAIN)")
    print("="*60)
    print(df.to_string(index=False))
    print(f"\n✅ Metrik tersimpan di: {csv_path}")

if __name__ == "__main__":
    main()
