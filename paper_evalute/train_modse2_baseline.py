import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

# Fix random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "public_dataset"))

from train_se2_by_dataset import SE2_CNNET
from train_comparison_models import FocalLoss, DiceLoss
import albumentations as A

def get_valid_path(rel_path):
    candidates = [
        os.path.expanduser(f"~/Clara/{rel_path}"),
        f"/raid/D13K48009/Clara/{rel_path}",
    ]
    for c in candidates:
        if os.path.exists(c): return c
    return candidates[0]

# --- Losses ---
class FocalDiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = FocalLoss(alpha=0.75, gamma=3.0)
        self.d = DiceLoss()
    def forward(self, l, t):
        return 0.5 * self.f(l, t) + 2.0 * self.d(l, t)

# --- HD95 ---
from scipy.ndimage import distance_transform_edt, binary_erosion
def surface_distances_2d(result, reference, voxelspacing=(0.45, 0.45)):
    result = result.astype(bool)
    reference = reference.astype(bool)
    res_borders = result ^ binary_erosion(result)
    ref_borders = reference ^ binary_erosion(reference)
    if not res_borders.any() or not ref_borders.any(): return np.nan
    dt_ref = distance_transform_edt(~ref_borders, sampling=voxelspacing)
    dt_res = distance_transform_edt(~res_borders, sampling=voxelspacing)
    dists = np.concatenate([dt_ref[res_borders], dt_res[ref_borders]])
    if len(dists) == 0: return np.nan
    return np.percentile(dists, 95)

def train_and_eval(dataset_name, train_loader, val_loader, test_loader, device, save_path):
    print(f"\n=========================================")
    print(f"🚀 TRAINING Mod-SE(2) [2D, No Boundary] on {dataset_name}")
    print(f"=========================================")
    
    model = SE2_CNNET(n_channels=1, n_classes=2, N=8, base_channels=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = FocalDiceLoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-7)
    scaler = torch.amp.GradScaler('cuda')
    
    EPOCHS = 100 # Adjust if needed
    best_iou = 0.0
    early_stop_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS} [Train]", leave=False)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        model.eval()
        tp=fp=fn=0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    preds = torch.argmax(F.softmax(model(imgs), 1), 1)
                pf, mf = preds.view(-1), masks.view(-1)
                tp += ((pf==1)&(mf==1)).sum().item()
                fp += ((pf==1)&(mf==0)).sum().item()
                fn += ((pf==0)&(mf==1)).sum().item()
                
        eps = 1e-7
        iou = tp / (tp + fp + fn + eps)
        dice = (2*tp) / (2*tp + fp + fn + eps)
        
        print(f"  Ep {epoch+1:>3} | Loss: {running_loss/len(train_loader):.4f} | Val Dice: {dice:.4f} | Val IoU: {iou:.4f}")
        
        if iou > best_iou:
            best_iou = iou
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            early_stop_counter += 1
            
        scheduler.step(iou)
        if early_stop_counter >= 20:
            print("  🛑 Early stopping triggered")
            break
            
    print(f"✅ Training Done! Best Val IoU: {best_iou:.4f}")
    
    # --- EVALUATION ---
    model.load_state_dict(torch.load(save_path))
    model.eval()
    tp=fp=fn=tn=0
    hd95_list = []
    
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc=f"Testing {dataset_name}"):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            preds = torch.argmax(F.softmax(logits, 1), 1)
            
            pf, mf = preds.view(-1), masks.view(-1)
            tp += ((pf==1)&(mf==1)).sum().item()
            fp += ((pf==1)&(mf==0)).sum().item()
            fn += ((pf==0)&(mf==1)).sum().item()
            tn += ((pf==0)&(mf==0)).sum().item()
            
            p_np = preds.cpu().numpy()
            m_np = masks.squeeze(1).cpu().numpy()
            for i in range(len(p_np)):
                if np.any(m_np[i]) and np.any(p_np[i]):
                    val = surface_distances_2d(p_np[i], m_np[i])
                    if not np.isnan(val): hd95_list.append(val)
                    
    eps = 1e-7
    res = {
        "Accuracy": (tp+tn)/(tp+tn+fp+fn+eps),
        "Precision": tp/(tp+fp+eps),
        "Recall": tp/(tp+fn+eps),
        "Dice Score": (2*tp)/(2*tp+fp+fn+eps),
        "IoU": tp/(tp+fp+fn+eps),
        "HD95 (mm)": np.median(hd95_list) if hd95_list else np.nan
    }
    
    print(f"\n🎯 FINAL RESULTS FOR MOD-SE(2) on {dataset_name}:")
    for k,v in res.items():
        print(f"- {k}: {v:.4f}")
    return res

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(get_valid_path("brain-ctc-seg/training/saved_models"), exist_ok=True)
    
    # 1. CECT
    from train_ablation import CTBrainAblationDataset, filter_df_by_dataset
    CSV_REPORT = get_valid_path("new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
    DATA_PATH = get_valid_path("local_ct_workspace_full")
    df = pd.read_csv(CSV_REPORT)
    df = filter_df_by_dataset(df, 'ctc', 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient')
    train_df = df.sample(frac=0.85, random_state=42)
    val_df = df.drop(train_df.index)
    
    train_set_cect = CTBrainAblationDataset(train_df, DATA_PATH, n_slices=1, transform=None)
    val_set_cect = CTBrainAblationDataset(val_df, DATA_PATH, n_slices=1, transform=None)
    
    train_loader_cect = DataLoader(train_set_cect, batch_size=16, shuffle=True, num_workers=2)
    val_loader_cect = DataLoader(val_set_cect, batch_size=16, shuffle=False, num_workers=2)
    
    train_and_eval("CECT (Table 4)", train_loader_cect, val_loader_cect, val_loader_cect, device, get_valid_path("brain-ctc-seg/training/saved_models/ModSE2_A2_cect.pth"))
    
    # 2. Kaggle Stroke
    import kagglehub
    from train_all_intra import get_kaggle_splits
    from torch.utils.data import Dataset
    import cv2
    
    class Kaggle1CDataset(Dataset):
        def __init__(self, samples): self.samples = samples
        def __len__(self): return len(self.samples)
        def __getitem__(self, idx):
            img_path, mask_path = self.samples[idx]
            img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (256, 256))
            mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (256, 256), interpolation=cv2.INTER_NEAREST)
            img = (img.astype(np.float32) - img.min()) / (img.max() - img.min() + 1e-7)
            mask = (mask > 127).astype(np.float32)
            return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(mask).long()
            
    root_dir = kagglehub.dataset_download("ozguraslank/brain-stroke-ct-dataset")
    train_samples, test_samples = get_kaggle_splits(root_dir)
    train_loader_stroke = DataLoader(Kaggle1CDataset(train_samples), batch_size=16, shuffle=True, num_workers=2)
    test_loader_stroke = DataLoader(Kaggle1CDataset(test_samples), batch_size=16, shuffle=False, num_workers=2)
    
    train_and_eval("Kaggle Stroke (Table 5)", train_loader_stroke, test_loader_stroke, test_loader_stroke, device, get_valid_path("brain-ctc-seg/training/saved_models/ModSE2_A2_stroke.pth"))
    
    # 3. Kaggle Hemorrhage
    from train_all_intra_hemorrhage import get_kaggle_hemorrhage_splits
    train_samples_h, test_samples_h = get_kaggle_hemorrhage_splits()
    train_loader_hemo = DataLoader(Kaggle1CDataset(train_samples_h), batch_size=16, shuffle=True, num_workers=2)
    test_loader_hemo = DataLoader(Kaggle1CDataset(test_samples_h), batch_size=16, shuffle=False, num_workers=2)
    
    train_and_eval("Kaggle Hemorrhage (Table 6)", train_loader_hemo, test_loader_hemo, test_loader_hemo, device, get_valid_path("brain-ctc-seg/training/saved_models/ModSE2_A2_hemo.pth"))
    
if __name__ == "__main__":
    main()
