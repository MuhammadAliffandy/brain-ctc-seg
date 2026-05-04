import os
import sys
import shutil
import random
import re

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import kagglehub
import albumentations as A

# E2CNN Specific Libraries
from escnn import gspaces
import escnn.nn as enn

# ==========================================
# 0. KAGGLE DOWNLOAD & PAIRING
# ==========================================

def download_and_pair_hemorrhage_data():
    print("\n" + "="*60)
    print("📥 STAGE 1: DOWNLOAD HEMORRHAGE DATASET FROM KAGGLE")
    print("="*60)
    
    try:
        download_path = kagglehub.dataset_download("vbookshelf/computed-tomography-ct-images")
        print(f"✅ Download successful! Cache: {download_path}")
    except Exception as e:
        print(f"❌ Failed to download dataset. Error: {e}")
        return None

    print("\n" + "="*60)
    print("⚙️ STAGE 2: PARSING IMAGES & MASKS (HEMORRHAGE)")
    print("="*60)
    
    all_files = []
    for root, dirs, files in os.walk(download_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.bmp', '.tif')):
                all_files.append(os.path.join(root, f))
                
    # Separate masks and images
    masks = [f for f in all_files if 'mask' in f.lower() or 'seg' in f.lower()]
    images = [f for f in all_files if f not in masks]
    
    valid_pairs = []
    for mask_path in masks:
        mask_name = os.path.basename(mask_path).lower()
        # Clean the Kaggle specific mask suffix e.g., '14_hge_seg.jpg' -> '14'
        clean_name = mask_name.replace('_hge_seg', '').replace('_seg', '').replace('_mask', '').replace('mask', '').split('.')[0]
        
        matched_img = None
        # We only want to search within the exact same patient folder to avoid mismatching "14.jpg" across different patients.
        parent_dir = os.path.dirname(mask_path)
        expected_img_path = os.path.join(parent_dir, f"{clean_name}.jpg")
        
        if os.path.exists(expected_img_path):
            valid_pairs.append((expected_img_path, mask_path))
            if expected_img_path in images:
                images.remove(expected_img_path)

    print(f"✅ Found {len(valid_pairs)} valid Image-Mask pairs (expected around 318)")
    return valid_pairs


# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================

class DoubleEquivariantConv(nn.Module):
    def __init__(self, in_type, out_type, mid_type=None):
        super().__init__()
        if not mid_type:
            mid_type = out_type
        self.double_conv = enn.SequentialModule(
            enn.R2Conv(in_type, mid_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(mid_type),
            enn.ReLU(mid_type, inplace=True),
            enn.R2Conv(mid_type, out_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True)
        )
    def forward(self, x): 
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_type, out_type):
        super().__init__()
        self.pool = enn.PointwiseMaxPool(in_type, kernel_size=2)
        self.conv = DoubleEquivariantConv(in_type, out_type)
    def forward(self, x): 
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_type, out_type):
        super().__init__()
        self.up = enn.R2Upsampling(in_type, scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleEquivariantConv(in_type + out_type, out_type)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = enn.tensor_directsum([x2, x1])
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_type, n_classes):
        super().__init__()
        gspace = in_type.gspace
        out_type = enn.FieldType(gspace, n_classes * [gspace.trivial_repr])
        self.conv = enn.R2Conv(in_type, out_type, kernel_size=1)
    def forward(self, x): 
        return self.conv(x)

class SE2_CNNET(nn.Module):
    def __init__(self, n_channels, n_classes, N=8, base_channels=24):
        super().__init__()
        self.r2_act = gspaces.rot2dOnR2(N=N)
        c = base_channels

        self.feat_type_in = enn.FieldType(self.r2_act, n_channels * [self.r2_act.trivial_repr])
        self.feat_type_1 = enn.FieldType(self.r2_act, c * [self.r2_act.regular_repr])
        self.feat_type_2 = enn.FieldType(self.r2_act, (c*2) * [self.r2_act.regular_repr])
        self.feat_type_3 = enn.FieldType(self.r2_act, (c*4) * [self.r2_act.regular_repr])
        self.feat_type_4 = enn.FieldType(self.r2_act, (c*8) * [self.r2_act.regular_repr])
        self.feat_type_5 = enn.FieldType(self.r2_act, (c*16) * [self.r2_act.regular_repr])

        self.inc = DoubleEquivariantConv(self.feat_type_in, self.feat_type_1)
        self.down1 = Down(self.feat_type_1, self.feat_type_2)
        self.down2 = Down(self.feat_type_2, self.feat_type_3)
        self.down3 = Down(self.feat_type_3, self.feat_type_4)
        self.down4 = Down(self.feat_type_4, self.feat_type_5)

        self.up1 = Up(self.feat_type_5, self.feat_type_4)
        self.up2 = Up(self.feat_type_4, self.feat_type_3)
        self.up3 = Up(self.feat_type_3, self.feat_type_2)
        self.up4 = Up(self.feat_type_2, self.feat_type_1)
        self.outc = OutConv(self.feat_type_1, n_classes)

    def forward(self, x):
        x_geom = enn.GeometricTensor(x, self.feat_type_in)
        x1 = self.inc(x_geom)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x).tensor

# ==========================================
# 2. LOSS FUNCTIONS (from train.py)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=3.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
    def forward(self, logits, targets):
        bce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, true_masks):
        nc = logits.shape[1]
        oh = F.one_hot(true_masks, nc).permute(0, 3, 1, 2).float()
        probs = F.softmax(logits, dim=1)
        inter = (probs[:, 1] * oh[:, 1]).sum(dim=(1, 2))
        union = probs[:, 1].sum(dim=(1, 2)) + oh[:, 1].sum(dim=(1, 2))
        return 1.0 - ((2. * inter + self.smooth) / (union + self.smooth)).mean()

class EdgeBoundaryLoss(nn.Module):
    def forward(self, logits, targets):
        tf = targets.float().unsqueeze(1)
        dilated = F.max_pool2d(tf, 5, 1, 2)
        eroded  = -F.max_pool2d(-tf, 5, 1, 2)
        bnd = (dilated - eroded).squeeze(1)
        base = F.cross_entropy(logits, targets, reduction='none')
        return (base * (1 + 5.0 * bnd)).mean()

class AdvancedCombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss(); self.dice = DiceLoss(); self.edge = EdgeBoundaryLoss()
    def forward(self, logits, targets):
        return self.focal(logits, targets) + self.dice(logits, targets) + 0.5 * self.edge(logits, targets)

# ==========================================
# 3. DATASET LOADER (HEMORRHAGE 2.5D SIMULATION)
# ==========================================
class HemorrhageCTDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
        
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        # Load Image and Mask
        img = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)
        
        if img.max() > 1.0: 
            img = img / 255.0
            
        mask = (mask > 127).astype(np.uint8)
        
        # Duplicate slice 3 times to simulate 2.5D input structure
        image_25d = np.stack([img, img, img], axis=-1)
        
        image_t = torch.from_numpy(image_25d).permute(2, 0, 1).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        
        # Resize to 256x256
        image_t = F.interpolate(image_t, size=(256, 256), mode='bilinear', align_corners=False)
        mask_t = F.interpolate(mask_t, size=(256, 256), mode='nearest')
        
        return image_t.squeeze(0), mask_t.squeeze(0).squeeze(0).long()

def calculate_metrics(preds, targets):
    preds = preds.view(-1)
    targets = targets.view(-1)
    tp = torch.sum((preds == 1) & (targets == 1)).item()
    fp = torch.sum((preds == 1) & (targets == 0)).item()
    fn = torch.sum((preds == 0) & (targets == 1)).item()
    tn = torch.sum((preds == 0) & (targets == 0)).item()
    return tp, fp, fn, tn

# ==========================================
# 4. EXTERNAL EVALUATION LOOP
# ==========================================
def evaluate_external_dataset():
    # 1. Download & Pair
    valid_pairs = download_and_pair_hemorrhage_data()
    if not valid_pairs:
        print("❌ Dataset pairing failed.")
        return
        
    # 2. Split Data deterministically (85/15)
    random.seed(42)
    random.shuffle(valid_pairs)
    
    split_idx = int(0.85 * len(valid_pairs))
    train_pairs = valid_pairs[:split_idx]
    test_pairs = valid_pairs[split_idx:]
    
    print("\n" + "="*60)
    print("📊 DATASET SPLIT PROFILE (HEMORRHAGE DATASET)")
    print("="*60)
    print(f"Total Valid Masked Images: {len(valid_pairs)}")
    print(f"Train Split (85%)        : {len(train_pairs)} images")
    print(f"Test/Eval Split (15%)    : {len(test_pairs)} images")
    print("="*60 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using Device: {device}\n")

    # 3. Load Dataset
    eval_dataset = HemorrhageCTDataset(test_pairs)
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    
    # 4. Load Model
    model = SE2_CNNET(n_channels=3, n_classes=2, N=8, base_channels=24).to(device)
    
    WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models_25D", "se2_unet_best_25D_Boundary.pth")
    if not os.path.exists(WEIGHTS_PATH):
        WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models_25D", "se2_unet_epoch_100.pth")
        
    if os.path.exists(WEIGHTS_PATH):
        try:
            # Handle possible 1-channel to 3-channel adaptation
            checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
            filter_key = 'inc.double_conv.0.weights'
            if filter_key in checkpoint and checkpoint[filter_key].shape[0] == 144:
                checkpoint[filter_key] = checkpoint[filter_key].repeat(3) / 3.0
                buf_key = 'inc.double_conv.0.filter'
                if buf_key in checkpoint:
                    checkpoint[buf_key] = checkpoint[buf_key].repeat(1, 3, 1, 1) / 3.0
            
            model.load_state_dict(checkpoint, strict=False)
            print(f"✅ Loaded weights from {WEIGHTS_PATH}")
        except Exception as e:
            print(f"❌ Failed to load weights: {e}")
            return
    else:
        print("❌ Warning: Model weights not found. Evaluation will fail.")
        return
        
    model.eval()

    # 5. Evaluate Metrics
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    print("\n🔍 Evaluating External Kaggle Dataset (Test Split 15%)...")
    with torch.no_grad():
        for images, targets in tqdm(eval_loader, desc="Calculating Metrics"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            tp, fp, fn, tn = calculate_metrics(preds, targets)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            
    eps = 1e-7
    dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + eps)
    iou = total_tp / (total_tp + total_fp + total_fn + eps)
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    
    print("\n" + "🌟"*20)
    print(" EXTERNAL BENCHMARK RESULTS (HEMORRHAGE KAGGLE DATASET)")
    print("🌟"*20)
    print(f"  Test Images Evaluated : {len(test_pairs)}")
    print(f"  Dice Score            : {dice:.4f}")
    print(f"  IoU (Jaccard)         : {iou:.4f}")
    print(f"  Precision             : {precision:.4f}")
    print(f"  Recall                : {recall:.4f}")
    print("─"*50)
    print("This evaluates how well the Tumor-trained SE(2) model")
    print("generalizes to Intracranial Hemorrhage detection.")

if __name__ == "__main__":
    evaluate_external_dataset()