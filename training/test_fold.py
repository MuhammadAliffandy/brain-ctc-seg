import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# E2CNN Specific Libraries
from escnn import gspaces
import escnn.nn as enn

# ==========================================
# 1. MODEL ARCHITECTURE (Must match exactly to load weights)
# ==========================================

class DoubleGConv(nn.Module):
    def __init__(self, in_type, out_type, mid_type=None):
        super().__init__()
        if not mid_type:
            mid_type = out_type
        self.double_gconv = enn.SequentialModule(
            enn.R2Conv(in_type, mid_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(mid_type),
            enn.ReLU(mid_type, inplace=True),
            enn.R2Conv(mid_type, out_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True)
        )
    def forward(self, x): return self.double_gconv(x)

class DownGConv(nn.Module):
    def __init__(self, in_type, out_type):
        super().__init__()
        self.pool = enn.PointwiseMaxPool(in_type, kernel_size=2)
        self.gconv = DoubleGConv(in_type, out_type)
    def forward(self, x): return self.gconv(self.pool(x))

class UpGDeconv(nn.Module):
    def __init__(self, in_type, out_type):
        super().__init__()
        self.up = enn.R2Upsampling(in_type, scale_factor=2, mode='bilinear', align_corners=True)
        self.gconv = DoubleGConv(in_type + out_type, out_type)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = enn.tensor_directsum([x2, x1])
        return self.gconv(x)

class OutConv(nn.Module):
    def __init__(self, in_type, n_classes):
        super().__init__()
        gspace = in_type.gspace
        out_type = enn.FieldType(gspace, n_classes * [gspace.trivial_repr])
        self.conv = enn.R2Conv(in_type, out_type, kernel_size=1) 
    def forward(self, x): return self.conv(x)

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

        self.inc = DoubleGConv(self.feat_type_in, self.feat_type_1)
        self.down1 = DownGConv(self.feat_type_1, self.feat_type_2)
        self.down2 = DownGConv(self.feat_type_2, self.feat_type_3)
        self.down3 = DownGConv(self.feat_type_3, self.feat_type_4)
        self.down4 = DownGConv(self.feat_type_4, self.feat_type_5)

        self.up1 = UpGDeconv(self.feat_type_5, self.feat_type_4)
        self.up2 = UpGDeconv(self.feat_type_4, self.feat_type_3)
        self.up3 = UpGDeconv(self.feat_type_3, self.feat_type_2)
        self.up4 = UpGDeconv(self.feat_type_2, self.feat_type_1)
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
# 2. DATASET LOADER
# ==========================================

class CTBrainDatasetTest(Dataset):
    def __init__(self, root_dir):
        self.slice_pairs = []
        patient_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        for patient_dir in patient_dirs:
            img_files = sorted([f for f in os.listdir(patient_dir) if f.endswith('_img.npy')])
            for img_name in img_files:
                img_path = os.path.join(patient_dir, img_name)
                mask_path = img_path.replace('_img.npy', '_mask.npy')
                if os.path.exists(mask_path):
                    self.slice_pairs.append((img_path, mask_path))

    def __len__(self):
        return len(self.slice_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.slice_pairs[idx]
        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.uint8)
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).long() 
        return image, mask

# ==========================================
# 3. EVALUATION & CLASSIFICATION REPORT
# ==========================================

def test_and_report():
    # PATHS
    LOCAL_DATA_PATH = os.path.expanduser("~/Clara/local_ct_workspace") 
    
    # KITA UJI MODEL TERBAIK DARI FOLD TERTENTU (misal Fold 1)
    # Ganti angka fold ini dengan fold yang memiliki Validation Loss terkecil dari log kamu
    MODEL_WEIGHTS = "se2_unet_best_fold_1.pth" 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using Device: {device}")
    
    # Load Data
    print("Loading test dataset...")
    test_set = CTBrainDatasetTest(LOCAL_DATA_PATH)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    print(f"✅ Found {len(test_set)} slices to evaluate.")

    # Load Model
    model = SE2_CNNET(n_channels=1, n_classes=2).to(device)
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"⚠️ Model weights not found: {MODEL_WEIGHTS}")
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()

    # Confusion Matrix accumulators
    TP = 0  # True Positive (Pred 1, Target 1)
    FP = 0  # False Positive (Pred 1, Target 0)
    TN = 0  # True Negative (Pred 0, Target 0)
    FN = 0  # False Negative (Pred 0, Target 1)

    print("🚀 Starting voxel-wise evaluation...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Flatten the tensors to 1D to compare voxel by voxel
            preds_flat = preds.view(-1)
            labels_flat = labels.view(-1)

            # Accumulate metrics
            TP += ((preds_flat == 1) & (labels_flat == 1)).sum().item()
            FP += ((preds_flat == 1) & (labels_flat == 0)).sum().item()
            TN += ((preds_flat == 0) & (labels_flat == 0)).sum().item()
            FN += ((preds_flat == 0) & (labels_flat == 1)).sum().item()

    # --- CALCULATE METRICS ---
    epsilon = 1e-7 # Prevent division by zero
    
    # Class 1 (Brain Target) Metrics
    precision_1 = TP / (TP + FP + epsilon)
    recall_1 = TP / (TP + FN + epsilon)
    f1_score_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1 + epsilon)
    iou_1 = TP / (TP + FP + FN + epsilon)

    # Class 0 (Background) Metrics
    precision_0 = TN / (TN + FN + epsilon)
    recall_0 = TN / (TN + FP + epsilon)
    f1_score_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0 + epsilon)
    iou_0 = TN / (TN + FP + FN + epsilon)

    # Global Metrics
    total_voxels = TP + FP + TN + FN
    accuracy = (TP + TN) / total_voxels
    macro_avg_precision = (precision_0 + precision_1) / 2
    macro_avg_recall = (recall_0 + recall_1) / 2
    macro_avg_f1 = (f1_score_0 + f1_score_1) / 2

    # --- PRINT CLASSIFICATION REPORT ---
    print("\n" + "="*60)
    print("VOXEL-WISE CLASSIFICATION REPORT (SE2-CNNET)")
    print("="*60)
    print(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support (Voxels)':<15}")
    print("-" * 60)
    print(f"{'0 (Background)':<15} | {precision_0:<10.4f} | {recall_0:<10.4f} | {f1_score_0:<10.4f} | {(TN + FP):<15,}")
    print(f"{'1 (Brain Target)':<15} | {precision_1:<10.4f} | {recall_1:<10.4f} | {f1_score_1:<10.4f} | {(TP + FN):<15,}")
    print("-" * 60)
    print(f"{'Accuracy':<15} | {' ':<10} | {' ':<10} | {accuracy:<10.4f} | {total_voxels:<15,}")
    print(f"{'Macro Avg':<15} | {macro_avg_precision:<10.4f} | {macro_avg_recall:<10.4f} | {macro_avg_f1:<10.4f} | {total_voxels:<15,}")
    print("="*60)
    print(f"🔥 Target Class IoU (Jaccard Index) : {iou_1:.4f} ({iou_1*100:.2f}%)")
    print(f"🎯 Target Class Dice Score (F1)     : {f1_score_1:.4f} ({f1_score_1*100:.2f}%)")
    print("="*60)

if __name__ == "__main__":
    test_and_report()