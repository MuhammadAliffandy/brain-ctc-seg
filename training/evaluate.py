import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# E2CNN Specific Libraries
from escnn import gspaces
import escnn.nn as enn

# ==========================================
# 1A. MODEL ARCHITECTURE: SE2-CNNET (OURS)
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
# 1B. MODEL ARCHITECTURE: STANDARD U-NET (BASELINE)
# ==========================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class StandardUNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_channels=24):
        super(StandardUNet, self).__init__()
        c = base_channels
        self.inc = DoubleConv(n_channels, c)
        self.down1 = Down(c, c*2)
        self.down2 = Down(c*2, c*4)
        self.down3 = Down(c*4, c*8)
        self.down4 = Down(c*8, c*16)
        self.up1 = Up(c*16 + c*8, c*8)
        self.up2 = Up(c*8 + c*4, c*4)
        self.up3 = Up(c*4 + c*2, c*2)
        self.up4 = Up(c*2 + c, c)
        self.outc = nn.Conv2d(c, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


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
# 3. VISUALIZATION FUNCTION
# ==========================================

def generate_robustness_proof(test_loader, se2_model, unet_model, device):
    """
    Scans the test loader for the slice with the highest number of lesions,
    runs inference on both models, and plots the side-by-side comparison.
    """
    print("\n" + "="*60)
    print("📸 GENERATING ROBUSTNESS VISUALIZATION PROOF...")
    print("="*60)
    
    max_lesions = -1
    best_image = None
    best_mask = None

    # 1. Scan for the slice with the most lesions
    for images, labels in test_loader:
        lesion_counts = labels.sum(dim=(1, 2))
        
        batch_max_idx = torch.argmax(lesion_counts).item()
        batch_max_val = lesion_counts[batch_max_idx].item()

        if batch_max_val > max_lesions:
            max_lesions = batch_max_val
            best_image = images[batch_max_idx].unsqueeze(0) 
            best_mask = labels[batch_max_idx].unsqueeze(0)

    if best_image is None:
        print("⚠️ No data found to generate visualization.")
        return

    print(f"🎯 Found complex slice with {max_lesions} lesion voxels. Running predictions...")
    
    best_image = best_image.to(device)
    se2_model.eval()
    unet_model.eval()
    
    # 2. Run Predictions
    with torch.no_grad():
        # Predict with SE2-CNNET (Ours)
        logits_se2 = se2_model(best_image)
        probs_se2 = F.softmax(logits_se2, dim=1)
        pred_se2 = torch.argmax(probs_se2, dim=1)
        
        # Predict with Standard U-Net (Baseline)
        logits_unet = unet_model(best_image)
        probs_unet = F.softmax(logits_unet, dim=1)
        pred_unet = torch.argmax(probs_unet, dim=1)

    # 3. Convert tensors to numpy arrays for Matplotlib
    img_np = best_image.cpu().squeeze().numpy()
    gt_np = best_mask.cpu().squeeze().numpy()
    pred_se2_np = pred_se2.cpu().squeeze().numpy()
    pred_unet_np = pred_unet.cpu().squeeze().numpy()
    
    # 4. Generate the plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # [Original Image]
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # [Ground Truth]
    axes[1].imshow(gt_np, cmap='gray')
    axes[1].set_title('Ground Truth', fontsize=14)
    axes[1].axis('off')
    
    # [Standard U-Net]
    axes[2].imshow(pred_unet_np, cmap='gray')
    axes[2].set_title('Standard U-Net\n(Baseline)', fontsize=14)
    axes[2].axis('off')
    
    # [SE2-CNNET (Ours)]
    axes[3].imshow(pred_se2_np, cmap='gray')
    axes[3].set_title('SE2-CNNET (Ours)', fontsize=14)
    axes[3].axis('off')
    
    plt.tight_layout()
    output_filename = "robustness_proof_visualization.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Visualization successfully saved to: {output_filename}")


# ==========================================
# 4. MAIN EVALUATION SCRIPT
# ==========================================

def evaluate_all():
    # PATHS
    LOCAL_DATA_PATH = os.path.expanduser("~/Clara/local_ct_workspace") 
    
    # MODEL WEIGHT PATHS
    SE2_MODEL_WEIGHTS = "se2_unet_best_fold_1.pth" 
    UNET_MODEL_WEIGHTS = "standard_unet_best_fold_1.pth" 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using Device: {device}")
    
    # Load Data
    print("Loading test dataset...")
    test_set = CTBrainDatasetTest(LOCAL_DATA_PATH)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    print(f"✅ Found {len(test_set)} slices to evaluate.")

    # Load SE2-CNNET Model
    print("Loading SE2-CNNET Model...")
    se2_model = SE2_CNNET(n_channels=1, n_classes=2).to(device)
    if not os.path.exists(SE2_MODEL_WEIGHTS):
        raise FileNotFoundError(f"⚠️ SE2 Model weights not found: {SE2_MODEL_WEIGHTS}")
    se2_model.load_state_dict(torch.load(SE2_MODEL_WEIGHTS, map_location=device))
    se2_model.eval()

    # Load Standard U-Net Model
    print("Loading Standard U-Net Model...")
    unet_model = StandardUNet(n_channels=1, n_classes=2, base_channels=24).to(device)
    if not os.path.exists(UNET_MODEL_WEIGHTS):
        raise FileNotFoundError(f"⚠️ U-Net Model weights not found: {UNET_MODEL_WEIGHTS}")
    unet_model.load_state_dict(torch.load(UNET_MODEL_WEIGHTS, map_location=device))
    unet_model.eval()

    # Confusion Matrix accumulators (For SE2-CNNET Report)
    TP, FP, TN, FN = 0, 0, 0, 0

    print("🚀 Starting voxel-wise evaluation for SE2-CNNET...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            logits = se2_model(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            preds_flat = preds.view(-1)
            labels_flat = labels.view(-1)

            TP += ((preds_flat == 1) & (labels_flat == 1)).sum().item()
            FP += ((preds_flat == 1) & (labels_flat == 0)).sum().item()
            TN += ((preds_flat == 0) & (labels_flat == 0)).sum().item()
            FN += ((preds_flat == 0) & (labels_flat == 1)).sum().item()

    # Calculate Metrics
    epsilon = 1e-7 
    precision_1 = TP / (TP + FP + epsilon)
    recall_1 = TP / (TP + FN + epsilon)
    f1_score_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1 + epsilon)
    iou_1 = TP / (TP + FP + FN + epsilon)

    precision_0 = TN / (TN + FN + epsilon)
    recall_0 = TN / (TN + FP + epsilon)
    f1_score_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0 + epsilon)

    total_voxels = TP + FP + TN + FN
    accuracy = (TP + TN) / total_voxels

    # Print Report
    print("\n" + "="*60)
    print("VOXEL-WISE CLASSIFICATION REPORT (SE2-CNNET)")
    print("="*60)
    print(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<15}")
    print("-" * 60)
    print(f"{'0 (Background)':<15} | {precision_0:<10.4f} | {recall_0:<10.4f} | {f1_score_0:<10.4f} | {(TN + FP):<15,}")
    print(f"{'1 (Brain Target)':<15} | {precision_1:<10.4f} | {recall_1:<10.4f} | {f1_score_1:<10.4f} | {(TP + FN):<15,}")
    print("-" * 60)
    print(f"{'Accuracy':<15} | {' ':<10} | {' ':<10} | {accuracy:<10.4f} | {total_voxels:<15,}")
    print("="*60)
    print(f"🔥 Target Class IoU (Jaccard Index) : {iou_1:.4f} ({iou_1*100:.2f}%)")
    print(f"🎯 Target Class Dice Score (F1)     : {f1_score_1:.4f} ({f1_score_1*100:.2f}%)")
    print("="*60)

    # CALL VISUALIZATION FUNCTION
    generate_robustness_proof(test_loader, se2_model, unet_model, device)

if __name__ == "__main__":
    evaluate_all()