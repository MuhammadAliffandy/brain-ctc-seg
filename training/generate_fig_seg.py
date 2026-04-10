import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# E2CNN Specific Libraries
from escnn import gspaces
import escnn.nn as enn

# ==========================================
# 1. MODEL ARCHITECTURES (Required to load weights)
# ==========================================

# --- SE2-CNNET (Ours) ---
class DoubleGConv(nn.Module):
    def __init__(self, in_type, out_type, mid_type=None):
        super().__init__()
        if not mid_type: mid_type = out_type
        self.double_gconv = enn.SequentialModule(
            enn.R2Conv(in_type, mid_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(mid_type), enn.ReLU(mid_type, inplace=True),
            enn.R2Conv(mid_type, out_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type), enn.ReLU(out_type, inplace=True)
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

# --- Standard U-Net (Baseline) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
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

    def __len__(self): return len(self.slice_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.slice_pairs[idx]
        image = torch.from_numpy(np.load(img_path).astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(np.load(mask_path).astype(np.uint8)).long() 
        return image, mask


# ==========================================
# 3. FIGURE GENERATION SCRIPT
# ==========================================
def generate_publication_figures():
    # PATHS (Adjust if necessary)
    LOCAL_DATA_PATH = os.path.expanduser("~/Clara/local_ct_workspace") 
    SE2_MODEL_WEIGHTS = "se2_unet_best_fold_1.pth" 
    UNET_MODEL_WEIGHTS = "standard_unet_best_fold_1.pth" 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using Device: {device}")
    
    # Load Models
    print("Loading Models...")
    se2_model = SE2_CNNET(n_channels=1, n_classes=2).to(device)
    se2_model.load_state_dict(torch.load(SE2_MODEL_WEIGHTS, map_location=device))
    se2_model.eval()

    unet_model = StandardUNet(n_channels=1, n_classes=2, base_channels=24).to(device)
    unet_model.load_state_dict(torch.load(UNET_MODEL_WEIGHTS, map_location=device))
    unet_model.eval()

    # Load Data
    print("Loading test dataset...")
    test_set = CTBrainDatasetTest(LOCAL_DATA_PATH)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Variables for ROC and Heatmap
    all_labels = []
    all_probs_se2 = []
    all_probs_unet = []
    
    max_lesions = -1
    best_image = None
    best_mask = None

    print("\n🚀 Scanning dataset for ROC data and complex slices...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Processing Slices"):
            images = images.to(device)
            labels = labels.to(device)
            
            # --- 1. Identify complex slice for Heatmap ---
            lesion_counts = labels.sum(dim=(1, 2))
            batch_max_idx = torch.argmax(lesion_counts).item()
            if lesion_counts[batch_max_idx].item() > max_lesions:
                max_lesions = lesion_counts[batch_max_idx].item()
                best_image = images[batch_max_idx].unsqueeze(0)
                best_mask = labels[batch_max_idx].unsqueeze(0)

            # --- 2. Collect Probability Data for ROC ---
            logits_se2 = se2_model(images)
            probs_se2_full = F.softmax(logits_se2, dim=1)
            
            logits_unet = unet_model(images)
            probs_unet_full = F.softmax(logits_unet, dim=1)
            
            # Sub-sample voxels (every 100th voxel) to prevent RAM crash
            labels_flat = labels.view(-1)
            subset_indices = torch.arange(0, labels_flat.size(0), 100)
            
            all_labels.extend(labels_flat[subset_indices].cpu().numpy())
            all_probs_se2.extend(probs_se2_full[:, 1, :, :].reshape(-1)[subset_indices].cpu().numpy())
            all_probs_unet.extend(probs_unet_full[:, 1, :, :].reshape(-1)[subset_indices].cpu().numpy())

    # ==========================================
    # FIGURE 1: ROC CURVE
    # ==========================================
    print("\n📈 Plotting ROC Curve...")
    fpr_se2, tpr_se2, _ = roc_curve(all_labels, all_probs_se2)
    roc_auc_se2 = auc(fpr_se2, tpr_se2)
    
    fpr_unet, tpr_unet, _ = roc_curve(all_labels, all_probs_unet)
    roc_auc_unet = auc(fpr_unet, tpr_unet)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr_unet, tpr_unet, color='darkorange', lw=2, label=f'Standard U-Net (AUC = {roc_auc_unet:.4f})')
    plt.plot(fpr_se2, tpr_se2, color='blue', lw=2, label=f'SE2-CNNET (AUC = {roc_auc_se2:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig("Fig_ROC_Curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ ROC Curve saved to: Fig_ROC_Curve.png")


    # ==========================================
    # FIGURE 2: HEATMAP OVERLAY
    # ==========================================
    print("\n🧠 Plotting Tumor Heatmap Overlay...")
    with torch.no_grad():
        # Get specific probabilities for the chosen complex slice
        prob_map_se2 = F.softmax(se2_model(best_image), dim=1)[0, 1, :, :].cpu().numpy()
        prob_map_unet = F.softmax(unet_model(best_image), dim=1)[0, 1, :, :].cpu().numpy()

    img_np = best_image.cpu().squeeze().numpy()
    gt_np = best_mask.cpu().squeeze().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    def plot_overlay(ax, title, prob_map):
        ax.imshow(img_np, cmap='gray') # Base Brain CT
        # Mask out low probabilities (< 10%) so the brain is visible
        masked_prob = np.ma.masked_where(prob_map < 0.1, prob_map) 
        heatmap = ax.imshow(masked_prob, cmap='jet', alpha=0.55, vmin=0, vmax=1)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        return heatmap

    # 1. Ground Truth
    axes[0].imshow(img_np, cmap='gray')
    masked_gt = np.ma.masked_where(gt_np == 0, gt_np)
    axes[0].imshow(masked_gt, cmap='autumn', alpha=0.55) 
    axes[0].set_title('Ground Truth Mask', fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Standard U-Net
    plot_overlay(axes[1], 'Standard U-Net\nProbability Heatmap', prob_map_unet)
    
    # 3. SE2-CNNET
    im = plot_overlay(axes[2], 'SE2-CNNET (Ours)\nProbability Heatmap', prob_map_se2)
    
    # Colorbar attached to the figure
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    cbar.set_label('Tumor Probability (0.0 - 1.0)', rotation=270, labelpad=20, fontsize=12)
    
    plt.savefig("Fig_Tumor_Heatmaps.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✅ Heatmap Overlays saved to: Fig_Tumor_Heatmaps.png")


if __name__ == "__main__":
    generate_publication_figures()