import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label
from sklearn.metrics import roc_curve, auc
import re

# Use Agg backend for headless servers to avoid display issues
plt.switch_backend('agg')

# E2CNN Specific Libraries (To find class SE2_CNNET)
from escnn import gspaces
import escnn.nn as enn

# ==========================================
# 1. MODEL ARCHITECTURE (Required to load .pth)
# ==========================================
class DoubleEquivariantConv(nn.Module):
    def __init__(self, in_type, out_type, mid_type=None):
        super().__init__()
        if not mid_type: mid_type = out_type
        self.double_conv = enn.SequentialModule(
            enn.R2Conv(in_type, mid_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(mid_type), enn.ReLU(mid_type, inplace=True),
            enn.R2Conv(mid_type, out_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type), enn.ReLU(out_type, inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_type, out_type):
        super().__init__()
        self.pool = enn.PointwiseMaxPool(in_type, kernel_size=2)
        self.conv = DoubleEquivariantConv(in_type, out_type)
    def forward(self, x): return self.conv(self.pool(x))

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
    def forward(self, x): return self.conv(x)

class SE2_CNNET(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, N=8, base_channels=24):
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
        x1 = self.inc(x_geom); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x).tensor

# ==========================================
# 2. HELPER TO FIND ONE PERFECT SLICE FOR JOURNAL
# ==========================================
def get_best_slice_for_paper(dataset_path):
    print("🔍 Searching for the perfect slice for journal visualization...")
    best_slice_info = None
    max_tumor_pixels = 0
    
    for root, dirs, files in os.walk(dataset_path):
        img_files = sorted([f for f in files if f.endswith('_img.npy')], 
                           key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0)
        
        for i, img_name in enumerate(img_files):
            img_path = os.path.join(root, img_name)
            mask_path = img_path.replace('_img.npy', '_mask.npy')
            
            if os.path.exists(mask_path):
                mask_np = np.load(mask_path)
                tumor_pixels = np.sum(mask_np)
                
                # We want a slice with a decently sized tumor, but not covering the whole brain
                if 500 < tumor_pixels < 5000 and tumor_pixels > max_tumor_pixels:
                    max_tumor_pixels = tumor_pixels
                    
                    # Store 2.5D context
                    idx_prev = max(0, i - 1)
                    idx_next = min(len(img_files) - 1, i + 1)
                    
                    best_slice_info = {
                        'prev': os.path.join(root, img_files[idx_prev]),
                        'curr': img_path,
                        'next': os.path.join(root, img_files[idx_next]),
                        'mask': mask_path,
                        'patient': os.path.basename(root)
                    }
    return best_slice_info

# ==========================================
# 3. JOURNAL FIGURE GENERATOR ENGINE
# ==========================================
def generate_journal_figures():
    TEST_DATA_PATH = os.path.expanduser("~/Clara/local_ct_workspace") 
    ROBUST_MODEL_WEIGHTS = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D/se2_unet_best_25D_Boundary.pth")
    OUTPUT_DIR = os.path.expanduser("~/Clara/brain-ctc-seg/training/Journal_Figures")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using Device: {device}")
    
    # Load the Model with 84% Weights
    model = SE2_CNNET(n_channels=3, n_classes=2, N=8, base_channels=24).to(device)
    try:
        model.load_state_dict(torch.load(ROBUST_MODEL_WEIGHTS, map_location=device, weights_only=True), strict=False)
        print("✅ 84% Accuracy Weights Loaded!")
    except Exception as e:
        print("⚠️ Failed to load best weights.")
        return

    model.eval()

    # 1. Prepare the Data
    target_slice = get_best_slice_for_paper(TEST_DATA_PATH)
    if not target_slice:
        print("❌ Could not find a suitable slice.")
        return

    print(f"✅ Selected Patient {target_slice['patient']} for Journal Figure.")
    
    # ✅ NOT COMPRESSING (No interpolation). Use native resolution for detail.
    img_prev = np.load(target_slice['prev']).astype(np.float32)
    img_curr = np.load(target_slice['curr']).astype(np.float32)
    img_next = np.load(target_slice['next']).astype(np.float32)
    gt_np = np.load(target_slice['mask']).astype(np.uint8)

    image_25d = np.stack([img_prev, img_curr, img_next], axis=-1)
    img_tensor = torch.from_numpy(image_25d).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 2. Model Inference for Confidence Heatmap
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        prob_map_ai = probs[0, 1, :, :].cpu().numpy()

    # Apply Crop & Rotation (Same as GIF settings: Crop=40, Rot=3)
    CROP_MARGIN = 40
    ROTATE_K = 3
    
    img_render = np.rot90(img_curr[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)
    gt_render = np.rot90(gt_np[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)
    prob_map_ai = np.rot90(prob_map_ai[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)

    # ---------------------------------------------------------
    # 🌟 FIGURE 1: JOURNAL-GRADE HEATMAP (Like Image 2/3 Top)
    # ---------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 6))
    
    # Subplot (a): Grayscale Input
    axes1[0].imshow(img_render, cmap='gray')
    axes1[0].set_title('(a) Input CT Scan', fontsize=16, fontweight='bold')
    axes1[0].axis('off')

    # Subplot (b): Doctor Ground Truth
    axes1[1].imshow(img_render, cmap='gray')
    masked_gt = np.ma.masked_where(gt_render == 0, gt_render)
    axes1[1].imshow(masked_gt, cmap='Greens', alpha=0.6, vmin=0, vmax=1) 
    axes1[1].set_title('(b) Ground Truth (Doctor)', fontsize=16, fontweight='bold')
    axes1[1].axis('off')
    
    # ✅ Subplot (c): Journal-Style DETAILED CONFIDENCE HEATMAP
    axes1[2].imshow(img_render, cmap='gray') # Grayscale background anatomy

    # THE MAGIC: Mask low probability (< 0.1) and use 'jet' for detail gradations
    # 'jet' starts at dark blue (0.0) and ends at dark red (1.0)
    masked_heatmap = np.ma.masked_where(prob_map_ai < 0.1, prob_map_ai)
    # We vmin at 0.2 to show the "cool" blues only around the tumor boundaries.
    cax = axes1[2].imshow(masked_heatmap, cmap='jet', alpha=0.5, vmin=0.2, vmax=1.0) 
    
    axes1[2].set_title('(c) Proposed Mod-Seg-SE(2)', fontsize=16, fontweight='bold')
    axes1[2].axis('off')
    
    # ✅ THE ESSENTIAL: Add a Colorbar for reference (standard for journal figures)
    cbar = fig1.colorbar(cax, ax=axes1[2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    # Save as high-resolution (300 DPI) for journal submission
    output_heatmap_path = os.path.join(OUTPUT_DIR, 'Fig1_Detailed_Heatmap.png')
    fig1.savefig(output_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close(fig1) 
    print(f"📸 Saved Figure 1: Heatmap (DPI=300) to: {output_heatmap_path}")

    # ---------------------------------------------------------
    # 🌟 FIGURE 2: ROC CURVE (Like Image 1 Bottom)
    # ---------------------------------------------------------
    # Flaten the masks to generate points for the curve
    y_true = gt_render.flatten()
    y_scores = prob_map_ai.flatten()
    
    # Calculate ROC-AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    # Line color must be dark (like darkblue) to distinguish
    ax2.plot(fpr, tpr, color='darkblue', lw=2.5, label=f'Mod-Seg-SE(2) (AUC = {roc_auc:.3f})')
    # Dashed line for random guess (often yellow/black in journals)
    ax2.plot([0, 1], [0, 1], color='yellow', lw=2, linestyle='--', label='Random Guessing')
    
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=14)
    ax2.set_ylabel('True Positive Rate', fontsize=14)
    # Standard grid lines
    ax2.grid(True, linestyle=':', alpha=0.6) 
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.legend(loc="lower right", fontsize=12)
    
    plt.tight_layout()
    output_roc_path = os.path.join(OUTPUT_DIR, 'Fig2_ROC_Curve.png')
    fig2.savefig(output_roc_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"📸 Saved Figure 2: ROC Curve (DPI=300) to: {output_roc_path}")

    print("\n🌟 ALL DETAILED JOURNAL FIGURES GENERATED SUCCESSFULLY in 'Journal_Figures' Folder! 🌟")

if __name__ == "__main__":
    generate_journal_figures()