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

from escnn import gspaces
import escnn.nn as enn

# ==========================================
# 1. CORE ARCHITECTURE (Required to load .pth)
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
# 3. JOURNAL FIGURE GENERATOR
# ==========================================
def generate_journal_figures():
    TEST_DATA_PATH = os.path.expanduser("~/Clara/local_ct_workspace") 
    ROBUST_MODEL_WEIGHTS = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D/se2_unet_best_25D_Boundary.pth")
    OUTPUT_DIR = os.path.expanduser("~/Clara/brain-ctc-seg/training/Journal_Figures")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SE2_CNNET(n_channels=3, n_classes=2, N=8, base_channels=24).to(device)
    model.load_state_dict(torch.load(ROBUST_MODEL_WEIGHTS, map_location=device, weights_only=True), strict=False)
    model.eval()

    # 1. Prepare the Data
    target_slice = get_best_slice_for_paper(TEST_DATA_PATH)
    if not target_slice:
        print("❌ Could not find a suitable slice.")
        return

    print(f"✅ Selected Patient {target_slice['patient']} for Journal Figures.")
    
    img_prev = np.load(target_slice['prev']).astype(np.float32)
    img_curr = np.load(target_slice['curr']).astype(np.float32)
    img_next = np.load(target_slice['next']).astype(np.float32)
    gt_np = np.load(target_slice['mask']).astype(np.uint8)

    image_25d = np.stack([img_prev, img_curr, img_next], axis=-1)
    img_tensor = torch.from_numpy(image_25d).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 2. Model Inference
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
    # FIGURE 1: CONFIDENCE HEATMAP WITH COLORBAR (Like Image 2)
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.imshow(img_render, cmap='gray')
    
    # Use 'jet' colormap for the heatmap, masking low confidence (< 0.1)
    masked_heatmap = np.ma.masked_where(prob_map_ai < 0.1, prob_map_ai)
    cax = ax1.imshow(masked_heatmap, cmap='jet', alpha=0.5, vmin=0.2, vmax=1.0)
    
    ax1.axis('off')
    ax1.set_title("Mod-Seg-SE(2)", fontsize=16, fontweight='bold')
    
    # Add Journal-style Colorbar
    cbar = fig1.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, 'Fig1_Confidence_Heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("📸 Saved Figure 1: Confidence Heatmap (Jet)")

    # ---------------------------------------------------------
    # FIGURE 2: SOLID MASK COMPARISON (Like Image 3)
    # ---------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    # (a) Input
    axes2[0].imshow(img_render, cmap='gray')
    axes2[0].set_title('Input', fontsize=18, fontweight='bold')
    axes2[0].text(0.5, -0.1, '(a)', transform=axes2[0].transAxes, fontsize=16, ha='center')
    axes2[0].axis('off')

    # (b) Ground Truth (White mask on black background)
    axes2[1].imshow(img_render, cmap='gray') # Show subtle background
    axes2[1].imshow(np.ma.masked_where(gt_render == 0, gt_render), cmap='gray', vmin=0, vmax=1, alpha=0.9)
    axes2[1].set_title('Ground Truth', fontsize=18, fontweight='bold')
    axes2[1].text(0.5, -0.1, '(b)', transform=axes2[1].transAxes, fontsize=16, ha='center')
    axes2[1].axis('off')

    # (c) Mod-Seg-SE(2) Output (Solid Red Mask)
    axes2[2].imshow(img_render, cmap='gray')
    binary_pred = (prob_map_ai >= 0.5).astype(int) # Threshold at 0.5 for solid shape
    axes2[2].imshow(np.ma.masked_where(binary_pred == 0, binary_pred), cmap='autumn', vmin=0, vmax=1, alpha=0.9)
    axes2[2].set_title('Mod-Seg-SE(2)', fontsize=18, fontweight='bold')
    axes2[2].text(0.5, -0.1, '(c)', transform=axes2[2].transAxes, fontsize=16, ha='center')
    axes2[2].axis('off')

    plt.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'Fig2_Solid_Mask_Comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("📸 Saved Figure 2: Solid Mask Comparison")

    # ---------------------------------------------------------
    # FIGURE 3: ROC CURVE (Like Image 1 Bottom)
    # ---------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    
    # Flatten the arrays to calculate True Positive Rate & False Positive Rate
    y_true = gt_render.flatten()
    y_scores = prob_map_ai.flatten()
    
    # Ensure there's at least one positive class to avoid ROC calculation errors
    if np.sum(y_true) > 0:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        ax3.plot(fpr, tpr, color='darkblue', lw=2, label=f'TPR_Mod-Seg-SE(2) (AUC = {roc_auc:.3f})')
        ax3.plot([0, 1], [0, 1], color='yellow', lw=2, linestyle='--', label='Random Picking')
        
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel('False Positive Rate', fontsize=14)
        ax3.set_ylabel('True Positive Rate', fontsize=14)
        ax3.tick_params(axis='both', which='major', labelsize=12)
        ax3.legend(loc="lower right", fontsize=12)
        ax3.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        fig3.savefig(os.path.join(OUTPUT_DIR, 'Fig3_ROC_Curve.png'), dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print("📸 Saved Figure 3: ROC Curve")
    else:
        print("⚠️ Skipped Figure 3: Selected slice does not contain GT tumors for ROC calculation.")

    print("\n🌟 ALL JOURNAL FIGURES GENERATED SUCCESSFULLY in 'Journal_Figures' Folder! 🌟")

if __name__ == "__main__":
    generate_journal_figures()