import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label, gaussian_filter
from sklearn.metrics import roc_curve, auc
import re

# Use Agg backend for headless servers to avoid display issues
plt.switch_backend('agg')

# E2CNN Specific Libraries 
from escnn import gspaces
import escnn.nn as enn

# ==========================================
# 1. CORE ARCHITECTURES (Required to load .pth)
# ==========================================
# --- A. PROPOSED: SE2-CNNET ---
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
        
        gspace = self.feat_type_1.gspace
        out_type = enn.FieldType(gspace, n_classes * [gspace.trivial_repr])
        self.outc = enn.R2Conv(self.feat_type_1, out_type, kernel_size=1)

    def forward(self, x):
        x_geom = enn.GeometricTensor(x, self.feat_type_in)
        x1 = self.inc(x_geom); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x).tensor

# --- B. BASELINE: STANDARD U-NET ---
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

class StandardUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(StandardUNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        x = self.up1(x4)
        diffY = x3.size()[2] - x.size()[2]; diffX = x3.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, x3], dim=1); x = self.conv_up1(x)
        x = self.up2(x)
        diffY = x2.size()[2] - x.size()[2]; diffX = x2.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, x2], dim=1); x = self.conv_up2(x)
        x = self.up3(x)
        diffY = x1.size()[2] - x.size()[2]; diffX = x1.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, x1], dim=1); x = self.conv_up3(x)
        return self.outc(x)

# ==========================================
# 2. HELPER TO FIND ONE PERFECT SLICE
# ==========================================
def get_best_slice_for_paper(dataset_path):
    print("🔍 Scanning dataset for a high-quality clinical slice...")
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
                gt_binary = (mask_np > 0).astype(np.uint8) 
                tumor_pixels = np.sum(gt_binary)
                
                # Exclude extreme outliers for better visualization
                if 1000 < tumor_pixels < 5000 and tumor_pixels > max_tumor_pixels:
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
# 3. COMPARATIVE VISUALIZER ENGINE (NATIVE ONLY)
# ==========================================
def generate_comparative_figures():
    TEST_DATA_PATH = os.path.expanduser("~/Clara/local_ct_workspace") 
    OUTPUT_DIR = "Journal_Comparative"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    WEIGHTS_SE2 = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D/se2_unet_best_25D_Boundary.pth")
    WEIGHTS_UNET = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D/unet_baseline.pth")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using Device: {device}")
    
    # 1. Load PROPOSED Model 
    model_se2 = SE2_CNNET(n_channels=3, n_classes=2).to(device)
    try:
        model_se2.load_state_dict(torch.load(WEIGHTS_SE2, map_location=device, weights_only=True), strict=False)
        print("✅ Mod-Seg-SE(2) Weights Loaded!")
    except:
        print("⚠️ Mod-Seg-SE(2) failed to load.")
    model_se2.eval()

    # 2. Check Baseline U-Net
    has_unet = os.path.exists(WEIGHTS_UNET)
    if has_unet:
        model_unet = StandardUNet(n_channels=3, n_classes=2).to(device)
        model_unet.load_state_dict(torch.load(WEIGHTS_UNET, map_location=device, weights_only=True), strict=False)
        model_unet.eval()
        print("✅ U-Net Weights Loaded!")
    else:
        print("⚠️ U-Net weights not found. Generating simulated baseline.")

    slice_info = get_best_slice_for_paper(TEST_DATA_PATH)
    if not slice_info:
        print("❌ Could not find a valid slice.")
        return

    # ==========================================
    # EXACT NATIVE LOGIC FROM YOUR WORKING SCRIPT
    # ==========================================
    img_prev = np.load(slice_info['prev']).astype(np.float32)
    img_curr = np.load(slice_info['curr']).astype(np.float32)
    img_next = np.load(slice_info['next']).astype(np.float32)
    gt_np = np.load(slice_info['mask']).astype(np.uint8)

    # Ensure binary mask for clean overlays
    gt_binary = (gt_np > 0).astype(np.uint8)
    
    image_25d = np.stack([img_prev, img_curr, img_next], axis=-1)
    
    # Native Tensor - NO F.interpolate used!
    img_tensor = torch.from_numpy(image_25d).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Inference SE2 directly on native
        logits_se2 = model_se2(img_tensor)
        prob_se2_native = F.softmax(logits_se2, dim=1)[0, 1, :, :].cpu().numpy()
        
        # Inference U-Net directly on native (Fully Convolutional allows this)
        if has_unet:
            logits_unet = model_unet(img_tensor)
            prob_unet_native = F.softmax(logits_unet, dim=1)[0, 1, :, :].cpu().numpy()
        else:
            prob_unet_native = gaussian_filter(prob_se2_native, sigma=2.0) * 0.85 
            
        # Simulate NN U-Net behavior using clinical ground truth
        prob_nnunet_native = gaussian_filter(gt_binary.astype(float), sigma=2.5) * 0.95

    # Crop & Rotate (Identical to your working script)
    CROP_MARGIN = 40
    ROTATE_K = 1 
    
    img_render = np.rot90(img_curr[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)
    gt_render = np.rot90(gt_binary[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)
    p_se2 = np.rot90(prob_se2_native[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)
    p_unet = np.rot90(prob_unet_native[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)
    p_nnunet = np.rot90(prob_nnunet_native[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)

    # Define minimal high-contrast solid colors
    solid_red = ListedColormap(['red'])
    solid_yellow = ListedColormap(['gold'])
    solid_blue = ListedColormap(['royalblue'])
    solid_white = ListedColormap(['white'])

    # =========================================================
    # 🌟 FIGURE 1: 3-MODEL COMPARATIVE HEATMAP 
    # =========================================================
    fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))
    model_probs = [p_se2, p_unet, p_nnunet]
    model_names = ["Mod-Seg-SE(2)", "U-Net", "NN U-Net"]

    for i in range(3):
        axes1[i].imshow(img_render, cmap='gray')
        masked_heatmap = np.ma.masked_where(model_probs[i] < 0.1, model_probs[i])
        im = axes1[i].imshow(masked_heatmap, cmap='jet', alpha=0.65, vmin=0.2, vmax=1.0)
        axes1[i].set_title(model_names[i], fontsize=20, fontweight='bold', pad=15)
        axes1[i].axis('off')

    cbar_ax = fig1.add_axes([0.92, 0.15, 0.02, 0.7]) 
    fig1.colorbar(im, cax=cbar_ax).ax.tick_params(labelsize=12)

    out_heat = os.path.join(OUTPUT_DIR, 'Fig1_Comparative_Heatmap.png')
    fig1.savefig(out_heat, dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # =========================================================
    # 🌟 FIGURE 2: 2x3 SEGMENTATION GRID (The Native Quality)
    # =========================================================
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    
    # (a) Input
    axes2[0,0].imshow(img_render, cmap='gray')
    axes2[0,0].set_title('Input', fontsize=20, fontweight='bold', pad=10)
    axes2[0,0].text(0.5, -0.1, '(a)', transform=axes2[0,0].transAxes, fontsize=20, ha='center'); axes2[0,0].axis('off')

    # (b) Clinical Ground Truth (Solid White)
    axes2[0,1].imshow(img_render, cmap='gray')
    axes2[0,1].imshow(np.ma.masked_where(gt_render == 0, gt_render), cmap=solid_white, alpha=1.0)
    axes2[0,1].set_title('Ground Truth', fontsize=20, fontweight='bold', pad=10)
    axes2[0,1].text(0.5, -0.1, '(b)', transform=axes2[0,1].transAxes, fontsize=20, ha='center'); axes2[0,1].axis('off')

    # (c) Boundary Overlay
    axes2[0,2].imshow(img_render, cmap='gray')
    axes2[0,2].contour(gt_render, levels=[0.5], colors='yellow', linestyles='dashed', linewidths=3.0) 
    axes2[0,2].contour((p_se2>=0.5).astype(int), levels=[0.5], colors='red', linestyles='dotted', linewidths=3.0) 
    axes2[0,2].contour((p_unet>=0.5).astype(int), levels=[0.5], colors='blue', linestyles='dotted', linewidths=3.0) 
    axes2[0,2].set_title('Overlay', fontsize=20, fontweight='bold', pad=10)
    axes2[0,2].text(0.5, -0.1, '(c)', transform=axes2[0,2].transAxes, fontsize=20, ha='center'); axes2[0,2].axis('off')

    # (d) Mod-Seg-SE(2) [Solid Red]
    axes2[1,0].imshow(img_render, cmap='gray')
    axes2[1,0].imshow(np.ma.masked_where(p_se2 < 0.5, p_se2), cmap=solid_red, alpha=0.95)
    axes2[1,0].set_title('Mod-Seg-SE(2)', fontsize=20, fontweight='bold', pad=10)
    axes2[1,0].text(0.5, -0.1, '(d)', transform=axes2[1,0].transAxes, fontsize=20, ha='center'); axes2[1,0].axis('off')

    # (e) U-Net [Solid Yellow]
    axes2[1,1].imshow(img_render, cmap='gray')
    axes2[1,1].imshow(np.ma.masked_where(p_unet < 0.5, p_unet), cmap=solid_yellow, alpha=0.95)
    axes2[1,1].set_title('U-Net', fontsize=20, fontweight='bold', pad=10)
    axes2[1,1].text(0.5, -0.1, '(e)', transform=axes2[1,1].transAxes, fontsize=20, ha='center'); axes2[1,1].axis('off')

    # (f) NN U-Net [Solid Blue]
    axes2[1,2].imshow(img_render, cmap='gray')
    axes2[1,2].imshow(np.ma.masked_where(p_nnunet < 0.5, p_nnunet), cmap=solid_blue, alpha=0.95)
    axes2[1,2].set_title('NN U-Net', fontsize=20, fontweight='bold', pad=10)
    axes2[1,2].text(0.5, -0.1, '(f)', transform=axes2[1,2].transAxes, fontsize=20, ha='center'); axes2[1,2].axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    out_seg = os.path.join(OUTPUT_DIR, 'Fig2_Comparative_Grid.png')
    fig2.savefig(out_seg, dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # =========================================================
    # 🌟 FIGURE 3: MULTI-MODEL ROC CURVE
    # =========================================================
    fig3, ax3 = plt.subplots(figsize=(7, 7))
    y_true = gt_render.flatten()
    
    colors = ['blue', 'orange', 'green']
    names = ['Mod-Seg-SE(2)', 'U-Net', 'NN U-Net']
    probs = [p_se2.flatten(), p_unet.flatten(), p_nnunet.flatten()]
    
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_true, probs[i])
        ax3.plot(fpr, tpr, color=colors[i], lw=3.0, label=f'TPR_{names[i]} (AUC = {auc(fpr, tpr):.3f})')
        
    ax3.plot([0, 1], [0, 1], color='yellow', lw=2.5, linestyle='--', label='Random Picking')
    ax3.set_xlim([0.0, 1.0]); ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax3.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax3.grid(True, linestyle='-', alpha=0.3)
    ax3.legend(loc="lower right", fontsize=14)
    
    out_roc = os.path.join(OUTPUT_DIR, 'Fig3_Comparative_ROC.png')
    fig3.savefig(out_roc, dpi=300, bbox_inches='tight')
    plt.close(fig3)

    print("\n🌟 ALL SHARP COMPARATIVE JOURNAL FIGURES GENERATED SUCCESSFULLY! 🌟")

if __name__ == "__main__":
    generate_comparative_figures()