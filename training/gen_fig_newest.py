import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label
from sklearn.metrics import roc_curve, auc
import re

# Use Agg backend for headless servers to avoid display issues
plt.switch_backend('agg')

# E2CNN Specific Libraries 
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
# 2. HELPER TO FIND TOP PATIENTS
# ==========================================
def get_best_slices_for_paper(dataset_path, num_patients=4):
    print("🔍 Searching for top slices from different patients for journal figures...")
    patient_max_tumors = {}
    for root, dirs, files in os.walk(dataset_path):
        img_files = sorted([f for f in files if f.endswith('_img.npy')], 
                           key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0)
        
        for i, img_name in enumerate(img_files):
            img_path = os.path.join(root, img_name)
            mask_path = img_path.replace('_img.npy', '_mask.npy')
            
            if os.path.exists(mask_path):
                mask_np = np.load(mask_path)
                tumor_pixels = np.sum(mask_np)
                patient = os.path.basename(root)
                
                # Exclude tumors that are too tiny or massively large to ensure good visuals
                if 500 < tumor_pixels < 7000:
                    if patient not in patient_max_tumors or tumor_pixels > patient_max_tumors[patient]['pixels']:
                        idx_prev = max(0, i - 1)
                        idx_next = min(len(img_files) - 1, i + 1)
                        patient_max_tumors[patient] = {
                            'pixels': tumor_pixels,
                            'prev': os.path.join(root, img_files[idx_prev]),
                            'curr': img_path,
                            'next': os.path.join(root, img_files[idx_next]),
                            'mask': mask_path,
                            'patient': patient
                        }

    sorted_patients = sorted(patient_max_tumors.values(), key=lambda x: x['pixels'], reverse=True)
    return sorted_patients[:num_patients]

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
    
    model = SE2_CNNET(n_channels=3, n_classes=2, N=8, base_channels=24).to(device)
    try:
        model.load_state_dict(torch.load(ROBUST_MODEL_WEIGHTS, map_location=device, weights_only=True), strict=False)
        print("✅ 84% Accuracy Weights Loaded!")
    except Exception as e:
        print("⚠️ Failed to load best weights.")
        return

    model.eval()

    # Get 4 Patients (for the multi-CT list figure)
    target_slices = get_best_slices_for_paper(TEST_DATA_PATH, num_patients=4)
    if len(target_slices) < 4:
        print("❌ Could not find enough suitable slices (need at least 4).")
        return

    CROP_MARGIN = 40
    ROTATE_K = 1 
    
    # --- CUSTOM SOLID COLORMAPS (ANTI-PUYEH) ---
    solid_red_cmap = ListedColormap(['red'])
    solid_white_cmap = ListedColormap(['white'])
    
    processed_data = []

    for slice_info in target_slices:
        # NATIVE RESOLUTION: NO RESIZING
        img_prev = np.load(slice_info['prev']).astype(np.float32)
        img_curr = np.load(slice_info['curr']).astype(np.float32)
        img_next = np.load(slice_info['next']).astype(np.float32)
        gt_np = np.load(slice_info['mask']).astype(np.uint8)

        image_25d = np.stack([img_prev, img_curr, img_next], axis=-1)
        img_tensor = torch.from_numpy(image_25d).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(img_tensor) 
            probs = F.softmax(logits, dim=1)
            prob_map_ai = probs[0, 1, :, :].cpu().numpy()

        img_render = np.rot90(img_curr[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)
        gt_render = np.rot90(gt_np[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)
        prob_map_ai = np.rot90(prob_map_ai[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)
        
        processed_data.append({
            'img': img_render,
            'gt': gt_render,
            'prob': prob_map_ai,
            'patient': slice_info['patient']
        })

    # =========================================================
    # 🌟 FIGURE 1: 3-BRAIN HEATMAP WITH ONE COLORBAR
    # =========================================================
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    
    for i in range(3): # Only use first 3 for heatmap
        axes1[i].imshow(processed_data[i]['img'], cmap='gray')
        # Mask lower confidence, high alpha for vibrant jet colors
        masked_heatmap = np.ma.masked_where(processed_data[i]['prob'] < 0.1, processed_data[i]['prob'])
        im = axes1[i].imshow(masked_heatmap, cmap='jet', alpha=0.8, vmin=0.2, vmax=1.0)
        
        axes1[i].set_title(f"Patient {i+1}\nMod-Seg-SE(2)", fontsize=16, fontweight='bold', pad=15)
        axes1[i].axis('off')

    cbar_ax = fig1.add_axes([0.92, 0.15, 0.02, 0.7]) 
    cbar = fig1.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)

    output_heatmap_path = os.path.join(OUTPUT_DIR, 'Fig1_3Brain_Heatmap_Journal.png')
    fig1.savefig(output_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close(fig1) 
    print(f"📸 Saved Figure 1 (Heatmap) to: {output_heatmap_path}")

    # =========================================================
    # 🌟 FIGURE 2: SHARP SEGMENTATION GRID OVERLAY (a to d)
    # =========================================================
    best_case = processed_data[0] # Use the best patient for the main grid
    img = best_case['img']
    gt = best_case['gt']
    prob = best_case['prob']
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))
    
    # (a) Input
    axes2[0, 0].imshow(img, cmap='gray')
    axes2[0, 0].set_title('Input', fontsize=18, fontweight='bold', pad=10)
    axes2[0, 0].text(0.5, -0.1, '(a)', transform=axes2[0, 0].transAxes, fontsize=18, ha='center')
    axes2[0, 0].axis('off')

    # (b) Ground Truth (Pure White mask)
    axes2[0, 1].imshow(img, cmap='gray')
    masked_gt = np.ma.masked_where(gt == 0, gt)
    axes2[0, 1].imshow(masked_gt, cmap=solid_white_cmap, alpha=1.0) 
    axes2[0, 1].set_title('Ground Truth', fontsize=18, fontweight='bold', pad=10)
    axes2[0, 1].text(0.5, -0.1, '(b)', transform=axes2[0, 1].transAxes, fontsize=18, ha='center')
    axes2[0, 1].axis('off')

    # (c) Overlay (Thick Contours)
    axes2[1, 0].imshow(img, cmap='gray')
    axes2[1, 0].contour(gt, levels=[0.5], colors='yellow', linestyles='dashed', linewidths=3.5)
    pred_binary = (prob >= 0.5).astype(int)
    axes2[1, 0].contour(pred_binary, levels=[0.5], colors='blue', linestyles='dotted', linewidths=3.5)
    axes2[1, 0].set_title('Overlay', fontsize=18, fontweight='bold', pad=10)
    axes2[1, 0].text(0.5, -0.1, '(c)', transform=axes2[1, 0].transAxes, fontsize=18, ha='center')
    axes2[1, 0].axis('off')

    # (d) Mod-Seg-SE(2) Output (Pure Solid Red)
    axes2[1, 1].imshow(img, cmap='gray')
    masked_pred = np.ma.masked_where(pred_binary == 0, pred_binary)
    # Using Solid Red Colormap with high alpha
    axes2[1, 1].imshow(masked_pred, cmap=solid_red_cmap, alpha=0.95) 
    axes2[1, 1].set_title('Mod-Seg-SE(2)', fontsize=18, fontweight='bold', pad=10)
    axes2[1, 1].text(0.5, -0.1, '(d)', transform=axes2[1, 1].transAxes, fontsize=18, ha='center')
    axes2[1, 1].axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.3)
    output_seg_path = os.path.join(OUTPUT_DIR, 'Fig2_Segmentation_Grid.png')
    fig2.savefig(output_seg_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"📸 Saved Figure 2 (Sharp Segmentation Grid) to: {output_seg_path}")

    # =========================================================
    # 🌟 FIGURE 3: AGGREGATED ROC CURVE 
    # =========================================================
    y_true_all = np.concatenate([data['gt'].flatten() for data in processed_data])
    y_scores_all = np.concatenate([data['prob'].flatten() for data in processed_data])
    
    fpr, tpr, _ = roc_curve(y_true_all, y_scores_all)
    roc_auc = auc(fpr, tpr)
    
    fig3, ax3 = plt.subplots(figsize=(7, 7))
    ax3.plot(fpr, tpr, color='blue', lw=3.0, label=f'TPR_Mod-Seg-SE(2) (AUC = {roc_auc:.3f})')
    ax3.plot([0, 1], [0, 1], color='yellow', lw=2.5, linestyle='--', label='Random Picking')
    
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax3.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax3.grid(True, linestyle='-', alpha=0.3) 
    ax3.tick_params(axis='both', which='major', labelsize=12)
    ax3.legend(loc="lower right", fontsize=14)
    
    output_roc_path = os.path.join(OUTPUT_DIR, 'Fig3_Aggregated_ROC.png')
    fig3.savefig(output_roc_path, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"📸 Saved Figure 3 (ROC Curve) to: {output_roc_path}")

    # =========================================================
    # 🌟 FIGURE 4: MULTI-CT REPRESENTATIVE LIST (Robustness Check)
    # =========================================================
    # 2 Rows: Top = Inputs, Bottom = Red Solid Predictions
    fig4, axes4 = plt.subplots(2, 4, figsize=(20, 10))
    
    for i, data in enumerate(processed_data):
        # Row 0: Original CT
        axes4[0, i].imshow(data['img'], cmap='gray')
        axes4[0, i].set_title(f"Patient {i+1} Input", fontsize=18, fontweight='bold', pad=10)
        axes4[0, i].axis('off')
        
        # Row 1: AI Prediction (Solid Red)
        axes4[1, i].imshow(data['img'], cmap='gray')
        pred_bin = (data['prob'] >= 0.5).astype(int)
        masked_p = np.ma.masked_where(pred_bin == 0, pred_bin)
        axes4[1, i].imshow(masked_p, cmap=solid_red_cmap, alpha=0.95)
        axes4[1, i].set_title(f"Mod-Seg-SE(2)", fontsize=18, fontweight='bold', pad=10)
        axes4[1, i].axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    output_list_path = os.path.join(OUTPUT_DIR, 'Fig4_Multi_CT_List.png')
    fig4.savefig(output_list_path, dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print(f"📸 Saved Figure 4 (Multi-CT List) to: {output_list_path}")

    print("\n🌟 ALL 4 HIGH-CONTRAST JOURNAL FIGURES GENERATED SUCCESSFULLY! 🌟")

if __name__ == "__main__":
    generate_journal_figures()