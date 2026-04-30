import os

# Set Kaggle Credentials (if needed)
os.environ["KAGGLE_USERNAME"] = "muhammadaliffandy"
os.environ["KAGGLE_KEY"] = "KGAT_6cf20e173408038efc8c307643a53392"

import torch
import shutil
import torch.nn as nn
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import re
plt.switch_backend('agg') 
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# E2CNN Specific Libraries
from escnn import gspaces
import escnn.nn as enn

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
# 2. DATASET LOADER (2.5D Spatial Context)
# ==========================================
class FullCTDataset(Dataset):
    def __init__(self, root_dir):
        self.all_samples = []
        self.patient_slices = {}
        
        for root, dirs, files in os.walk(root_dir):
            img_files = [f for f in files if f.endswith('_img.npy')]
            if not img_files:
                continue
                
            img_files = sorted(img_files, key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0)
            
            valid_pairs = []
            for img_name in img_files:
                img_path = os.path.join(root, img_name)
                mask_path = img_path.replace('_img.npy', '_mask.npy')
                if os.path.exists(mask_path):
                    valid_pairs.append((img_path, mask_path))
            
            if valid_pairs:
                patient = os.path.basename(root)
                self.patient_slices[patient] = valid_pairs
                for i in range(len(valid_pairs)):
                    self.all_samples.append((patient, i))

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        patient, slice_idx = self.all_samples[idx]
        slices = self.patient_slices[patient]
        
        idx_prev = max(0, slice_idx - 1)
        idx_next = min(len(slices) - 1, slice_idx + 1)
        
        img_prev = np.load(slices[idx_prev][0]).astype(np.float32)
        img_curr = np.load(slices[slice_idx][0]).astype(np.float32)
        img_next = np.load(slices[idx_next][0]).astype(np.float32)
        
        mask = np.load(slices[slice_idx][1]).astype(np.uint8) 
        
        # Stack into 3 channels exactly like train.py
        image_25d = np.stack([img_prev, img_curr, img_next], axis=-1)
        
        # Ensure mask is exactly 0 and 1
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)

        image_tensor = torch.from_numpy(image_25d).permute(2, 0, 1).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() 
        
        # Fix U-Net Crash: Model requires dims divisible by 16 (since it has 4 downsample layers)
        # Resizing to 256x256 matches the standard 2.5D training resolution
        TARGET_SIZE = (256, 256)
        image_tensor = F.interpolate(image_tensor, size=TARGET_SIZE, mode='bilinear', align_corners=False)
        mask_tensor = F.interpolate(mask_tensor, size=TARGET_SIZE, mode='nearest')
        
        image_tensor = image_tensor.squeeze(0)
        mask_tensor = mask_tensor.squeeze(0).squeeze(0).long()
        
        filename = os.path.basename(slices[slice_idx][0])
        return image_tensor, mask_tensor, filename

# ==========================================
# 3. GLOBAL EVALUATION ENGINE & CLIENT REPORTER
# ==========================================
def calculate_tumor_size(binary_mask, pixel_spacing_mm=0.5, slice_thickness_mm=1.0):
    pixel_count = np.sum(binary_mask == 1)
    area_mm2 = pixel_count * (pixel_spacing_mm ** 2)
    volume_mm3 = area_mm2 * slice_thickness_mm
    
    return {
        "Pixel_Count": int(pixel_count),
        "Area_cm2": round(area_mm2 / 100.0, 4),
        "Volume_cm3": round(volume_mm3 / 1000.0, 4)
    }

def evaluate_all():
    PUBLIC_DATA_PATH = os.path.expanduser("~/Clara/public_dataset_npy")
    if not os.path.exists(PUBLIC_DATA_PATH):
        print("❌ Dataset tidak ditemukan. Harap pastikan data NPY sudah ada.")
        return

    # --- NEW CLIENT DIRECTORY ---
    CLIENT_REPORTS_DIR = os.path.expanduser("~/Clara/client_reports_finetuned")
    VISUALS_DIR = os.path.join(CLIENT_REPORTS_DIR, "visual_proofs")
    os.makedirs(VISUALS_DIR, exist_ok=True)
    
    report_data_list = []

    # --- USE NEW FINETUNED WEIGHTS ---
    MODEL_WEIGHTS_PATH = "saved_models_25D/se2_unet_best_25D_Boundary.pth" 
    BATCH_SIZE = 4 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SE2_CNNET(n_channels=3, n_classes=2, N=8, base_channels=24).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True), strict=False)
        print(f"✅ Loaded NEW FINETUNED weights from {MODEL_WEIGHTS_PATH}")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return
        
    model.eval()

    dataset = FullCTDataset(PUBLIC_DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, pin_memory=True)

    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0

    print("\n⚡ Running Evaluation & Generating Client Reports (Finetuned Version)...")
    
    with torch.no_grad():
        for images, masks, filenames in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                logits = model(images)
                
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1) 
            
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)
            total_tp += torch.sum((preds_flat == 1) & (masks_flat == 1)).item()
            total_tn += torch.sum((preds_flat == 0) & (masks_flat == 0)).item()
            total_fp += torch.sum((preds_flat == 1) & (masks_flat == 0)).item()
            total_fn += torch.sum((preds_flat == 0) & (masks_flat == 1)).item()

            for i in range(len(filenames)):
                filename = filenames[i].replace('_img.npy', '')
                true_mask_np = masks[i].cpu().numpy()
                pred_mask_np = preds[i].cpu().numpy()
                img_np = images[i][1].cpu().numpy() # 2.5D middle slice
                
                size_metrics = calculate_tumor_size(pred_mask_np)
                
                report_data_list.append({
                    "Slice_ID": filename,
                    "Has_Ground_Truth_Tumor": "Yes" if np.sum(true_mask_np) > 0 else "No",
                    "AI_Detected_Tumor": "Yes" if size_metrics["Pixel_Count"] > 0 else "No",
                    "Estimated_Area_cm2": size_metrics["Area_cm2"],
                    "Estimated_Volume_cm3": size_metrics["Volume_cm3"],
                    "Detected_Pixels": size_metrics["Pixel_Count"]
                })
                
                if np.sum(true_mask_np) > 0 or size_metrics["Pixel_Count"] > 0:
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    
                    # --- MATPLOTLIB BACKGROUND FIX ---
                    true_masked = np.ma.masked_where(true_mask_np == 0, true_mask_np)
                    pred_masked = np.ma.masked_where(pred_mask_np == 0, pred_mask_np)
                    
                    axes[0].imshow(img_np, cmap='gray')
                    axes[0].set_title('Original CT')
                    axes[0].axis('off')
                    
                    axes[1].imshow(img_np, cmap='gray')
                    axes[1].imshow(true_masked, cmap='Greens', alpha=0.6, vmin=0, vmax=1) 
                    axes[1].set_title('Doctor Ground Truth')
                    axes[1].axis('off')
                    
                    axes[2].imshow(img_np, cmap='gray')
                    axes[2].imshow(pred_masked, cmap='Reds', alpha=0.6, vmin=0, vmax=1) 
                    axes[2].set_title(f'AI Prediction\nVol: {size_metrics["Volume_cm3"]} cm3')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(VISUALS_DIR, f"{filename}_proof.jpg"), dpi=150, bbox_inches='tight')
                    plt.close(fig) 

    # --- SAVE CSV REPORT ---
    report_df = pd.DataFrame(report_data_list)
    csv_path = os.path.join(CLIENT_REPORTS_DIR, "Tumor_Size_Estimation_Report_Finetuned.csv")
    report_df.to_csv(csv_path, index=False)

    # --- PRINT GLOBAL METRICS ---
    epsilon = 1e-6 
    global_dice = (2.0 * total_tp) / ((2.0 * total_tp) + total_fp + total_fn + epsilon)
    global_iou = total_tp / (total_tp + total_fp + total_fn + epsilon)

    print("\n" + "🌟"*20)
    print(f"  CLIENT DELIVERABLES READY! (FINETUNED)")
    print("🌟"*20)
    print(f"🔥 Global Dice Score : {global_dice:.4f} ({(global_dice*100):.2f}%)")
    print(f"🎯 Global IoU Score  : {global_iou:.4f} ({(global_iou*100):.2f}%)")
    print(f"\n📁 VISUAL PROOFS SAVED TO : {VISUALS_DIR}")
    print(f"📄 EXCEL/CSV REPORT SAVED TO: {csv_path}")

if __name__ == "__main__":
    evaluate_all()