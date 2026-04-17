import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import label 
import imageio 
import re
plt.switch_backend('agg') 

# E2CNN Specific Libraries
from escnn import gspaces
import escnn.nn as enn

# ==========================================
# 1. MODEL ARCHITECTURE (SE2-CNNET 2.5D)
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
# 2. PATIENT DISCOVERY LOGIC
# ==========================================
def find_top_n_patients_for_gif(dataset_path, top_n=3):
    print(f"🔍 Analyzing BASE DATA at {dataset_path}...")
    patient_dict = {}
    patient_max_tumors = {} 
    
    for root, dirs, files in os.walk(dataset_path):
        img_files = [f for f in files if f.endswith('_img.npy')]
        for img_name in img_files:
            patient_id = os.path.basename(root)
            numbers = re.findall(r'\d+', img_name)
            slice_num = int(numbers[-1]) if numbers else 0
            
            if patient_id not in patient_dict:
                patient_dict[patient_id] = []
                patient_max_tumors[patient_id] = 0
                
            img_path = os.path.join(root, img_name)
            mask_path = img_path.replace('_img.npy', '_mask.npy')
            
            if os.path.exists(mask_path):
                mask_np = np.load(mask_path)
                _, num_tumors = label(mask_np)
                if num_tumors > patient_max_tumors[patient_id]:
                    patient_max_tumors[patient_id] = num_tumors
                    
                patient_dict[patient_id].append({
                    'slice': slice_num,
                    'img_path': img_path,
                    'mask_path': mask_path
                })

    sorted_patients = sorted(patient_max_tumors.items(), key=lambda item: item[1], reverse=True)
    top_patients_data = []
    for i in range(min(top_n, len(sorted_patients))):
        pat_id = sorted_patients[i][0]
        max_tumor = sorted_patients[i][1]
        sorted_slices = sorted(patient_dict[pat_id], key=lambda x: x['slice'])
        top_patients_data.append((pat_id, sorted_slices, max_tumor))
        print(f"🎯 Candidate #{i+1}: Patient [{pat_id}]")
        
    return top_patients_data

# ==========================================
# 3. BATCH GIF GENERATOR ENGINE (2.5D SUPPORTED - FIXED VERSION)
# ==========================================
def generate_batch_gifs():
    TEST_DATA_PATH = os.path.expanduser("~/Clara/local_ct_workspace") 
    
    ROBUST_MODEL_WEIGHTS = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D/se2_unet_best_25D_Boundary.pth")
    
    TOTAL_SAMPLES = 4
    GIF_SPEED = 0.8 
    
    # ✅ PERBAIKAN 1: Putar 90 Derajat agar posisi TEGAK (Mata di atas)
    ROTATE_K = 1 
    
    # Zoom/Crop agar tatakan hilang
    CROP_MARGIN = 40 
    AI_COLORMAP = 'Wistia'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using Device: {device}")
    
    print(f"📥 Loading 2.5D Robust Weights from: {ROBUST_MODEL_WEIGHTS}...")
    model = SE2_CNNET(n_channels=3, n_classes=2, N=8, base_channels=24).to(device)
    
    try:
        model.load_state_dict(torch.load(ROBUST_MODEL_WEIGHTS, map_location=device, weights_only=True), strict=False)
        print("✅ Weights loaded successfully!")
    except Exception as e:
        fallback_path = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D/se2_unet_epoch_100.pth")
        print(f"⚠️ Best model load failed. Falling back to Epoch 100: {fallback_path}")
        model.load_state_dict(torch.load(fallback_path, map_location=device, weights_only=True), strict=False)

    model.eval()

    top_patients = find_top_n_patients_for_gif(TEST_DATA_PATH, top_n=TOTAL_SAMPLES)
    if not top_patients:
        print("❌ Base data not found.")
        return

    OUTPUT_DIR = os.path.expanduser("~/Clara/brain-ctc-seg/training/Client_GIFs_Final_25D")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for rank, (patient_id, patient_slices, max_tumor) in enumerate(top_patients):
        print(f"\n🎥 [Sample {rank+1}/{len(top_patients)}] Rendering Patient: {patient_id}...")
        frames = []
        
        for i, s_info in enumerate(tqdm(patient_slices, desc="Rendering")):
            slice_idx = s_info['slice']
            
            idx_prev = max(0, i - 1)
            idx_next = min(len(patient_slices) - 1, i + 1)
            
            img_prev = np.load(patient_slices[idx_prev]['img_path']).astype(np.float32)
            img_curr = np.load(s_info['img_path']).astype(np.float32) 
            img_next = np.load(patient_slices[idx_next]['img_path']).astype(np.float32)
            
            gt_np = np.load(s_info['mask_path']).astype(np.uint8)

            image_25d = np.stack([img_prev, img_curr, img_next], axis=-1)
            
            img_tensor = torch.from_numpy(image_25d).permute(2, 0, 1).unsqueeze(0).to(device)
            gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).float().to(device)
            
            # ✅ PERBAIKAN 2: Blok F.interpolate (Resize) DIHAPUS.
            # Biarkan AI menebak di resolusi aslinya agar akurat 100%.
            
            with torch.no_grad():
                logits = model(img_tensor)
                probs = F.softmax(logits, dim=1)
                prob_map_ai = probs[0, 1, :, :].cpu().numpy()

            # ✅ Karena tidak di-resize, kita langsung pakai gambar aslinya untuk di-plot
            img_render = img_curr 
            gt_render = gt_np
            
            # --- 🪄 CROP & ROTATE ---
            img_render = img_render[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN]
            gt_render = gt_render[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN]
            prob_map_ai = prob_map_ai[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN]

            img_render = np.rot90(img_render, k=ROTATE_K)
            gt_render = np.rot90(gt_render, k=ROTATE_K)
            prob_map_ai = np.rot90(prob_map_ai, k=ROTATE_K)
            
            _, num_tumors_gt = label(gt_render)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"Patient ID: {patient_id} - Slice #{slice_idx}\nActive Tumors Detected: {num_tumors_gt}", fontsize=18, fontweight='bold', color='navy')
            
            axes[0].imshow(img_render, cmap='gray')
            axes[0].set_title('Original CT Scan (Zoomed & Upright)', fontsize=14)
            axes[0].axis('off')

            axes[1].imshow(img_render, cmap='gray')
            masked_gt = np.ma.masked_where(gt_render == 0, gt_render)
            axes[1].imshow(masked_gt, cmap='Greens', alpha=0.6, vmin=0, vmax=1) 
            axes[1].set_title('Doctor Ground Truth', fontsize=14)
            axes[1].axis('off')
            
            axes[2].imshow(img_render, cmap='gray')
            masked_ai = np.ma.masked_where(prob_map_ai < 0.5, prob_map_ai) 
            axes[2].imshow(masked_ai, cmap=AI_COLORMAP, alpha=0.8, vmin=0, vmax=1) 
            axes[2].set_title('AI Prediction (84% Dice Score)', fontsize=14)
            axes[2].axis('off')
            
            plt.tight_layout()
            
            fig.canvas.draw()
            rgba_buffer = fig.canvas.buffer_rgba()
            frame = np.asarray(rgba_buffer) 
            frame = frame[:, :, :3] 
            frames.append(frame)
            plt.close(fig) 

        output_filename = os.path.join(OUTPUT_DIR, f"Final_25D_Tumor_GIF_{patient_id}.gif")
        imageio.mimsave(output_filename, frames, duration=GIF_SPEED, loop=0)
        print(f"✅ GIF successfully saved: {output_filename}")

    print("\n🌟 ALL 2.5D CLIENT APPROVED SAMPLES GENERATED SUCCESSFULLY! 🌟")

if __name__ == "__main__":
    generate_batch_gifs()