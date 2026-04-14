import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import label 
import imageio 
plt.switch_backend('agg') 

# E2CNN Specific Libraries
from escnn import gspaces
import escnn.nn as enn

# ==========================================
# 1. MODEL ARCHITECTURE (SE2-CNNET)
# ==========================================
# (Arsitektur tetap sama, saya persingkat di komentar agar Anda tinggal copy-paste yang lama jika mau, 
# TAPI di bawah ini sudah saya sertakan full agar aman)
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
# 2. LOGIC PENEMU BANYAK PASIEN (TOP N)
# ==========================================
def find_top_n_patients_for_gif(dataset_path, top_n=3):
    print(f"🔍 Menganalisis semua pasien untuk mencari TOP {top_n} kandidat GIF terbaik...")
    
    patient_dict = {}
    patient_max_tumors = {} # Untuk menyimpan rekor tiap pasien
    
    for root, dirs, files in os.walk(dataset_path):
        img_files = [f for f in files if f.endswith('_img.npy')]
        for img_name in img_files:
            parts = img_name.split('_')
            if len(parts) >= 3:
                patient_id = f"{parts[0]}_{parts[1]}" 
                slice_num = int(parts[2]) 
                
                if patient_id not in patient_dict:
                    patient_dict[patient_id] = []
                    patient_max_tumors[patient_id] = 0
                    
                img_path = os.path.join(root, img_name)
                mask_path = img_path.replace('_img.npy', '_mask.npy')
                
                if os.path.exists(mask_path):
                    # Hitung tumor langsung saat scanning
                    mask_np = np.load(mask_path)
                    _, num_tumors = label(mask_np)
                    if num_tumors > patient_max_tumors[patient_id]:
                        patient_max_tumors[patient_id] = num_tumors
                        
                    patient_dict[patient_id].append({
                        'slice': slice_num,
                        'img_path': img_path,
                        'mask_path': mask_path
                    })

    # Urutkan pasien berdasarkan jumlah tumor terbanyak
    sorted_patients = sorted(patient_max_tumors.items(), key=lambda item: item[1], reverse=True)
    
    top_patients_data = []
    for i in range(min(top_n, len(sorted_patients))):
        pat_id = sorted_patients[i][0]
        max_tumor = sorted_patients[i][1]
        
        # Ambil semua slice pasien tersebut dan urutkan
        sorted_slices = sorted(patient_dict[pat_id], key=lambda x: x['slice'])
        top_patients_data.append((pat_id, sorted_slices, max_tumor))
        print(f"🎯 Kandidat #{i+1}: {pat_id} (Rekor: {max_tumor} tumor terpisah)")
        
    return top_patients_data

# ==========================================
# 3. BATCH GIF GENERATOR ENGINE
# ==========================================
def generate_batch_gifs():
    TEST_DATA_PATH = os.path.expanduser("~/Clara/public_dataset_npy") 
    ROBUST_MODEL_WEIGHTS = "se2_unet_best_robust.pth" 
    TOTAL_SAMPLES = 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using Device: {device}")
    
    print(f"📥 Loading Robust Weights from: {ROBUST_MODEL_WEIGHTS}...")
    model = SE2_CNNET(n_channels=1, n_classes=2, N=8, base_channels=24).to(device)
    model.load_state_dict(torch.load(ROBUST_MODEL_WEIGHTS, map_location=device, weights_only=True), strict=False)
    model.eval()

    # Ambil list pasien terbaik
    top_patients = find_top_n_patients_for_gif(TEST_DATA_PATH, top_n=TOTAL_SAMPLES)
    if not top_patients:
        print("❌ Data tidak ditemukan.")
        return

    # Buat folder khusus agar GIF-nya rapi tidak berceceran
    OUTPUT_DIR = os.path.expanduser("~/Clara/brain-ctc-seg/training/Client_GIFs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # LOOP UNTUK MASING-MASING PASIEN
    for rank, (patient_id, patient_slices, max_tumor) in enumerate(top_patients):
        print(f"\n🎥 [Sample {rank+1}/{len(top_patients)}] Merender Pasien: {patient_id} ({len(patient_slices)} frame)...")
        frames = []
        
        for s_info in tqdm(patient_slices, desc=f"Rendering {patient_id}"):
            slice_idx = s_info['slice']
            
            img_np = np.load(s_info['img_path']).astype(np.float32)
            gt_np = np.load(s_info['mask_path']).astype(np.uint8)
            
            if len(img_np.shape) == 2: img_np = np.expand_dims(img_np, axis=0)
            
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)
            gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).float().to(device)
            
            TARGET_SIZE = (256, 256)
            img_tensor = F.interpolate(img_tensor, size=TARGET_SIZE, mode='bilinear', align_corners=False)
            gt_tensor = F.interpolate(gt_tensor, size=TARGET_SIZE, mode='nearest')
            
            with torch.no_grad():
                logits = model(img_tensor)
                probs = F.softmax(logits, dim=1)
                prob_map_ai = probs[0, 1, :, :].cpu().numpy()

            img_render = img_tensor.squeeze().cpu().numpy()
            gt_render = gt_tensor.squeeze().cpu().numpy()
            
            _, num_tumors_gt = label(gt_render)

            # Plotting Frame
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"Patient {patient_id} - Slice #{slice_idx}\nActive Tumors: {num_tumors_gt}", fontsize=18, fontweight='bold', color='navy')
            
            axes[0].imshow(img_render, cmap='gray')
            axes[0].set_title('Original CT Scan', fontsize=14)
            axes[0].axis('off')

            axes[1].imshow(img_render, cmap='gray')
            masked_gt = np.ma.masked_where(gt_render == 0, gt_render)
            axes[1].imshow(masked_gt, cmap='Greens', alpha=0.6, vmin=0, vmax=1) 
            axes[1].set_title('Doctor Ground Truth', fontsize=14)
            axes[1].axis('off')
            
            axes[2].imshow(img_render, cmap='gray')
            # Kita naikkan threshold ke 0.3 (30%) agar videonya bersih dari noise debu
            masked_ai = np.ma.masked_where(prob_map_ai < 0.3, prob_map_ai)
            axes[2].imshow(masked_ai, cmap='Reds', alpha=0.6, vmin=0, vmax=1) 
            axes[2].set_title('AI Prediction', fontsize=14)
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # MATPLOTLIB BUFFER FIX (YANG BARU)
            fig.canvas.draw()
            rgba_buffer = fig.canvas.buffer_rgba()
            frame = np.asarray(rgba_buffer) 
            frame = frame[:, :, :3] 
            frames.append(frame)
            
            plt.close(fig) 

        # SIMPAN GIF PER PASIEN
        output_filename = os.path.join(OUTPUT_DIR, f"Tumor_Progression_{patient_id}.gif")
        imageio.mimsave(output_filename, frames, duration=0.3, loop=0)
        print(f"✅ GIF tersimpan: {output_filename}")

    print("\n🌟 SEMUA SAMPEL SELESAI DIBUAT! 🌟")
    print(f"📁 Silakan download semua file GIF dari folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_batch_gifs()