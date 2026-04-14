import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import label 
import imageio # LIBRARY BARU UNTUK BIKIN GIF
plt.switch_backend('agg') # Aman untuk server DGX tanpa layar

# E2CNN Specific Libraries
from escnn import gspaces
import escnn.nn as enn

# ==========================================
# 1. MODEL ARCHITECTURE (SE2-CNNET)
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
# 2. LOGIC PENEMU PASIEN TERBAIK UNTUK GIF
# ==========================================
def find_best_patient_for_gif(dataset_path):
    print("🔍 Menganalisis semua pasien untuk mencari kandidat GIF terbaik...")
    
    patient_dict = {}
    
    # Kumpulkan semua file berdasarkan Patient ID
    for root, dirs, files in os.walk(dataset_path):
        img_files = [f for f in files if f.endswith('_img.npy')]
        for img_name in img_files:
            # Format file: Patient_130_12_img.npy
            parts = img_name.split('_')
            if len(parts) >= 3:
                patient_id = f"{parts[0]}_{parts[1]}" # "Patient_130"
                slice_num = int(parts[2]) # "12" -> Ini kedalaman slice-nya
                
                if patient_id not in patient_dict:
                    patient_dict[patient_id] = []
                    
                img_path = os.path.join(root, img_name)
                mask_path = img_path.replace('_img.npy', '_mask.npy')
                
                if os.path.exists(mask_path):
                    patient_dict[patient_id].append({
                        'slice': slice_num,
                        'img_path': img_path,
                        'mask_path': mask_path
                    })

    # Cari pasien yang punya slice dengan jumlah tumor terbanyak (misal 4+)
    best_patient_id = None
    max_tumor_in_any_slice = 0
    
    for patient_id, slices in patient_dict.items():
        for s in slices:
            mask_np = np.load(s['mask_path'])
            _, num_tumors = label(mask_np)
            if num_tumors > max_tumor_in_any_slice:
                max_tumor_in_any_slice = num_tumors
                best_patient_id = patient_id

    print(f"🎯 Pasien Terpilih: {best_patient_id} (Memiliki slice dengan {max_tumor_in_any_slice} tumor terpisah!)")
    
    # Kembalikan daftar slice pasien tersebut, diurutkan dari atas kepala ke leher
    sorted_slices = sorted(patient_dict[best_patient_id], key=lambda x: x['slice'])
    return sorted_slices

# ==========================================
# 3. GIF GENERATOR ENGINE
# ==========================================
def generate_gif_stitching():
    TEST_DATA_PATH = os.path.expanduser("~/Clara/public_dataset_npy") 
    ROBUST_MODEL_WEIGHTS = "se2_unet_best_robust.pth" 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using Device: {device}")
    
    print(f"📥 Loading Robust Weights from: {ROBUST_MODEL_WEIGHTS}...")
    model = SE2_CNNET(n_channels=1, n_classes=2, N=8, base_channels=24).to(device)
    model.load_state_dict(torch.load(ROBUST_MODEL_WEIGHTS, map_location=device, weights_only=True), strict=False)
    model.eval()

    # Cari urutan slice pasien terbaik
    patient_slices = find_best_patient_for_gif(TEST_DATA_PATH)
    if not patient_slices:
        print("❌ Data tidak ditemukan.")
        return

    frames = []
    
    print(f"🎥 Membuat {len(patient_slices)} Frame untuk Video GIF...")
    
    for s_info in tqdm(patient_slices, desc="Rendering Frames"):
        slice_idx = s_info['slice']
        
        # Load & Resize
        img_np = np.load(s_info['img_path']).astype(np.float32)
        gt_np = np.load(s_info['mask_path']).astype(np.uint8)
        
        if len(img_np.shape) == 2: img_np = np.expand_dims(img_np, axis=0)
        
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)
        gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).float().to(device)
        
        TARGET_SIZE = (256, 256)
        img_tensor = F.interpolate(img_tensor, size=TARGET_SIZE, mode='bilinear', align_corners=False)
        gt_tensor = F.interpolate(gt_tensor, size=TARGET_SIZE, mode='nearest')
        
        # AI Prediction
        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)
            prob_map_ai = probs[0, 1, :, :].cpu().numpy()

        img_render = img_tensor.squeeze().cpu().numpy()
        gt_render = gt_tensor.squeeze().cpu().numpy()
        
        # Hitung jumlah tumor aktif di slice ini untuk judul
        _, num_tumors_gt = label(gt_render)

        # Plotting Frame
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"3D Progression Scan - Slice #{slice_idx}\nActive Tumors: {num_tumors_gt}", fontsize=18, fontweight='bold', color='navy')
        
        axes[0].imshow(img_render, cmap='gray')
        axes[0].set_title('Original CT Scan', fontsize=14)
        axes[0].axis('off')

        axes[1].imshow(img_render, cmap='gray')
        masked_gt = np.ma.masked_where(gt_render == 0, gt_render)
        axes[1].imshow(masked_gt, cmap='Greens', alpha=0.6, vmin=0, vmax=1) 
        axes[1].set_title('Doctor Ground Truth', fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(img_render, cmap='gray')
        masked_ai = np.ma.masked_where(prob_map_ai < 0.1, prob_map_ai) # Threshold 10%
        axes[2].imshow(masked_ai, cmap='Reds', alpha=0.6, vmin=0, vmax=1) 
        axes[2].set_title('AI Prediction (Robust)', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Convert Matplotlib Figure to RGB array for GIF
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        
        plt.close(fig) # Cegah Memory Leak

    # JAHIT MENJADI GIF
    output_filename = "3D_Tumor_Progression.gif"
    print(f"\n🎬 Menjahit frame menjadi file {output_filename}...")
    
    # duration = 0.3 artinya jarak antar frame adalah 0.3 detik. 
    # Semakin kecil angkanya, videonya akan semakin cepat.
    imageio.mimsave(output_filename, frames, duration=0.3, loop=0)
    
    print(f"✅ BINGO! Video GIF berhasil dibuat dan tersimpan sebagai: {output_filename}")
    print("-> Segera download file ini dan presentasikan ke Profesor Anda!")

if __name__ == "__main__":
    generate_gif_stitching()