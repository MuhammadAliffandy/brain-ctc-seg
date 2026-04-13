import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import label 
plt.switch_backend('agg') # Aman untuk DGX

# E2CNN Specific Libraries
from escnn import gspaces
import escnn.nn as enn

# ==========================================
# 1. MODEL ARCHITECTURE (SE2-CNNET)
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
# 2. DATASET LOADER (KAGGLE PUBLIC DATASET)
# ==========================================
# ==========================================
# 2. DATASET LOADER (KAGGLE PUBLIC DATASET)
# ==========================================
class SimpleDatasetTest(Dataset):
    def __init__(self, root_dir):
        self.slice_pairs = []
        for root, dirs, files in os.walk(root_dir):
            img_files = sorted([f for f in files if f.endswith('_img.npy')])
            for img_name in img_files:
                img_path = os.path.join(root, img_name)
                mask_path = img_path.replace('_img.npy', '_mask.npy')
                if os.path.exists(mask_path):
                    self.slice_pairs.append((img_path, mask_path))

    def __len__(self): return len(self.slice_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.slice_pairs[idx]
        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.uint8)
        
        # Add channel dim if missing
        if len(image.shape) == 2: image = np.expand_dims(image, axis=0)
            
        # Temporarily add Batch dimension for interpolation
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() 
        
        # MAGIC FIX: Force resize to 256x256 (Divisible by 16) to prevent U-Net shape mismatch
        TARGET_SIZE = (256, 256)
        image_tensor = F.interpolate(image_tensor, size=TARGET_SIZE, mode='bilinear', align_corners=False)
        mask_tensor = F.interpolate(mask_tensor, size=TARGET_SIZE, mode='nearest')
        
        # Remove Batch dimension back to original shape
        image_tensor = image_tensor.squeeze(0) 
        mask_tensor = mask_tensor.squeeze(0).squeeze(0).long()
        
        return image_tensor, mask_tensor

# ==========================================
# 3. CLIENT VISUALIZATION ENGINE
# ==========================================
def generate_client_final_visual():
    # PATHS
    # Kita tes pakai data Kaggle (public_dataset_npy) untuk membuktikan model kebal Domain Gap
    TEST_DATA_PATH = os.path.expanduser("~/Clara/public_dataset_npy") 
    
    # KITA GUNAKAN WEIGHTS TERBARU YANG ROBUST!
    ROBUST_MODEL_WEIGHTS = "se2_unet_best_robust.pth" 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using Device: {device}")
    
    print(f"📥 Loading Robust Weights from: {ROBUST_MODEL_WEIGHTS}...")
    model = SE2_CNNET(n_channels=1, n_classes=2, N=8, base_channels=24).to(device)
    
    # strict=False agar tidak error masalah cache wigner_D_matrix
    model.load_state_dict(torch.load(ROBUST_MODEL_WEIGHTS, map_location=device, weights_only=True), strict=False)
    model.eval()

    print("🔍 Scanning test dataset...")
    test_set = SimpleDatasetTest(TEST_DATA_PATH)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4)

    best_image = None
    best_mask = None
    target_tumor_count = 4
    found_target = False
    max_tumors_found = 0
    fallback_image = None
    fallback_mask = None

    print(f"🚀 Hunting for a slice with EXACTLY {target_tumor_count} separate tumors...")
    
    for images, labels in tqdm(test_loader, desc="Hunting"):
        for b in range(labels.size(0)):
            gt_mask_np = labels[b].cpu().numpy()
            
            # Hitung jumlah tumor terpisah
            labeled_array, num_features = label(gt_mask_np)
            
            if num_features > max_tumors_found:
                max_tumors_found = num_features
                fallback_image = images[b].unsqueeze(0)
                fallback_mask = labels[b].unsqueeze(0)
            
            if num_features >= target_tumor_count:
                best_image = images[b].unsqueeze(0)
                best_mask = labels[b].unsqueeze(0)
                target_tumor_count = num_features 
                found_target = True
                break
                
        if found_target:
            print(f"\n🎯 BINGO! Found a slice with {target_tumor_count} distinct tumors.")
            break

    if not found_target:
        print(f"\n⚠️ Could not find exactly {target_tumor_count}. Using the one with {max_tumors_found} tumors.")
        best_image = fallback_image
        best_mask = fallback_mask
        target_tumor_count = max_tumors_found

    best_image = best_image.to(device)
    best_mask = best_mask.to(device)

    # ==========================================
    # PLOTTING
    # ==========================================
    print("🧠 Rendering Visualization...")
    with torch.no_grad():
        logits = model(best_image)
        probs = F.softmax(logits, dim=1)
        prob_map_ai = probs[0, 1, :, :].cpu().numpy() # Ambil probabilitas kelas 1 (Tumor)

    img_np = best_image.cpu().squeeze().numpy()
    gt_np = best_mask.cpu().squeeze().numpy()
    
    # Bikin kanvas gambar
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Robust Model Validation: Detecting {target_tumor_count} Separate Tumors", fontsize=16, fontweight='bold')
    
    # 1. Original CT
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('Original CT Scan', fontsize=14)
    axes[0].axis('off')

    # 2. Ground Truth Dokter
    axes[1].imshow(img_np, cmap='gray')
    masked_gt = np.ma.masked_where(gt_np == 0, gt_np) # Trik transparan
    axes[1].imshow(masked_gt, cmap='Greens', alpha=0.6, vmin=0, vmax=1) 
    axes[1].set_title(f'Ground Truth\n({target_tumor_count} Tumors)', fontsize=14)
    axes[1].axis('off')
    
    # 3. AI Prediction
    axes[2].imshow(img_np, cmap='gray')
    masked_ai = np.ma.masked_where(prob_map_ai < 0.5, prob_map_ai) # Threshold 0.5
    axes[2].imshow(masked_ai, cmap='Reds', alpha=0.6, vmin=0, vmax=1) 
    axes[2].set_title('AI Prediction\n(SE2-CNNET Robust)', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    save_filename = f"Final_Client_Proof_{target_tumor_count}_Tumors.png"
    plt.savefig(save_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Success! Please download: {save_filename} and send it to your client!")

if __name__ == "__main__":
    generate_client_final_visual()