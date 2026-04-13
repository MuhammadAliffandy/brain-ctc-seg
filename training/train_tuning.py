import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
# 2. BALANCED DATASET LOADER
# ==========================================
class BalancedCTDataset(Dataset):
    def __init__(self, root_dir):
        print(f"🔍 Analyzing dataset in {root_dir}...")
        tumor_pairs = []
        healthy_pairs = []
        
        # Scan all files
        for root, dirs, files in tqdm(list(os.walk(root_dir)), desc="Scanning & Filtering"):
            img_files = sorted([f for f in files if f.endswith('_img.npy')])
            for img_name in img_files:
                img_path = os.path.join(root, img_name)
                mask_path = img_path.replace('_img.npy', '_mask.npy')
                
                if os.path.exists(mask_path):
                    # Check if mask has any tumor pixels
                    mask_array = np.load(mask_path)
                    if np.sum(mask_array) > 0:
                        tumor_pairs.append((img_path, mask_path))
                    else:
                        healthy_pairs.append((img_path, mask_path))
        
        print(f"📊 Found {len(tumor_pairs)} Tumor slices and {len(healthy_pairs)} Healthy slices.")
        
        # STRATEGI BALANCING: Ambil data sehat HANYA sebanyak data tumor
        random.seed(42)
        random.shuffle(healthy_pairs)
        balanced_healthy = healthy_pairs[:len(tumor_pairs)]
        
        self.slice_pairs = tumor_pairs + balanced_healthy
        random.shuffle(self.slice_pairs) # Acak urutannya
        
        print(f"⚖️ BALANCED DATASET CREATED: Total {len(self.slice_pairs)} slices for Fine-Tuning.")

    def __len__(self):
        return len(self.slice_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.slice_pairs[idx]
        
        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.uint8)
        
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
            
        image_tensor = torch.from_numpy(image).unsqueeze(0) 
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() 
        
        # Resize to 256x256 to save memory and match standard UNet
        TARGET_SIZE = (256, 256) 
        image_tensor = F.interpolate(image_tensor, size=TARGET_SIZE, mode='bilinear', align_corners=False)
        mask_tensor = F.interpolate(mask_tensor, size=TARGET_SIZE, mode='nearest')
        
        image_tensor = image_tensor.squeeze(0) 
        mask_tensor = mask_tensor.squeeze(0).squeeze(0).long() 
        
        return image_tensor, mask_tensor

# ==========================================
# 3. FINE-TUNING ENGINE
# ==========================================
def combined_dice_bce_loss(preds, targets):
    """Kombinasi Loss agar model pintar membedakan batas tumor"""
    # 1. Cross Entropy (BCE)
    bce = F.cross_entropy(preds, targets)
    
    # 2. Dice Loss
    probs = F.softmax(preds, dim=1)[:, 1, ...] # Get probabilities for class 1 (Tumor)
    targets_float = targets.float()
    
    smooth = 1e-6
    intersection = (probs * targets_float).sum()
    dice_loss = 1 - (2. * intersection + smooth) / (probs.sum() + targets_float.sum() + smooth)
    
    return bce + dice_loss

def fine_tune_model():
    # --- CONFIGURATION ---
    DATA_PATH = os.path.expanduser("~/Clara/public_dataset_npy")
    OLD_WEIGHTS_PATH = "se2_unet_epoch_100.pth" 
    NEW_WEIGHTS_PATH = "se2_unet_finetuned.pth"
    
    EPOCHS = 50         # Hanya 10 Epoch karena ini Transfer Learning!
    BATCH_SIZE = 8      # Aman untuk H100
    LEARNING_RATE = 5e-6 # Sangat kecil agar ilmu lama tidak hilang
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Hardware accelerated on: {device}")

    # --- SETUP DATA ---
    dataset = BalancedCTDataset(DATA_PATH)
    if len(dataset) == 0:
        print("❌ Dataset kosong! Pastikan folder preprocessing sudah benar.")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=4, pin_memory=True)

    # --- LOAD MODEL & OLD WEIGHTS ---
    model = SE2_CNNET(n_channels=1, n_classes=2, N=8, base_channels=24).to(device)
    try:
        model.load_state_dict(torch.load(OLD_WEIGHTS_PATH, map_location=device, weights_only=True))
        print(f"✅ Pre-trained weights loaded from {OLD_WEIGHTS_PATH}")
    except Exception as e:
        print(f"❌ Gagal memuat weights: {e}")
        return

    # --- SETUP OPTIMIZER ---
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print("\n" + "🔥"*20)
    print(f"  STARTING FINE-TUNING FOR {EPOCHS} EPOCHS")
    print("🔥"*20)

    # --- TRAINING LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, masks in progress_bar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(images)
            loss = combined_dice_bce_loss(logits, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"📈 Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f}")

    # --- SAVE NEW WEIGHTS ---
    torch.save(model.state_dict(), NEW_WEIGHTS_PATH)
    print("\n" + "🌟"*20)
    print("  FINE-TUNING COMPLETE!")
    print("🌟"*20)
    print(f"💾 Model baru Anda telah disimpan sebagai: {NEW_WEIGHTS_PATH}")
    print("-> Selanjutnya, Anda bisa me-run skrip Evaluasi Laporan Klien dengan model baru ini!")

if __name__ == "__main__":
    fine_tune_model()