import os
import sys
import datetime
import zipfile
import shutil
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

# =========================================
# 1. DATA PREPARATION (DGX Local NVMe Optimized)
# ==========================================

def prepare_local_data(gdrive_dir, local_extract_dir):
    """
    Checks if data needs to be copied/extracted from rclone mount to local server storage.
    """
    os.makedirs(local_extract_dir, exist_ok=True)
    
    zip_files = [f for f in os.listdir(gdrive_dir) if f.endswith('.zip')]
    if zip_files:
        print(f"📦 Found {len(zip_files)} .zip files. Extracting to {local_extract_dir}...")
        for z_file in tqdm(zip_files, desc="Extracting Zips"):
            patient_name = z_file.replace('.zip', '')
            target_folder = os.path.join(local_extract_dir, patient_name)
            if not os.path.exists(target_folder):
                try:
                    with zipfile.ZipFile(os.path.join(gdrive_dir, z_file), 'r') as zip_ref:
                        zip_ref.extractall(target_folder)
                except Exception as e:
                    print(f"⚠️ Error extracting {z_file}: {e}")
        print("✅ Zip extraction complete!")
        return local_extract_dir
    
    sub_folders = [f for f in os.listdir(gdrive_dir) if os.path.isdir(os.path.join(gdrive_dir, f))]
    if sub_folders:
        print(f"📁 Found standard folders. Copying to {local_extract_dir}...")
        for folder in tqdm(sub_folders, desc="Copying Folders"):
            src = os.path.join(gdrive_dir, folder)
            dst = os.path.join(local_extract_dir, folder)
            if not os.path.exists(dst):
                shutil.copytree(src, dst)
        print("✅ Data copy complete!")
        return local_extract_dir
        
    print("⚠️ No valid data found in the GDrive path!")
    return local_extract_dir

class CTBrainDataset(Dataset):
    """
    Modified for Slice-Level Cross Validation.
    Receives a direct list of (image_path, mask_path) tuples.
    """
    def __init__(self, slice_pairs):
        self.slice_pairs = slice_pairs

    def __len__(self):
        return len(self.slice_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.slice_pairs[idx]
        try:
            image = np.load(img_path).astype(np.float32)
            mask = np.load(mask_path).astype(np.uint8)
            image = torch.from_numpy(image).unsqueeze(0)
            mask = torch.from_numpy(mask).long() 
            return image, mask
        except Exception as e:
            # Fallback if a specific slice is corrupted
            random_idx = random.randint(0, len(self.slice_pairs) - 1)
            return self.__getitem__(random_idx)

# ==========================================
# 2. STANDARD U-NET ARCHITECTURE (BASELINE)
# Uses standard PyTorch 2D Convolutions
# ==========================================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Using bilinear interpolation matching the original GDeconv logic
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Pad if dimensions don't match exactly due to pooling
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # Concatenate along the channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class StandardUNet(nn.Module):
    """Standard Baseline U-Net Model"""
    def __init__(self, n_channels, n_classes, base_channels=24):
        super(StandardUNet, self).__init__()
        c = base_channels
        
        self.inc = DoubleConv(n_channels, c)
        self.down1 = Down(c, c*2)
        self.down2 = Down(c*2, c*4)
        self.down3 = Down(c*4, c*8)
        self.down4 = Down(c*8, c*16)
        
        # In channels = incoming upsampled features (c*16) + skip connection features (c*8)
        self.up1 = Up(c*16 + c*8, c*8)
        self.up2 = Up(c*8 + c*4, c*4)
        self.up3 = Up(c*4 + c*2, c*2)
        self.up4 = Up(c*2 + c, c)
        
        self.outc = nn.Conv2d(c, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return self.outc(x)

# ==========================================
# 3. LOSS FUNCTIONS
# ==========================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true_masks):
        num_classes = logits.shape[1]
        true_masks_one_hot = F.one_hot(true_masks, num_classes).permute(0, 3, 1, 2).float()
        probs = F.softmax(logits, dim=1)
        
        probs_target = probs[:, 1, :, :]
        true_target = true_masks_one_hot[:, 1, :, :]
        
        intersection = (probs_target * true_target).sum(dim=(1, 2))
        union = probs_target.sum(dim=(1, 2)) + true_target.sum(dim=(1, 2))
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice_score.mean()

class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return (self.weight_ce * ce) + (self.weight_dice * dice)

# ==========================================
# 4. SLICE-LEVEL 10-FOLD CROSS VALIDATION
# ==========================================

def train():
    GDRIVE_ROOT = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive")
    GDRIVE_DATA_DIR = os.path.join(GDRIVE_ROOT, "Dataset_CT_Preprocessed_NPY") 
    CSV_REPORT = os.path.join(GDRIVE_ROOT, "Dataset_CT_Report.csv")
    LOCAL_DATA_PATH = os.path.expanduser("~/Clara/local_ct_workspace") 
    
    local_root = prepare_local_data(GDRIVE_DATA_DIR, LOCAL_DATA_PATH)

    # HYPERPARAMETERS
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4
    ACCUMULATION_STEPS = 8
    EPOCHS = 20
    K_FOLDS = 10 
    NUM_CLASSES = 2     
    INPUT_CHANNELS = 1  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device configured: {device}")

    if not os.path.exists(CSV_REPORT):
        raise FileNotFoundError(f"Cannot find CSV report at {CSV_REPORT}")
    df = pd.read_csv(CSV_REPORT)
    patient_col = 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'
    
    # 1. GATHER ALL SLICES INTO A GLOBAL LIST
    print("Scanning directories for valid slices...")
    all_slices_global = []
    for patient in df[patient_col].unique():
        patient_dir = os.path.join(local_root, patient)
        if os.path.exists(patient_dir):
            img_files = sorted([f for f in os.listdir(patient_dir) if f.endswith('_img.npy')])
            for img_name in img_files:
                img_path = os.path.join(patient_dir, img_name)
                mask_path = img_path.replace('_img.npy', '_mask.npy')
                if os.path.exists(mask_path):
                    all_slices_global.append((img_path, mask_path))

    total_slices = len(all_slices_global)
    print(f"✅ Found {total_slices} valid image-mask pairs in total.")
    
    all_slices_global = np.array(all_slices_global)

    # 2. INITIALIZE SLICE-LEVEL K-FOLD
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_slices_global)):
        print(f"\n{'='*40}")
        print(f"🚀 STARTING FOLD {fold + 1}/{K_FOLDS}")
        print(f"{'='*40}")

        train_pairs = all_slices_global[train_idx].tolist()
        val_pairs = all_slices_global[val_idx].tolist()

        train_set = CTBrainDataset(train_pairs)
        val_set = CTBrainDataset(val_pairs)

        num_workers = min(os.cpu_count(), 16) if os.cpu_count() else 4
        
        train_loader = DataLoader(
            train_set, batch_size=BATCH_SIZE, shuffle=True, 
            pin_memory=True, num_workers=num_workers, 
            prefetch_factor=2, persistent_workers=True
        )
        val_loader = DataLoader(
            val_set, batch_size=BATCH_SIZE, shuffle=False, 
            pin_memory=True, num_workers=num_workers,
            prefetch_factor=2, persistent_workers=True
        )

        # Initialize the Baseline Standard U-Net
        model = StandardUNet(n_channels=INPUT_CHANNELS, n_classes=NUM_CLASSES, base_channels=24).to(device)
        class_weights = torch.tensor([1.0, 50.0]).to(device)
        criterion = CombinedLoss(weight_ce=1.0, weight_dice=1.0, class_weights=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scaler = torch.amp.GradScaler('cuda')

        best_val_loss = float('inf')

        # Epoch Loop
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()

            pbar_train = tqdm(train_loader, desc=f"Fold {fold+1} - Epoch {epoch+1}/{EPOCHS} [Train]")
            for i, (images, labels) in enumerate(pbar_train):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss = loss / ACCUMULATION_STEPS 

                scaler.scale(loss).backward()

                if (i + 1) % ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                running_loss += loss.item() * ACCUMULATION_STEPS
                pbar_train.set_postfix({'loss': loss.item() * ACCUMULATION_STEPS})
                del outputs, loss

            if len(train_loader) % ACCUMULATION_STEPS != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            avg_train_loss = running_loss / len(train_loader)

            # Validation Phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc=f"Fold {fold+1} - Epoch {epoch+1}/{EPOCHS} [Val]")
                for images, labels in pbar_val:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with torch.amp.autocast('cuda'): 
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    pbar_val.set_postfix({'val_loss': loss.item()})
                    del outputs, loss

            avg_val_loss = val_loss / len(val_loader)
            print(f"Fold {fold+1} | Epoch {epoch+1}/{EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Saved name changed to standard_unet_best_fold
                save_path = f'standard_unet_best_fold_{fold+1}.pth'
                torch.save(model.state_dict(), save_path)
                print(f"🌟 Best Baseline U-Net saved for Fold {fold+1} with Val Loss: {best_val_loss:.4f}")
                
            torch.cuda.empty_cache()
        
        fold_results.append(best_val_loss)
        print(f"✅ Fold {fold+1} Complete. Best Val Loss: {best_val_loss:.4f}")

    print(f"\n{'='*40}")
    print(f"🎉 STANDARD U-NET 10-FOLD CV FINISHED!")
    print(f"Average Best Validation Loss across 10 folds: {sum(fold_results)/K_FOLDS:.4f}")
    print(f"{'='*40}")

# ==========================================
# 5. LOGGING UTILITY
# ==========================================

class Logger:
    def __init__(self, filename, stream):
        self.terminal = stream
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Log filename updated to reflect standard unet
    log_filename = f"training_unet_baseline_cv_log_{timestamp}.txt"
    sys.stdout = Logger(log_filename, sys.stdout)
    sys.stderr = Logger(log_filename, sys.stderr)
    
    print(f"📝 Logging Baseline U-Net CV terminal output to {log_filename}")
    train()