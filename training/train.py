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
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# E2CNN Specific Libraries
from escnn import gspaces
import escnn.nn as enn

# ==========================================
# 1. DATA PREPARATION (DGX Local NVMe Optimized)
# ==========================================

def prepare_local_data(gdrive_dir, local_extract_dir):
    """
    Checks if data needs to be copied/extracted from rclone mount to local server storage.
    Prevents massive I/O bottlenecks during training.
    """
    os.makedirs(local_extract_dir, exist_ok=True)
    
    # Check if data is stored as .zip files (from our chunking method)
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
            else:
                pass # Already extracted
        print("✅ Zip extraction complete!")
        return local_extract_dir
    
    # Alternatively, if data is just standard folders, copy them
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
    Robust Loader for preprocessed .npy slices.
    Scans directory dynamically to prevent FileNotFoundError.
    """
    def __init__(self, dataframe, root_dir):
        self.root_dir = root_dir
        self.slice_pairs = []

        # Support both CSV column name variants
        patient_col = 'Patient_Folder' if 'Patient_Folder' in dataframe.columns else 'Patient'

        for patient in dataframe[patient_col].unique():
            patient_dir = os.path.join(root_dir, patient)
            
            if os.path.exists(patient_dir):
                # Fetch all valid image slices
                img_files = sorted([f for f in os.listdir(patient_dir) if f.endswith('_img.npy')])
                
                for img_name in img_files:
                    img_path = os.path.join(patient_dir, img_name)
                    mask_path = img_path.replace('_img.npy', '_mask.npy')

                    if os.path.exists(mask_path):
                        self.slice_pairs.append((img_path, mask_path))

    def __len__(self):
        return len(self.slice_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.slice_pairs[idx]

        try:
            # Load arrays
            image = np.load(img_path).astype(np.float32)
            mask = np.load(mask_path).astype(np.uint8)

            # Reshape for PyTorch (C, H, W)
            image = torch.from_numpy(image).unsqueeze(0)
            mask = torch.from_numpy(mask).long() 

            return image, mask
        except Exception as e:
            # Safety fallback mechanism
            random_idx = random.randint(0, len(self.slice_pairs) - 1)
            return self.__getitem__(random_idx)

# ==========================================
# 2. EQUIVARIANT MODEL COMPONENTS (SE2-CNNET)
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
# 4. TRAINING EXECUTION
# ==========================================

def train():
    # ==========================================
    # PATH DEFINITIONS (Adjusted based on actual DGX terminal output)
    # ==========================================
    # Define the absolute path to where rclone is mounted
    GDRIVE_ROOT = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive")
    
    # Path to the preprocessed .npy folders inside GDrive
    GDRIVE_DATA_DIR = os.path.join(GDRIVE_ROOT, "Dataset_CT_Preprocessed_NPY") 
    CSV_REPORT = os.path.join(GDRIVE_ROOT, "Dataset_CT_Report.csv")
    
    # Fast local NVMe storage on DGX for maximum DataLoader speed
    LOCAL_DATA_PATH = os.path.expanduser("~/Clara/local_ct_workspace") 
    
    # Extract/Copy data to local server storage
    local_root = prepare_local_data(GDRIVE_DATA_DIR, LOCAL_DATA_PATH)

    # HYPERPARAMETERS
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8 # H100 has 80GB VRAM, you can try bumping this to 16 if memory permits
    ACCUMULATION_STEPS = 4  
    EPOCHS = 100
    VALIDATION_SPLIT = 0.15
    NUM_CLASSES = 2     
    INPUT_CHANNELS = 1  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device configured: {device}")

    # Load Report
    if not os.path.exists(CSV_REPORT):
        raise FileNotFoundError(f"Cannot find CSV report at {CSV_REPORT}")
    df = pd.read_csv(CSV_REPORT)

    # Patient-level Split (Prevents data leakage)
    train_df = df.sample(frac=(1 - VALIDATION_SPLIT), random_state=42)
    val_df = df.drop(train_df.index)

    print("Preparing Datasets...")
    train_set = CTBrainDataset(train_df, local_root)
    val_set = CTBrainDataset(val_df, local_root)

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

    print(f"📊 Data ready: {len(train_set)} training slices, {len(val_set)} validation slices. Workers: {num_workers}")

    # Model Setup
    model = SE2_CNNET(n_channels=INPUT_CHANNELS, n_classes=NUM_CLASSES, N=8, base_channels=24).to(device)

    # Loss & Optimizer
    class_weights = torch.tensor([1.0, 50.0]).to(device)
    criterion = CombinedLoss(weight_ce=1.0, weight_dice=1.0, class_weights=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Modern AMP configuration to avoid deprecation warnings
    scaler = torch.amp.GradScaler('cuda')

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
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

        # Flush remaining gradients
        if len(train_loader) % ACCUMULATION_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train_loss = running_loss / len(train_loader)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
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
        print(f"Epoch {epoch+1}/{EPOCHS} -> Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f'se2_unet_epoch_{epoch+1}.pth')
            
        torch.cuda.empty_cache()

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
    # Generate timestamp for unique log file name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{timestamp}.txt"
    
    # Redirect stdout and stderr
    sys.stdout = Logger(log_filename, sys.stdout)
    sys.stderr = Logger(log_filename, sys.stderr)
    
    print(f"📝 Logging terminal output to {log_filename}")
    train()