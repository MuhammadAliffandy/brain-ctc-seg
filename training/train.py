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
from tqdm import tqdm

# The "Vaccine" for Domain Gap
import albumentations as A

# E2CNN Specific Libraries
from escnn import gspaces
import escnn.nn as enn

# ==========================================
# 1. DATA PREPARATION 
# ==========================================

def prepare_local_data(gdrive_dir, local_extract_dir):
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

# ==========================================
# 2. DATASET LOADER WITH AUGMENTATION
# ==========================================

class CTBrainDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform 
        self.slice_pairs = []

        patient_col = 'Patient_Folder' if 'Patient_Folder' in dataframe.columns else 'Patient'

        for patient in dataframe[patient_col].unique():
            patient_dir = os.path.join(root_dir, patient)
            if os.path.exists(patient_dir):
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
            image = np.load(img_path).astype(np.float32)
            mask = np.load(mask_path).astype(np.uint8)

            # Apply Albumentations Transformations
            if self.transform is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

            # Reshape for PyTorch (Channels, Height, Width)
            image = torch.from_numpy(image).unsqueeze(0)
            mask = torch.from_numpy(mask).long() 

            return image, mask
        except Exception as e:
            # Fallback to a random index if file is corrupted
            random_idx = random.randint(0, len(self.slice_pairs) - 1)
            return self.__getitem__(random_idx)

# ==========================================
# 3. SE2-CNNET ARCHITECTURE
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
# 4. LOSS FUNCTIONS
# ==========================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

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
    def __init__(self, weight_focal=1.0, weight_dice=1.0):
        super(CombinedLoss, self).__init__()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.focal = FocalLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return (self.weight_focal * self.focal(logits, targets)) + (self.weight_dice * self.dice(logits, targets))

# ==========================================
# 5. METRIC CALCULATION (FOR REPORTING)
# ==========================================

def calculate_metrics_tensors(preds, targets):
    """Calculates True Positives, False Positives, False Negatives, True Negatives"""
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    tp = torch.sum((preds == 1) & (targets == 1)).item()
    fp = torch.sum((preds == 1) & (targets == 0)).item()
    fn = torch.sum((preds == 0) & (targets == 1)).item()
    tn = torch.sum((preds == 0) & (targets == 0)).item()
    
    return tp, fp, fn, tn

# ==========================================
# 6. TRAINING EXECUTION
# ==========================================

def train():
    # PATH DEFINITIONS
    GDRIVE_ROOT = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive")
    GDRIVE_DATA_DIR = os.path.join(GDRIVE_ROOT, "Dataset_CT_Preprocessed_NPY") 
    CSV_REPORT = os.path.join(GDRIVE_ROOT, "Dataset_CT_Report.csv")
    LOCAL_DATA_PATH = os.path.expanduser("~/Clara/local_ct_workspace") 
    
    local_root = prepare_local_data(GDRIVE_DATA_DIR, LOCAL_DATA_PATH)

    # HYPERPARAMETERS
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8 
    ACCUMULATION_STEPS = 4  
    EPOCHS = 100
    VALIDATION_SPLIT = 0.15

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device configured: {device}")

    if not os.path.exists(CSV_REPORT):
        raise FileNotFoundError(f"Cannot find CSV report at {CSV_REPORT}")
    df = pd.read_csv(CSV_REPORT)

    train_df = df.sample(frac=(1 - VALIDATION_SPLIT), random_state=42)
    val_df = df.drop(train_df.index)

    # 🧬 AUGMENTATION SETUP (WARNINGS FIXED)
    print("🧬 Setting up Data Augmentation Pipelines...")
    train_transform = A.Compose([
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.06, 0.06), rotate=(-15, 15), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3), # Removed invalid var_limit
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.HorizontalFlip(p=0.5)
    ])
    
    val_transform = None 

    print("Preparing Datasets...")
    train_set = CTBrainDataset(train_df, local_root, transform=train_transform)
    val_set = CTBrainDataset(val_df, local_root, transform=val_transform)

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

    print(f"📊 Data ready: {len(train_set)} training slices, {len(val_set)} validation slices.")

    # Model Setup
    model = SE2_CNNET(n_channels=1, n_classes=2, N=8, base_channels=24).to(device)

    # Loss & Optimizer (FIXED)
    criterion = CombinedLoss(weight_focal=1.0, weight_dice=1.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) 
    
    scaler = torch.amp.GradScaler('cuda')

    best_val_dice = 0.0 # Track best DICE score instead of loss
    
    # TRAINING LOOP
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
            pbar_train.set_postfix({'loss': f"{loss.item() * ACCUMULATION_STEPS:.4f}"})
            
            del outputs, loss

        if len(train_loader) % ACCUMULATION_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train_loss = running_loss / len(train_loader)

        # VALIDATION LOOP & METRICS REPORTING
        model.eval()
        val_loss = 0.0
        
        # Metric Trackers
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
            for images, labels in pbar_val:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'): 
                    logits = model(images)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                pbar_val.set_postfix({'val_loss': f"{loss.item():.4f}"})
                
                # Calculate metrics for current batch
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                tp, fp, fn, tn = calculate_metrics_tensors(preds, labels)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn
                
                del logits, loss

        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate Global Metrics for the Report
        epsilon = 1e-7
        epoch_dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + epsilon)
        epoch_iou = total_tp / (total_tp + total_fp + total_fn + epsilon)
        epoch_precision = total_tp / (total_tp + total_fp + epsilon)
        epoch_recall = total_tp / (total_tp + total_fn + epsilon)
        
        # PRINT THE REPORT
        print(f"\n📉 Epoch {epoch+1} Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"================== CLASSIFICATION & SEGMENTATION REPORT ==================")
        print(f"  • Global Dice Score : {epoch_dice:.4f}")
        print(f"  • Global IoU        : {epoch_iou:.4f}")
        print(f"  • Precision         : {epoch_precision:.4f} (Ability to avoid false alarms)")
        print(f"  • Recall            : {epoch_recall:.4f} (Ability to find all tumors)")
        print(f"==========================================================================")

        # AUTO-SAVE BEST MODEL BASED ON DICE SCORE
        if epoch_dice > best_val_dice:
            best_val_dice = epoch_dice
            torch.save(model.state_dict(), 'se2_unet_best_robust.pth')
            print(f"🌟 New Best Model Saved! (Dice Score improved to: {best_val_dice:.4f})")

        # Save checkpoint periodically
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f'se2_unet_epoch_{epoch+1}.pth')
            
        torch.cuda.empty_cache()

# ==========================================
# 7. LOGGING UTILITY
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
    log_filename = f"training_log_robust_{timestamp}.txt"
    
    sys.stdout = Logger(log_filename, sys.stdout)
    sys.stderr = Logger(log_filename, sys.stderr)
    
    print(f"📝 Logging terminal output to {log_filename}")
    train()