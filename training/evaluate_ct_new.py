import os
import sys
import shutil
import random
import re

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import kagglehub
import albumentations as A

# E2CNN Specific Libraries
from escnn import gspaces
import escnn.nn as enn

# ==========================================
# 0. KAGGLE DOWNLOAD & PREPROCESSING (INFERENCE ONLY)
# ==========================================

def download_and_prepare_inference_data():
    print("\n" + "="*50)
    print("📥 STAGE 1: DOWNLOAD DATASET FROM KAGGLE")
    print("="*50)
    
    try:
        download_path = kagglehub.dataset_download("trainingdatapro/computed-tomography-ct-of-the-brain")
        print(f"✅ Download successful! Cache: {download_path}")
    except Exception as e:
        print(f"❌ Failed to download dataset. Error: {e}")
        return None

    # Create target directory for NPY files
    TARGET_DIR = os.path.expanduser("~/Clara/inference_dataset_npy")
    
    import shutil
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    print("\n" + "="*50)
    print("⚙️ STAGE 2: PREPROCESSING FOR INFERENCE (NO MASKS)")
    print("="*50)
    
    processed_count = 0
    
    # Scan through the directories (aneurysm, cancer, tumor)
    for root, dirs, files in tqdm(list(os.walk(download_path)), desc="Scanning & Converting"):
        
        # Filter only JPG files
        img_files = [f for f in files if f.lower().endswith('.jpg')]
        
        for img_name in img_files:
            base_name = img_name.split('.')[0]
            img_path = os.path.join(root, img_name)
            category = os.path.basename(root) # e.g., 'tumor', 'cancer'
            
            try:
                # 1. Load CT Image
                img_array = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
                
                # Normalize if necessary
                if img_array.max() > 1.0: 
                    img_array = img_array / 255.0
                
                # 2. Save to NPY with category prefix
                unique_name = f"{category}_{base_name}"
                np.save(os.path.join(TARGET_DIR, f"{unique_name}_img.npy"), img_array)
                processed_count += 1
                
            except Exception as e:
                pass

    print(f"\n✅ Preprocessing Complete!")
    print(f"📊 Total Images Ready for Inference: {processed_count}")
    print(f"💾 Saved at: {TARGET_DIR}")
    
    return TARGET_DIR

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
# 2. LOSS FUNCTIONS (from train.py)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=3.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
    def forward(self, logits, targets):
        bce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, true_masks):
        nc = logits.shape[1]
        oh = F.one_hot(true_masks, nc).permute(0, 3, 1, 2).float()
        probs = F.softmax(logits, dim=1)
        inter = (probs[:, 1] * oh[:, 1]).sum(dim=(1, 2))
        union = probs[:, 1].sum(dim=(1, 2)) + oh[:, 1].sum(dim=(1, 2))
        return 1.0 - ((2. * inter + self.smooth) / (union + self.smooth)).mean()

class EdgeBoundaryLoss(nn.Module):
    def forward(self, logits, targets):
        tf = targets.float().unsqueeze(1)
        dilated = F.max_pool2d(tf, 5, 1, 2)
        eroded  = -F.max_pool2d(-tf, 5, 1, 2)
        bnd = (dilated - eroded).squeeze(1)
        base = F.cross_entropy(logits, targets, reduction='none')
        return (base * (1 + 5.0 * bnd)).mean()

class AdvancedCombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss(); self.dice = DiceLoss(); self.edge = EdgeBoundaryLoss()
    def forward(self, logits, targets):
        return self.focal(logits, targets) + self.dice(logits, targets) + 0.5 * self.edge(logits, targets)

# ==========================================
# 3. TRAINING DATASET LOADER
# ==========================================
class CTBrain25DDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.root_dir = root_dir; self.transform = transform
        self.patient_slices = {}; self.all_samples = []
        patient_col = 'Patient_Folder' if 'Patient_Folder' in dataframe.columns else 'Patient'
        for patient in dataframe[patient_col].unique():
            patient_dir = os.path.join(root_dir, patient)
            if os.path.exists(patient_dir):
                img_files = sorted([f for f in os.listdir(patient_dir) if f.endswith('_img.npy')],
                                   key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0)
                valid_pairs = []
                for img_name in img_files:
                    ip = os.path.join(patient_dir, img_name)
                    mp = ip.replace('_img.npy', '_mask.npy')
                    if os.path.exists(mp): valid_pairs.append((ip, mp))
                if valid_pairs:
                    self.patient_slices[patient] = valid_pairs
                    for i in range(len(valid_pairs)): self.all_samples.append((patient, i))
    def __len__(self): return len(self.all_samples)
    def __getitem__(self, idx):
        patient, slice_idx = self.all_samples[idx]
        slices = self.patient_slices[patient]
        ip = max(0, slice_idx - 1); nx = min(len(slices) - 1, slice_idx + 1)
        try:
            img_prev = np.load(slices[ip][0]).astype(np.float32)
            img_curr = np.load(slices[slice_idx][0]).astype(np.float32)
            img_next = np.load(slices[nx][0]).astype(np.float32)
            mask = np.load(slices[slice_idx][1]).astype(np.uint8)
            image_25d = np.stack([img_prev, img_curr, img_next], axis=-1)
            if self.transform:
                aug = self.transform(image=image_25d, mask=mask)
                image_25d = aug['image']; mask = aug['mask']
            return torch.from_numpy(image_25d).permute(2, 0, 1), torch.from_numpy(mask).long()
        except:
            return self.__getitem__(random.randint(0, len(self.all_samples) - 1))

# ==========================================
# 4. TRAINING PIPELINE
# ==========================================
def train_model(model_save_path):
    print("\n" + "="*60)
    print("🚀 TRAINING SE2-CNNET (2.5D) — 100 EPOCHS")
    print("="*60)

    GDRIVE_DATA_DIR = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive/Dataset_CT_Preprocessed_NPY")
    CSV_REPORT      = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
    LOCAL_DATA_PATH = os.path.expanduser("~/Clara/local_ct_workspace")

    os.makedirs(LOCAL_DATA_PATH, exist_ok=True)
    if not os.listdir(LOCAL_DATA_PATH):
        print("📦 Copying dataset locally...")
        for folder in tqdm(os.listdir(GDRIVE_DATA_DIR), desc="Copying"):
            src = os.path.join(GDRIVE_DATA_DIR, folder)
            dst = os.path.join(LOCAL_DATA_PATH, folder)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)

    LEARNING_RATE = 1e-4; BATCH_SIZE = 8; ACCUMULATION_STEPS = 4; EPOCHS = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")

    df = pd.read_csv(CSV_REPORT)
    train_df = df.sample(frac=0.85, random_state=42)
    val_df   = df.drop(train_df.index)

    train_transform = A.Compose([
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.06, 0.06), rotate=(-15, 15), p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.GaussNoise(p=0.3), A.HorizontalFlip(p=0.5)
    ])

    train_set = CTBrain25DDataset(train_df, LOCAL_DATA_PATH, transform=train_transform)
    val_set   = CTBrain25DDataset(val_df,   LOCAL_DATA_PATH)
    nw = min(os.cpu_count() or 4, 16)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True, num_workers=nw, persistent_workers=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=nw, persistent_workers=True)
    print(f"📊 Train: {len(train_set)} slices | Val: {len(val_set)} slices")

    model = SE2_CNNET(n_channels=3, n_classes=2, N=8, base_channels=24).to(device)
    criterion = AdvancedCombinedLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    best_iou = 0.0
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(EPOCHS):
        model.train(); optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels) / ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            pbar.set_postfix({'loss': f"{loss.item() * ACCUMULATION_STEPS:.4f}"})
            del outputs, loss

        model.eval()
        total_tp = total_fp = total_fn = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
                total_tp += torch.sum((preds == 1) & (labels == 1)).item()
                total_fp += torch.sum((preds == 1) & (labels == 0)).item()
                total_fn += torch.sum((preds == 0) & (labels == 1)).item()
                del logits

        eps = 1e-7
        epoch_iou = total_tp / (total_tp + total_fp + total_fn + eps)
        epoch_dice = (2*total_tp) / (2*total_tp + total_fp + total_fn + eps)
        print(f"📉 Epoch {epoch+1} | Dice: {epoch_dice:.4f} | IoU: {epoch_iou:.4f}")

        if epoch_iou > best_iou:
            best_iou = epoch_iou
            torch.save(model.state_dict(), model_save_path)
            print(f"🌟 Best model saved → {model_save_path}")
        if (epoch + 1) % 10 == 0:
            ck = model_save_path.replace('.pth', f'_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), ck)
        torch.cuda.empty_cache()

    print(f"\n✅ Training complete! Best IoU: {best_iou:.4f}")
    return model_save_path

# ==========================================
# 5. INFERENCE DATASET LOADER
# ==========================================

class InferenceCTDataset(Dataset):
    """
    Loads images for prediction generation with 2.5D spatial context.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.all_samples = []
        self.category_slices = {}
        
        print(f"🔍 Scanning directory for inference data: {root_dir}...")
        
        img_files = [f for f in os.listdir(root_dir) if f.endswith('_img.npy')]
        
        for img_name in img_files:
            category = img_name.split('_')[0] 
            if category not in self.category_slices:
                self.category_slices[category] = []
            self.category_slices[category].append(img_name)
            
        for category in self.category_slices:
            self.category_slices[category] = sorted(self.category_slices[category], 
                key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0)
            
            for i in range(len(self.category_slices[category])):
                self.all_samples.append((category, i))
                
        print(f"✅ Found a total of {len(self.all_samples)} valid images.")

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        category, slice_idx = self.all_samples[idx]
        slices = self.category_slices[category]
        
        idx_prev = max(0, slice_idx - 1)
        idx_next = min(len(slices) - 1, slice_idx + 1)
        
        img_prev = np.load(os.path.join(self.root_dir, slices[idx_prev])).astype(np.float32)
        img_curr = np.load(os.path.join(self.root_dir, slices[slice_idx])).astype(np.float32)
        img_next = np.load(os.path.join(self.root_dir, slices[idx_next])).astype(np.float32)
        
        image_25d = np.stack([img_prev, img_curr, img_next], axis=-1)
        image_tensor = torch.from_numpy(image_25d).permute(2, 0, 1).unsqueeze(0) 
        
        TARGET_SIZE = (256, 256) 
        image_tensor = F.interpolate(image_tensor, size=TARGET_SIZE, mode='bilinear', align_corners=False)
        image_tensor = image_tensor.squeeze(0)
        
        filename = slices[slice_idx]
        return image_tensor, filename

# ==========================================
# 3. INFERENCE ENGINE (SAVE PREDICTIONS)
# ==========================================

def run_inference():
    # --- 1. SETUP DATASET ---
    INFERENCE_DATA_PATH = download_and_prepare_inference_data()
    if not INFERENCE_DATA_PATH:
        return

    # --- CONFIGURATION ---
    # NAMA MODEL DIKEMBALIKAN KE VERSI 2.5D
    MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models_25D", "se2_unet_best_25D_Boundary.pth")
    OUTPUT_DIR = os.path.expanduser("~/Clara/inference_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Kept small to prevent OOM
    BATCH_SIZE = 4 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Hardware accelerated on: {device}")

    # --- LOAD MODEL ---
    model = SE2_CNNET(n_channels=3, n_classes=2, N=8, base_channels=24).to(device)
    try:
        # Load weights into CPU first to modify them safely
        checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True)
        
        # 🔄 DYNAMIC 1-CHANNEL TO 3-CHANNEL ADAPTATION
        if 'inc.double_conv.0.weights' in checkpoint and checkpoint['inc.double_conv.0.weights'].shape[0] == 144:
            print(f"🔄 Adapting 1-Channel Weights from {MODEL_WEIGHTS_PATH} to 3-Channel 2.5D Architecture...")
            checkpoint['inc.double_conv.0.weights'] = checkpoint['inc.double_conv.0.weights'].repeat(3) / 3.0
            
            if 'inc.double_conv.0.filter' in checkpoint:
                checkpoint['inc.double_conv.0.filter'] = checkpoint['inc.double_conv.0.filter'].repeat(1, 3, 1, 1) / 3.0
                
        model.load_state_dict(checkpoint, strict=False)
        print(f"✅ Successfully Loaded & Adapted weights from {MODEL_WEIGHTS_PATH}")
    except Exception as e:
        print(f"❌ Critical Error loading weights: {e}")
        return
        
    model.eval()

    # --- SETUP DATA LOADER ---
    dataset = InferenceCTDataset(INFERENCE_DATA_PATH)
    if len(dataset) == 0:
        print("❌ No data found!")
        return
        
    num_workers = min(os.cpu_count(), 8) if os.cpu_count() else 4
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)

    print(f"\n⚡ Beginning inference. Predictions will be saved to: {OUTPUT_DIR}")
    
    # --- INFERENCE LOOP ---
    with torch.no_grad():
        for images, filenames in tqdm(dataloader, desc="Generating Predictions"):
            images = images.to(device, non_blocking=True)
            
            # Predict
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1) # Shape: [Batch, H, W]
            
            # Save predictions as images
            # Convert 1s (tumor) to 255 (white) for clear visibility
            preds_np = preds.cpu().numpy().astype(np.uint8) * 255 
            
            for i in range(len(filenames)):
                pred_img = Image.fromarray(preds_np[i])
                
                # Create a clean filename for the prediction
                base_name = filenames[i].replace('_img.npy', '_prediction.jpg')
                save_path = os.path.join(OUTPUT_DIR, base_name)
                
                pred_img.save(save_path)

    # --- PRINT REPORT ---
    print("\n" + "🌟"*20)
    print("  INFERENCE COMPLETE")
    print("🌟"*20)
    print(f"✅ Successfully generated and saved {len(dataset)} prediction masks.")
    print(f"📁 Location: {OUTPUT_DIR}")
    print("-> You can download this folder to visually inspect your model's performance!")

if __name__ == "__main__":
    WEIGHTS = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "saved_models_25D", "se2_unet_best_25D_Boundary.pth"
    )
    if not os.path.exists(WEIGHTS):
        print("⚠️  No weights found — starting training first...")
        train_model(WEIGHTS)
    else:
        print(f"✅ Weights found: {WEIGHTS}")
    run_inference()