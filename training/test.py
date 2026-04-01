import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
# 2. GLOBAL DATASET LOADER
# ==========================================

class FullCTDataset(Dataset):
    """
    Scans the entire local workspace for all folders and slice pairs.
    """
    def __init__(self, root_dir):
        self.slice_pairs = []
        
        print("🔍 Scanning all directories for evaluation data...")
        # Get all subdirectories (CT_, CT_2, etc.)
        sub_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        
        for folder in sub_folders:
            folder_path = os.path.join(root_dir, folder)
            # Find all image arrays
            img_files = sorted([f for f in os.listdir(folder_path) if f.endswith('_img.npy')])
            
            for img_name in img_files:
                img_path = os.path.join(folder_path, img_name)
                mask_path = img_path.replace('_img.npy', '_mask.npy')
                
                # Verify that the corresponding mask actually exists
                if os.path.exists(mask_path):
                    self.slice_pairs.append((img_path, mask_path))
                    
        print(f"✅ Found a total of {len(self.slice_pairs)} valid image-mask pairs across {len(sub_folders)} folders.")

    def __len__(self):
        return len(self.slice_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.slice_pairs[idx]
        
        # Load numpy arrays
        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.uint8)
        
        # Convert to tensors (C, H, W for image)
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).long()
        
        return image_tensor, mask_tensor

# ==========================================
# 3. GLOBAL EVALUATION ENGINE
# ==========================================

def evaluate_all():
    # --- CONFIGURATION ---
    MODEL_WEIGHTS_PATH = "se2_unet_epoch_100.pth" 
    LOCAL_DATA_PATH = os.path.expanduser("~/Clara/local_ct_workspace")
    BATCH_SIZE = 32 # Large batch size for inference on DGX H100
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Hardware accelerated on: {device}")

    # --- LOAD MODEL ---
    model = SE2_CNNET(n_channels=1, n_classes=2, N=8, base_channels=24).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True))
        print(f"✅ Loaded weights from {MODEL_WEIGHTS_PATH}")
    except Exception as e:
        print(f"❌ Critical Error loading weights: {e}")
        return
        
    model.eval()

    # --- SETUP DATA LOADER ---
    dataset = FullCTDataset(LOCAL_DATA_PATH)
    if len(dataset) == 0:
        print("❌ No data found! Please check the LOCAL_DATA_PATH.")
        return
        
    num_workers = min(os.cpu_count(), 8) if os.cpu_count() else 4
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)

    # Variables to accumulate global confusion matrix values
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    print("\n⚡ Beginning massive parallel evaluation...")
    
    # --- INFERENCE LOOP ---
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating Dataset"):
            # Move to GPU
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Predict
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1) 
            
            # Flatten tensors for metric calculation
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)
            
            # Accumulate metrics
            total_tp += torch.sum((preds_flat == 1) & (masks_flat == 1)).item()
            total_tn += torch.sum((preds_flat == 0) & (masks_flat == 0)).item()
            total_fp += torch.sum((preds_flat == 1) & (masks_flat == 0)).item()
            total_fn += torch.sum((preds_flat == 0) & (masks_flat == 1)).item()

    # --- CALCULATE FINAL GLOBAL METRICS ---
    epsilon = 1e-6 # Prevent division by zero
    
    global_dice = (2.0 * total_tp) / ((2.0 * total_tp) + total_fp + total_fn + epsilon)
    global_iou = total_tp / (total_tp + total_fp + total_fn + epsilon)
    global_acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + epsilon)
    global_prec = total_tp / (total_tp + total_fp + epsilon)
    global_rec = total_tp / (total_tp + total_fn + epsilon)

    # --- PRINT REPORT ---
    print("\n" + "🌟"*20)
    print(f"  GLOBAL PERFORMANCE REPORT (Across {len(dataset)} Slices)")
    print("🌟"*20)
    print(f"🔥 Global Dice Score : {global_dice:.4f} ({(global_dice*100):.2f}%)")
    print(f"🎯 Global IoU Score  : {global_iou:.4f} ({(global_iou*100):.2f}%)")
    print(f"✅ Global Accuracy   : {global_acc:.4f} ({(global_acc*100):.2f}%)")
    print(f"📌 Global Precision  : {global_prec:.4f} ({(global_prec*100):.2f}%)")
    print(f"🔍 Global Recall     : {global_rec:.4f} ({(global_rec*100):.2f}%)")
    print("🌟"*20 + "\n")

if __name__ == "__main__":
    evaluate_all()