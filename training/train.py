import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import os
import traceback
from torch.cuda.amp import autocast, GradScaler
import shutil

# Pustaka khusus untuk E2CNN
from escnn import gspaces
import escnn.nn as enn

# Normal Loader Data

# class CTMultiFolderDataset(Dataset):
#     def __init__(self, root_dir):
#         self.root_dir = root_dir
#         self.samples = []

#         # Daftar sub-folder yang ingin diambil (CT_, CT_2, dll)
#         sub_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

#         print("🔍 Mulai menyisir data di semua folder...")

#         for folder in sub_folders:
#             folder_path = os.path.join(root_dir, folder)
#             # Ambil semua file citra asli (bukan mask)
#             images = sorted([f for f in os.listdir(folder_path) if f.endswith('.nii.gz') and '.seg.' not in f])

#             for img_name in images:
#                 mask_name = img_name.replace('.nii.gz', '.seg.nii.gz')
#                 img_full_path = os.path.join(folder_path, img_name)
#                 mask_full_path = os.path.join(folder_path, mask_name)

#                 # Hanya masukkan jika pasangan mask-nya ada
#                 if os.path.exists(mask_full_path):
#                     self.samples.append((img_full_path, mask_full_path))

#         print(f"✅ Total ditemukan {len(self.samples)} pasangan data dari {len(sub_folders)} folder.")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#       img_path, mask_path = self.samples[idx]
#       image = nib.load(img_path).get_fdata()
#       mask = nib.load(mask_path).get_fdata()

#       mid = image.shape[2] // 2
#       img_slice = image[:, :, mid]
#       mask_slice = mask[:, :, mid]

#       img_tensor = torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0) # [1, H, W]
#       mask_tensor = torch.tensor(mask_slice, dtype=torch.long).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

#       # --- RESIZE DI SINI ---
#       img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(256, 256), mode='bilinear').squeeze(0).squeeze(0).unsqueeze(0)
#       mask_tensor = F.interpolate(mask_tensor.float(), size=(256, 256), mode='nearest').long().squeeze()
#       # ----------------------

#       if img_tensor.max() > img_tensor.min():
#           img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

#       return img_tensor, mask_tensor


#Data loader for partial data

import os
import random # Pastikan import random di bagian atas file kamu
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class CTMultiFolderDataset(Dataset):
    def __init__(self, root_dir, target_folder=None):
        """
        Modified loader for preprocessed PyTorch (.pt) files with Partial Mode.
        root_dir: Path to 'Dataset_CT_Preprocessed'
        target_folder: (Optional) Name of a specific patient folder to load.
        """
        self.root_dir = root_dir
        self.samples = []
        
        # List all sub-folders in the preprocessed directory
        all_sub_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

        # --- PARTIAL MODE LOGIC ---
        if target_folder:
            if target_folder in all_sub_folders:
                sub_folders = [target_folder]
                print(f"🎯 Partial Mode Active: Loading data only from folder '{target_folder}'")
            else:
                print(f"⚠️ Warning: Folder '{target_folder}' not found! Falling back to scanning all folders...")
                sub_folders = all_sub_folders
        else:
            sub_folders = all_sub_folders
            print("🔍 Scanning all preprocessed patient folders...")

        for folder in sub_folders:
            folder_path = os.path.join(root_dir, folder)
            # Find all preprocessed slices (.pt) in the specific folder
            pt_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pt')])

            for pt_name in pt_files:
                full_path = os.path.join(folder_path, pt_name)
                # Double check file existence and size
                if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
                    self.samples.append(full_path)

        print(f"✅ Ready! Found {len(self.samples)} valid slices in selected directory.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]

        try:
            # Load the preprocessed dictionary: {'image': tensor, 'mask': tensor}
            data = torch.load(file_path)
            
            img_tensor = data['image'] # Already normalized and resized
            mask_tensor = data['mask']   # Already resized to 256x256

            # Ensure the tensors have the correct shape for Conv2D (C, H, W)
            if img_tensor.dim() == 2:
                img_tensor = img_tensor.unsqueeze(0)
            
            # Mask should be long type for CrossEntropyLoss
            mask_tensor = mask_tensor.long()

            return img_tensor, mask_tensor

        except Exception as e:
            print(f"\n❌ Error loading .pt file: {file_path}")
            # Safety fallback: get another random sample
            random_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(random_idx)


class DoubleEquivariantConv(nn.Module):
    """Blok konvolusi ganda yang equivariant."""
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
    """Blok downsampling menggunakan MaxPool diikuti DoubleEquivariantConv."""
    def __init__(self, in_type, out_type):
        super().__init__()
        self.pool = enn.PointwiseMaxPool(in_type, kernel_size=2)
        self.conv = DoubleEquivariantConv(in_type, out_type)
    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)

class Up(nn.Module):
    """Blok upsampling diikuti penggabungan skip connection dan DoubleEquivariantConv."""
    def __init__(self, in_type, out_type):
        super().__init__()
        self.up = enn.R2Upsampling(in_type, scale_factor=2, mode='bilinear', align_corners=True)
        # Tipe input untuk konvolusi adalah gabungan dari tensor setelah upsampling dan tensor dari skip connection
        self.conv = DoubleEquivariantConv(in_type + out_type, out_type)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Menggabungkan tensor dari skip connection (x2) dan tensor yang di-upsample (x1)
        x = enn.tensor_directsum([x2, x1])
        return self.conv(x)

class OutConv(nn.Module):
    """Konvolusi 1x1 di akhir untuk memetakan fitur ke jumlah kelas output."""
    def __init__(self, in_type, n_classes):
        super().__init__()
        gspace = in_type.gspace
        # Tipe output adalah trivial representation, karena output segmentasi harus invarian terhadap rotasi
        out_type = enn.FieldType(gspace, n_classes * [gspace.trivial_repr])
        self.conv = enn.R2Conv(in_type, out_type, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class SE2_CNNET(nn.Module):
    """
    Arsitektur U-Net Equivariant SE(2) untuk segmentasi.
    N: Jumlah rotasi diskrit yang akan dipertimbangkan (misal, N=8 untuk rotasi kelipatan 45 derajat).
    base_channels: Jumlah channel dasar pada lapisan pertama.
    """
    def __init__(self, n_channels, n_classes, N=8, base_channels=24):
        super().__init__()
        self.r2_act = gspaces.rot2dOnR2(N=N)
        c = base_channels

        # Mendefinisikan tipe field untuk setiap level kedalaman U-Net
        self.feat_type_in = enn.FieldType(self.r2_act, n_channels * [self.r2_act.trivial_repr])
        self.feat_type_1 = enn.FieldType(self.r2_act, c * [self.r2_act.regular_repr])
        self.feat_type_2 = enn.FieldType(self.r2_act, (c*2) * [self.r2_act.regular_repr])
        self.feat_type_3 = enn.FieldType(self.r2_act, (c*4) * [self.r2_act.regular_repr])
        self.feat_type_4 = enn.FieldType(self.r2_act, (c*8) * [self.r2_act.regular_repr])
        self.feat_type_5 = enn.FieldType(self.r2_act, (c*16) * [self.r2_act.regular_repr])

        # Encoder Path
        self.inc = DoubleEquivariantConv(self.feat_type_in, self.feat_type_1)
        self.down1 = Down(self.feat_type_1, self.feat_type_2)
        self.down2 = Down(self.feat_type_2, self.feat_type_3)
        self.down3 = Down(self.feat_type_3, self.feat_type_4)
        self.down4 = Down(self.feat_type_4, self.feat_type_5)

        # Decoder Path
        self.up1 = Up(self.feat_type_5, self.feat_type_4)
        self.up2 = Up(self.feat_type_4, self.feat_type_3)
        self.up3 = Up(self.feat_type_3, self.feat_type_2)
        self.up4 = Up(self.feat_type_2, self.feat_type_1)

        # Output Layer
        self.outc = OutConv(self.feat_type_1, n_classes)

    def forward(self, x):
        # Konversi input tensor menjadi GeometricTensor
        x_geom = enn.GeometricTensor(x, self.feat_type_in)

        # Encoder
        x1 = self.inc(x_geom)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Mengembalikan tensor biasa untuk dihitung loss-nya
        logits = self.outc(x).tensor
        return logits

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true_masks):
        # logits shape: (B, C, H, W). true_masks shape: (B, H, W)
        
        # 1. Konversi true_masks ke One-Hot Encoding
        num_classes = logits.shape[1]
        true_masks_one_hot = F.one_hot(true_masks, num_classes).permute(0, 3, 1, 2).float()
        
        # 2. Hitung probability dengan Softmax
        probs = F.softmax(logits, dim=1)
        
        # 3. Hitung Dice Score untuk kelas target (index 1)
        # Kita abaikan kelas background (index 0) agar model fokus ke target
        probs_target = probs[:, 1, :, :]
        true_target = true_masks_one_hot[:, 1, :, :]
        
        intersection = (probs_target * true_target).sum(dim=(1, 2))
        union = probs_target.sum(dim=(1, 2)) + true_target.sum(dim=(1, 2))
        
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 4. Kembalikan 1 - rata-rata Dice
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


def train():
    # ==========================================
    # ARCHITECTURAL DECISION 1: I/O OPTIMIZATION
    # Move data from slow network drive to local high-speed NVMe/SSD
    # ==========================================
    ORIGINAL_GDRIVE_PATH = "../../../Gdrive_new/Dataset_CT_Preprocessed"
    LOCAL_DATA_PATH = "/content/Dataset_CT_Preprocessed_Local" # Change if not on Colab
    
    if not os.path.exists(LOCAL_DATA_PATH):
        print(f"🚀 Funday AI Core: Copying dataset to local storage to prevent I/O bottlenecks. Please wait...")
        shutil.copytree(ORIGINAL_GDRIVE_PATH, LOCAL_DATA_PATH)
        print("✅ Data copy complete!")
    else:
        print("✅ Local dataset already exists.")

    # 1. Update Path & Parameters
    ROOT_DATA_PATH = LOCAL_DATA_PATH # Use the local path!
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4      
    EPOCHS = 10
    VALIDATION_SPLIT = 0.15
    NUM_CLASSES = 2     
    INPUT_CHANNELS = 1  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 2. Dataset Loader Setup
    print("Preparing CT Scan Multi-Folder Dataset...")
    full_dataset = CTMultiFolderDataset(root_dir=ROOT_DATA_PATH, target_folder="CT_") 

    n_val = int(len(full_dataset) * VALIDATION_SPLIT)
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])

    # ==========================================
    # ARCHITECTURAL DECISION 2: DATALOADER ACCELERATION
    # Enable multiprocessing for data loading. 
    # ==========================================
    # Get optimal number of CPU cores for workers
    num_workers = min(os.cpu_count(), 8) if os.cpu_count() else 2
    
    train_loader = DataLoader(
        train_set, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True,
        num_workers=num_workers, # Distribute loading to multiple CPU cores
        prefetch_factor=2,       # Pre-fetch batches to eliminate GPU idle time
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2,
        persistent_workers=True
    )

    print(f"Data ready: {len(train_set)} training, {len(val_set)} validation. Workers: {num_workers}")

    # ==========================================
    # ARCHITECTURAL DECISION 3: MODEL COMPLEXITY
    # If it's still too slow, consider reducing N to 4 or base_channels to 16 
    # during the prototyping phase to speed up research iterations.
    # ==========================================
    model = SE2_CNNET(n_channels=INPUT_CHANNELS, n_classes=NUM_CLASSES, N=8, base_channels=24)
    model.to(device)

    # Memberikan bobot lebih berat ke kelas target (index 1) karena imbalanced
    class_weights = torch.tensor([1.0, 50.0]).to(device)
    criterion = CombinedLoss(weight_ce=1.0, weight_dice=1.0, class_weights=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scaler = GradScaler()
    ACCUMULATION_STEPS = 8  

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for i, (images, labels) in enumerate(pbar_train):
            # Non-blocking transfer is faster when pin_memory is True
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / ACCUMULATION_STEPS 

            scaler.scale(loss).backward()

            # Step optimization
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * ACCUMULATION_STEPS
            pbar_train.set_postfix({'loss': loss.item() * ACCUMULATION_STEPS})

            # Force delete for memory management
            del outputs, loss

        # ==========================================
        # EDGE CASE FIX: Flush remaining gradients
        # If the dataloader length isn't perfectly divisible by ACCUMULATION_STEPS,
        # we must step the optimizer for the remaining batches at the end of epoch.
        # ==========================================
        if len(train_loader) % ACCUMULATION_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train_loss = running_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
            for images, labels in pbar_val:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(): 
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                pbar_val.set_postfix({'val_loss': loss.item()})
                del outputs, loss

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} -> Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
        torch.cuda.empty_cache()

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
    import sys
    import datetime
    
    # Generate timestamp for unique log file name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{timestamp}.txt"
    
    # Redirect stdout and stderr
    sys.stdout = Logger(log_filename, sys.stdout)
    sys.stderr = Logger(log_filename, sys.stderr)
    
    print(f"Mulai menyimpan log terminal ke {log_filename}")
    
    train()
    
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import os
import traceback
from torch.cuda.amp import autocast, GradScaler
import shutil

# Pustaka khusus untuk E2CNN
from escnn import gspaces
import escnn.nn as enn

# Normal Loader Data

# class CTMultiFolderDataset(Dataset):
#     def __init__(self, root_dir):
#         self.root_dir = root_dir
#         self.samples = []

#         # Daftar sub-folder yang ingin diambil (CT_, CT_2, dll)
#         sub_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

#         print("🔍 Mulai menyisir data di semua folder...")

#         for folder in sub_folders:
#             folder_path = os.path.join(root_dir, folder)
#             # Ambil semua file citra asli (bukan mask)
#             images = sorted([f for f in os.listdir(folder_path) if f.endswith('.nii.gz') and '.seg.' not in f])

#             for img_name in images:
#                 mask_name = img_name.replace('.nii.gz', '.seg.nii.gz')
#                 img_full_path = os.path.join(folder_path, img_name)
#                 mask_full_path = os.path.join(folder_path, mask_name)

#                 # Hanya masukkan jika pasangan mask-nya ada
#                 if os.path.exists(mask_full_path):
#                     self.samples.append((img_full_path, mask_full_path))

#         print(f"✅ Total ditemukan {len(self.samples)} pasangan data dari {len(sub_folders)} folder.")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#       img_path, mask_path = self.samples[idx]
#       image = nib.load(img_path).get_fdata()
#       mask = nib.load(mask_path).get_fdata()

#       mid = image.shape[2] // 2
#       img_slice = image[:, :, mid]
#       mask_slice = mask[:, :, mid]

#       img_tensor = torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0) # [1, H, W]
#       mask_tensor = torch.tensor(mask_slice, dtype=torch.long).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

#       # --- RESIZE DI SINI ---
#       img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(256, 256), mode='bilinear').squeeze(0).squeeze(0).unsqueeze(0)
#       mask_tensor = F.interpolate(mask_tensor.float(), size=(256, 256), mode='nearest').long().squeeze()
#       # ----------------------

#       if img_tensor.max() > img_tensor.min():
#           img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

#       return img_tensor, mask_tensor


#Data loader for partial data

import os
import random # Pastikan import random di bagian atas file kamu
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class CTMultiFolderDataset(Dataset):
    def __init__(self, root_dir, target_folder=None):
        """
        Modified loader for preprocessed PyTorch (.pt) files with Partial Mode.
        root_dir: Path to 'Dataset_CT_Preprocessed'
        target_folder: (Optional) Name of a specific patient folder to load.
        """
        self.root_dir = root_dir
        self.samples = []
        
        # List all sub-folders in the preprocessed directory
        all_sub_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

        # --- PARTIAL MODE LOGIC ---
        if target_folder:
            if target_folder in all_sub_folders:
                sub_folders = [target_folder]
                print(f"🎯 Partial Mode Active: Loading data only from folder '{target_folder}'")
            else:
                print(f"⚠️ Warning: Folder '{target_folder}' not found! Falling back to scanning all folders...")
                sub_folders = all_sub_folders
        else:
            sub_folders = all_sub_folders
            print("🔍 Scanning all preprocessed patient folders...")

        for folder in sub_folders:
            folder_path = os.path.join(root_dir, folder)
            # Find all preprocessed slices (.pt) in the specific folder
            pt_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pt')])

            for pt_name in pt_files:
                full_path = os.path.join(folder_path, pt_name)
                # Double check file existence and size
                if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
                    self.samples.append(full_path)

        print(f"✅ Ready! Found {len(self.samples)} valid slices in selected directory.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]

        try:
            # Load the preprocessed dictionary: {'image': tensor, 'mask': tensor}
            data = torch.load(file_path)
            
            img_tensor = data['image'] # Already normalized and resized
            mask_tensor = data['mask']   # Already resized to 256x256

            # Ensure the tensors have the correct shape for Conv2D (C, H, W)
            if img_tensor.dim() == 2:
                img_tensor = img_tensor.unsqueeze(0)
            
            # Mask should be long type for CrossEntropyLoss
            mask_tensor = mask_tensor.long()

            return img_tensor, mask_tensor

        except Exception as e:
            print(f"\n❌ Error loading .pt file: {file_path}")
            # Safety fallback: get another random sample
            random_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(random_idx)


class DoubleEquivariantConv(nn.Module):
    """Blok konvolusi ganda yang equivariant."""
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
    """Blok downsampling menggunakan MaxPool diikuti DoubleEquivariantConv."""
    def __init__(self, in_type, out_type):
        super().__init__()
        self.pool = enn.PointwiseMaxPool(in_type, kernel_size=2)
        self.conv = DoubleEquivariantConv(in_type, out_type)
    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)

class Up(nn.Module):
    """Blok upsampling diikuti penggabungan skip connection dan DoubleEquivariantConv."""
    def __init__(self, in_type, out_type):
        super().__init__()
        self.up = enn.R2Upsampling(in_type, scale_factor=2, mode='bilinear', align_corners=True)
        # Tipe input untuk konvolusi adalah gabungan dari tensor setelah upsampling dan tensor dari skip connection
        self.conv = DoubleEquivariantConv(in_type + out_type, out_type)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Menggabungkan tensor dari skip connection (x2) dan tensor yang di-upsample (x1)
        x = enn.tensor_directsum([x2, x1])
        return self.conv(x)

class OutConv(nn.Module):
    """Konvolusi 1x1 di akhir untuk memetakan fitur ke jumlah kelas output."""
    def __init__(self, in_type, n_classes):
        super().__init__()
        gspace = in_type.gspace
        # Tipe output adalah trivial representation, karena output segmentasi harus invarian terhadap rotasi
        out_type = enn.FieldType(gspace, n_classes * [gspace.trivial_repr])
        self.conv = enn.R2Conv(in_type, out_type, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class SE2_CNNET(nn.Module):
    """
    Arsitektur U-Net Equivariant SE(2) untuk segmentasi.
    N: Jumlah rotasi diskrit yang akan dipertimbangkan (misal, N=8 untuk rotasi kelipatan 45 derajat).
    base_channels: Jumlah channel dasar pada lapisan pertama.
    """
    def __init__(self, n_channels, n_classes, N=8, base_channels=24):
        super().__init__()
        self.r2_act = gspaces.rot2dOnR2(N=N)
        c = base_channels

        # Mendefinisikan tipe field untuk setiap level kedalaman U-Net
        self.feat_type_in = enn.FieldType(self.r2_act, n_channels * [self.r2_act.trivial_repr])
        self.feat_type_1 = enn.FieldType(self.r2_act, c * [self.r2_act.regular_repr])
        self.feat_type_2 = enn.FieldType(self.r2_act, (c*2) * [self.r2_act.regular_repr])
        self.feat_type_3 = enn.FieldType(self.r2_act, (c*4) * [self.r2_act.regular_repr])
        self.feat_type_4 = enn.FieldType(self.r2_act, (c*8) * [self.r2_act.regular_repr])
        self.feat_type_5 = enn.FieldType(self.r2_act, (c*16) * [self.r2_act.regular_repr])

        # Encoder Path
        self.inc = DoubleEquivariantConv(self.feat_type_in, self.feat_type_1)
        self.down1 = Down(self.feat_type_1, self.feat_type_2)
        self.down2 = Down(self.feat_type_2, self.feat_type_3)
        self.down3 = Down(self.feat_type_3, self.feat_type_4)
        self.down4 = Down(self.feat_type_4, self.feat_type_5)

        # Decoder Path
        self.up1 = Up(self.feat_type_5, self.feat_type_4)
        self.up2 = Up(self.feat_type_4, self.feat_type_3)
        self.up3 = Up(self.feat_type_3, self.feat_type_2)
        self.up4 = Up(self.feat_type_2, self.feat_type_1)

        # Output Layer
        self.outc = OutConv(self.feat_type_1, n_classes)

    def forward(self, x):
        # Konversi input tensor menjadi GeometricTensor
        x_geom = enn.GeometricTensor(x, self.feat_type_in)

        # Encoder
        x1 = self.inc(x_geom)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Mengembalikan tensor biasa untuk dihitung loss-nya
        logits = self.outc(x).tensor
        return logits

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true_masks):
        # logits shape: (B, C, H, W). true_masks shape: (B, H, W)
        
        # 1. Konversi true_masks ke One-Hot Encoding
        num_classes = logits.shape[1]
        true_masks_one_hot = F.one_hot(true_masks, num_classes).permute(0, 3, 1, 2).float()
        
        # 2. Hitung probability dengan Softmax
        probs = F.softmax(logits, dim=1)
        
        # 3. Hitung Dice Score untuk kelas target (index 1)
        # Kita abaikan kelas background (index 0) agar model fokus ke target
        probs_target = probs[:, 1, :, :]
        true_target = true_masks_one_hot[:, 1, :, :]
        
        intersection = (probs_target * true_target).sum(dim=(1, 2))
        union = probs_target.sum(dim=(1, 2)) + true_target.sum(dim=(1, 2))
        
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 4. Kembalikan 1 - rata-rata Dice
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


def train():
    # ==========================================
    # ARCHITECTURAL DECISION 1: I/O OPTIMIZATION
    # Move data from slow network drive to local high-speed NVMe/SSD
    # ==========================================
    ORIGINAL_GDRIVE_PATH = "../../../Gdrive_new/Dataset_CT_Preprocessed"
    #LOCAL_DATA_PATH = "/content/Dataset_CT_Preprocessed_Local" # Change if not on Colab
    
    #if not os.path.exists(LOCAL_DATA_PATH):
    #    print(f"🚀 Funday AI Core: Copying dataset to local storage to prevent I/O bottlenecks. Please wait...")
    #    shutil.copytree(ORIGINAL_GDRIVE_PATH, LOCAL_DATA_PATH)
    #    print("✅ Data copy complete!")
    #else:
    #    print("✅ Local dataset already exists.")

    # 1. Update Path & Parameters
    #ROOT_DATA_PATH = LOCAL_DATA_PATH # Use the local path!
    ROOT_DATA_PATH = ORIGINAL_GDRIVE_PATH
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 16      
    EPOCHS = 100
    VALIDATION_SPLIT = 0.15
    NUM_CLASSES = 2     
    INPUT_CHANNELS = 1  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 2. Dataset Loader Setup
    print("Preparing CT Scan Multi-Folder Dataset...")
    full_dataset = CTMultiFolderDataset(root_dir=ROOT_DATA_PATH, target_folder="CT_") 

    n_val = int(len(full_dataset) * VALIDATION_SPLIT)
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])

    # ==========================================
    # ARCHITECTURAL DECISION 2: DATALOADER ACCELERATION
    # Enable multiprocessing for data loading. 
    # ==========================================
    # Get optimal number of CPU cores for workers
    num_workers = min(os.cpu_count(), 8) if os.cpu_count() else 2
    
    train_loader = DataLoader(
        train_set, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True,
        num_workers=num_workers, # Distribute loading to multiple CPU cores
        prefetch_factor=2,       # Pre-fetch batches to eliminate GPU idle time
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2,
        persistent_workers=True
    )

    print(f"Data ready: {len(train_set)} training, {len(val_set)} validation. Workers: {num_workers}")

    # ==========================================
    # ARCHITECTURAL DECISION 3: MODEL COMPLEXITY
    # If it's still too slow, consider reducing N to 4 or base_channels to 16 
    # during the prototyping phase to speed up research iterations.
    # ==========================================
    model = SE2_CNNET(n_channels=INPUT_CHANNELS, n_classes=NUM_CLASSES, N=8, base_channels=24)
    model.to(device)

    # Memberikan bobot lebih berat ke kelas target (index 1) karena imbalanced
    class_weights = torch.tensor([1.0, 50.0]).to(device)
    criterion = CombinedLoss(weight_ce=1.0, weight_dice=1.0, class_weights=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scaler = GradScaler()
    ACCUMULATION_STEPS = 8  

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for i, (images, labels) in enumerate(pbar_train):
            # Non-blocking transfer is faster when pin_memory is True
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / ACCUMULATION_STEPS 

            scaler.scale(loss).backward()

            # Step optimization
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * ACCUMULATION_STEPS
            pbar_train.set_postfix({'loss': loss.item() * ACCUMULATION_STEPS})

            # Force delete for memory management
            del outputs, loss

        # ==========================================
        # EDGE CASE FIX: Flush remaining gradients
        # If the dataloader length isn't perfectly divisible by ACCUMULATION_STEPS,
        # we must step the optimizer for the remaining batches at the end of epoch.
        # ==========================================
        if len(train_loader) % ACCUMULATION_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train_loss = running_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
            for images, labels in pbar_val:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(): 
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                pbar_val.set_postfix({'val_loss': loss.item()})
                del outputs, loss

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} -> Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        #torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
        if (epoch+1) % 10 ==0:
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
        torch.cuda.empty_cache()

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
    import sys
    import datetime
    
    # Generate timestamp for unique log file name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{timestamp}.txt"
    
    # Redirect stdout and stderr
    sys.stdout = Logger(log_filename, sys.stdout)
    sys.stderr = Logger(log_filename, sys.stderr)
    
    print(f"Mulai menyimpan log terminal ke {log_filename}")
    
    train()