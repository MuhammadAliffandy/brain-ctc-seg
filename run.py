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

# Pustaka khusus untuk E2CNN
from escnn import gspaces
import escnn.nn as enn


class CTMultiFolderDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []

        # Daftar sub-folder yang ingin diambil (CT_, CT_2, dll)
        sub_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

        print("🔍 Mulai menyisir data di semua folder...")

        for folder in sub_folders:
            folder_path = os.path.join(root_dir, folder)
            # Ambil semua file citra asli (bukan mask)
            images = sorted([f for f in os.listdir(folder_path) if f.endswith('.nii.gz') and '.seg.' not in f])

            for img_name in images:
                mask_name = img_name.replace('.nii.gz', '.seg.nii.gz')
                img_full_path = os.path.join(folder_path, img_name)
                mask_full_path = os.path.join(folder_path, mask_name)

                # Hanya masukkan jika pasangan mask-nya ada
                if os.path.exists(mask_full_path):
                    self.samples.append((img_full_path, mask_full_path))

        print(f"✅ Total ditemukan {len(self.samples)} pasangan data dari {len(sub_folders)} folder.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
      img_path, mask_path = self.samples[idx]
      image = nib.load(img_path).get_fdata()
      mask = nib.load(mask_path).get_fdata()

      mid = image.shape[2] // 2
      img_slice = image[:, :, mid]
      mask_slice = mask[:, :, mid]

      img_tensor = torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0) # [1, H, W]
      mask_tensor = torch.tensor(mask_slice, dtype=torch.long).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

      # --- RESIZE DI SINI ---
      img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(256, 256), mode='bilinear').squeeze(0).squeeze(0).unsqueeze(0)
      mask_tensor = F.interpolate(mask_tensor.float(), size=(256, 256), mode='nearest').long().squeeze()
      # ----------------------

      if img_tensor.max() > img_tensor.min():
          img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

      return img_tensor, mask_tensor


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

def train():
    # 1. Update Path & Parameter
    ROOT_DATA_PATH = "/content/drive/MyDrive/CT Brain Data"
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4      # Jika nanti Cuda Out of Memory, turunkan ke 2
    EPOCHS = 10
    VALIDATION_SPLIT = 0.15
    NUM_CLASSES = 2     # CT Scan kamu biasanya 2 (0: Background, 1: Target)
    INPUT_CHANNELS = 1  # WAJIB 1 untuk CT Scan

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 2. Ganti Dataset Loader ke CTMultiFolderDataset
    print("Mempersiapkan dataset CT Scan Multi-Folder...")
    if not os.path.exists(ROOT_DATA_PATH):
         print(f"⚠️ Warning: Directory '{ROOT_DATA_PATH}' not found. Please update ROOT_DATA_PATH.")
         return

    full_dataset = CTMultiFolderDataset(root_dir=ROOT_DATA_PATH)

    # Split Data
    n_val = int(len(full_dataset) * VALIDATION_SPLIT)
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])

    # Data Loader (pin_memory=True sangat membantu kecepatan di Colab)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    print(f"Data siap: {len(train_set)} training, {len(val_set)} validasi.")

    # 3. Inisialisasi Model dengan 1 Channel
    model = SE2_CNNET(n_channels=INPUT_CHANNELS, n_classes=NUM_CLASSES)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    scaler = GradScaler()
    ACCUMULATION_STEPS = 8  # Update bobot setiap 8 batch (Batch efektif = 8)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for i, (images, labels) in enumerate(pbar_train):
            images, labels = images.to(device), labels.to(device)

            # 1. Forward pass dengan Autocast (Hemat VRAM)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / ACCUMULATION_STEPS # Normalisasi loss

            # 2. Backward pass dengan Scaler
            scaler.scale(loss).backward()

            # 3. Step hanya setelah akumulasi terpenuhi
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * ACCUMULATION_STEPS
            pbar_train.set_postfix({'loss': loss.item() * ACCUMULATION_STEPS})

            # Bersihkan variabel agar VRAM tidak menumpuk
            del outputs, loss

        avg_train_loss = running_loss / len(train_loader)

        # --- VALIDASI ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validasi]")
            for images, labels in pbar_val:
                images, labels = images.to(device), labels.to(device)

                with autocast(): # Tetap pakai autocast di validasi agar cepat
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                pbar_val.set_postfix({'val_loss': loss.item()})
                del outputs, loss

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} -> Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Simpan model ke file
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

        # Paksa GPU buang sampah tiap ganti epoch
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train()