import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from escnn import gspaces
import escnn.nn as enn

# ==========================================
# 1. MODEL ARCHITECTURES
# ==========================================
class SE2_CNNET(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, N=8, base_channels=24):
        super(SE2_CNNET, self).__init__()
        self.r2_act = gspaces.rot2dOnR2(N=N)
        self.feat_type_in  = enn.FieldType(self.r2_act, n_channels * [self.r2_act.trivial_repr])
        self.feat_type_out = enn.FieldType(self.r2_act, n_classes * [self.r2_act.trivial_repr])
        
        from benchmarking import DoubleEquivariantConv, Down, Up
        
        self.inc   = DoubleEquivariantConv(self.feat_type_in, enn.FieldType(self.r2_act, base_channels * [self.r2_act.regular_repr]))
        self.down1 = Down(enn.FieldType(self.r2_act, base_channels * [self.r2_act.regular_repr]), 
                                     enn.FieldType(self.r2_act, base_channels*2 * [self.r2_act.regular_repr]))
        self.down2 = Down(enn.FieldType(self.r2_act, base_channels*2 * [self.r2_act.regular_repr]), 
                                     enn.FieldType(self.r2_act, base_channels*4 * [self.r2_act.regular_repr]))
        self.down3 = Down(enn.FieldType(self.r2_act, base_channels*4 * [self.r2_act.regular_repr]), 
                                     enn.FieldType(self.r2_act, base_channels*8 * [self.r2_act.regular_repr]))
        self.down4 = Down(enn.FieldType(self.r2_act, base_channels*8 * [self.r2_act.regular_repr]), 
                                     enn.FieldType(self.r2_act, base_channels*16 * [self.r2_act.regular_repr]))

        self.up1 = Up(enn.FieldType(self.r2_act, base_channels*16 * [self.r2_act.regular_repr]), 
                                 enn.FieldType(self.r2_act, base_channels*8 * [self.r2_act.regular_repr]))
        self.up2 = Up(enn.FieldType(self.r2_act, base_channels*8 * [self.r2_act.regular_repr]), 
                                 enn.FieldType(self.r2_act, base_channels*4 * [self.r2_act.regular_repr]))
        self.up3 = Up(enn.FieldType(self.r2_act, base_channels*4 * [self.r2_act.regular_repr]), 
                                 enn.FieldType(self.r2_act, base_channels*2 * [self.r2_act.regular_repr]))
        self.up4 = Up(enn.FieldType(self.r2_act, base_channels*2 * [self.r2_act.regular_repr]), 
                                 enn.FieldType(self.r2_act, base_channels * [self.r2_act.regular_repr]))

        self.outc = enn.R2Conv(enn.FieldType(self.r2_act, base_channels * [self.r2_act.regular_repr]), 
                               self.feat_type_out, kernel_size=1, bias=False)

    def forward(self, x):
        x_geom = enn.GeometricTensor(x, self.feat_type_in)
        x1 = self.inc(x_geom); x2 = self.down1(x1); x3 = self.down2(x2)
        x4 = self.down3(x3); x5 = self.down4(x4)
        x  = self.up1(x5, x4); x = self.up2(x, x3)
        x  = self.up3(x, x2);  x = self.up4(x, x1)
        return self.outc(x).tensor

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class StandardUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(StandardUNet, self).__init__()
        self.inc    = DoubleConv(n_channels, 64)
        self.down1  = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2  = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3  = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.up1    = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1  = DoubleConv(512, 256)
        self.up2    = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2  = DoubleConv(256, 128)
        self.up3    = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3  = DoubleConv(128, 64)
        self.outc   = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x);  x2 = self.down1(x1)
        x3 = self.down2(x2); x4 = self.down3(x3)
        x  = self.up1(x4); x = torch.cat([x, x3], dim=1); x = self.conv1(x)
        x  = self.up2(x);  x = torch.cat([x, x2], dim=1); x = self.conv2(x)
        x  = self.up3(x);  x = torch.cat([x, x1], dim=1); x = self.conv3(x)
        return self.outc(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(AttentionUNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConv(n_channels, 64)
        self.Conv2 = DoubleConv(64, 128)
        self.Conv3 = DoubleConv(128, 256)
        self.Conv4 = DoubleConv(256, 512)

        self.Up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = DoubleConv(512, 256)

        self.Up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = DoubleConv(256, 128)

        self.Up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = DoubleConv(128, 64)

        self.Conv_1x1 = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)

        d4 = self.Up4(e4)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv_1x1(d2)
        return out

import torchvision.models.segmentation as seg_models

class DeepLabV3_ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = seg_models.deeplabv3_resnet50(weights=None, num_classes=2)
    def forward(self, x):
        return self.model(x)['out']

class FCN_ResNet101(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = seg_models.fcn_resnet101(weights=None, num_classes=2)
    def forward(self, x):
        return self.model(x)['out']

# ==========================================
# 2. DATASET LOADER
# ==========================================
class SpeedTestDataset(Dataset):
    def __init__(self, root_dir, max_samples=200):
        self.all_samples = []
        self.patient_slices = {}
        
        print("🔍 Scanning dataset for speed test...")
        for root, _, files in os.walk(root_dir):
            img_files = [f for f in files if f.endswith('_img.npy')]
            if not img_files: continue
            
            img_files = sorted(img_files, key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0)
            patient = os.path.basename(root)
            self.patient_slices[patient] = [os.path.join(root, f) for f in img_files]
            
            for i in range(len(self.patient_slices[patient])):
                self.all_samples.append((patient, i))
                
        # Limit samples so we don't take forever, just enough for stable average
        if len(self.all_samples) > max_samples:
            self.all_samples = self.all_samples[:max_samples]

    def __len__(self): return len(self.all_samples)

    def __getitem__(self, idx):
        patient, slice_idx = self.all_samples[idx]
        slices = self.patient_slices[patient]
        
        idx_prev = max(0, slice_idx - 1)
        idx_next = min(len(slices) - 1, slice_idx + 1)
        
        img_prev = np.load(slices[idx_prev]).astype(np.float32)
        img_curr = np.load(slices[slice_idx]).astype(np.float32)
        img_next = np.load(slices[idx_next]).astype(np.float32)
        
        image_25d = np.stack([img_prev, img_curr, img_next], axis=-1)
        image_tensor = torch.from_numpy(image_25d).permute(2, 0, 1).unsqueeze(0)
        
        image_tensor = F.interpolate(image_tensor, size=(256, 256), mode='bilinear', align_corners=False)
        return image_tensor.squeeze(0)

# ==========================================
# 3. MEASUREMENT LOGIC
# ==========================================
def measure_speed(model, dataloader, device):
    model.eval()
    
    # 1. GPU WARMUP (Crucial for accurate timing)
    print("   🔥 Warming up GPU...")
    with torch.no_grad():
        for i, images in enumerate(dataloader):
            if i >= 5: break
            _ = model(images.to(device))
            
    if torch.cuda.is_available(): torch.cuda.synchronize()
    
    # 2. ACTUAL MEASUREMENT
    total_time = 0.0
    total_images = 0
    
    print("   ⏱️  Running benchmark...")
    with torch.no_grad():
        for images in tqdm(dataloader, desc="   Processing"):
            images = images.to(device)
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            start_time = time.time()
            
            _ = model(images)
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_images += images.size(0)
            
    return total_time / total_images

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def run_benchmark():
    print("\n" + "="*70)
    print("⏱️  INFERENCE SPEED BENCHMARK (SEGMENTATION)")
    print("="*70 + "\n")
    
    LOCAL_DATA_PATH = os.path.expanduser("~/Clara/local_ct_workspace")
    if not os.path.exists(LOCAL_DATA_PATH):
        print(f"❌ Cannot find local data path: {LOCAL_DATA_PATH}")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}\n")
    
    # Batch size 1 simulates sequential real-world inference perfectly
    dataset = SpeedTestDataset(LOCAL_DATA_PATH, max_samples=200)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    models = {
        "Mod-Seg-SE(2)": SE2_CNNET(n_channels=3, n_classes=2, N=8, base_channels=24).to(device),
        "Standard U-Net": StandardUNet(n_channels=3, n_classes=2).to(device),
        "Attention U-Net": AttentionUNet(n_channels=3, n_classes=2).to(device),
        "ResNet50 (DeepLabV3)": DeepLabV3_ResNet50().to(device),
        "ResNet101 (FCN)": FCN_ResNet101().to(device)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"🚀 Benchmarking {name}...")
        avg_time = measure_speed(model, dataloader, device)
        results.append({
            "Model Name": name,
            "Inference (seconds/image)": round(avg_time, 4)
        })
        print(f"   ✅ Done: {round(avg_time, 4)} seconds/image\n")
        
    # Print Table
    print("="*50)
    print("📋 TABLE 7: Average inference time per image")
    print("="*50)
    print(f"{'Model Name':<25} | {'Inference (seconds/image)'}")
    print("-" * 50)
    for r in results:
        print(f"{r['Model Name']:<25} | {r['Inference (seconds/image)']}")
    print("-" * 50)

if __name__ == "__main__":
    run_benchmark()
