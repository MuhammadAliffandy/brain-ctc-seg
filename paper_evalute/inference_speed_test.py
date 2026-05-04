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

class FCN_VGG(nn.Module):
    def __init__(self, vgg_model, n_classes=2):
        super().__init__()
        self.features = vgg_model.features
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, n_classes, kernel_size=1)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

def get_vgg16_fcn():
    import torchvision.models as models
    return FCN_VGG(models.vgg16(weights=None), 2)

def get_vgg19_fcn():
    import torchvision.models as models
    return FCN_VGG(models.vgg19(weights=None), 2)

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
        "VGG16 (FCN)": get_vgg16_fcn().to(device),
        "VGG19 (FCN)": get_vgg19_fcn().to(device),
        "ResNet50 (DeepLabV3)": DeepLabV3_ResNet50().to(device),
        "ResNet101 (FCN)": FCN_ResNet101().to(device)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"🚀 Benchmarking {name}...")
        # Hitung jumlah parameter dalam hitungan Juta (Millions)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        
        avg_time = measure_speed(model, dataloader, device)
        results.append({
            "Model Name": name,
            "Parameters (M)": round(params, 2),
            "Inference (seconds/image)": round(avg_time, 4)
        })
        print(f"   ✅ Done: {round(avg_time, 4)} seconds | Params: {round(params, 2)} M\n")
        
    # Print Table
    print("="*75)
    print("📋 TABLE 7: Average inference time & Model Complexity")
    print("="*75)
    print(f"{'Model Name':<25} | {'Parameters (M)':<15} | {'Inference (seconds/image)'}")
    print("-" * 75)
    for r in results:
        print(f"{r['Model Name']:<25} | {str(r['Parameters (M)']) + ' M':<15} | {r['Inference (seconds/image)']}")
    print("-" * 75)

if __name__ == "__main__":
    run_benchmark()
