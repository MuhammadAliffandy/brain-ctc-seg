import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import kagglehub

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
# 2. INFERENCE DATASET LOADER
# ==========================================

class InferenceCTDataset(Dataset):
    """
    Loads only images for prediction generation.
    """
    def __init__(self, root_dir):
        self.image_files = []
        
        print(f"🔍 Scanning directory for inference data: {root_dir}...")
        
        for root, dirs, files in os.walk(root_dir):
            img_files = sorted([f for f in files if f.endswith('_img.npy')])
            for img_name in img_files:
                self.image_files.append(os.path.join(root, img_name))
                
        print(f"✅ Found a total of {len(self.image_files)} valid images.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        filename = os.path.basename(img_path)
        
        # Load numpy array
        image = np.load(img_path).astype(np.float32)
        
        # Ensure channel dimension (1, H, W)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
            
        # Convert to tensor and add batch dimension temporarily for interpolation
        image_tensor = torch.from_numpy(image).unsqueeze(0) 
        
        # 🪄 RESIZE MAGIC (Adjust 256 to your training size if needed)
        TARGET_SIZE = (256, 256) 
        image_tensor = F.interpolate(image_tensor, size=TARGET_SIZE, mode='bilinear', align_corners=False)
        image_tensor = image_tensor.squeeze(0)
        
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
    # NAMA MODEL DIKEMBALIKAN KE VERSI AWAL
    MODEL_WEIGHTS_PATH = "se2_unet_epoch_100.pth" 
    OUTPUT_DIR = os.path.expanduser("~/Clara/inference_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Kept small to prevent OOM
    BATCH_SIZE = 4 
    
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
    run_inference()