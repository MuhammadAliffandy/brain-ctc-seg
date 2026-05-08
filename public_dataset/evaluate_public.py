import os
import sys
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
plt.switch_backend('agg')

# Ensure kagglehub is installed
try:
    import kagglehub
except ImportError:
    print("❌ Error: kagglehub library not found.")
    print("Please install it first: pip install kagglehub[pandas-datasets]")
    sys.exit(1)

# Import Models from our training module
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "paper_evalute"))
from train_comparison_models import HarmonicNet, StandardUNet, nnUNet, AttentionUNet, TransUNet
from evaluate_trained_models import SE2_CNNET, load_se2_weights

class PublicKaggleDataset(Dataset):
    """
    Dataset loader dinamis untuk Kaggle Stroke Dataset.
    Karena struktur pastinya belum diketahui, script ini mencoba mencari
    file gambar secara rekursif dan mencari file mask yang berkesesuaian.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        
        # Cari semua gambar (PNG/JPG)
        all_images = glob.glob(os.path.join(root_dir, '**', '*.*'), recursive=True)
        img_exts = ['.png', '.jpg', '.jpeg']
        
        images = [f for f in all_images if any(f.lower().endswith(ext) for ext in img_exts)]
        
        # Pisahkan mana yang gambar input dan mana yang mask
        # Asumsi umum: file mask mengandung kata 'mask' di namanya
        masks = [f for f in images if 'mask' in os.path.basename(f).lower()]
        inputs = [f for f in images if 'mask' not in os.path.basename(f).lower()]
        
        # Pairing logic (Mencocokkan nama file input dengan mask)
        for img_path in inputs:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            # Cari mask yang cocok dengan nama gambar (misal: image_1.png -> image_1_mask.png)
            matching_mask = next((m for m in masks if base_name in m), None)
            
            if matching_mask:
                self.samples.append((img_path, matching_mask))
        
        if len(self.samples) == 0:
            print("\n⚠️  WARNING: Tidak dapat menemukan pasangan image-mask.")
            print(f"Ditemukan {len(inputs)} input images dan {len(masks)} mask images.")
            print("Perlu menyesuaikan Dataset Loader jika Kaggle menggunakan struktur spesifik.")
        else:
            print(f"📊 Ditemukan {len(self.samples)} pasangan data untuk evaluasi.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        # Load dengan OpenCV
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            # Fallback jika rusak
            img = np.zeros((256, 256), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            
        # Resize ke ukuran standar model kita (256x256)
        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Binarize mask
        mask = (mask > 127).astype(np.uint8)
        
        # Normalisasi Image (Min-Max)
        img_float = img.astype(np.float32)
        if img_float.max() > img_float.min():
            img_norm = (img_float - img_float.min()) / (img_float.max() - img_float.min())
        else:
            img_norm = img_float
            
        # Karena model kita butuh 3 channel (2.5D), kita duplikasi channelnya
        img_3c = np.stack([img_norm, img_norm, img_norm], axis=0)
        
        return torch.from_numpy(img_3c), torch.from_numpy(mask).long()

def get_models_dict(device):
    # Load bobot yang sudah kita training sebelumnya (Zero-Shot)
    weights_dir = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D")
    
    models = {
        "Mod-Seg-SE(2)": (SE2_CNNET, os.path.join(weights_dir, "se2_unet_ct_best.pth"), True),
        "HarmonicNet": (HarmonicNet, os.path.join(weights_dir, "harmonic_net_ct_best.pth"), False),
        "nnU-Net": (nnUNet, os.path.join(weights_dir, "nn_unet_ct_best.pth"), False),
        "Attention U-Net": (AttentionUNet, os.path.join(weights_dir, "attention_unet_ct_best.pth"), False),
        "TransUNet": (TransUNet, os.path.join(weights_dir, "trans_unet_ct_best.pth"), False),
        "Standard U-Net": (StandardUNet, os.path.join(weights_dir, "standard_unet_ct_best.pth"), False),
    }
    return models

def evaluate():
    print("="*60)
    print("📥 1. DOWNLOADING KAGGLE DATASET")
    print("="*60)
    dataset_path = kagglehub.dataset_download("ozguraslank/brain-stroke-ct-dataset")
    print(f"✅ Dataset Path: {dataset_path}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}\n")

    dataset = PublicKaggleDataset(dataset_path)
    if len(dataset) == 0:
        print("❌ Evaluasi dihentikan karena Dataset Loader tidak bisa mem-parsing struktur Kaggle.")
        return
        
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    models_dict = get_models_dict(device)
    
    results = []

    for name, (ModelClass, weight_path, is_se2) in models_dict.items():
        print(f"\n🚀 Evaluating {name}...")
        if not os.path.exists(weight_path):
            print(f"⚠️  Weight tidak ditemukan: {weight_path} -> SKIP")
            continue
            
        if is_se2:
            model = load_se2_weights(ModelClass, weight_path, device)
        else:
            model = ModelClass(n_channels=3, n_classes=2).to(device)
            try:
                model.load_state_dict(torch.load(weight_path, map_location=device))
            except Exception as e:
                print(f"⚠️  Error loading weights for {name}: {e} -> SKIP")
                continue
                
        model.eval()
        
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for imgs, masks in tqdm(loader, desc=f"{name}", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    logits = model(imgs)
                
                preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
                
                pf = preds.view(-1)
                mf = masks.view(-1)
                
                tp += ((pf == 1) & (mf == 1)).sum().item()
                fp += ((pf == 1) & (mf == 0)).sum().item()
                fn += ((pf == 0) & (mf == 1)).sum().item()
                tn += ((pf == 0) & (mf == 0)).sum().item()
                
        eps = 1e-7
        dice = (2 * tp) / (2 * tp + fp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        
        print(f"   => Dice: {dice:.4f} | IoU: {iou:.4f} | Acc: {accuracy:.4f}")
        
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall (Sens.)": recall,
            "F1 (Dice)": dice,
            "IoU": iou
        })
        
    df = pd.DataFrame(results)
    save_path = os.path.join(os.path.dirname(__file__), "public_eval_metrics.csv")
    df.to_csv(save_path, index=False)
    
    print("\n" + "="*60)
    print("📊 FINAL RESULTS ON KAGGLE DATASET")
    print("="*60)
    print(df.to_string(index=False))
    print(f"\n✅ Tersimpan di: {save_path}")

if __name__ == "__main__":
    evaluate()
