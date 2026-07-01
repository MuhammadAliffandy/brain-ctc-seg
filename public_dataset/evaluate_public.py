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
        
        # Temukan letak persis folder External_Test secara dinamis
        external_test_dir = None
        for r, d, f in os.walk(root_dir):
            if "External_Test" in d:
                external_test_dir = os.path.join(r, "External_Test")
                break
                
        if not external_test_dir:
            print(f"\n⚠️  WARNING: Folder External_Test tidak ditemukan di dalam {root_dir}.")
            return
            
        png_dir = os.path.join(external_test_dir, "PNG")
        mask_dir = os.path.join(external_test_dir, "MASKS")
        
        if not os.path.exists(png_dir) or not os.path.exists(mask_dir):
            print(f"\n⚠️  WARNING: Folder PNG atau MASKS tidak ditemukan di {external_test_dir}.")
            print("Perlu menyesuaikan nama folder jika dataset berbeda.")
            return

        # Load all PNGs in the External_Test/PNG folder
        inputs = sorted(glob.glob(os.path.join(png_dir, "*.png")))
        
        # Pairing logic
        for img_path in inputs:
            base_name = os.path.basename(img_path)
            mask_path_exact = os.path.join(mask_dir, base_name)
            
            if os.path.exists(mask_path_exact):
                # Nama file mask persis sama dengan nama image
                self.samples.append((img_path, mask_path_exact))
            else:
                # Fallback: Cari mask yang mengandung nama image
                name_without_ext = os.path.splitext(base_name)[0]
                possible_masks = glob.glob(os.path.join(mask_dir, f"*{name_without_ext}*.png"))
                if possible_masks:
                    self.samples.append((img_path, possible_masks[0]))
        
        if len(self.samples) == 0:
            print("\n⚠️  WARNING: Tidak dapat menemukan pasangan image-mask.")
            print(f"Ditemukan {len(inputs)} input images di PNG, tapi tidak ada mask yang cocok di MASKS.")
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
    print("📥 1. DOWNLOADING / LOCATING KAGGLE DATASET")
    print("="*60)
    
    LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    
    if not os.path.exists(LOCAL_DATA_DIR):
        print("Mendownload dataset ke cache Kaggle...")
        cache_path = kagglehub.dataset_download("ozguraslank/brain-stroke-ct-dataset")
        print(f"✅ Downloaded to cache: {cache_path}")
        
        print(f"📦 Memindahkan dataset ke workspace lokal: {LOCAL_DATA_DIR}")
        import shutil
        shutil.copytree(cache_path, LOCAL_DATA_DIR)
        print("✅ Proses copy selesai!")
    else:
        print(f"✅ Dataset sudah ada di lokal: {LOCAL_DATA_DIR}")

    dataset_path = LOCAL_DATA_DIR
    
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
                    probs = F.softmax(logits, dim=1)[:, 1]
                
                preds = (probs > 0.80).long()
                
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
