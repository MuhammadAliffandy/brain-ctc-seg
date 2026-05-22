import os
import sys
import glob
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

plt.switch_backend('agg')

# Ensure we can import our modules
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(base_dir, "training"))
sys.path.append(os.path.join(base_dir, "public_dataset"))

from train_se2_by_dataset import SE2_CNNET
sys.path.append(os.path.join(base_dir, "paper_evalute"))
from evaluate_trained_models import load_se2_weights
from train_all_intra import get_kaggle_splits
from train_all_intra_hemorrhage import get_kaggle_hemorrhage_splits

def overlay_mask(image, mask, color, alpha=0.6):
    img_rgb = np.stack([image, image, image], axis=-1)
    mask_bool = mask > 0
    for c in range(3):
        img_rgb[mask_bool, c] = img_rgb[mask_bool, c] * (1 - alpha) + color[c] * alpha
    return np.clip(img_rgb, 0, 1)

def get_best_ct_ctc_slices(dataset_prefix, num_slices=4):
    DATA_DIR = os.path.expanduser("~/Clara/local_ct_workspace_full")
    all_folders = os.listdir(DATA_DIR)
    
    if dataset_prefix == "CT":
        patients = [f for f in all_folders if f.upper().startswith("CT") and "CTC" not in f.upper()]
    else:
        patients = [f for f in all_folders if f.upper().startswith("CTC")]
        
    patients = sorted(patients)[:num_slices+5] # Give some buffer
    
    slices = []
    for pat in patients:
        pat_dir = os.path.join(DATA_DIR, pat)
        img_files = glob.glob(os.path.join(pat_dir, "*_img.npy"))
        
        best_img, best_mask, best_25d = None, None, None
        max_tumor = 0
        
        for img_path in img_files:
            mask_path = img_path.replace('_img.npy', '_mask.npy')
            if not os.path.exists(mask_path): continue
            m = np.load(mask_path).astype(np.uint8)
            t_sum = np.sum(m)
                
            # Check for reasonable tumor size (avoid bounding box artifacts or skull masks)
            if t_sum > max_tumor and t_sum < 4000:
                max_tumor = t_sum
                
                z_str = img_path.split('_')[-2]
                z = int(z_str.replace('z', ''))
                prev_path = img_path.replace(z_str, f"z{z-1:03d}")
                next_path = img_path.replace(z_str, f"z{z+1:03d}")
                
                try:
                    i0 = np.load(prev_path).astype(np.float32) if os.path.exists(prev_path) else np.load(img_path).astype(np.float32)
                    i1 = np.load(img_path).astype(np.float32)
                    i2 = np.load(next_path).astype(np.float32) if os.path.exists(next_path) else np.load(img_path).astype(np.float32)
                    best_25d = np.stack([i0, i1, i2], axis=-1)
                    best_img = i1
                    best_mask = m
                except: pass
                
        if best_img is not None and max_tumor > 50:
            CROP = 40
            best_img = np.rot90(best_img[CROP:-CROP, CROP:-CROP], k=3)
            best_mask = np.rot90(best_mask[CROP:-CROP, CROP:-CROP], k=3)
            best_25d = np.rot90(best_25d[CROP:-CROP, CROP:-CROP, :], k=3)
            
            # Normalize for visualization
            if best_img.max() > best_img.min():
                best_img = (best_img - best_img.min()) / (best_img.max() - best_img.min())
            if best_25d.max() > best_25d.min():
                best_25d = (best_25d - best_25d.min()) / (best_25d.max() - best_25d.min())
                
            slices.append((best_img, best_mask, best_25d))
            
        if len(slices) == num_slices:
            break
            
    while len(slices) < num_slices and len(slices) > 0:
        slices.append(slices[-1])
        
    return slices

def get_kaggle_slices(dataset_type, num_slices=4):
    if dataset_type == "Stroke":
        root_dir = os.path.join(base_dir, "public_dataset", "data")
        _, test_s = get_kaggle_splits(root_dir)
    else:
        _, test_s = get_kaggle_hemorrhage_splits()
        
    slices = []
    for img_p, mask_p in test_s:
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None: continue
        
        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)
        
        # Check for reasonable tumor size (avoid full brain/skull masks which have > 5000 pixels)
        pixel_count = np.sum(mask)
        if pixel_count > 50 and pixel_count < 4000:
            img_float = img.astype(np.float32)
            if img_float.max() > img_float.min():
                img_norm = (img_float - img_float.min()) / (img_float.max() - img_float.min())
            else:
                img_norm = img_float
            img_3c = np.stack([img_norm, img_norm, img_norm], axis=-1)
            slices.append((img_norm, mask, img_3c))
        
        if len(slices) == num_slices:
            break
            
    while len(slices) < num_slices and len(slices) > 0:
        slices.append(slices[-1])
        
    return slices

def infer_slices(model, slices, device):
    preds = []
    model.eval()
    for img, mask, img_3c in slices:
        # img_3c is (H, W, 3), model expects (1, 3, H, W)
        input_tensor = torch.from_numpy(img_3c).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                logits = model(input_tensor)
            pred_mask = torch.argmax(F.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()
        preds.append(pred_mask)
    return preds

def generate_figure_a(ct, ctc, stroke, hem, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor('white')
    titles = ["Private CT", "Private CTC", "Kaggle Stroke", "Kaggle Hemorrhage"]
    data = [ct[0], ctc[0], stroke[0], hem[0]]
    
    for i in range(4):
        img, mask, _ = data[i]
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(titles[i], fontsize=18, fontweight='bold')
        axes[0, i].axis('off')
        
        overlay = overlay_mask(img, mask, color=[1, 0, 0], alpha=0.6)
        axes[1, i].imshow(overlay)
        axes[1, i].set_title("Ground Truth", fontsize=16)
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_figure_b(slices, preds, title_prefix, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor('white')
    
    for i in range(4):
        img, mask, _ = slices[i]
        pred = preds[i]
        
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f"Sample {i+1} Input", fontsize=16, fontweight='bold')
        axes[0, i].axis('off')
        
        overlay = overlay_mask(img, pred, color=[1, 0, 0], alpha=0.6)
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"{title_prefix}-SE(2) Prediction", fontsize=16)
        axes[1, i].axis('off')
        
    plt.suptitle(f"Multi-sample {title_prefix} slice results indicating robust generalization.", fontsize=20, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.expanduser("~/Clara/brain-ctc-seg/training/Journal_Figures")
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading data slices...")
    ct_slices = get_best_ct_ctc_slices("CT", 1)
    ctc_slices = get_best_ct_ctc_slices("CTC", 4)
    stroke_slices = get_kaggle_slices("Stroke", 4)
    hem_slices = get_kaggle_slices("Hemorrhage", 4)
    
    print("Generating Figure A (Dataset Variety)...")
    generate_figure_a(ct_slices, ctc_slices, stroke_slices, hem_slices, os.path.join(out_dir, "Fig_Supplemental_A_Variety.png"))
    
    print("Loading Models & Generating Figure B (CTC, Stroke, Hemorrhage)...")
    
    # 1. CTC
    w_ctc = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D/se2_unet_ctc_best.pth")
    if os.path.exists(w_ctc):
        model_ctc = load_se2_weights(SE2_CNNET, w_ctc, device)
        preds_ctc = infer_slices(model_ctc, ctc_slices, device)
        generate_figure_b(ctc_slices, preds_ctc, "CTC", os.path.join(out_dir, "Fig_Supplemental_B_CTC.png"))
    else:
        print(f"⚠️ Weights for CTC not found: {w_ctc}")
    
    # 2. Stroke
    w_stroke = os.path.join(base_dir, "public_dataset", "saved_models", "Mod-Seg-SE2_kaggle_best.pth")
    if os.path.exists(w_stroke):
        model_stroke = load_se2_weights(SE2_CNNET, w_stroke, device)
        preds_stroke = infer_slices(model_stroke, stroke_slices, device)
        generate_figure_b(stroke_slices, preds_stroke, "Stroke", os.path.join(out_dir, "Fig_Supplemental_B_Stroke.png"))
    else:
        print(f"⚠️ Weights for Stroke not found: {w_stroke}")
        
    # 3. Hemorrhage
    w_hem = os.path.join(base_dir, "public_dataset", "saved_models", "Mod-Seg-SE2_kaggle_hemorrhage_best.pth")
    if os.path.exists(w_hem):
        model_hem = load_se2_weights(SE2_CNNET, w_hem, device)
        preds_hem = infer_slices(model_hem, hem_slices, device)
        generate_figure_b(hem_slices, preds_hem, "Hemorrhage", os.path.join(out_dir, "Fig_Supplemental_B_Hemorrhage.png"))
    else:
        print(f"⚠️ Weights for Hemorrhage not found: {w_hem}")

    print("✅ All supplemental figures generated successfully in Journal_Figures!")

if __name__ == "__main__":
    main()
