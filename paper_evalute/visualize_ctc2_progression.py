"""
visualize_ctc2_progression.py
=============================
Script khusus untuk mengekstrak dan memvisualisasikan slice 65-70 dari pasien CTC 2.
Permintaan Klien: Membuktikan model mampu menangkap perkembangan jumlah tumor (1, 3, hingga 4) 
seiring berjalannya slice.
"""

import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.colors import ListedColormap

# Import arsitektur SE2 dari script evaluasi
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.append(os.path.dirname(__file__))
from evaluate_trained_models import SE2_CNNET, load_se2_weights

def overlay_mask(image, mask, color, alpha=0.5):
    """
    Menumpuk (overlay) mask berwarna di atas gambar grayscale.
    color: list RGB, e.g., [1, 0, 0] untuk merah.
    """
    img_rgb = np.stack([image, image, image], axis=-1)
    mask_bool = mask > 0
    for c in range(3):
        img_rgb[mask_bool, c] = img_rgb[mask_bool, c] * (1 - alpha) + color[c] * alpha
    return np.clip(img_rgb, 0, 1)

def main():
    # ─── 1. KONFIGURASI PATH ───
    DATA_DIR = os.path.expanduser("~/Clara/local_ct_workspace_full")
    WEIGHT_PATH = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D/se2_unet_ctc_best.pth")
    SAVE_PATH = os.path.expanduser("~/Clara/brain-ctc-seg/training/Journal_Figures/CTC2_Progression_65_70.png")
    
    # Range slice yang diminta klien
    TARGET_SLICES = [65, 66, 67, 68, 69, 70]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Visualisasi Perkembangan Tumor CTC 2 (Device: {device})")

    # ─── 2. CARI FOLDER PASIEN ───
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Data directory {DATA_DIR} tidak ditemukan!")
        sys.exit(1)

    all_folders = os.listdir(DATA_DIR)
    # Cari folder yang mengandung "CTC" dan angka "2" (misal: CTC_2, CTC_002, dll)
    patient_folder = None
    for f in all_folders:
        if "CTC" in f.upper() and ("_2" in f or "_002" in f or " 2" in f):
            patient_folder = f
            break
            
    if patient_folder is None:
        print("❌ Error: Folder pasien CTC 2 tidak ditemukan di workspace!")
        # Fallback: ambil folder CTC pertama yang ada buat testing jika benar-benar hilang
        fallback = [f for f in all_folders if "CTC" in f.upper()]
        if fallback:
            patient_folder = fallback[0]
            print(f"⚠️  Fallback menggunakan folder: {patient_folder}")
        else:
            sys.exit(1)
            
    patient_path = os.path.join(DATA_DIR, patient_folder)
    print(f"📂 Menggunakan data pasien: {patient_folder}")

    # ─── 3. LOAD MODEL ───
    if not os.path.exists(WEIGHT_PATH):
        # Fallback jika model CTC spesifik belum selesai, coba pakai model all atau epoch 100
        fallback_weights = [
            os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D/se2_unet_epoch_100.pth"),
            os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D/se2_unet_best_25D_Boundary.pth")
        ]
        for fw in fallback_weights:
            if os.path.exists(fw):
                WEIGHT_PATH = fw
                print(f"⚠️  Weight CTC spesifik tidak ditemukan. Fallback ke: {WEIGHT_PATH}")
                break
        else:
            print("❌ Error: Weight model tidak ditemukan sama sekali!")
            sys.exit(1)

    model = SE2_CNNET(n_channels=3, n_classes=2).to(device)
    model = load_se2_weights(model, WEIGHT_PATH, device)
    model.eval()
    print("✅ Model loaded successfully.")

    # ─── 4. PROSES INFERENCE & PLOTTING ───
    # Layout Horizontal: 3 Baris (Input, GT, Pred) x N Kolom (Slices)
    fig, axes = plt.subplots(nrows=3, ncols=len(TARGET_SLICES), figsize=(4 * len(TARGET_SLICES), 12))
    fig.patch.set_facecolor('black')
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for i, z in enumerate(TARGET_SLICES):
        ax_input = axes[0, i]
        ax_gt    = axes[1, i]
        ax_pred  = axes[2, i]
        
        for ax in [ax_input, ax_gt, ax_pred]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Judul Kolom (Slice Number) di baris paling atas
        ax_input.set_title(f"Slice {z}", color='yellow', fontsize=14, fontweight='bold', pad=10)

        # Format slice ke string (z065, z066, dst)
        z_str = f"z{z:03d}"
        
        # Cari file numpy
        img_files = glob.glob(os.path.join(patient_path, f"*{z_str}_img.npy"))
        if not img_files:
            ax_input.text(0.5, 0.5, f"Not Found", color='white', ha='center', va='center')
            continue
            
        img_path = img_files[0]
        mask_path = img_path.replace('_img.npy', '_mask.npy')
        
        # Format tetangga untuk 2.5D context
        z_prev_str = f"z{z-1:03d}"
        z_next_str = f"z{z+1:03d}"
        
        prev_path = img_path.replace(z_str, z_prev_str)
        next_path = img_path.replace(z_str, z_next_str)
        
        # Load raw numpy arrays (Stack 2.5D)
        try:
            i0 = np.load(prev_path).astype(np.float32) if os.path.exists(prev_path) else np.load(img_path).astype(np.float32)
            i1 = np.load(img_path).astype(np.float32)
            i2 = np.load(next_path).astype(np.float32) if os.path.exists(next_path) else np.load(img_path).astype(np.float32)
            
            img_25d = np.stack([i0, i1, i2], axis=-1)  # (H, W, 3)
            gt_mask = np.load(mask_path).astype(np.uint8)   # (H, W)
        except Exception as e:
            print(f"⚠️ Gagal meload slice {z}: {e}"); continue

        # Normalisasi visualisasi gambar (tengah channel z)
        mid_img = img_25d[:, :, 1]
        if mid_img.max() > mid_img.min():
            mid_img = (mid_img - mid_img.min()) / (mid_img.max() - mid_img.min())

        # Inference menggunakan Model
        input_tensor = torch.from_numpy(img_25d).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                logits = model(input_tensor)
            pred_mask = torch.argmax(F.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()

        # Cropping & Rotation
        CROP_MARGIN = 40
        ROTATE_K = 3  # K=3 memutar ke arah sebaliknya (flip/berlawanan) agar mata di atas
        
        mid_img = np.rot90(mid_img[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)
        gt_mask = np.rot90(gt_mask[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)
        pred_mask = np.rot90(pred_mask[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)

        # Buat overlay
        gt_overlay = overlay_mask(mid_img, gt_mask, color=[1, 0, 0])      # Merah untuk Ground Truth
        pred_overlay = overlay_mask(mid_img, pred_mask, color=[0, 1, 0])  # Hijau untuk Prediction

        # Tampilkan
        ax_input.imshow(mid_img, cmap='gray')
        ax_gt.imshow(gt_overlay)
        ax_pred.imshow(pred_overlay)

    # Label Baris
    axes[0, 0].set_ylabel("Input Image", color='white', fontsize=16, fontweight='bold', labelpad=20)
    axes[1, 0].set_ylabel("Ground Truth", color='red', fontsize=16, fontweight='bold', labelpad=20)
    axes[2, 0].set_ylabel("Mod-Seg-SE(2)", color='green', fontsize=16, fontweight='bold', labelpad=20)

    # Simpan
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"\n✅ Visualisasi berhasil disimpan di: {SAVE_PATH}")
    print("Silakan download file tersebut untuk diberikan ke klien!")

if __name__ == "__main__":
    main()
