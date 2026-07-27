import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nibabel as nib
import scipy.ndimage
plt.switch_backend('agg')

# Import arsitektur SE2 dari script evaluasi
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from evaluate_trained_models import SE2_CNNET, load_se2_weights

def overlay_mask(image, mask, color, alpha=0.5):
    img_rgb = np.stack([image, image, image], axis=-1)
    mask_bool = mask > 0
    for c in range(3):
        img_rgb[mask_bool, c] = img_rgb[mask_bool, c] * (1 - alpha) + color[c] * alpha
    return np.clip(img_rgb, 0, 1)

def get_pixel_spacing():
    """
    Mencari original NIfTI untuk membaca exact pixel spacing dari metadata CT scan.
    Jika tidak ketemu, gunakan asumsi klinis standar (0.45 mm x 0.45 mm).
    """
    raw_dir = os.path.expanduser("~/Clara/new_drive/CT Brain Data/Original Data")
    if os.path.exists(raw_dir):
        # Cari file ctc 2
        for f in os.listdir(raw_dir):
            if "CTC" in f.upper() and ("_2" in f or "_002" in f or " 2" in f):
                patient_dir = os.path.join(raw_dir, f)
                nii_files = [x for x in os.listdir(patient_dir) if x.endswith('.nii.gz') and '.seg.' not in x]
                if nii_files:
                    nii_path = os.path.join(patient_dir, nii_files[0])
                    try:
                        img_obj = nib.load(nii_path)
                        header = img_obj.header
                        pixdim = header['pixdim']
                        # pixdim[1] is x spacing, pixdim[2] is y spacing in mm
                        sx, sy = float(pixdim[1]), float(pixdim[2])
                        # Validasi nilai rasional (0.1 mm - 2.0 mm)
                        if 0.1 < sx < 2.0 and 0.1 < sy < 2.0:
                            print(f"📏 Ditemukan pixel spacing dari NIfTI: {sx:.3f} mm x {sy:.3f} mm")
                            return sx, sy
                    except Exception as e:
                        print(f"⚠️ Gagal membaca header NIfTI: {e}")
    
    # Fallback clinical standard for head CT
    print("⚠️ NIfTI original tidak ditemukan/tidak valid. Menggunakan estimasi standar Head CT: 0.45 mm x 0.45 mm")
    return 0.45, 0.45

def main():
    DATA_DIR = os.path.expanduser("~/Clara/local_ct_workspace_full")
    WEIGHT_PATH = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D/se2_unet_ctc_best.pth")
    SAVE_PATH = os.path.expanduser("~/Clara/brain-ctc-seg/training/Journal_Figures/Tumor_Size_Estimation.png")
    
    TARGET_SLICES = [65, 66]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Estimasi Ukuran Tumor CTC 2 (Device: {device})")

    # Ambil resolusi pixel
    sp_x, sp_y = get_pixel_spacing()
    pixel_area_mm2 = sp_x * sp_y

    all_folders = os.listdir(DATA_DIR)
    patient_folder = next((f for f in all_folders if "CTC" in f.upper() and ("_2" in f or "_002" in f or " 2" in f)), None)
    if not patient_folder:
        print("❌ Error: Folder pasien CTC 2 tidak ditemukan di workspace!")
        return
        
    patient_path = os.path.join(DATA_DIR, patient_folder)
    print(f"📂 Memproses: {patient_folder}")

    # Load Model
    if not os.path.exists(WEIGHT_PATH):
        print(f"❌ Error: Model weights not found at {WEIGHT_PATH}")
        return

    model = load_se2_weights(SE2_CNNET, WEIGHT_PATH, device)
    model.eval()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i, z in enumerate(TARGET_SLICES):
        ax_gt    = axes[0, i]
        ax_pred  = axes[1, i]
        
        for ax in [ax_gt, ax_pred]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('black')

        z_str = f"z{z:03d}"
        img_files = glob.glob(os.path.join(patient_path, f"*{z_str}_img.npy"))
        if not img_files:
            continue
            
        img_path = img_files[0]
        mask_path = img_path.replace('_img.npy', '_mask.npy')
        
        prev_path = img_path.replace(z_str, f"z{z-1:03d}")
        next_path = img_path.replace(z_str, f"z{z+1:03d}")
        
        try:
            i0 = np.load(prev_path).astype(np.float32) if os.path.exists(prev_path) else np.load(img_path).astype(np.float32)
            i1 = np.load(img_path).astype(np.float32)
            i2 = np.load(next_path).astype(np.float32) if os.path.exists(next_path) else np.load(img_path).astype(np.float32)
            
            img_25d = np.stack([i0, i1, i2], axis=-1)
            gt_mask = np.load(mask_path).astype(np.uint8)
        except Exception as e:
            print(f"⚠️ Gagal meload slice {z}: {e}"); continue

        # Normalization
        mid_img = img_25d[:, :, 1]
        if mid_img.max() > mid_img.min():
            mid_img = (mid_img - mid_img.min()) / (mid_img.max() - mid_img.min())

        if img_25d.max() > img_25d.min():
            img_25d_norm = (img_25d - img_25d.min()) / (img_25d.max() - img_25d.min())
        else:
            img_25d_norm = img_25d

        # Inference
        input_tensor = torch.from_numpy(img_25d_norm).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                logits = model(input_tensor)
            pred_mask = torch.argmax(F.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()

        # Cropping & Rotation
        CROP_MARGIN = 40
        ROTATE_K = 3
        
        mid_img = np.rot90(mid_img[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)
        gt_mask = np.rot90(gt_mask[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)
        pred_mask = np.rot90(pred_mask[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)

        # Hitung Ukuran (Pixels -> mm^2 -> cm^2) untuk OVERALL
        gt_pixels = np.sum(gt_mask == 1)
        pred_pixels = np.sum(pred_mask == 1)
        
        gt_area_mm2 = gt_pixels * pixel_area_mm2
        pred_area_mm2 = pred_pixels * pixel_area_mm2
        
        # Overlays
        gt_overlay = overlay_mask(mid_img, gt_mask, color=[1, 0, 0])
        pred_overlay = overlay_mask(mid_img, pred_mask, color=[0, 1, 0])

        ax_gt.imshow(gt_overlay)
        ax_pred.imshow(pred_overlay)

        # Annotate individual tumors for Ground Truth
        gt_labels, gt_num = scipy.ndimage.label(gt_mask)
        for idx in range(1, gt_num + 1):
            blob = (gt_labels == idx)
            pixels = np.sum(blob)
            if pixels < 5: continue
            area_mm2 = pixels * pixel_area_mm2
            y_idx, x_idx = np.where(blob)
            cy, cx = int(np.mean(y_idx)), int(np.mean(x_idx))
            
            top_y = np.min(y_idx)
            
            # Geser teks ke atas tumor agar tidak menutupi, background dibuat lebih transparan
            ax_gt.text(cx, top_y - 12, f"{area_mm2:.1f} mm²", color='white', 
                       fontsize=9, fontweight='bold', ha='center', va='center',
                       bbox=dict(facecolor='darkred', alpha=0.25, edgecolor='none', pad=2))

        # Annotate individual tumors for Prediction
        pred_labels, pred_num = scipy.ndimage.label(pred_mask)
        for idx in range(1, pred_num + 1):
            blob = (pred_labels == idx)
            pixels = np.sum(blob)
            if pixels < 5: continue
            area_mm2 = pixels * pixel_area_mm2
            y_idx, x_idx = np.where(blob)
            cy, cx = int(np.mean(y_idx)), int(np.mean(x_idx))
            
            top_y = np.min(y_idx)
            
            # Geser teks ke atas tumor agar tidak menutupi, background dibuat lebih transparan
            ax_pred.text(cx, top_y - 12, f"{area_mm2:.1f} mm²", color='white', 
                       fontsize=9, fontweight='bold', ha='center', va='center',
                       bbox=dict(facecolor='darkgreen', alpha=0.25, edgecolor='none', pad=2))

        # Annotations di gambar (Hanya menampilkan slice info, area total bisa dihilangkan atau dipertahankan)
        ax_gt.set_title(f"Slice {z} (Ground Truth)", fontsize=14, fontweight='bold', pad=10, color='darkred')
        ax_pred.set_title(f"Slice {z} (SE(2) Prediction)", fontsize=14, fontweight='bold', pad=10, color='darkgreen')

        # Print to console
        print(f"\n--- SLICE {z} ---")
        print(f"Ground Truth : {gt_pixels} pixels -> {gt_area_mm2:.2f} mm² ({gt_area_mm2/100:.3f} cm²)")
        print(f"Prediksi SE2 : {pred_pixels} pixels -> {pred_area_mm2:.2f} mm² ({pred_area_mm2/100:.3f} cm²)")
        error_margin = abs(gt_area_mm2 - pred_area_mm2)
        print(f"Error Margin : {error_margin:.2f} mm²")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualisasi Ukuran Tumor berhasil disimpan di:\n{SAVE_PATH}")

if __name__ == "__main__":
    main()
