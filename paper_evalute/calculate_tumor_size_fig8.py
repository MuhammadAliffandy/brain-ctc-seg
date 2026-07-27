"""
calculate_tumor_size_fig8.py
============================
Reproduce Fig. 8 style figure:
  - Center: Ground Truth (red border) and CT-SE(2) Prediction (green border)
            with contour OUTLINES (not filled overlays)
  - Left col: 2 thumbnail crops from GT — Largest & Smallest lesion (red contour)
  - Right col: 2 thumbnail crops from Pred — Largest & Smallest lesion (green contour)
  - Labels below each thumbnail: "Largest • X mm²" / "Smallest • X mm²"

Usage (on DGX):
    CUDA_VISIBLE_DEVICES=5 python calculate_tumor_size_fig8.py
"""

import os, sys, glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import nibabel as nib
import scipy.ndimage

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from evaluate_trained_models import SE2_CNNET, load_se2_weights


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def get_pixel_spacing():
    raw_dir = os.path.expanduser("~/Clara/new_drive/CT Brain Data/Original Data")
    if os.path.exists(raw_dir):
        for f in os.listdir(raw_dir):
            if "CTC" in f.upper() and ("_2" in f or "_002" in f or " 2" in f):
                patient_dir = os.path.join(raw_dir, f)
                nii_files = [x for x in os.listdir(patient_dir)
                             if x.endswith('.nii.gz') and '.seg.' not in x]
                if nii_files:
                    try:
                        hdr = nib.load(os.path.join(patient_dir, nii_files[0])).header
                        sx, sy = float(hdr['pixdim'][1]), float(hdr['pixdim'][2])
                        if 0.1 < sx < 2.0 and 0.1 < sy < 2.0:
                            print(f"📏 Pixel spacing: {sx:.3f} x {sy:.3f} mm")
                            return sx, sy
                    except Exception as e:
                        print(f"⚠️ NIfTI header error: {e}")
    print("⚠️ Using fallback: 0.45 x 0.45 mm")
    return 0.45, 0.45


def get_blobs(mask, pixel_area_mm2, min_px=5):
    """Return list of dicts with blob info sorted by area descending."""
    labeled, n = scipy.ndimage.label(mask)
    blobs = []
    for idx in range(1, n + 1):
        blob = (labeled == idx)
        px = int(np.sum(blob))
        if px < min_px:
            continue
        ys, xs = np.where(blob)
        blobs.append(dict(
            pixels=px,
            area_mm2=px * pixel_area_mm2,
            mask=blob,
            cy=int(np.mean(ys)), cx=int(np.mean(xs)),
            y0=int(ys.min()), y1=int(ys.max()),
            x0=int(xs.min()), x1=int(xs.max()),
        ))
    blobs.sort(key=lambda b: b['area_mm2'], reverse=True)
    return blobs


def crop_blob(img_gray, blob, pad=20, target_size=None):
    """Crop a square patch around a blob with padding. Returns (img_crop, mask_crop)."""
    H, W = img_gray.shape
    y0 = max(0, blob['y0'] - pad)
    y1 = min(H, blob['y1'] + pad)
    x0 = max(0, blob['x0'] - pad)
    x1 = min(W, blob['x1'] + pad)
    # Make square
    h, w = y1 - y0, x1 - x0
    side = max(h, w)
    cy = (y0 + y1) // 2; cx = (x0 + x1) // 2
    y0 = max(0, cy - side // 2); y1 = min(H, y0 + side)
    x0 = max(0, cx - side // 2); x1 = min(W, x0 + side)
    img_crop  = img_gray[y0:y1, x0:x1]
    mask_crop = blob['mask'][y0:y1, x0:x1]
    return img_crop, mask_crop


# ─────────────────────────────────────────────────────────────────
# DRAW THUMBNAIL
# ─────────────────────────────────────────────────────────────────
def draw_thumbnail(ax, img_crop, mask_crop, color, label_text, fontcolor):
    """Show cropped lesion with colored contour outline only (no fill)."""
    ax.imshow(img_crop, cmap='gray', vmin=0, vmax=1)
    ax.contour(mask_crop, levels=[0.5], colors=[color], linewidths=2.0)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(label_text, color=fontcolor, fontsize=9, labelpad=4)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    DATA_DIR    = os.path.expanduser("~/Clara/local_ct_workspace_full")
    WEIGHT_PATH = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D/se2_unet_ctc_best.pth")
    SAVE_PATH   = os.path.expanduser("~/Clara/brain-ctc-seg/training/Journal_Figures/Fig8_Lesion_Area.png")
    TARGET_SLICE = 65          # best multifocal slice
    CROP_MARGIN  = 40
    ROTATE_K     = 3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Fig8 Generator (Device: {device})")

    sp_x, sp_y = get_pixel_spacing()
    pixel_area_mm2 = sp_x * sp_y

    # ── Locate patient folder ──
    all_folders = os.listdir(DATA_DIR)
    patient_folder = next(
        (f for f in all_folders if "CTC" in f.upper() and
         ("_2" in f or "_002" in f or " 2" in f)), None)
    if not patient_folder:
        print("❌ Folder pasien CTC 2 tidak ditemukan!"); return

    patient_path = os.path.join(DATA_DIR, patient_folder)
    z = TARGET_SLICE
    z_str = f"z{z:03d}"

    img_files = glob.glob(os.path.join(patient_path, f"*{z_str}_img.npy"))
    if not img_files:
        print(f"❌ Slice {z} tidak ditemukan!"); return

    img_path  = img_files[0]
    mask_path = img_path.replace('_img.npy', '_mask.npy')
    prev_path = img_path.replace(z_str, f"z{z-1:03d}")
    next_path = img_path.replace(z_str, f"z{z+1:03d}")

    i0 = np.load(prev_path).astype(np.float32) if os.path.exists(prev_path) else np.load(img_path).astype(np.float32)
    i1 = np.load(img_path).astype(np.float32)
    i2 = np.load(next_path).astype(np.float32) if os.path.exists(next_path) else np.load(img_path).astype(np.float32)
    img_25d = np.stack([i0, i1, i2], axis=-1)
    gt_mask_raw = np.load(mask_path).astype(np.uint8)

    # ── Inference ──
    if not os.path.exists(WEIGHT_PATH):
        print(f"❌ Weights not found: {WEIGHT_PATH}"); return
    model = load_se2_weights(SE2_CNNET, WEIGHT_PATH, device)
    model.eval()

    if img_25d.max() > img_25d.min():
        img_25d_norm = (img_25d - img_25d.min()) / (img_25d.max() - img_25d.min())
    else:
        img_25d_norm = img_25d

    tensor = torch.from_numpy(img_25d_norm).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            logits = model(tensor)
        pred_mask_raw = torch.argmax(F.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # ── Crop & Rotate ──
    mid_img   = i1.copy()
    if mid_img.max() > mid_img.min():
        mid_img = (mid_img - mid_img.min()) / (mid_img.max() - mid_img.min())
    mid_img   = np.rot90(mid_img[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)
    gt_mask   = np.rot90(gt_mask_raw[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)
    pred_mask = np.rot90(pred_mask_raw[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)

    # ── Blobs ──
    gt_blobs   = get_blobs(gt_mask,   pixel_area_mm2)
    pred_blobs = get_blobs(pred_mask, pixel_area_mm2)

    if len(gt_blobs) < 2 or len(pred_blobs) < 2:
        print("⚠️ Kurang dari 2 blob ditemukan. Coba slice lain."); return

    gt_largest   = gt_blobs[0];   gt_smallest   = gt_blobs[-1]
    pred_largest = pred_blobs[0]; pred_smallest = pred_blobs[-1]

    print(f"GT   — Largest: {gt_largest['area_mm2']:.1f} mm²  | Smallest: {gt_smallest['area_mm2']:.1f} mm²")
    print(f"Pred — Largest: {pred_largest['area_mm2']:.1f} mm² | Smallest: {pred_smallest['area_mm2']:.1f} mm²")

    # ── Build Figure ──
    # Layout: [thumb_col] [GT_panel] [Pred_panel] [thumb_col]
    fig = plt.figure(figsize=(14, 7), facecolor='white')
    gs  = gridspec.GridSpec(
        2, 4,
        width_ratios=[1, 3, 3, 1],
        height_ratios=[1, 1],
        wspace=0.05, hspace=0.35,
        left=0.04, right=0.96, top=0.88, bottom=0.06
    )

    # ── GT main panel (spans 2 rows) ──
    ax_gt = fig.add_subplot(gs[:, 1])
    ax_gt.imshow(mid_img, cmap='gray')
    for blob in gt_blobs:
        ax_gt.contour(blob['mask'], levels=[0.5], colors=['red'], linewidths=2.0)
    ax_gt.set_title("Ground truth", fontsize=13, fontweight='bold', pad=6)
    ax_gt.set_xticks([]); ax_gt.set_yticks([])
    for spine in ax_gt.spines.values():
        spine.set_edgecolor('red'); spine.set_linewidth(3)

    # ── Pred main panel (spans 2 rows) ──
    ax_pred = fig.add_subplot(gs[:, 2])
    ax_pred.imshow(mid_img, cmap='gray')
    for blob in pred_blobs:
        ax_pred.contour(blob['mask'], levels=[0.5], colors=['limegreen'], linewidths=2.0)
    ax_pred.set_title("CT-SE(2)", fontsize=13, fontweight='bold', pad=6)
    ax_pred.set_xticks([]); ax_pred.set_yticks([])
    for spine in ax_pred.spines.values():
        spine.set_edgecolor('limegreen'); spine.set_linewidth(3)

    # ── GT thumbnails (left col) ──
    ax_gt_large  = fig.add_subplot(gs[0, 0])
    ax_gt_small  = fig.add_subplot(gs[1, 0])

    img_gl, msk_gl = crop_blob(mid_img, gt_largest)
    img_gs, msk_gs = crop_blob(mid_img, gt_smallest)

    draw_thumbnail(ax_gt_large, img_gl, msk_gl, 'red', f"Largest  •  {gt_largest['area_mm2']:.1f} mm²", 'red')
    draw_thumbnail(ax_gt_small, img_gs, msk_gs, 'red', f"Smallest  •  {gt_smallest['area_mm2']:.1f} mm²", 'red')

    # ── Pred thumbnails (right col) ──
    ax_pred_large = fig.add_subplot(gs[0, 3])
    ax_pred_small = fig.add_subplot(gs[1, 3])

    img_pl, msk_pl = crop_blob(mid_img, pred_largest)
    img_ps, msk_ps = crop_blob(mid_img, pred_smallest)

    draw_thumbnail(ax_pred_large, img_pl, msk_pl, 'limegreen', f"Largest  •  {pred_largest['area_mm2']:.1f} mm²", 'green')
    draw_thumbnail(ax_pred_small, img_ps, msk_ps, 'limegreen', f"Smallest  •  {pred_smallest['area_mm2']:.1f} mm²", 'green')

    # ── Save ──
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Fig. 8 tersimpan di:\n{SAVE_PATH}")


if __name__ == "__main__":
    main()
