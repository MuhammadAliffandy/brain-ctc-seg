"""
comparative_inference_all_datasets.py
======================================
Fig. 5 style figure — 1 representative result from EACH of the 4 datasets.

Layout (5 rows × 4 columns):
  Col:   CT          CTC         Stroke      Hemorrhage
  R1:    Input       Input       Input       Input
  R2:    Ground Truth  GT        GT          GT
  R3:    Overlay     Overlay     Overlay     Overlay
  R4:    CT-SE(2)    CT-SE(2)    CT-SE(2)    CT-SE(2)
  R5:    Standard    Standard    Standard    Standard
         U-Net       U-Net       U-Net       U-Net

Each panel has a green inset zoom box highlighting the lesion ROI.

Usage (on DGX):
    CUDA_VISIBLE_DEVICES=5 python comparative_inference_all_datasets.py
"""

import os, sys, glob, re, random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import kagglehub

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from evaluate_trained_models import SE2_CNNET, load_se2_weights

# ─────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURES (Standard UNet & nnUNet)
# ─────────────────────────────────────────────────────────────────
import torch.nn as nn

class _DC(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
        )
    def forward(self, x): return self.seq(x)

class StandardUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        self.inc = _DC(n_channels, 64)
        self.d1 = nn.Sequential(nn.MaxPool2d(2), _DC(64, 128))
        self.d2 = nn.Sequential(nn.MaxPool2d(2), _DC(128, 256))
        self.d3 = nn.Sequential(nn.MaxPool2d(2), _DC(256, 512))
        self.u1 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.c1 = _DC(512, 256)
        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.c2 = _DC(256, 128)
        self.u3 = nn.ConvTranspose2d(128, 64,  2, stride=2); self.c3 = _DC(128, 64)
        self.out = nn.Conv2d(64, n_classes, 1)
    def _pc(self, x, s):
        dy = s.size(2)-x.size(2); dx = s.size(3)-x.size(3)
        return torch.cat([s, F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])], 1)
    def forward(self, x):
        x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3)
        x=self.c1(self._pc(self.u1(x4),x3))
        x=self.c2(self._pc(self.u2(x),x2))
        x=self.c3(self._pc(self.u3(x),x1))
        return self.out(x)

class _NNC(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.InstanceNorm2d(o), nn.LeakyReLU(0.01, True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.InstanceNorm2d(o), nn.LeakyReLU(0.01, True),
        )
    def forward(self, x): return self.seq(x)

class nnUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        ch = [32, 64, 128, 256, 320]
        self.enc  = nn.ModuleList([_NNC(n_channels if i==0 else ch[i-1], ch[i]) for i in range(5)])
        self.pool = nn.MaxPool2d(2)
        self.ups  = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i-1], 2, stride=2) for i in range(4, 0, -1)])
        self.dec  = nn.ModuleList([_NNC(ch[i-1]*2, ch[i-1]) for i in range(4, 0, -1)])
        self.out  = nn.Conv2d(ch[0], n_classes, 1)
    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            if i < 4: skips.append(x); x = self.pool(x)
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips)):
            x = up(x)
            dy = skip.size(2)-x.size(2); dx = skip.size(3)-x.size(3)
            x = dec(torch.cat([skip, F.pad(x,[dx//2,dx-dx//2,dy//2,dy-dy//2])], 1))
        return self.out(x)


# ─────────────────────────────────────────────────────────────────
# DATA LOADING HELPERS
# ─────────────────────────────────────────────────────────────────
SAVE_DIR_NPY   = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D")
SAVE_DIR_INTRA = os.path.expanduser("~/Clara/brain-ctc-seg/public_dataset/saved_models")
CROP_MARGIN, ROTATE_K = 40, 3


def load_model(ModelClass, weight_path, device, is_se2=False):
    if is_se2:
        return load_se2_weights(ModelClass, weight_path, device)
    model = ModelClass(n_channels=3, n_classes=2).to(device)
    ckpt  = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def infer(model, tensor, device):
    model.eval()
    tensor = tensor.to(device)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            logits = model(tensor)
        pred = torch.argmax(F.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def find_best_npy_slice(data_dir, prefix, min_px=200, max_px=10000):
    """Find the NPY slice with the most representative lesion for a given dataset prefix."""
    best = None
    for folder in sorted(os.listdir(data_dir)):
        if not folder.upper().startswith(prefix.upper()):
            continue
        fpath = os.path.join(data_dir, folder)
        imgs = sorted([f for f in os.listdir(fpath) if f.endswith('_img.npy')],
                      key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0)
        for i, fname in enumerate(imgs):
            img_path  = os.path.join(fpath, fname)
            mask_path = img_path.replace('_img.npy', '_mask.npy')
            if not os.path.exists(mask_path): continue
            n_px = int(np.sum(np.load(mask_path)))
            if min_px < n_px < max_px:
                if best is None or n_px > best['px']:
                    i_prev = max(0, i-1); i_next = min(len(imgs)-1, i+1)
                    best = dict(
                        px=n_px, folder=folder,
                        prev=os.path.join(fpath, imgs[i_prev]),
                        curr=img_path,
                        next=os.path.join(fpath, imgs[i_next]),
                        mask=mask_path,
                    )
    return best


def load_npy_sample(sample_info):
    """Load a 2.5D NPY sample and return (img_gray, img_25d_norm, gt_mask)."""
    i0 = np.load(sample_info['prev']).astype(np.float32)
    i1 = np.load(sample_info['curr']).astype(np.float32)
    i2 = np.load(sample_info['next']).astype(np.float32)
    mask = np.load(sample_info['mask']).astype(np.uint8)

    img_25d = np.stack([i0, i1, i2], axis=-1)
    if img_25d.max() > img_25d.min():
        img_25d_norm = (img_25d - img_25d.min()) / (img_25d.max() - img_25d.min())
    else:
        img_25d_norm = img_25d

    mid = i1.copy()
    if mid.max() > mid.min(): mid = (mid - mid.min()) / (mid.max() - mid.min())

    # Crop + rotate to match training view
    mid  = np.rot90(mid[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN],  k=ROTATE_K)
    mask = np.rot90(mask[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)
    img_25d_norm_crop = np.rot90(img_25d_norm[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)

    tensor = torch.from_numpy(img_25d_norm_crop).permute(2, 0, 1).unsqueeze(0)
    return mid, tensor, mask


def find_best_kaggle_sample(download_path, mask_keywords=('mask', 'seg'), min_px=50):
    """Find a good sample from a Kaggle image/mask directory."""
    all_files = []
    for root, _, files in os.walk(download_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.bmp')):
                all_files.append(os.path.join(root, f))

    masks  = [f for f in all_files if any(k in f.lower() for k in mask_keywords)]
    random.seed(42); random.shuffle(masks)

    for mask_path in masks:
        m_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m_img is None: continue
        if np.sum(m_img > 127) < min_px: continue

        # Try to find matching image
        parent = os.path.dirname(mask_path)
        base = os.path.basename(mask_path)
        clean = base.lower().replace('_hge_seg','').replace('_seg','').replace('_mask','').replace('mask','').split('.')[0]
        for ext in ['.jpg', '.png', '.bmp']:
            candidate = os.path.join(parent, clean + ext)
            if os.path.exists(candidate):
                return candidate, mask_path
    return None, None


def load_kaggle_sample(img_path, mask_path):
    """Load a 2D kaggle image, return (img_gray, tensor_3c, gt_mask)."""
    img  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    img  = cv2.resize(img,  (256, 256)).astype(np.float32)
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8)

    if img.max() > img.min(): img = (img - img.min()) / (img.max() - img.min())
    img_3c = np.stack([img, img, img], axis=0)  # [3,H,W]
    tensor = torch.from_numpy(img_3c).unsqueeze(0)  # [1,3,H,W]
    return img, tensor, mask


# ─────────────────────────────────────────────────────────────────
# FIGURE DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    'se2':      'red',
    'standard': 'gold',
    'nn':       'dodgerblue',
}

ROW_LABELS = ['Input', 'Ground Truth', 'Overlay', 'CT-SE(2)', 'Standard U-Net']


def find_lesion_roi(mask, pad=15):
    """Return (y0,y1,x0,x1) bounding box around all positive pixels."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        H, W = mask.shape
        return 0, H, 0, W
    y0 = max(0, ys.min()-pad); y1 = min(mask.shape[0], ys.max()+pad)
    x0 = max(0, xs.min()-pad); x1 = min(mask.shape[1], xs.max()+pad)
    return y0, y1, x0, x1


def draw_panel(ax, img_gray, gt_mask=None, pred_mask=None,
               pred_color=None, mode='input', show_inset=True):
    """
    Draw one panel on ax.
    mode: 'input' | 'gt' | 'overlay' | 'pred'
    """
    ax.imshow(img_gray, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

    H, W = img_gray.shape

    if mode == 'gt' and gt_mask is not None:
        overlay = np.zeros((H, W, 4))
        overlay[gt_mask > 0] = [1, 1, 1, 0.85]
        ax.imshow(overlay)

    elif mode == 'overlay' and gt_mask is not None and pred_mask is not None:
        # GT = yellow dashed contour, Pred = cyan dotted contour
        ax.contour(gt_mask,   levels=[0.5], colors=['yellow'], linewidths=1.5, linestyles='dashed')
        ax.contour(pred_mask, levels=[0.5], colors=['cyan'],   linewidths=1.5, linestyles='dotted')

    elif mode == 'pred' and pred_mask is not None and pred_color is not None:
        overlay = np.zeros((H, W, 4))
        color_map = {'red':[1,0,0], 'gold':[1,0.85,0], 'dodgerblue':[0.12,0.56,1]}
        rgb = color_map.get(pred_color, [1,0,0])
        overlay[pred_mask > 0] = rgb + [0.75]
        ax.imshow(overlay)

    # ── Green inset zoom box ──
    if show_inset:
        ref_mask = gt_mask if gt_mask is not None else pred_mask
        if ref_mask is not None and ref_mask.sum() > 0:
            y0, y1, x0, x1 = find_lesion_roi(ref_mask, pad=12)
            bw = x1 - x0; bh = y1 - y0
            rect = patches.Rectangle((x0, y0), bw, bh,
                                      linewidth=1.5, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

            # Inset axes in top-right corner
            inset_size = 0.35
            axin = ax.inset_axes([1.0 - inset_size - 0.02, 1.0 - inset_size - 0.02,
                                   inset_size, inset_size])
            axin.imshow(img_gray[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=1)
            if mode == 'gt' and gt_mask is not None:
                sub_ov = np.zeros((y1-y0, x1-x0, 4))
                sub_ov[gt_mask[y0:y1, x0:x1] > 0] = [1,1,1,0.85]
                axin.imshow(sub_ov)
            elif mode == 'overlay':
                if gt_mask is not None:
                    axin.contour(gt_mask[y0:y1,x0:x1],   levels=[0.5], colors=['yellow'], linewidths=1.5, linestyles='dashed')
                if pred_mask is not None:
                    axin.contour(pred_mask[y0:y1,x0:x1], levels=[0.5], colors=['cyan'],   linewidths=1.5, linestyles='dotted')
            elif mode == 'pred' and pred_mask is not None and pred_color:
                color_map = {'red':[1,0,0], 'gold':[1,0.85,0], 'dodgerblue':[0.12,0.56,1]}
                rgb = color_map.get(pred_color, [1,0,0])
                sub_ov = np.zeros((y1-y0, x1-x0, 4))
                sub_ov[pred_mask[y0:y1,x0:x1] > 0] = rgb + [0.75]
                axin.imshow(sub_ov)

            axin.set_xticks([]); axin.set_yticks([])
            for sp in axin.spines.values():
                sp.set_edgecolor('lime'); sp.set_linewidth(1.5)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    DATA_DIR  = os.path.expanduser("~/Clara/local_ct_workspace_full")
    OUT_PATH  = os.path.expanduser("~/Clara/brain-ctc-seg/training/Journal_Figures/Fig5_Comparative_All_Datasets.png")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Comparative All Datasets Figure (Device: {device})")

    # ── Dataset configs ──
    datasets = [
        dict(name='CT',          prefix='CT_',  source='npy',
             se2_w=os.path.join(SAVE_DIR_NPY,   'se2_unet_ct_best.pth'),
             std_w=os.path.join(SAVE_DIR_NPY,   'standard_unet_ct_best.pth')),
        dict(name='CTC',         prefix='CTC_', source='npy',
             se2_w=os.path.join(SAVE_DIR_NPY,   'se2_unet_ctc_best.pth'),
             std_w=os.path.join(SAVE_DIR_NPY,   'standard_unet_ctc_best.pth')),
        dict(name='Stroke',      prefix=None,   source='kaggle_stroke',
             se2_w=os.path.join(SAVE_DIR_INTRA, 'Mod-Seg-SE2_kaggle_best.pth'),
             std_w=os.path.join(SAVE_DIR_INTRA, 'Standard_U-Net_kaggle_best.pth')),
        dict(name='Hemorrhage',  prefix=None,   source='kaggle_hemo',
             se2_w=os.path.join(SAVE_DIR_INTRA, 'Mod-Seg-SE2_kaggle_hemorrhage_best.pth'),
             std_w=os.path.join(SAVE_DIR_INTRA, 'Standard_U-Net_kaggle_hemorrhage_best.pth')),
    ]

    # ── Collect data for each dataset ──
    dataset_data = []
    for ds in datasets:
        print(f"\n📂 Loading {ds['name']}...")
        entry = dict(name=ds['name'])

        if ds['source'] == 'npy':
            sample = find_best_npy_slice(DATA_DIR, ds['prefix'])
            if sample is None:
                print(f"  ⚠️ No slice found for {ds['name']}, skipping"); continue
            img_gray, tensor, gt_mask = load_npy_sample(sample)

        elif ds['source'] == 'kaggle_stroke':
            dl_path = kagglehub.dataset_download("ozcangundes/brain-stroke-ct-dataset")
            img_path, mask_path = find_best_kaggle_sample(dl_path)
            if img_path is None:
                print(f"  ⚠️ No stroke sample found, skipping"); continue
            img_gray, tensor, gt_mask = load_kaggle_sample(img_path, mask_path)

        elif ds['source'] == 'kaggle_hemo':
            dl_path = kagglehub.dataset_download("vbookshelf/computed-tomography-ct-images")
            img_path, mask_path = find_best_kaggle_sample(dl_path, mask_keywords=('mask','hge_seg','seg'))
            if img_path is None:
                print(f"  ⚠️ No hemorrhage sample found, skipping"); continue
            img_gray, tensor, gt_mask = load_kaggle_sample(img_path, mask_path)

        # ── Load models & infer ──
        pred_se2 = pred_std = None

        if os.path.exists(ds['se2_w']):
            try:
                m = load_model(SE2_CNNET, ds['se2_w'], device, is_se2=True)
                pred_se2 = infer(m, tensor, device)
                del m; torch.cuda.empty_cache()
                print(f"  ✅ SE2 inference done")
            except Exception as e:
                print(f"  ⚠️ SE2 error: {e}")
        else:
            print(f"  ⚠️ SE2 weights not found: {ds['se2_w']}")

        if os.path.exists(ds['std_w']):
            try:
                m = load_model(StandardUNet, ds['std_w'], device)
                pred_std = infer(m, tensor, device)
                del m; torch.cuda.empty_cache()
                print(f"  ✅ Standard UNet inference done")
            except Exception as e:
                print(f"  ⚠️ Standard UNet error: {e}")
        else:
            print(f"  ⚠️ Standard UNet weights not found: {ds['std_w']}")

        entry.update(dict(img=img_gray, gt=gt_mask, se2=pred_se2, std=pred_std))
        dataset_data.append(entry)

    if not dataset_data:
        print("❌ No dataset loaded successfully."); return

    n_cols = len(dataset_data)
    n_rows = 5  # Input, GT, Overlay, SE2, Standard

    # ── Build Figure ──
    fig = plt.figure(figsize=(4.5 * n_cols, 4.5 * n_rows), facecolor='white')
    gs  = gridspec.GridSpec(n_rows, n_cols,
                             wspace=0.05, hspace=0.08,
                             left=0.06, right=0.99, top=0.96, bottom=0.02)

    for col_idx, entry in enumerate(dataset_data):
        img   = entry['img']
        gt    = entry['gt']
        se2   = entry.get('se2')
        std   = entry.get('std')

        panels = [
            dict(mode='input',   gt_mask=gt,  pred_mask=None, pred_color=None),
            dict(mode='gt',      gt_mask=gt,  pred_mask=None, pred_color=None),
            dict(mode='overlay', gt_mask=gt,  pred_mask=se2,  pred_color=None),
            dict(mode='pred',    gt_mask=gt,  pred_mask=se2,  pred_color='red'),
            dict(mode='pred',    gt_mask=gt,  pred_mask=std,  pred_color='gold'),
        ]

        for row_idx, panel in enumerate(panels):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            draw_panel(ax, img, **panel)

            # Column header (dataset name) on top row only
            if row_idx == 0:
                ax.set_title(entry['name'], fontsize=14, fontweight='bold', pad=5)

            # Row labels on leftmost column only
            if col_idx == 0:
                ax.set_ylabel(ROW_LABELS[row_idx], fontsize=11, rotation=90,
                              labelpad=5, va='center')

            # Panel letter label
            letter = chr(ord('a') + row_idx * n_cols + col_idx)
            ax.text(0.04, 0.04, f"({letter})", transform=ax.transAxes,
                    color='white', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.45, edgecolor='none', pad=1.5))

    plt.savefig(OUT_PATH, dpi=250, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Fig. 5 saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
