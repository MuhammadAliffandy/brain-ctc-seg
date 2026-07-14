"""
inference_figure.py
====================
Generate journal-quality inference figures dari model_epoch_100.pth
(atau model .pth manapun yang ada di folder training/).

Figures yang dihasilkan:
  Fig1 — Heatmap probability overlay (3 patients)
  Fig2 — Segmentation grid: Input / GT / Overlay / Prediction (2x2)
  Fig3 — ROC Curve (Mod-Seg-SE(2))
  Fig4 — Multi-CT robustness panel (4 patients, 2 rows)

Usage:
    # Otomatis pakai model_epoch_100.pth dari folder training/
    python inference_figure.py

    # Custom model dan data path
    python inference_figure.py --model model_epoch_100.pth --data ~/Clara/local_ct_workspace
    python inference_figure.py --model model_epoch_100.pth --data ~/Clara/local_ct_workspace_full
    python inference_figure.py --model model_epoch_100.pth --data ~/Clara/new_drive/CT\ Brain\ Data/MyDrive/Dataset_CT_Preprocessed_NPY

    # Ganti threshold (default 0.80 sesuai paper)
    python inference_figure.py --threshold 0.5

    # N=8 rotations (default), base_channels=24 (default)
    python inference_figure.py --N 8 --base_channels 24

Output:
    ~/Clara/brain-ctc-seg/training/Inference_Figures/
        Fig1_Heatmap.png
        Fig2_Segmentation_Grid.png
        Fig3_ROC.png
        Fig4_MultiCT.png
"""

import os
import re
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve, auc

# E2CNN
from escnn import gspaces
import escnn.nn as enn


# ================================================================
# 1. MODEL ARCHITECTURE (SE2 — must match checkpoint)
# ================================================================
class DoubleEquivariantConv(nn.Module):
    def __init__(self, in_type, out_type, mid_type=None):
        super().__init__()
        if not mid_type:
            mid_type = out_type
        self.double_conv = enn.SequentialModule(
            enn.R2Conv(in_type, mid_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(mid_type), enn.ReLU(mid_type, inplace=True),
            enn.R2Conv(mid_type, out_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type), enn.ReLU(out_type, inplace=True)
        )
    def forward(self, x): return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_type, out_type):
        super().__init__()
        self.pool = enn.PointwiseMaxPool(in_type, kernel_size=2)
        self.conv = DoubleEquivariantConv(in_type, out_type)
    def forward(self, x): return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_type, out_type):
        super().__init__()
        self.up   = enn.R2Upsampling(in_type, scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleEquivariantConv(in_type + out_type, out_type)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x  = enn.tensor_directsum([x2, x1])
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_type, n_classes):
        super().__init__()
        gspace   = in_type.gspace
        out_type = enn.FieldType(gspace, n_classes * [gspace.trivial_repr])
        self.conv = enn.R2Conv(in_type, out_type, kernel_size=1)
    def forward(self, x): return self.conv(x)


class SE2_CNNET(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, N=8, base_channels=24):
        super().__init__()
        self.r2_act    = gspaces.rot2dOnR2(N=N)
        c              = base_channels
        trivial        = self.r2_act.trivial_repr
        regular        = self.r2_act.regular_repr

        def ft(n): return enn.FieldType(self.r2_act, n * [regular])

        self.feat_type_in = enn.FieldType(self.r2_act, n_channels * [trivial])
        self.ft1 = ft(c);    self.ft2 = ft(c*2)
        self.ft3 = ft(c*4);  self.ft4 = ft(c*8); self.ft5 = ft(c*16)

        self.inc   = DoubleEquivariantConv(self.feat_type_in, self.ft1)
        self.down1 = Down(self.ft1, self.ft2)
        self.down2 = Down(self.ft2, self.ft3)
        self.down3 = Down(self.ft3, self.ft4)
        self.down4 = Down(self.ft4, self.ft5)
        self.up1   = Up(self.ft5, self.ft4)
        self.up2   = Up(self.ft4, self.ft3)
        self.up3   = Up(self.ft3, self.ft2)
        self.up4   = Up(self.ft2, self.ft1)
        self.outc  = OutConv(self.ft1, n_classes)

    def forward(self, x):
        x_g = enn.GeometricTensor(x, self.feat_type_in)
        x1  = self.inc(x_g)
        x2  = self.down1(x1); x3 = self.down2(x2)
        x4  = self.down3(x3); x5 = self.down4(x4)
        x   = self.up1(x5, x4); x = self.up2(x, x3)
        x   = self.up3(x, x2);  x = self.up4(x, x1)
        return self.outc(x).tensor


# ================================================================
# 2. DATA HELPERS
# ================================================================
def find_data_root():
    """Cari dataset CT secara otomatis di lokasi umum."""
    candidates = [
        "~/Clara/local_ct_workspace_full",
        "~/Clara/local_ct_workspace",
        "~/Clara/new_drive/CT Brain Data/MyDrive/Dataset_CT_Preprocessed_NPY",
    ]
    for c in candidates:
        p = os.path.expanduser(c)
        if os.path.isdir(p):
            return p
    return None


def get_best_slices(dataset_path: str, num_patients: int = 4,
                    min_px: int = 300, max_px: int = 8000):
    """
    Cari slice terbaik (dengan lesion cukup besar tapi tidak terlalu besar)
    dari setiap pasien yang berbeda.
    """
    print(f"🔍 Scanning slices di: {dataset_path}")
    patient_best = {}

    for root, _, files in os.walk(dataset_path):
        img_files = sorted(
            [f for f in files if f.endswith('_img.npy')],
            key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0
        )
        if not img_files:
            continue

        for i, fname in enumerate(img_files):
            img_path  = os.path.join(root, fname)
            mask_path = img_path.replace('_img.npy', '_mask.npy')
            if not os.path.exists(mask_path):
                continue

            n_px    = int(np.sum(np.load(mask_path)))
            patient = os.path.basename(root)

            if min_px < n_px < max_px:
                if patient not in patient_best or n_px > patient_best[patient]['pixels']:
                    i_prev = max(0, i - 1)
                    i_next = min(len(img_files) - 1, i + 1)
                    patient_best[patient] = {
                        'pixels':  n_px,
                        'prev':    os.path.join(root, img_files[i_prev]),
                        'curr':    img_path,
                        'next':    os.path.join(root, img_files[i_next]),
                        'mask':    mask_path,
                        'patient': patient,
                    }

    result = sorted(patient_best.values(), key=lambda x: x['pixels'], reverse=True)
    print(f"   ✅ Ditemukan {len(result)} pasien dengan lesion visible")
    return result[:num_patients]


def run_inference(model, device, slice_info: dict, threshold: float = 0.80,
                  crop: int = 30):
    """Load slice, run model, return rendered arrays."""
    prev  = np.load(slice_info['prev']).astype(np.float32)
    curr  = np.load(slice_info['curr']).astype(np.float32)
    nxt   = np.load(slice_info['next']).astype(np.float32)
    mask  = np.load(slice_info['mask']).astype(np.uint8)

    img_25d    = np.stack([prev, curr, nxt], axis=-1)          # [H, W, 3]
    img_tensor = torch.from_numpy(img_25d).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 3, H, W]

    with torch.no_grad():
        logits = model(img_tensor)
        probs  = F.softmax(logits, dim=1)
        prob_map = probs[0, 1].cpu().numpy()                   # [H, W]

    def crop_rot(arr):
        if crop > 0:
            arr = arr[crop:-crop, crop:-crop]
        return np.rot90(arr, k=1)

    return {
        'img':     crop_rot(curr),
        'gt':      crop_rot(mask),
        'prob':    crop_rot(prob_map),
        'pred':    (crop_rot(prob_map) >= threshold).astype(np.uint8),
        'patient': slice_info['patient'],
    }


# ================================================================
# 3. FIGURE GENERATORS
# ================================================================
SOLID_RED   = ListedColormap(['red'])
SOLID_WHITE = ListedColormap(['white'])
SOLID_CYAN  = ListedColormap(['cyan'])

BG_DARK  = '#0a0a14'
BG_PANEL = '#111122'


def fig1_heatmap(data_list: list, out_path: str):
    """Fig 1: Probability heatmap overlay untuk 3 pasien."""
    n = min(3, len(data_list))
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), facecolor=BG_DARK)
    if n == 1:
        axes = [axes]

    last_im = None
    for i in range(n):
        d = data_list[i]
        ax = axes[i]
        ax.set_facecolor(BG_PANEL)
        ax.imshow(d['img'], cmap='gray')
        masked = np.ma.masked_where(d['prob'] < 0.1, d['prob'])
        last_im = ax.imshow(masked, cmap='jet', alpha=0.80, vmin=0.2, vmax=1.0)
        ax.set_title(f"Patient {i+1}  —  Mod-Seg-SE(2)",
                     color='white', fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')

    if last_im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.70])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.ax.tick_params(colors='white', labelsize=11)
        cbar.outline.set_edgecolor('white')

    fig.suptitle("Probability Heatmap — Lesion Detection",
                 color='white', fontsize=16, fontweight='bold', y=1.01)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=BG_DARK)
    plt.close(fig)
    print(f"  📸 Fig1 saved → {out_path}")


def fig2_seg_grid(data: dict, out_path: str):
    """Fig 2: 2×2 segmentation grid (Input / GT / Overlay / Prediction)."""
    img  = data['img']
    gt   = data['gt']
    prob = data['prob']
    pred = data['pred']

    fig, axes = plt.subplots(2, 2, figsize=(11, 11), facecolor=BG_DARK)

    def base(ax):
        ax.set_facecolor(BG_PANEL)
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    labels = ['(a)', '(b)', '(c)', '(d)']
    titles = ['Input CT', 'Ground Truth', 'Overlay (GT vs Pred)', 'Mod-Seg-SE(2) Output']

    # (a) Input
    ax = axes[0, 0]
    base(ax)

    # (b) Ground Truth — white overlay
    ax = axes[0, 1]
    base(ax)
    ax.imshow(np.ma.masked_where(gt == 0, gt), cmap=SOLID_WHITE, alpha=0.95)

    # (c) Overlay — GT yellow dashed, Pred cyan dotted
    ax = axes[1, 0]
    base(ax)
    ax.contour(gt,   levels=[0.5], colors='yellow', linestyles='dashed', linewidths=3.0)
    ax.contour(pred, levels=[0.5], colors='cyan',   linestyles='dotted', linewidths=3.0)
    # legend patch
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color='yellow', lw=2.5, linestyle='dashed', label='Ground Truth'),
        Line2D([0], [0], color='cyan',   lw=2.5, linestyle='dotted', label='Prediction'),
    ]
    ax.legend(handles=legend_els, loc='lower right',
              facecolor='#111122', labelcolor='white', fontsize=11, framealpha=0.85)

    # (d) Prediction — solid red
    ax = axes[1, 1]
    base(ax)
    ax.imshow(np.ma.masked_where(pred == 0, pred), cmap=SOLID_RED, alpha=0.92)

    # Titles & labels
    flat_axes = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]
    for ax, title, lbl in zip(flat_axes, titles, labels):
        ax.set_title(title, color='white', fontsize=16, fontweight='bold', pad=8)
        ax.text(0.5, -0.04, lbl, transform=ax.transAxes,
                color='#aaaacc', fontsize=15, ha='center')

    plt.subplots_adjust(wspace=0.06, hspace=0.18)
    fig.suptitle(f"Segmentation Result — {data['patient']}",
                 color='white', fontsize=17, fontweight='bold', y=1.01)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=BG_DARK)
    plt.close(fig)
    print(f"  📸 Fig2 saved → {out_path}")


def fig3_roc(data_list: list, out_path: str):
    """Fig 3: Aggregated ROC curve dari semua slice."""
    y_true   = np.concatenate([d['gt'].flatten()   for d in data_list])
    y_scores = np.concatenate([d['prob'].flatten()  for d in data_list])

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 7), facecolor=BG_DARK)
    ax.set_facecolor(BG_PANEL)

    ax.plot(fpr, tpr, color='#ff6b6b', lw=3.0,
            label=f'Mod-Seg-SE(2)  (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#aaaaaa', lw=1.8, linestyle='--', label='Random Guess')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)',
                  color='white', fontsize=13, labelpad=8)
    ax.set_ylabel('True Positive Rate (Sensitivity)',
                  color='white', fontsize=13, labelpad=8)
    ax.set_title('ROC Curve — Mod-Seg-SE(2)', color='white', fontsize=15, fontweight='bold', pad=12)
    ax.tick_params(colors='white', labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    ax.grid(True, color='#222244', linewidth=0.7, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', facecolor='#111122', labelcolor='white',
              fontsize=12, framealpha=0.85)

    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=BG_DARK)
    plt.close(fig)
    print(f"  📸 Fig3 saved → {out_path}")


def fig4_multi_ct(data_list: list, out_path: str):
    """Fig 4: Multi-CT robustness panel — 2 rows (Input / Prediction) × N patients."""
    n   = len(data_list)
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 11), facecolor=BG_DARK)

    for i, d in enumerate(data_list):
        # Row 0: Input
        ax = axes[0, i]
        ax.set_facecolor(BG_PANEL)
        ax.imshow(d['img'], cmap='gray')
        ax.set_title(f"Patient {i+1}\nInput CT",
                     color='white', fontsize=14, fontweight='bold', pad=8)
        ax.axis('off')

        # Row 1: Prediction (red)
        ax = axes[1, i]
        ax.set_facecolor(BG_PANEL)
        ax.imshow(d['img'], cmap='gray')
        ax.imshow(np.ma.masked_where(d['pred'] == 0, d['pred']),
                  cmap=SOLID_RED, alpha=0.92)
        ax.set_title(f"Mod-Seg-SE(2)\nPrediction",
                     color='white', fontsize=14, fontweight='bold', pad=8)
        ax.axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.12)
    fig.suptitle("Robustness Check — Multiple Patients",
                 color='white', fontsize=17, fontweight='bold', y=1.01)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=BG_DARK)
    plt.close(fig)
    print(f"  📸 Fig4 saved → {out_path}")


# ================================================================
# 4. MAIN
# ================================================================
def main():
    # ── Default paths ──
    SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_MDL = os.path.join(SCRIPT_DIR, "model_epoch_100.pth")
    DEFAULT_OUT = os.path.expanduser("~/Clara/brain-ctc-seg/training/Inference_Figures")

    parser = argparse.ArgumentParser(
        description="Generate inference figures dari model .pth"
    )
    parser.add_argument('--model',       type=str, default=DEFAULT_MDL,
                        help=f"Path ke .pth checkpoint (default: {DEFAULT_MDL})")
    parser.add_argument('--data',        type=str, default=None,
                        help="Path ke dataset CT (NPY). Auto-detect jika tidak diisi.")
    parser.add_argument('--out',         type=str, default=DEFAULT_OUT,
                        help=f"Output folder untuk figures (default: {DEFAULT_OUT})")
    parser.add_argument('--threshold',   type=float, default=0.80,
                        help="Threshold binarisasi prediksi (default: 0.80)")
    parser.add_argument('--N',           type=int, default=8,
                        help="N rotations SE2 model (default: 8)")
    parser.add_argument('--base_channels', type=int, default=24,
                        help="base_channels SE2 model (default: 24)")
    parser.add_argument('--n_patients',  type=int, default=4,
                        help="Jumlah pasien untuk figures (default: 4)")
    parser.add_argument('--crop',        type=int, default=30,
                        help="Crop margin (px) dari tepi image (default: 30)")
    args = parser.parse_args()

    # ── Model path ──
    model_path = os.path.expanduser(args.model)
    if not os.path.exists(model_path):
        # Fallback: cari model_epoch_*.pth terbaru di SCRIPT_DIR
        candidates = sorted(glob.glob(os.path.join(SCRIPT_DIR, "model_epoch_*.pth")))
        if candidates:
            model_path = candidates[-1]
            print(f"⚠️  model_epoch_100.pth tidak ditemukan, pakai: {os.path.basename(model_path)}")
        else:
            print(f"❌ Model tidak ditemukan: {model_path}")
            print("   Tentukan --model /path/to/model.pth")
            return

    # ── Data path ──
    data_path = os.path.expanduser(args.data) if args.data else find_data_root()
    if data_path is None or not os.path.isdir(data_path):
        print("❌ Dataset tidak ditemukan. Tentukan --data /path/ke/dataset")
        print("   Contoh lokasi: ~/Clara/local_ct_workspace_full")
        return

    # ── Output dir ──
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  🧠 Inference Figure Generator")
    print(f"{'='*65}")
    print(f"  Model   : {model_path}")
    print(f"  Data    : {data_path}")
    print(f"  Output  : {out_dir}")
    print(f"  Threshold: {args.threshold}")
    print(f"  N={args.N}, base_channels={args.base_channels}")
    print(f"{'='*65}\n")

    # ── Load model ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  🖥️  Device: {device}")

    model = SE2_CNNET(n_channels=3, n_classes=2,
                      N=args.N, base_channels=args.base_channels).to(device)
    ckpt  = torch.load(model_path, map_location=device, weights_only=True)
    result = model.load_state_dict(ckpt, strict=False)
    print(f"  ✅ Model loaded: {os.path.basename(model_path)}")
    if result.missing_keys:
        print(f"     Missing keys  : {len(result.missing_keys)}")
    if result.unexpected_keys:
        print(f"     Unexpected    : {len(result.unexpected_keys)}")
    model.eval()
    print()

    # ── Get best slices ──
    slices = get_best_slices(data_path, num_patients=args.n_patients)
    if len(slices) < 2:
        print("❌ Tidak cukup slice dengan lesion visible (minimal 2 pasien).")
        print("   Coba turunkan --n_patients atau cek dataset path.")
        return
    print()

    # ── Run inference ──
    print("  🔄 Running inference...")
    data_list = []
    for si in slices:
        d = run_inference(model, device, si,
                          threshold=args.threshold, crop=args.crop)
        data_list.append(d)
        print(f"     Patient: {d['patient'][:30]:30s} | "
              f"lesion px={si['pixels']} | "
              f"pred px={(d['pred']==1).sum()}")
    print()

    # ── Generate figures ──
    print("  🎨 Generating figures...\n")
    fig1_heatmap(data_list[:3], os.path.join(out_dir, "Fig1_Heatmap.png"))
    fig2_seg_grid(data_list[0], os.path.join(out_dir, "Fig2_Segmentation_Grid.png"))
    fig3_roc(data_list,         os.path.join(out_dir, "Fig3_ROC.png"))
    fig4_multi_ct(data_list,    os.path.join(out_dir, "Fig4_MultiCT.png"))

    print(f"\n{'='*65}")
    print(f"  🌟 Semua figures tersimpan di: {out_dir}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
