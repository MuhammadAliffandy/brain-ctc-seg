"""
comparative_inference_all_datasets.py
======================================
IEEE-style Fig. 5 — 4 datasets × 7 columns table:
  [Dataset] | [Input] | [Ground Truth] | [nnU-Net] | [Attn U-Net] | [TransUNet] | [Standard U-Net] | [Proposed CT-SE(2)]

- Ground Truth column: blue background tint
- Proposed column: green background tint

Usage (on DGX):
    CUDA_VISIBLE_DEVICES=5 python comparative_inference_all_datasets.py
"""

import os, sys, glob, re, random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import kagglehub

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from evaluate_trained_models import SE2_CNNET, load_se2_weights

# ─────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURES
# ─────────────────────────────────────────────────────────────────
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
        self.d1 = nn.Sequential(nn.MaxPool2d(2), _DC(64,  128))
        self.d2 = nn.Sequential(nn.MaxPool2d(2), _DC(128, 256))
        self.d3 = nn.Sequential(nn.MaxPool2d(2), _DC(256, 512))
        self.u1 = nn.ConvTranspose2d(512,256,2,stride=2); self.c1=_DC(512,256)
        self.u2 = nn.ConvTranspose2d(256,128,2,stride=2); self.c2=_DC(256,128)
        self.u3 = nn.ConvTranspose2d(128,64, 2,stride=2); self.c3=_DC(128,64)
        self.out = nn.Conv2d(64, n_classes, 1)
    def _pc(self, x, s):
        dy=s.size(2)-x.size(2); dx=s.size(3)-x.size(3)
        return torch.cat([s, F.pad(x,[dx//2,dx-dx//2,dy//2,dy-dy//2])],1)
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
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.InstanceNorm2d(o), nn.LeakyReLU(0.01,True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.InstanceNorm2d(o), nn.LeakyReLU(0.01,True),
        )
    def forward(self, x): return self.seq(x)

class nnUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        ch=[32,64,128,256,320]
        self.enc=nn.ModuleList([_NNC(n_channels if i==0 else ch[i-1],ch[i]) for i in range(5)])
        self.pool=nn.MaxPool2d(2)
        self.ups=nn.ModuleList([nn.ConvTranspose2d(ch[i],ch[i-1],2,stride=2) for i in range(4,0,-1)])
        self.dec=nn.ModuleList([_NNC(ch[i-1]*2,ch[i-1]) for i in range(4,0,-1)])
        self.out=nn.Conv2d(ch[0],n_classes,1)
    def forward(self, x):
        skips=[]
        for i,enc in enumerate(self.enc):
            x=enc(x)
            if i<4: skips.append(x); x=self.pool(x)
        for up,dec,skip in zip(self.ups,self.dec,reversed(skips)):
            x=up(x)
            dy=skip.size(2)-x.size(2); dx=skip.size(3)-x.size(3)
            x=dec(torch.cat([skip,F.pad(x,[dx//2,dx-dx//2,dy//2,dy-dy//2])],1))
        return self.out(x)

class _AttnGate(nn.Module):
    def __init__(self, g, x, mid):
        super().__init__()
        self.Wg=nn.Conv2d(g,mid,1); self.Wx=nn.Conv2d(x,mid,1)
        self.psi=nn.Sequential(nn.Conv2d(mid,1,1),nn.Sigmoid())
    def forward(self, g, x):
        a=self.psi(F.relu(self.Wg(g)+self.Wx(x),True))
        return x*F.interpolate(a,size=x.shape[2:],mode='nearest')

class AttentionUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        self.inc=_DC(n_channels,64)
        self.d1=nn.Sequential(nn.MaxPool2d(2),_DC(64,128))
        self.d2=nn.Sequential(nn.MaxPool2d(2),_DC(128,256))
        self.d3=nn.Sequential(nn.MaxPool2d(2),_DC(256,512))
        self.u1=nn.ConvTranspose2d(512,256,2,stride=2); self.a1=_AttnGate(256,256,128); self.c1=_DC(512,256)
        self.u2=nn.ConvTranspose2d(256,128,2,stride=2); self.a2=_AttnGate(128,128,64);  self.c2=_DC(256,128)
        self.u3=nn.ConvTranspose2d(128,64, 2,stride=2); self.a3=_AttnGate(64,64,32);    self.c3=_DC(128,64)
        self.out=nn.Conv2d(64,n_classes,1)
    def _pc(self, x, s):
        dy=s.size(2)-x.size(2); dx=s.size(3)-x.size(3)
        return torch.cat([s,F.pad(x,[dx//2,dx-dx//2,dy//2,dy-dy//2])],1)
    def forward(self, x):
        x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3)
        u=self.u1(x4); x=self.c1(self._pc(u,self.a1(u,x3)))
        u=self.u2(x);  x=self.c2(self._pc(u,self.a2(u,x2)))
        u=self.u3(x);  x=self.c3(self._pc(u,self.a3(u,x1)))
        return self.out(x)

class _TransBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.n1=nn.LayerNorm(dim); self.attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.n2=nn.LayerNorm(dim); self.mlp=nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim))
    def forward(self, x):
        B,C,H,W=x.shape; t=x.flatten(2).transpose(1,2)
        t=t+self.attn(self.n1(t),self.n1(t),self.n1(t))[0]
        t=t+self.mlp(self.n2(t))
        return t.transpose(1,2).reshape(B,C,H,W)

class TransUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        self.inc=_DC(n_channels,64)
        self.d1=nn.Sequential(nn.MaxPool2d(2),_DC(64,128))
        self.d2=nn.Sequential(nn.MaxPool2d(2),_DC(128,256))
        self.d3=nn.Sequential(nn.MaxPool2d(2),_DC(256,512))
        self.trans=nn.Sequential(_TransBlock(512),_TransBlock(512))
        self.u1=nn.ConvTranspose2d(512,256,2,stride=2); self.c1=_DC(512,256)
        self.u2=nn.ConvTranspose2d(256,128,2,stride=2); self.c2=_DC(256,128)
        self.u3=nn.ConvTranspose2d(128,64, 2,stride=2); self.c3=_DC(128,64)
        self.out=nn.Conv2d(64,n_classes,1)
    def _pc(self, x, s):
        dy=s.size(2)-x.size(2); dx=s.size(3)-x.size(3)
        return torch.cat([s,F.pad(x,[dx//2,dx-dx//2,dy//2,dy-dy//2])],1)
    def forward(self, x):
        x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3)
        x4=self.trans(x4)
        x=self.c1(self._pc(self.u1(x4),x3))
        x=self.c2(self._pc(self.u2(x),x2))
        x=self.c3(self._pc(self.u3(x),x1))
        return self.out(x)


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
SAVE_DIR_NPY   = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D")
SAVE_DIR_INTRA = os.path.expanduser("~/Clara/brain-ctc-seg/public_dataset/saved_models")
CROP_MARGIN, ROTATE_K = 40, 3

# Column definitions: (header_label, bg_color, text_color)
COLUMNS = [
    ("Input",             None,              'black'),
    ("Ground Truth",      '#dbeeff',         'black'),   # blue tint
    ("nnU-Net",           None,              'black'),
    ("Attention U-Net",   None,              'black'),
    ("TransUNet",         None,              'black'),
    ("Standard U-Net",    None,              'black'),
    ("Proposed\nCT-SE(2)",'#d4f5d4',         '#006600'), # green tint
]

DATASET_NAMES = ['CT', 'CTC', 'Stroke', 'Hemorrhage']


# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────
def load_model_safe(ModelClass, weight_path, device, is_se2=False):
    if not os.path.exists(weight_path):
        print(f"  ⚠️ Missing: {os.path.basename(weight_path)}")
        return None
    try:
        if is_se2:
            m = load_se2_weights(ModelClass, weight_path, device)
        else:
            m = ModelClass(n_channels=3, n_classes=2).to(device)
            m.load_state_dict(torch.load(weight_path, map_location=device, weights_only=False), strict=False)
        m.eval()
        print(f"  ✅ Loaded {os.path.basename(weight_path)}")
        return m
    except Exception as e:
        print(f"  ❌ Error loading {os.path.basename(weight_path)}: {e}")
        return None


def infer_safe(model, tensor, device):
    if model is None: return None
    try:
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                logits = model(tensor.to(device))
            return torch.argmax(F.softmax(logits,dim=1),dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    except Exception as e:
        print(f"  ❌ Inference error: {e}")
        return None


def find_best_npy_slice(data_dir, prefix, min_px=200, max_px=15000):
    best = None
    for folder in sorted(os.listdir(data_dir)):
        if not folder.upper().startswith(prefix.upper()): continue
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
                    i_prev = max(0,i-1); i_next = min(len(imgs)-1,i+1)
                    best = dict(px=n_px,
                                prev=os.path.join(fpath,imgs[i_prev]),
                                curr=img_path,
                                next=os.path.join(fpath,imgs[i_next]),
                                mask=mask_path)
    return best


def load_npy_sample(s):
    i0=np.load(s['prev']).astype(np.float32)
    i1=np.load(s['curr']).astype(np.float32)
    i2=np.load(s['next']).astype(np.float32)
    mask=np.load(s['mask']).astype(np.uint8)
    img_25d=np.stack([i0,i1,i2],axis=-1)
    if img_25d.max()>img_25d.min():
        img_25d_n=(img_25d-img_25d.min())/(img_25d.max()-img_25d.min())
    else: img_25d_n=img_25d
    mid=i1.copy()
    if mid.max()>mid.min(): mid=(mid-mid.min())/(mid.max()-mid.min())
    mid  = np.rot90(mid[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN],   k=ROTATE_K).copy()
    mask = np.rot90(mask[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN],  k=ROTATE_K).copy()
    inp_c= np.rot90(img_25d_n[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K).copy()
    tensor = torch.from_numpy(inp_c).permute(2, 0, 1).unsqueeze(0)
    return mid, tensor, mask


def find_best_kaggle_sample(root, mask_kw=('mask','seg','hge'), min_px=50):
    all_files=[]
    for r,_,files in os.walk(root):
        for f in files:
            if f.lower().endswith(('.jpg','.png','.bmp')):
                all_files.append(os.path.join(r,f))
    masks=[f for f in all_files if any(k in f.lower() for k in mask_kw)]
    random.seed(42); random.shuffle(masks)
    
    for mp in masks:
        m=cv2.imread(mp,cv2.IMREAD_GRAYSCALE)
        if m is None or np.sum(m>127)<min_px: continue
        
        parent = os.path.dirname(mp)
        grandparent = os.path.dirname(parent)
        base = os.path.basename(mp)
        clean = base.lower()
        for k in ['_hge_seg','_seg','_mask','mask','seg']:
            clean = clean.replace(k,'')
        clean = clean.split('.')[0]
        
        # Possible image paths:
        # 1. Same folder as mask (Hemorrhage)
        # 2. Sibling folder named "images" or "PNG" (Stroke)
        candidates = []
        for ext in ['.jpg','.png','.bmp']:
            candidates.append(os.path.join(parent, clean+ext))
            candidates.append(os.path.join(parent, base)) # Sometimes name is identical
            for sibling in ['PNG', 'images', 'Image', 'Images']:
                candidates.append(os.path.join(grandparent, sibling, clean+ext))
                candidates.append(os.path.join(grandparent, sibling, base))
        
        for cand in candidates:
            if cand != mp and os.path.exists(cand):
                return cand, mp
                
    return None, None


def load_kaggle_sample(ip,mp):
    img =cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
    mask=cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    img =cv2.resize(img, (256,256)).astype(np.float32)
    mask=cv2.resize(mask,(256,256),interpolation=cv2.INTER_NEAREST)
    mask=(mask>127).astype(np.uint8)
    if img.max()>img.min(): img=(img-img.min())/(img.max()-img.min())
    t=torch.from_numpy(np.stack([img,img,img],axis=0)).unsqueeze(0)
    return img,t,mask


# ─────────────────────────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────────────────────────
def draw_image_panel(ax, img_gray, pred_mask=None, is_gt=False, bg_color=None):
    """Render one panel: grayscale CT + optional segmentation overlay."""
    H, W = img_gray.shape

    if bg_color:
        ax.set_facecolor(bg_color)

    ax.imshow(img_gray, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#aaaaaa'); sp.set_linewidth(0.8)

    if is_gt and pred_mask is not None:
        # White filled overlay for GT
        ov = np.zeros((H, W, 4))
        ov[pred_mask > 0] = [1, 1, 1, 0.90]
        ax.imshow(ov)

    elif pred_mask is not None:
        # Colored filled overlay for predictions
        ov = np.zeros((H, W, 4))
        ov[pred_mask > 0] = [1, 0.2, 0.2, 0.75]   # red for all model predictions
        ax.imshow(ov)


def draw_empty_panel(ax, text, bg_color=None):
    ax.set_xticks([]); ax.set_yticks([])
    if bg_color: ax.set_facecolor(bg_color)
    for sp in ax.spines.values():
        sp.set_edgecolor('#aaaaaa'); sp.set_linewidth(0.8)
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha='center', va='center', fontsize=9, color='#888888', style='italic')


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    DATA_DIR  = os.path.expanduser("~/Clara/local_ct_workspace_full")
    OUT_PATH  = os.path.expanduser("~/Clara/brain-ctc-seg/training/Journal_Figures/Fig5_Comparative_All_Datasets.png")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 IEEE Fig. 5 Generator (Device: {device})")

    # ── Dataset configs ──
    datasets = [
        dict(name='CT',         source='npy',           prefix='CT_',
             weights=dict(
                 nn    =os.path.join(SAVE_DIR_NPY,   'nn_unet_ct_best.pth'),
                 attn  =os.path.join(SAVE_DIR_NPY,   'attention_unet_ct_best.pth'),
                 trans =os.path.join(SAVE_DIR_NPY,   'trans_unet_ct_best.pth'),
                 std   =os.path.join(SAVE_DIR_NPY,   'standard_unet_ct_best.pth'),
                 se2   =os.path.join(SAVE_DIR_NPY,   'se2_unet_ct_best.pth'),
             )),
        dict(name='CTC',        source='npy',           prefix='CTC_',
             weights=dict(
                 nn    =os.path.join(SAVE_DIR_NPY,   'nn_unet_ctc_best.pth'),
                 attn  =os.path.join(SAVE_DIR_NPY,   'attention_unet_ctc_best.pth'),
                 trans =os.path.join(SAVE_DIR_NPY,   'trans_unet_ctc_best.pth'),
                 std   =os.path.join(SAVE_DIR_NPY,   'standard_unet_ctc_best.pth'),
                 se2   =os.path.join(SAVE_DIR_NPY,   'se2_unet_ctc_best.pth'),
             )),
        dict(name='Stroke',     source='kaggle_stroke', prefix=None,
             weights=dict(
                 nn    =os.path.join(SAVE_DIR_INTRA, 'nnU-Net_kaggle_best.pth'),
                 attn  =os.path.join(SAVE_DIR_INTRA, 'Attention_U-Net_kaggle_best.pth'),
                 trans =os.path.join(SAVE_DIR_INTRA, 'TransUNet_kaggle_best.pth'),
                 std   =os.path.join(SAVE_DIR_INTRA, 'Standard_U-Net_kaggle_best.pth'),
                 se2   =os.path.join(SAVE_DIR_INTRA, 'Mod-Seg-SE2_kaggle_best.pth'),
             )),
        dict(name='Hemorrhage', source='kaggle_hemo',   prefix=None,
             weights=dict(
                 nn    =os.path.join(SAVE_DIR_INTRA, 'nnU-Net_kaggle_hemorrhage_best.pth'),
                 attn  =os.path.join(SAVE_DIR_INTRA, 'Attention_U-Net_kaggle_hemorrhage_best.pth'),
                 trans =os.path.join(SAVE_DIR_INTRA, 'TransUNet_kaggle_hemorrhage_best.pth'),
                 std   =os.path.join(SAVE_DIR_INTRA, 'Standard_U-Net_kaggle_hemorrhage_best.pth'),
                 se2   =os.path.join(SAVE_DIR_INTRA, 'Mod-Seg-SE2_kaggle_hemorrhage_best.pth'),
             )),
    ]

    # ── Figure Layout ──
    n_rows = len(datasets)
    # 8 axes cols: [dataset_label, input, gt, nn, attn, trans, std, se2]
    n_img_cols = 7
    fig_w = 2.6 * n_img_cols + 1.4  # +1.4 for label column
    fig_h = 3.0 * n_rows + 0.8      # +0.8 for header

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

    # Outer grid: header row + data rows
    outer_gs = gridspec.GridSpec(
        n_rows + 1, n_img_cols + 1,  # +1 row for header, +1 col for label
        figure=fig,
        width_ratios=[1.2] + [2.6]*n_img_cols,
        height_ratios=[0.5] + [3.0]*n_rows,
        hspace=0.04, wspace=0.03,
        left=0.04, right=0.99, top=0.98, bottom=0.02
    )

    # ── Column headers (row 0) ──
    ax_corner = fig.add_subplot(outer_gs[0, 0])
    ax_corner.axis('off')
    ax_corner.text(0.5, 0.5, 'Dataset', transform=ax_corner.transAxes,
                   ha='center', va='center', fontsize=11, fontweight='bold')

    col_keys = ['nn', 'attn', 'trans', 'std', 'se2']
    for ci, (col_label, bg, fg) in enumerate(COLUMNS):
        ax_hdr = fig.add_subplot(outer_gs[0, ci + 1])
        ax_hdr.axis('off')
        if bg:
            ax_hdr.set_facecolor(bg)
            ax_hdr.set_axis_on()
            ax_hdr.set_xticks([]); ax_hdr.set_yticks([])
            for sp in ax_hdr.spines.values():
                sp.set_visible(False)
        ax_hdr.text(0.5, 0.5, col_label, transform=ax_hdr.transAxes,
                    ha='center', va='center', fontsize=10, fontweight='bold', color=fg,
                    linespacing=1.3)

    # ── Process each dataset row ──
    for row_idx, ds in enumerate(datasets):
        print(f"\n{'='*55}")
        print(f"  📂 Dataset: {ds['name']}")
        print(f"{'='*55}")

        # ── Load sample ──
        if ds['source'] == 'npy':
            sample = find_best_npy_slice(DATA_DIR, ds['prefix'])
            if sample is None:
                print(f"  ⚠️ No valid slice found for {ds['name']}")
                img_gray = tensor = gt_mask = None
            else:
                img_gray, tensor, gt_mask = load_npy_sample(sample)

        elif ds['source'] == 'kaggle_stroke':
            # Bypass kagglehub download API to avoid 403 Forbidden
            dl = os.path.expanduser("~/.cache/kagglehub/datasets/ozcangundes/brain-stroke-ct-dataset")
            ip, mp = find_best_kaggle_sample(dl)
            if ip is None:
                print("  ⚠️ No stroke sample found in cache"); img_gray=tensor=gt_mask=None
            else:
                img_gray, tensor, gt_mask = load_kaggle_sample(ip, mp)

        elif ds['source'] == 'kaggle_hemo':
            # Bypass kagglehub download API to avoid 403 Forbidden
            dl = os.path.expanduser("~/.cache/kagglehub/datasets/vbookshelf/computed-tomography-ct-images")
            ip, mp = find_best_kaggle_sample(dl, mask_kw=('mask','hge_seg','seg'))
            if ip is None:
                print("  ⚠️ No hemorrhage sample found in cache"); img_gray=tensor=gt_mask=None
            else:
                img_gray, tensor, gt_mask = load_kaggle_sample(ip, mp)

        # ── Dataset label axis ──
        ax_lbl = fig.add_subplot(outer_gs[row_idx + 1, 0])
        ax_lbl.axis('off')
        ax_lbl.text(0.5, 0.5, ds['name'], transform=ax_lbl.transAxes,
                    ha='center', va='center', fontsize=12, fontweight='bold', rotation=0)

        # ── Infer all models ──
        preds = {}
        if img_gray is not None:
            model_map = [
                ('nn',   nnUNet,        False),
                ('attn', AttentionUNet, False),
                ('trans',TransUNet,     False),
                ('std',  StandardUNet,  False),
                ('se2',  SE2_CNNET,     True),
            ]
            for key, ModelClass, is_se2 in model_map:
                m = load_model_safe(ModelClass, ds['weights'][key], device, is_se2)
                preds[key] = infer_safe(m, tensor, device)
                if m is not None: del m; torch.cuda.empty_cache()

        # ── Draw panels ──
        panel_specs = [
            # (col_idx_in_figure, data_key_or_special, is_gt, bg_color)
            (1, 'input',  False, None),
            (2, 'gt',     True,  '#dbeeff'),
            (3, 'nn',     False, None),
            (4, 'attn',   False, None),
            (5, 'trans',  False, None),
            (6, 'std',    False, None),
            (7, 'se2',    False, '#d4f5d4'),
        ]

        for (ci, key, is_gt, bg) in panel_specs:
            ax = fig.add_subplot(outer_gs[row_idx + 1, ci])

            if img_gray is None:
                draw_empty_panel(ax, 'N/A', bg_color=bg or 'white')
                continue

            if key == 'input':
                draw_image_panel(ax, img_gray, bg_color=bg)
            elif key == 'gt':
                draw_image_panel(ax, img_gray, pred_mask=gt_mask, is_gt=True, bg_color=bg)
            else:
                pred = preds.get(key)
                if pred is None:
                    draw_empty_panel(ax, 'N/A', bg_color=bg or 'white')
                else:
                    draw_image_panel(ax, img_gray, pred_mask=pred, is_gt=False, bg_color=bg)

            # Add border highlight for SE2 column
            if ci == 7:
                for sp in ax.spines.values():
                    sp.set_edgecolor('#228B22'); sp.set_linewidth(1.8)

    plt.savefig(OUT_PATH, dpi=250, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
