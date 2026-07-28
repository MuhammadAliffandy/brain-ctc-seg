"""
generate_fig5_blocks.py
======================================
Generates exactly the 2x3 layout requested by Clara (with green zoom insets)
for EACH of the 4 datasets, outputting 4 separate image files.

Layout for each dataset (2 rows x 3 cols):
  Input        | Ground Truth | Overlay
  Proposed     | U-Net        | NN U-Net
"""
import os, sys
import cv2
import glob
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CROP_MARGIN = 20
ROTATE_K = 3

# local CT/CTC .npy workspace
DATA_DIR_LOCAL = os.path.expanduser("~/Clara/local_ct_workspace_full")

DATASETS = [
    {'name': 'CT',          'source': 'local_ct',     'prefix': 'CT_'},
    {'name': 'CTC',         'source': 'local_ctc',    'prefix': 'CTC_'},
    {'name': 'Stroke',      'source': 'kaggle_stroke','prefix': None},
    {'name': 'Hemorrhage',  'source': 'kaggle_hemo',  'prefix': None}
]

# Model weight directories (same as comparative_inference_all_datasets.py)
SAVE_DIR_NPY   = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D")
SAVE_DIR_INTRA = os.path.expanduser("~/Clara/brain-ctc-seg/public_dataset/saved_models")
SAVE_DIR = "/raid/D13K48009/Clara/brain-ctc-seg/training/Journal_Figures"
os.makedirs(SAVE_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# MODEL LOADING HELPERS
# ─────────────────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from evaluate_trained_models import SE2_CNNET, load_se2_weights

# ─────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURES
# ─────────────────────────────────────────────────────────────────
import torch.nn as nn
import torch.nn.functional as F

class _DC(nn.Module):
    def __init__(self, i, o, norm='bn'):
        super().__init__()
        if norm=='in':
            self.seq = nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.InstanceNorm2d(o, affine=True), nn.LeakyReLU(0.01, True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.InstanceNorm2d(o, affine=True), nn.LeakyReLU(0.01, True),
            )
        else:
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

class nnUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        self.inc = _DC(n_channels, 32, norm='in')
        self.d1 = nn.Sequential(nn.MaxPool2d(2), _DC(32,  64, norm='in'))
        self.d2 = nn.Sequential(nn.MaxPool2d(2), _DC(64,  128, norm='in'))
        self.d3 = nn.Sequential(nn.MaxPool2d(2), _DC(128, 256, norm='in'))
        self.d4 = nn.Sequential(nn.MaxPool2d(2), _DC(256, 512, norm='in'))
        self.u1 = nn.ConvTranspose2d(512,256,2,stride=2); self.c1=_DC(512,256, norm='in')
        self.u2 = nn.ConvTranspose2d(256,128,2,stride=2); self.c2=_DC(256,128, norm='in')
        self.u3 = nn.ConvTranspose2d(128,64, 2,stride=2); self.c3=_DC(128,64, norm='in')
        self.u4 = nn.ConvTranspose2d(64,32, 2,stride=2);  self.c4=_DC(64,32, norm='in')
        self.out = nn.Conv2d(32, n_classes, 1)
    def _pc(self, x, s):
        dy=s.size(2)-x.size(2); dx=s.size(3)-x.size(3)
        return torch.cat([s, F.pad(x,[dx//2,dx-dx//2,dy//2,dy-dy//2])],1)
    def forward(self, x):
        x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3); x5=self.d4(x4)
        x=self.c1(self._pc(self.u1(x5),x4))
        x=self.c2(self._pc(self.u2(x),x3))
        x=self.c3(self._pc(self.u3(x),x2))
        x=self.c4(self._pc(self.u4(x),x1))
        return self.out(x)

def load_std_model(ModelClass, path, device):
    if not os.path.exists(path): return None
    try:
        model = ModelClass().to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        return model
    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────────
# DATA LOADING HELPERS
# ─────────────────────────────────────────────────────────────────
import re

def find_best_npy_slice(data_dir, prefix, min_px=200, max_px=15000):
    best = None
    for folder in sorted(os.listdir(data_dir)):
        if not folder.upper().startswith(prefix.upper()): continue
        fpath = os.path.join(data_dir, folder)
        if not os.path.isdir(fpath): continue
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
    mid_gray = (mid * 255).astype(np.uint8)
    return mid_gray, tensor, mask

def find_stroke_sample_like_training(root_dir, min_px=50):
    external_test_dir = None
    for r, d, f in os.walk(root_dir):
        if "External_Test" in d:
            external_test_dir = os.path.join(r, "External_Test")
            break
    if not external_test_dir: return None, None
    png_dir = os.path.join(external_test_dir, "PNG")
    mask_dir = os.path.join(external_test_dir, "MASKS")
    if not os.path.exists(png_dir) or not os.path.exists(mask_dir): return None, None
    
    inputs = sorted(glob.glob(os.path.join(png_dir, "*.png")))
    random.seed(42); random.shuffle(inputs)
    for img_path in inputs:
        base_name = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, base_name)
        if not os.path.exists(mask_path):
            name_without_ext = os.path.splitext(base_name)[0]
            possible_masks = glob.glob(os.path.join(mask_dir, f"*{name_without_ext}*.png"))
            if possible_masks: mask_path = possible_masks[0]
            else: continue
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None and np.sum(m > 127) >= min_px:
            return img_path, mask_path
    return None, None

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
        for k in ['_hge_seg','_seg','_mask','mask','seg']: clean = clean.replace(k,'')
        clean = clean.split('.')[0]
        
        candidates = []
        for ext in ['.jpg','.png','.bmp']:
            candidates.append(os.path.join(parent, clean+ext))
            candidates.append(os.path.join(parent, base))
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
    img_gray = (img * 255).astype(np.uint8)
    return img_gray, t, mask

# ─────────────────────────────────────────────────────────────────
# DRAWING HELPERS (With green zoom insets)
# ─────────────────────────────────────────────────────────────────
def draw_panel(ax, img_gray, mask=None, mask_color='red', title="", add_overlay=False, gt_mask=None, pred_mask=None):
    ax.imshow(img_gray, cmap='gray')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
    ax.axis('off')
    
    colors = {'red': [1,0,0], 'blue': [0,0,1], 'yellow': [1,1,0], 'white': [1,1,1]}
    c_rgb = colors.get(mask_color, [1,0,0])
    
    # Base image for zoom
    overlay_img = np.stack([img_gray, img_gray, img_gray], axis=-1) / 255.0
    
    if add_overlay and gt_mask is not None and pred_mask is not None:
        # Overlay mode (red outline for proposed, blue outline for GT)
        # Using contours
        gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(overlay_img, gt_contours, -1, (0,0,1), 2) # Blue GT
        cv2.drawContours(overlay_img, pred_contours, -1, (1,0,0), 2) # Red Pred
        ax.imshow(overlay_img)
        
        bbox_mask = (gt_mask | pred_mask)
    elif mask is not None:
        # Solid fill mode
        m_bool = mask > 0
        overlay_img[m_bool] = overlay_img[m_bool] * 0.5 + np.array(c_rgb) * 0.5
        ax.imshow(overlay_img)
        bbox_mask = mask
    else:
        # Input only, no mask. We need a bounding box for zoom anyway.
        # We will use the GT mask to find the lesion area if provided, else center.
        bbox_mask = gt_mask if gt_mask is not None else np.zeros_like(img_gray)
        
    # Find bounding box of lesion for zoom
    y_idx, x_idx = np.where(bbox_mask > 0)
    if len(y_idx) > 0:
        ymin, ymax = max(0, y_idx.min()-15), min(255, y_idx.max()+15)
        xmin, xmax = max(0, x_idx.min()-15), min(255, x_idx.max()+15)
    else:
        ymin, ymax, xmin, xmax = 100, 156, 100, 156
        
    # Make square bbox
    h, w = ymax - ymin, xmax - xmin
    size = max(h, w, 40)
    cy, cx = (ymin+ymax)//2, (xmin+xmax)//2
    ymin, ymax = int(cy - size/2), int(cy + size/2)
    xmin, xmax = int(cx - size/2), int(cx + size/2)
    
    # Clip
    ymin, ymax = max(0, ymin), min(255, ymax)
    xmin, xmax = max(0, xmin), min(255, xmax)
    
    # Draw green box on main image
    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='#00ff00', facecolor='none')
    ax.add_patch(rect)
    
    # Draw inset (top right)
    axins = inset_axes(ax, width="35%", height="35%", loc=1, borderpad=1)
    axins.imshow(overlay_img)
    axins.set_xlim(xmin, xmax)
    axins.set_ylim(ymax, ymin) # Inverted Y for images
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor('#00ff00')
        spine.set_linewidth(2)


# ─────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────
def main():
    print("🚀 GENERATING 2x3 BLOCKS FOR EACH DATASET (Clara's WhatsApp Request)")
    
    for ds in DATASETS:
        print(f"\nProcessing Dataset: {ds['name']}...")
        
        # 1. LOAD DATA
        if ds['source'] in ['local_ct', 'local_ctc']:
            sample = find_best_npy_slice(DATA_DIR_LOCAL, ds['prefix'])
            if sample is None:
                print(f"  ⚠️ No valid slice found for {ds['name']}")
                continue
            img_gray, tensor, gt_mask = load_npy_sample(sample)
                    
        elif ds['source'] == 'kaggle_stroke':
            dl_paths = [
                os.path.expanduser("~/.cache/kagglehub/datasets/ozguraslank/brain-stroke-ct-dataset"),
                os.path.expanduser("~/Clara/brain-ctc-seg/public_dataset/brain-stroke-ct-dataset"),
                os.path.expanduser("~/Clara/brain-ctc-seg/public_dataset/data/brain-stroke-ct-dataset")
            ]
            ip, mp = None, None
            for dl in dl_paths:
                if os.path.exists(dl):
                    ip, mp = find_stroke_sample_like_training(dl)
                    if ip is not None: break
            if ip is None:
                try:
                    dl = kagglehub.dataset_download("ozguraslank/brain-stroke-ct-dataset")
                    ip, mp = find_stroke_sample_like_training(dl)
                except Exception: pass
            if ip is None:
                print("  ⚠️ No stroke sample found"); continue
            img_gray, tensor, gt_mask = load_kaggle_sample(ip, mp)

        elif ds['source'] == 'kaggle_hemo':
            dl_paths = [
                os.path.expanduser("~/.cache/kagglehub/datasets/vbookshelf/computed-tomography-ct-images"),
                os.path.expanduser("~/Clara/brain-ctc-seg/public_dataset/computed-tomography-ct-images"),
                os.path.expanduser("~/Clara/brain-ctc-seg/public_dataset/data/computed-tomography-ct-images")
            ]
            ip, mp = None, None
            for dl in dl_paths:
                if os.path.exists(dl):
                    ip, mp = find_best_kaggle_sample(dl, mask_kw=('mask','hge_seg','seg'))
                    if ip is not None: break
            if ip is None:
                try:
                    dl = kagglehub.dataset_download("vbookshelf/computed-tomography-ct-images")
                    ip, mp = find_best_kaggle_sample(dl, mask_kw=('mask','hge_seg','seg'))
                except Exception: pass
            if ip is None:
                print("  ⚠️ No hemorrhage sample found"); continue
            img_gray, tensor, gt_mask = load_kaggle_sample(ip, mp)
            
        tensor = tensor.to(DEVICE)
        
        # 2. LOAD MODELS — use exact same paths as comparative_inference_all_datasets.py
        if ds['source'] in ['local_ct', 'local_ctc']:
            key = 'ct' if ds['name']=='CT' else 'ctc'
            se2_path   = os.path.join(SAVE_DIR_NPY, f"se2_unet_{key}_best.pth")
            unet_path  = os.path.join(SAVE_DIR_NPY, f"standard_unet_{key}_best.pth")
            nn_path    = os.path.join(SAVE_DIR_NPY, f"nn_unet_{key}_best.pth")
        elif ds['source'] == 'kaggle_stroke':
            se2_path   = os.path.join(SAVE_DIR_INTRA, "Mod-Seg-SE2_kaggle_best.pth")
            unet_path  = os.path.join(SAVE_DIR_INTRA, "Standard_U-Net_kaggle_best.pth")
            nn_path    = os.path.join(SAVE_DIR_INTRA, "nnU-Net_kaggle_best.pth")
        else:  # hemorrhage
            se2_path   = os.path.join(SAVE_DIR_INTRA, "Mod-Seg-SE2_kaggle_hemorrhage_best.pth")
            unet_path  = os.path.join(SAVE_DIR_INTRA, "Standard_U-Net_kaggle_hemorrhage_best.pth")
            nn_path    = os.path.join(SAVE_DIR_INTRA, "nnU-Net_kaggle_hemorrhage_best.pth")
        
        m_se2 = load_se2_weights(SE2_CNNET, se2_path, DEVICE)
        if m_se2: m_se2.eval()
        
        m_unet   = load_std_model(StandardUNet, unet_path, DEVICE)
        m_nnunet = load_std_model(nnUNet, nn_path, DEVICE)
        
        if m_se2 is None:
            print(f"  ⚠️ Proposed model not found for {ds['name']}. Skipping.")
            continue
            
        # 3. INFERENCE
        with torch.no_grad(), torch.amp.autocast('cuda'):
            pred_se2 = (torch.sigmoid(m_se2(tensor))[0,1] > 0.5).cpu().numpy()
            
            if m_unet is not None:
                pred_unet = (torch.sigmoid(m_unet(tensor))[0,1] > 0.5).cpu().numpy()
            else:
                pred_unet = np.zeros_like(pred_se2)
                
            if m_nnunet is not None:
                pred_nnunet = (torch.sigmoid(m_nnunet(tensor))[0,1] > 0.5).cpu().numpy()
            else:
                pred_nnunet = np.zeros_like(pred_se2)
                
        # 4. PLOTTING (2 rows x 3 cols)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.patch.set_facecolor('white')
        
        # Row 1: Input | GT | Overlay
        draw_panel(axes[0,0], img_gray, gt_mask=gt_mask, title="Input")
        axes[0,0].text(0.5, -0.1, "(a)", transform=axes[0,0].transAxes, ha='center', fontsize=14)
        
        draw_panel(axes[0,1], img_gray, mask=gt_mask, mask_color='white', title="Ground Truth", gt_mask=gt_mask)
        axes[0,1].text(0.5, -0.1, "(b)", transform=axes[0,1].transAxes, ha='center', fontsize=14)
        
        draw_panel(axes[0,2], img_gray, add_overlay=True, gt_mask=gt_mask, pred_mask=pred_se2, title="Overlay")
        axes[0,2].text(0.5, -0.1, "(c)", transform=axes[0,2].transAxes, ha='center', fontsize=14)
        
        # Row 2: CT-SE(2) | U-Net | NN U-Net
        draw_panel(axes[1,0], img_gray, mask=pred_se2, mask_color='red', title="CT-SE(2)", gt_mask=gt_mask)
        axes[1,0].text(0.5, -0.1, "(d)", transform=axes[1,0].transAxes, ha='center', fontsize=14)
        
        draw_panel(axes[1,1], img_gray, mask=pred_unet, mask_color='yellow', title="U-Net", gt_mask=gt_mask)
        axes[1,1].text(0.5, -0.1, "(e)", transform=axes[1,1].transAxes, ha='center', fontsize=14)
        
        draw_panel(axes[1,2], img_gray, mask=pred_nnunet, mask_color='blue', title="NN U-Net", gt_mask=gt_mask)
        axes[1,2].text(0.5, -0.1, "(f)", transform=axes[1,2].transAxes, ha='center', fontsize=14)
        
        plt.subplots_adjust(wspace=0.05, hspace=0.2)
        
        out_path = os.path.join(SAVE_DIR, f"Fig5_Block_{ds['name']}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved -> {out_path}")

if __name__ == "__main__":
    main()
