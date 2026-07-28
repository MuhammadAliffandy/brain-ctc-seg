"""
generate_fig5_blocks.py
======================================
Generates exactly the 2x3 layout requested by Clara (with green zoom insets)
for EACH of the 4 datasets, outputting 4 separate image files.

Layout for each dataset (2 rows x 3 cols):
  Input        | Ground Truth | Overlay
  Proposed     | U-Net        | NN U-Net
"""
import os
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

DATASETS = [
    {'name': 'CT', 'source': 'local_ct', 'path': '../data/test'},
    {'name': 'CTC', 'source': 'local_ctc', 'path': '../data_ctc/test'},
    {'name': 'Stroke', 'source': 'kaggle_stroke', 'path': ''},
    {'name': 'Hemorrhage', 'source': 'kaggle_hemo', 'path': ''}
]

MODELS_DIR = "/raid/D13K48009/Clara/brain-ctc-seg/training/Best_Models"
SAVE_DIR = "/raid/D13K48009/Clara/brain-ctc-seg/training/Journal_Figures"
os.makedirs(SAVE_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# MODEL LOADING HELPERS
# ─────────────────────────────────────────────────────────────────
def auto_detect_base_channels(state_dict):
    for key in ['inc.double_conv.0.weight', 'inc.double_conv.0.weights', 'inc.double_conv.0.filter']:
        if key in state_dict:
            return state_dict[key].shape[0]
    return 32

def load_model(ModelClass, path, device, is_se2=False):
    if not os.path.exists(path):
        return None
    try:
        sd = torch.load(path, map_location=device, weights_only=True)
        if is_se2:
            bc = auto_detect_base_channels(sd)
            model = ModelClass(n_channels=3, n_classes=2, base_channels=bc).to(device)
            model.load_state_dict(sd, strict=False)
        else:
            model = ModelClass(n_channels=3, n_classes=2).to(device)
            model.load_state_dict(sd, strict=False)
        model.eval()
        return model
    except Exception as e:
        return None

# Import models
import sys
sys.path.append("/raid/D13K48009/Clara/brain-ctc-seg")
try:
    from src.models.unet import UNet
    from src.models.nnunet import nnUNet
    from src.models.se2_unet import SE2UNet
except ImportError:
    print("Run this from paper_evalute folder!")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────
# DATA LOADING HELPERS
# ─────────────────────────────────────────────────────────────────
def load_npy_sample(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()
    img_25d = data['image_25d']
    mid = img_25d[..., 1]
    mask = data['mask']
    
    img_25d_n = (img_25d - img_25d.min()) / (img_25d.max() - img_25d.min() + 1e-8)
    
    mid = np.rot90(mid[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K).copy()
    mask = np.rot90(mask[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K).copy()
    inp_c = np.rot90(img_25d_n[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN, :], k=ROTATE_K).copy()
    
    mid = cv2.resize(mid, (256, 256))
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    inp_c = cv2.resize(inp_c, (256, 256))
    
    mask = (mask > 0).astype(np.uint8)
    tensor = torch.from_numpy(inp_c).permute(2, 0, 1).unsqueeze(0)
    
    # Normalize gray for visualization
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
            base_dir = os.path.expanduser(f"~/Clara/brain-ctc-seg/{ds['path']}")
            samples = glob.glob(os.path.join(base_dir, "*.npy"))
            if not samples:
                print(f"  ⚠️ No local samples found in {base_dir}")
                continue
            random.seed(42); random.shuffle(samples)
            sample_path = None
            # Find one with a decent sized lesion
            for s in samples:
                mg, t, gm = load_npy_sample(s)
                if np.sum(gm) > 100:
                    sample_path = s
                    img_gray, tensor, gt_mask = mg, t, gm
                    break
                    
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
        
        # 2. LOAD MODELS
        prefix = 'ct' if ds['name']=='CT' else 'ctc' if ds['name']=='CTC' else 'kaggle_stroke' if ds['name']=='Stroke' else 'kaggle_hemo'
        
        m_se2 = load_model(SE2UNet, os.path.join(MODELS_DIR, f"se2_unet_{prefix}_best.pth"), DEVICE, is_se2=True)
        m_unet = load_model(UNet, os.path.join(MODELS_DIR, f"standard_unet_{prefix}_best.pth"), DEVICE, is_se2=False)
        m_nnunet = load_model(nnUNet, os.path.join(MODELS_DIR, f"nnUNet_{prefix}_best.pth"), DEVICE, is_se2=False)
        
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
