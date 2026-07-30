import os, sys, glob, re, random
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import kagglehub

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
SAVE_DIR_NPY   = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D")
SAVE_DIR_INTRA = os.path.expanduser("~/Clara/brain-ctc-seg/public_dataset/saved_models")
CROP_MARGIN, ROTATE_K = 40, 3

# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────
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
    
    mid=i1.copy()
    mid  = np.rot90(mid[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN],   k=ROTATE_K).copy()
    mask = np.rot90(mask[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN],  k=ROTATE_K).copy()
    
    return mid, mask

def find_best_kaggle_sample(root, mask_kw=('mask','seg','hge'), min_px=50, seed=42):
    all_files=[]
    for r,_,files in os.walk(root):
        for f in files:
            if f.lower().endswith(('.jpg','.png','.bmp')):
                all_files.append(os.path.join(r,f))
    masks=[f for f in all_files if any(k in f.lower() for k in mask_kw)]
    random.seed(seed); random.shuffle(masks)
    
    for mp in masks:
        m=cv2.imread(mp,cv2.IMREAD_GRAYSCALE)
        if m is None or np.sum(m>127)<min_px: continue
        
        base = os.path.basename(mp)
        parent = os.path.dirname(mp)
        grandparent = os.path.dirname(parent)
        
        clean = base.lower()
        for k in ['_hge_seg','_seg','_mask','mask','seg']:
            clean = clean.replace(k,'')
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

def find_stroke_sample_like_training(root_dir, min_px=50):
    external_test_dir = None
    for r, d, f in os.walk(root_dir):
        if "External_Test" in d:
            external_test_dir = os.path.join(r, "External_Test")
            break
    if not external_test_dir:
        return None, None
    
    png_dir = os.path.join(external_test_dir, "PNG")
    mask_dir = os.path.join(external_test_dir, "MASKS")
    if not os.path.exists(png_dir) or not os.path.exists(mask_dir):
        return None, None
        
    inputs = sorted(glob.glob(os.path.join(png_dir, "*.png")))
    random.seed(42); random.shuffle(inputs)
    
    for img_path in inputs:
        base_name = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, base_name)
        if not os.path.exists(mask_path):
            name_without_ext = os.path.splitext(base_name)[0]
            possible_masks = glob.glob(os.path.join(mask_dir, f"*{name_without_ext}*.png"))
            if possible_masks:
                mask_path = possible_masks[0]
            else:
                continue
                
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None and np.sum(m > 127) >= min_px:
            return img_path, mask_path
    return None, None

def load_kaggle_sample(ip,mp):
    img =cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
    mask=cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    img =cv2.resize(img, (256,256)).astype(np.float32)
    mask=cv2.resize(mask,(256,256),interpolation=cv2.INTER_NEAREST)
    mask=(mask>127).astype(np.uint8)
    img = img / 255.0
    return img, mask

# ─────────────────────────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────────────────────────
def draw_image_panel(ax, img_gray, gt_mask=None):
    H, W = img_gray.shape
    ax.imshow(img_gray, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    if gt_mask is not None:
        # Blue filled overlay for GT like in the user's screenshot
        ov = np.zeros((H, W, 4))
        ov[gt_mask > 0] = [0.1, 0.4, 0.9, 0.6]  # Blue with some transparency
        ax.imshow(ov)

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    DATA_DIR  = os.path.expanduser("~/Clara/local_ct_workspace_full")
    OUT_PATH  = os.path.expanduser("~/Clara/brain-ctc-seg/training/Journal_Figures/Dataset_Samples_2x4_HD.png")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    datasets = [
        dict(name='NCCT (NTUH)',         source='npy',           prefix='CT_'),
        dict(name='CECT (NTUH)',        source='npy',           prefix='CTC_'),
        dict(name='Stroke (Kaggle)',     source='kaggle_stroke', prefix=None),
        dict(name='Hemorrhage (Kaggle)', source='kaggle_hemo',   prefix=None),
    ]

    fig = plt.figure(figsize=(10, 5), facecolor='white')
    outer_gs = gridspec.GridSpec(
        3, 5,
        figure=fig,
        width_ratios=[0.5, 2.5, 2.5, 2.5, 2.5],
        height_ratios=[0.3, 2.5, 2.5],
        hspace=0.05, wspace=0.05,
        left=0.02, right=0.98, top=0.95, bottom=0.05
    )

    # Column Headers
    for ci, ds in enumerate(datasets):
        ax = fig.add_subplot(outer_gs[0, ci + 1])
        ax.axis('off')
        ax.text(0.5, 0.5, ds['name'], transform=ax.transAxes, ha='center', va='center', fontsize=14)

    # Row Headers
    for ri, row_name in enumerate(['Input', 'Groundtruth']):
        ax = fig.add_subplot(outer_gs[ri + 1, 0])
        ax.axis('off')
        ax.text(0.5, 0.5, row_name, transform=ax.transAxes, ha='center', va='center', fontsize=14, rotation=90)

    # Draw panels
    for ci, ds in enumerate(datasets):
        if ds['source'] == 'npy':
            sample = find_best_npy_slice(DATA_DIR, ds['prefix'])
            if sample:
                print(f"  ✅ {ds['name']}: found npy slice with {sample['px']}px tumor")
            img_gray, gt_mask = load_npy_sample(sample) if sample else (None, None)
        elif ds['source'] == 'kaggle_stroke':
            dl_paths = [
                os.path.expanduser("~/.cache/kagglehub/datasets/ozguraslank/brain-stroke-ct-dataset"),
                os.path.expanduser("~/Clara/brain-ctc-seg/public_dataset/brain-stroke-ct-dataset"),
                os.path.expanduser("~/Clara/brain-ctc-seg/public_dataset/data/brain-stroke-ct-dataset")
            ]
            ip, mp = None, None
            for dl in dl_paths:
                print(f"  🔍 Checking stroke path: {dl} -> exists={os.path.exists(dl)}")
                if os.path.exists(dl):
                    ip, mp = find_stroke_sample_like_training(dl)
                    if ip is not None:
                        print(f"  ✅ Stroke sample found: {ip}")
                        break
            if ip is None:
                try:
                    dl = kagglehub.dataset_download("ozguraslank/brain-stroke-ct-dataset")
                    ip, mp = find_stroke_sample_like_training(dl)
                except Exception as e:
                    print(f"  ⚠️ kagglehub stroke error: {e}")
            if ip is None:
                print(f"  ❌ No stroke sample found!")
            img_gray, gt_mask = load_kaggle_sample(ip, mp) if ip else (None, None)
        elif ds['source'] == 'kaggle_hemo':
            dl_paths = [
                os.path.expanduser("~/.cache/kagglehub/datasets/vbookshelf/computed-tomography-ct-images"),
                os.path.expanduser("~/Clara/brain-ctc-seg/public_dataset/computed-tomography-ct-images"),
                os.path.expanduser("~/Clara/brain-ctc-seg/public_dataset/data/computed-tomography-ct-images")
            ]
            ip, mp = None, None
            for dl in dl_paths:
                print(f"  🔍 Checking hemo path: {dl} -> exists={os.path.exists(dl)}")
                if os.path.exists(dl):
                    ip, mp = find_best_kaggle_sample(dl, mask_kw=('mask','hge_seg','seg'), seed=99)
                    if ip is not None:
                        print(f"  ✅ Hemorrhage sample found: {ip}")
                        break
            if ip is None:
                try:
                    dl = kagglehub.dataset_download("vbookshelf/computed-tomography-ct-images")
                    ip, mp = find_best_kaggle_sample(dl, mask_kw=('mask','hge_seg','seg'), seed=99)
                except Exception as e:
                    print(f"  ⚠️ kagglehub hemo error: {e}")
            if ip is None:
                print(f"  ❌ No hemorrhage sample found!")
            img_gray, gt_mask = load_kaggle_sample(ip, mp) if ip else (None, None)

        if img_gray is None:
            print(f"  ⚠️ Skipping {ds['name']}: no data loaded")
            continue

        # Draw Input (row 1)
        ax_in = fig.add_subplot(outer_gs[1, ci + 1])
        draw_image_panel(ax_in, img_gray, gt_mask=None)

        # Draw Groundtruth (row 2)
        ax_gt = fig.add_subplot(outer_gs[2, ci + 1])
        draw_image_panel(ax_gt, img_gray, gt_mask=gt_mask)

    print(f"  💾 Saving HD Figure to {OUT_PATH}")
    plt.savefig(OUT_PATH, dpi=600, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
