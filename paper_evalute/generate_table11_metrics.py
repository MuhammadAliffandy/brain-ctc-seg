import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from evaluate_trained_models import SE2_CNNET, load_se2_weights
import scipy.ndimage

def get_pixel_spacing():
    return 0.45, 0.45  # Assuming fallback since we don't have raw NIfTI paths easily

def get_blobs(mask):
    labeled, n = scipy.ndimage.label(mask)
    blobs = []
    for idx in range(1, n + 1):
        blob = (labeled == idx)
        px = int(np.sum(blob))
        if px < 5: continue
        blobs.append({'pixels': px, 'mask': blob})
    blobs.sort(key=lambda b: b['pixels'], reverse=True)
    return blobs

def surface_distances_2d(result, reference, voxelspacing=(0.45, 0.45)):
    result = result.astype(bool)
    reference = reference.astype(bool)
    res_borders = result ^ binary_erosion(result)
    ref_borders = reference ^ binary_erosion(reference)
    
    if not res_borders.any() or not ref_borders.any():
        return np.nan
        
    dt_ref = distance_transform_edt(~ref_borders, sampling=voxelspacing)
    dt_res = distance_transform_edt(~res_borders, sampling=voxelspacing)
    
    dists = np.concatenate([dt_ref[res_borders], dt_res[ref_borders]])
    if len(dists) == 0:
        return np.nan
    return np.percentile(dists, 95)

def run_table11():
    DATA_DIR = os.path.expanduser("~/Clara/local_ct_workspace_full")
    # For Table 11, it says CT-SE(2). Let's use A8_ct_best.pth which is the Mod-SE(2) / CT-SE(2) retrained.
    WEIGHT_PATH = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_ablation/A8_ct_best.pth")
    TARGET_SLICE = 65
    CROP_MARGIN = 40
    ROTATE_K = 3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print(" 🔬 GENERATING TABLE 11 (LESION-WISE HD95 for Fig 8)")
    print("="*60)

    sp_x, sp_y = get_pixel_spacing()
    
    all_folders = os.listdir(DATA_DIR)
    patient_folder = next((f for f in all_folders if "CTC" in f.upper() and ("_2" in f or "_002" in f or " 2" in f)), None)
    if not patient_folder:
        print("❌ Patient folder CTC 2 not found!"); return
        
    patient_path = os.path.join(DATA_DIR, patient_folder)
    z_str = f"z{TARGET_SLICE:03d}"
    img_files = glob.glob(os.path.join(patient_path, f"*{z_str}_img.npy"))
    
    if not img_files:
        print(f"❌ Slice {TARGET_SLICE} not found!"); return
        
    img_path = img_files[0]
    mask_path = img_path.replace('_img.npy', '_mask.npy')
    prev_path = img_path.replace(z_str, f"z{TARGET_SLICE-1:03d}")
    next_path = img_path.replace(z_str, f"z{TARGET_SLICE+1:03d}")

    i0 = np.load(prev_path).astype(np.float32) if os.path.exists(prev_path) else np.load(img_path).astype(np.float32)
    i1 = np.load(img_path).astype(np.float32)
    i2 = np.load(next_path).astype(np.float32) if os.path.exists(next_path) else np.load(img_path).astype(np.float32)
    img_25d = np.stack([i0, i1, i2], axis=-1)
    gt_mask_raw = np.load(mask_path).astype(np.uint8)

    model = SE2_CNNET(n_channels=3, n_classes=2, N=8, base_channels=32).to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device, weights_only=True))
    model.eval()

    img_25d_norm = (img_25d - img_25d.min()) / (img_25d.max() - img_25d.min()) if img_25d.max() > img_25d.min() else img_25d
    tensor = torch.from_numpy(img_25d_norm).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            logits = model(tensor)
        pred_mask_raw = torch.argmax(F.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    gt_mask = np.rot90(gt_mask_raw[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)
    pred_mask = np.rot90(pred_mask_raw[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN], k=ROTATE_K)

    gt_blobs = get_blobs(gt_mask)
    pred_blobs = get_blobs(pred_mask)
    
    # We expect 4 lesions as per Table 11
    print(f"Found {len(gt_blobs)} GT lesions, {len(pred_blobs)} Pred lesions.")
    
    # Calculate HD95 per lesion (match by maximum overlap)
    total_hd95 = surface_distances_2d(pred_mask, gt_mask, voxelspacing=(sp_x, sp_y))
    
    print("\n  Lesion | GT (px) | Pred (px) | HD95 (mm)")
    print("-" * 45)
    
    for idx, g_blob in enumerate(gt_blobs[:4]):
        g_mask = g_blob['mask']
        best_p_mask = None
        best_intersect = 0
        
        for p_blob in pred_blobs:
            p_mask = p_blob['mask']
            intersect = np.logical_and(g_mask, p_mask).sum()
            if intersect > best_intersect:
                best_intersect = intersect
                best_p_mask = p_mask
                
        if best_p_mask is not None:
            hd95 = surface_distances_2d(best_p_mask, g_mask, voxelspacing=(sp_x, sp_y))
            print(f"    {idx+1}    |   {g_blob['pixels']:4d}  |   {best_p_mask.sum():4d}    |  {hd95:.2f}")
        else:
            print(f"    {idx+1}    |   {g_blob['pixels']:4d}  |   Missed    |  NaN")
            
    print("-" * 45)
    print(f"  TOTAL SLICE-LEVEL HD95: {total_hd95:.2f} mm\n")
    print("Run this to complete Table 11!")

if __name__ == "__main__":
    run_table11()
