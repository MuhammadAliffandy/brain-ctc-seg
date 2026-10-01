import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, binary_erosion
import re
from torch.utils.data import Dataset, DataLoader
import cv2
import kagglehub

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from train_ablation import filter_df_by_dataset
from train_comparison_models import StandardUNet, HarmonicNet, nnUNet, AttentionUNet, TransUNet
from evaluate_trained_models import SE2_CNNET

def get_valid_path(rel_path):
    candidates = [
        os.path.expanduser(f"~/Clara/{rel_path}"),
        f"/raid/D13K48009/Clara/{rel_path}",
    ]
    for c in candidates:
        if os.path.exists(c): return c
    return candidates[0]

# --- 1. Dataloader for CECT (NTUH) ---
class PatientVolumeDataset(Dataset):
    def __init__(self, patient_id, root_dir, n_slices=3):
        self.patient_dir = os.path.join(root_dir, patient_id)
        self.n_slices = n_slices
        self.slices = []
        if os.path.exists(self.patient_dir):
            img_files = sorted(
                [f for f in os.listdir(self.patient_dir) if f.endswith('_img.npy')],
                key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0
            )
            for img_name in img_files:
                img_path = os.path.join(self.patient_dir, img_name)
                mask_path = img_path.replace('_img.npy', '_mask.npy')
                if os.path.exists(mask_path):
                    self.slices.append((img_path, mask_path))

    def __len__(self): return len(self.slices)

    def __getitem__(self, idx):
        img_path, mask_path = self.slices[idx]
        img = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        if self.n_slices > 1:
            stack = []
            half = self.n_slices // 2
            for offset in range(-half, half + 1):
                neighbor_idx = max(0, min(idx + offset, len(self.slices) - 1))
                n_img_path, _ = self.slices[neighbor_idx]
                n_img = np.load(n_img_path).astype(np.float32)
                stack.append(n_img)
            img = np.stack(stack, axis=0)
        else:
            img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)
        return torch.from_numpy(img), torch.from_numpy(mask)

# --- 2. Dataloader for Kaggle (2D slices repeated into 2.5D) ---
class PublicKaggleDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        external_test_dir = None
        for r, d, f in os.walk(root_dir):
            if "External_Test" in d or "PNG" in d:
                if "PNG" in d and "MASKS" in d:
                    external_test_dir = r
                    break
                elif "External_Test" in d:
                    external_test_dir = os.path.join(r, "External_Test")
                    break

        if not external_test_dir: return

        png_dir = os.path.join(external_test_dir, "PNG")
        mask_dir = os.path.join(external_test_dir, "MASKS")
        if not os.path.exists(png_dir) or not os.path.exists(mask_dir): return

        inputs = sorted(glob.glob(os.path.join(png_dir, "*.png")))
        for img_path in inputs:
            base_name = os.path.basename(img_path)
            mask_path_exact = os.path.join(mask_dir, base_name)
            if os.path.exists(mask_path_exact):
                self.samples.append((img_path, mask_path_exact))
            else:
                name_without_ext = os.path.splitext(base_name)[0]
                possible_masks = glob.glob(os.path.join(mask_dir, f"*{name_without_ext}*.png"))
                if possible_masks: self.samples.append((img_path, possible_masks[0]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            img = np.zeros((256, 256), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        mask = (mask > 0).astype(np.float32)
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-7)
        # Duplicate to 3 channels to simulate 2.5D
        image_25d = np.stack([img_norm, img_norm, img_norm], axis=0)

        return torch.from_numpy(image_25d).float(), torch.from_numpy(mask).unsqueeze(0).float()

class PublicHemorrhageDataset(Dataset):
    def __init__(self, root_dir):
        all_files = []
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.bmp', '.tif')):
                    all_files.append(os.path.join(root, f))
                    
        masks = [f for f in all_files if 'mask' in f.lower() or 'seg' in f.lower()]
        images = [f for f in all_files if f not in masks]
        
        self.samples = []
        for mask_path in masks:
            mask_name = os.path.basename(mask_path).lower()
            clean_name = mask_name.replace('_hge_seg', '').replace('_seg', '').replace('_mask', '').replace('mask', '').split('.')[0]
            parent_dir = os.path.dirname(mask_path)
            expected_img_path = os.path.join(parent_dir, f"{clean_name}.jpg")
            if not os.path.exists(expected_img_path):
                 expected_img_path = os.path.join(parent_dir, f"{clean_name}.png")
            
            if os.path.exists(expected_img_path):
                self.samples.append((expected_img_path, mask_path))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            img, mask = np.zeros((256, 256), dtype=np.uint8), np.zeros((256, 256), dtype=np.uint8)

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        mask = (mask > 127).astype(np.float32)
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-7)
        image_25d = np.stack([img_norm, img_norm, img_norm], axis=0)

        return torch.from_numpy(image_25d).float(), torch.from_numpy(mask).unsqueeze(0).float()

def surface_distances_3d(result, reference, voxelspacing=(2.5, 0.45, 0.45)):
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

def run_evaluation(table_name, dataset_key, weight_suffix, is_kaggle=False):
    print("\n" + "="*60)
    print(f" 🔬 GENERATING HD95 FOR {table_name}")
    print("="*60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    MODELS = [
        ("Mod-Seg-SE(2)", f"se2_unet_{weight_suffix}_best.pth", SE2_CNNET, True),
        ("HarmonicNet", f"harmonic_net_{weight_suffix}_best.pth", HarmonicNet, True),
        ("nnU-Net", f"nn_unet_{weight_suffix}_best.pth", nnUNet, False),
        ("Standard U-Net", f"standard_unet_{weight_suffix}_best.pth", StandardUNet, False),
        ("Attention U-Net", f"attention_unet_{weight_suffix}_best.pth", AttentionUNet, False),
        ("TransUNet", f"trans_unet_{weight_suffix}_best.pth", TransUNet, False)
    ]
    
    if is_kaggle:
        MODELS[0] = ("Mod-Seg-SE(2)", f"Mod-Seg-SE2_{weight_suffix}_best.pth", SE2_CNNET, True)
        MODELS[3] = ("Standard U-Net", f"Standard_U-Net_{weight_suffix}_best.pth", StandardUNet, False)
        MODELS[4] = ("Attention U-Net", f"Attention_U-Net_{weight_suffix}_best.pth", AttentionUNet, False)
        MODELS[2] = ("nnU-Net", f"nnU-Net_{weight_suffix}_best.pth", nnUNet, False)
        MODELS[5] = ("TransUNet", f"TransUNet_{weight_suffix}_best.pth", TransUNet, False)
    
    # Weights directory
    DIR_W = get_valid_path("brain-ctc-seg/public_dataset/saved_models") if is_kaggle else get_valid_path("brain-ctc-seg/training/saved_models_25D")
    
    results = []
    
    # Prepare Dataloaders
    if not is_kaggle:
        CSV_REPORT = get_valid_path("new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
        DATA_PATH = get_valid_path("local_ct_workspace_full")
        df = pd.read_csv(CSV_REPORT)
        pc = 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'
        df = filter_df_by_dataset(df, 'ctc', pc) # CECT is mapped to CTC
        train_df = df.sample(frac=0.85, random_state=42)
        val_df = df.drop(train_df.index)
        val_patients = val_df[pc].unique()
    else:
        # Download kaggle
        print(f"Downloading Kaggle dataset for {table_name}...")
        kaggle_id = "vbookshelf/computed-tomography-ct-images" if weight_suffix == "kaggle_hemorrhage" else "ozguraslank/brain-stroke-ct-dataset"
        try:
            if weight_suffix == "kaggle_hemorrhage":
                sys.path.append(os.path.join(os.path.dirname(__file__), "..", "public_dataset"))
                from train_all_intra_hemorrhage import get_kaggle_hemorrhage_splits, IntraHemorrhageDataset
                _, test_samples = get_kaggle_hemorrhage_splits(test_size=0.15, seed=42)
                test_loader = DataLoader(IntraHemorrhageDataset(test_samples), batch_size=8, shuffle=False)
            else:
                path = kagglehub.dataset_download(kaggle_id)
                test_loader = DataLoader(PublicKaggleDataset(path), batch_size=8, shuffle=False)
        except Exception as e:
            print(f"Kaggle download failed: {e}")
            return

    for name, weight_name, ModelClass, is_se2 in MODELS:
        weight_path = os.path.join(DIR_W, weight_name)
        if not os.path.exists(weight_path):
            print(f"⚠️ Missing {weight_name}. Skipping...")
            continue

        model = ModelClass(n_channels=3, n_classes=2, N=8, base_channels=32).to(device) if is_se2 and name != "HarmonicNet" else ModelClass(n_channels=3, n_classes=2, N=4, base_channels=32).to(device) if name == "HarmonicNet" else ModelClass(n_channels=3, n_classes=2).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True), strict=False)
        model.eval()

        hd95_list = []
        with torch.no_grad():
            if not is_kaggle:
                for patient in tqdm(val_patients, desc=f"Evaluating {name}"):
                    dataset = PatientVolumeDataset(patient, DATA_PATH, n_slices=3)
                    if len(dataset) == 0: continue
                    loader = DataLoader(dataset, batch_size=8, shuffle=False)
                    vol_preds, vol_masks = [], []
                    for imgs, masks in loader:
                        imgs = imgs.to(device)
                        orig_h, orig_w = masks.shape[-2:]
                        if not is_se2: imgs = F.interpolate(imgs, size=(256, 256), mode='bilinear', align_corners=False)
                        logits = model(imgs)
                        preds = torch.argmax(F.softmax(logits, 1), 1)
                        if not is_se2: preds = F.interpolate(preds.unsqueeze(1).float(), size=(orig_h, orig_w), mode='nearest').squeeze(1).long()
                        vol_preds.append(preds.cpu().numpy())
                        vol_masks.append(masks.squeeze(1).cpu().numpy())
                    vol_preds, vol_masks = np.concatenate(vol_preds, axis=0), np.concatenate(vol_masks, axis=0)
                    if np.any(vol_masks) and np.any(vol_preds):
                        val = surface_distances_3d(vol_preds, vol_masks)
                        if not np.isnan(val): hd95_list.append(val)
            else:
                for imgs, masks in tqdm(test_loader, desc=f"Evaluating {name}"):
                    imgs = imgs.to(device)
                    logits = model(imgs)
                    preds = torch.argmax(F.softmax(logits, 1), 1).cpu().numpy()
                    masks = masks.squeeze(1).cpu().numpy()
                    for i in range(len(preds)):
                        if np.any(masks[i]) and np.any(preds[i]):
                            val = surface_distances_2d(preds[i], masks[i])
                            if not np.isnan(val): hd95_list.append(val)

        if len(hd95_list) > 0:
            median_hd95 = np.median(hd95_list)
            print(f"✅ {name} -> HD95: {median_hd95:.2f} mm")
            results.append((name, median_hd95))
        else:
            print(f"❌ {name} -> Failed (NaN)")
            results.append((name, np.nan))

    print("\n" + "-"*40)
    for name, val in results: print(f"{name:30s} : {val:.2f} mm")

if __name__ == "__main__":
    run_evaluation("TABLE 4 (CECT NTUH)", "ctc", "ctc", is_kaggle=False)
    run_evaluation("TABLE 5 (Kaggle Stroke)", "kaggle", "kaggle", is_kaggle=True)
    run_evaluation("TABLE 6 (Kaggle Hemorrhage)", "kaggle_hemorrhage", "kaggle_hemorrhage", is_kaggle=True)
