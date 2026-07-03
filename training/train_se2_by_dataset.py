"""
train_se2_by_dataset.py
=======================
Exact replica of train.py with --dataset argument for CT/CTC separation.
v4: Reverted LR to 1e-4 (stable). Added CLAHE + Sharpen augmentations for CT dataset only
    to improve local contrast and edge definition on non-contrast CT scans.

Usage:
    python train_se2_by_dataset.py --dataset ct    # Train on CT_* patients only
    python train_se2_by_dataset.py --dataset ctc   # Train on CTC_*/CTW_* patients only
    python train_se2_by_dataset.py --dataset all   # All patients (same as original train.py)

Weights saved to:
    saved_models_25D/se2_unet_ct_best.pth
    saved_models_25D/se2_unet_ctc_best.pth
    saved_models_25D/se2_unet_all_best.pth
"""

import os, sys, re, argparse, datetime, zipfile, shutil, random
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from escnn import gspaces
import escnn.nn as enn


# ================================================================
# DATA PREPARATION
# ================================================================
def prepare_local_data(gdrive_dir, local_extract_dir):
    os.makedirs(local_extract_dir, exist_ok=True)
    zip_files = [f for f in os.listdir(gdrive_dir) if f.endswith('.zip')]
    if zip_files:
        print(f"📦 Found {len(zip_files)} .zip files. Extracting...")
        for z_file in tqdm(zip_files, desc="Extracting Zips"):
            patient_name = z_file.replace('.zip', '')
            target_folder = os.path.join(local_extract_dir, patient_name)
            if not os.path.exists(target_folder):
                try:
                    with zipfile.ZipFile(os.path.join(gdrive_dir, z_file), 'r') as zr:
                        zr.extractall(target_folder)
                except Exception as e:
                    print(f"⚠️ Error extracting {z_file}: {e}")
        return local_extract_dir
    sub_folders = [f for f in os.listdir(gdrive_dir) if os.path.isdir(os.path.join(gdrive_dir, f))]
    if sub_folders:
        print(f"📁 Copying Folders to {local_extract_dir}...")
        for folder in tqdm(sub_folders, desc="Copying Folders"):
            src = os.path.join(gdrive_dir, folder)
            dst = os.path.join(local_extract_dir, folder)
            if not os.path.exists(dst):
                shutil.copytree(src, dst)
    return local_extract_dir


# ================================================================
# DATASET — identical to train.py, NO resize
# ================================================================
class CTBrain25DDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.root_dir  = root_dir
        self.transform = transform
        self.patient_slices = {}
        self.all_samples    = []
        patient_col = 'Patient_Folder' if 'Patient_Folder' in dataframe.columns else 'Patient'
        for patient in dataframe[patient_col].unique():
            patient_dir = os.path.join(root_dir, patient)
            if not os.path.exists(patient_dir):
                continue
            img_files = sorted(
                [f for f in os.listdir(patient_dir) if f.endswith('_img.npy')],
                key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0
            )
            valid_pairs = []
            for img_name in img_files:
                img_path  = os.path.join(patient_dir, img_name)
                mask_path = img_path.replace('_img.npy', '_mask.npy')
                if os.path.exists(mask_path):
                    valid_pairs.append((img_path, mask_path))
            if valid_pairs:
                self.patient_slices[patient] = valid_pairs
                for i in range(len(valid_pairs)):
                    self.all_samples.append((patient, i))

    def __len__(self): return len(self.all_samples)

    def __getitem__(self, idx):
        patient, slice_idx = self.all_samples[idx]
        slices  = self.patient_slices[patient]
        idx_prev = max(0, slice_idx - 1)
        idx_next = min(len(slices) - 1, slice_idx + 1)
        try:
            i0 = np.load(slices[idx_prev][0]).astype(np.float32)
            i1 = np.load(slices[slice_idx][0]).astype(np.float32)
            i2 = np.load(slices[idx_next][0]).astype(np.float32)
            m  = np.load(slices[slice_idx][1]).astype(np.uint8)
            image_25d = np.stack([i0, i1, i2], axis=-1)
            
            # NORMALIZATION FIX: Min-Max scale to [0, 1] to prevent dead gradients
            if image_25d.max() > image_25d.min():
                image_25d = (image_25d - image_25d.min()) / (image_25d.max() - image_25d.min())
                
            if self.transform is not None:
                aug       = self.transform(image=image_25d, mask=m)
                image_25d = aug['image']
                m         = aug['mask']
            return torch.from_numpy(image_25d).permute(2, 0, 1), torch.from_numpy(m).long()
        except Exception:
            return self.__getitem__(random.randint(0, len(self.all_samples) - 1))


# ================================================================
# SE2_CNNET ARCHITECTURE — exact copy from train.py
# ================================================================
class DoubleEquivariantConv(nn.Module):
    def __init__(self, in_type, out_type, mid_type=None):
        super().__init__()
        if not mid_type: mid_type = out_type
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
    def __init__(self, n_channels=3, n_classes=2, N=8, base_channels=32):
        super().__init__()
        self.r2_act = gspaces.rot2dOnR2(N=N)
        c = base_channels
        self.feat_type_in = enn.FieldType(self.r2_act, n_channels * [self.r2_act.trivial_repr])
        self.feat_type_1  = enn.FieldType(self.r2_act, c      * [self.r2_act.regular_repr])
        self.feat_type_2  = enn.FieldType(self.r2_act, (c*2)  * [self.r2_act.regular_repr])
        self.feat_type_3  = enn.FieldType(self.r2_act, (c*4)  * [self.r2_act.regular_repr])
        self.feat_type_4  = enn.FieldType(self.r2_act, (c*8)  * [self.r2_act.regular_repr])
        self.feat_type_5  = enn.FieldType(self.r2_act, (c*16) * [self.r2_act.regular_repr])
        self.inc   = DoubleEquivariantConv(self.feat_type_in, self.feat_type_1)
        self.down1 = Down(self.feat_type_1, self.feat_type_2)
        self.down2 = Down(self.feat_type_2, self.feat_type_3)
        self.down3 = Down(self.feat_type_3, self.feat_type_4)
        self.down4 = Down(self.feat_type_4, self.feat_type_5)
        self.up1   = Up(self.feat_type_5, self.feat_type_4)
        self.up2   = Up(self.feat_type_4, self.feat_type_3)
        self.up3   = Up(self.feat_type_3, self.feat_type_2)
        self.up4   = Up(self.feat_type_2, self.feat_type_1)
        self.outc  = OutConv(self.feat_type_1, n_classes)

    def forward(self, x):
        x_geom = enn.GeometricTensor(x, self.feat_type_in)
        x1 = self.inc(x_geom)
        x2 = self.down1(x1); x3 = self.down2(x2)
        x4 = self.down3(x3); x5 = self.down4(x4)
        x  = self.up1(x5, x4); x = self.up2(x, x3)
        x  = self.up3(x, x2);  x = self.up4(x, x1)
        return self.outc(x).tensor


# ================================================================
# LOSS FUNCTIONS — identical to train.py
# ================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=3.0):
        super().__init__(); self.alpha = alpha; self.gamma = gamma
    def forward(self, logits, targets):
        bce = F.cross_entropy(logits, targets, reduction='none')
        pt  = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()

class EdgeBoundaryLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
    def forward(self, logits, targets):
        tf      = targets.float().unsqueeze(1)
        dilated = F.max_pool2d(tf, kernel_size=5, stride=1, padding=2)
        eroded  = -F.max_pool2d(-tf, kernel_size=5, stride=1, padding=2)
        bnd     = (dilated - eroded).squeeze(1)
        base    = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        return (base * (1 + 5.0 * bnd)).mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5): super().__init__(); self.smooth = smooth
    def forward(self, logits, true_masks):
        nc    = logits.shape[1]
        oh    = F.one_hot(true_masks, nc).permute(0, 3, 1, 2).float()
        probs = F.softmax(logits, dim=1)
        inter = (probs[:, 1] * oh[:, 1]).sum(dim=(1, 2))
        union = probs[:, 1].sum(dim=(1, 2)) + oh[:, 1].sum(dim=(1, 2))
        return 1.0 - ((2. * inter + self.smooth) / (union + self.smooth)).mean()

class TverskyLoss(nn.Module):
    """Generalization of DiceLoss. alpha penalizes FP, beta penalizes FN.
    V6: alpha=0.55 (gentle FP penalty, balanced). alpha=0.6 was too aggressive
    and caused Recall to drop from 0.83 to 0.67, hurting IoU."""
    def __init__(self, alpha=0.55, beta=0.45, smooth=1e-5):
        super().__init__()
        self.alpha = alpha; self.beta = beta; self.smooth = smooth
    def forward(self, logits, targets):
        nc    = logits.shape[1]
        oh    = F.one_hot(targets, nc).permute(0, 3, 1, 2).float()
        probs = F.softmax(logits, dim=1)[:, 1]
        tp    = (probs * oh[:, 1]).sum(dim=(1, 2))
        fp    = (probs * (1 - oh[:, 1])).sum(dim=(1, 2))
        fn    = ((1 - probs) * oh[:, 1]).sum(dim=(1, 2))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()

class AdvancedCombinedLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.focal = FocalLoss(alpha=0.75, gamma=3.0)
        self.dice  = DiceLoss()
        self.edge  = EdgeBoundaryLoss(class_weights=class_weights)
    def forward(self, logits, targets):
        # V4 (proven best): FocalLoss suppresses easy negatives, DiceLoss(2.0) directly
        # optimizes for IoU/Dice, EdgeBoundaryLoss sharpens boundary precision.
        return 0.5 * self.focal(logits, targets) + 2.0 * self.dice(logits, targets) + 0.5 * self.edge(logits, targets)


# ================================================================
# METRICS
# ================================================================
def calculate_metrics_tensors(preds, targets):
    preds = preds.view(-1); targets = targets.view(-1)
    tp = torch.sum((preds == 1) & (targets == 1)).item()
    fp = torch.sum((preds == 1) & (targets == 0)).item()
    fn = torch.sum((preds == 0) & (targets == 1)).item()
    tn = torch.sum((preds == 0) & (targets == 0)).item()
    return tp, fp, fn, tn


# ================================================================
# FILTER HELPERS
# ================================================================
def filter_df_by_dataset(df, dataset_key, patient_col='Patient_Folder'):
    """Filter DataFrame rows by CT_* or CTC_*/CTW_* folder prefix."""
    if dataset_key == 'ct':
        mask = df[patient_col].str.startswith('CT_')
    elif dataset_key == 'ctc':
        mask = df[patient_col].str.startswith('CTC_') | df[patient_col].str.startswith('CTW_')
    else:  # 'all'
        mask = pd.Series([True] * len(df), index=df.index)
    return df[mask]


# ================================================================
# TRAINING
# ================================================================
def train(dataset_key: str):
    GDRIVE_ROOT   = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive")
    GDRIVE_DATA   = os.path.join(GDRIVE_ROOT, "Dataset_CT_Preprocessed_NPY")
    CSV_REPORT    = os.path.join(GDRIVE_ROOT, "Dataset_CT_Report.csv")
    LOCAL_DATA    = os.path.expanduser("~/Clara/local_ct_workspace_full")
    PROJECT_ROOT  = os.path.expanduser("~/Clara/brain-ctc-seg/training")
    MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "saved_models_25D")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    LEARNING_RATE = 1e-4; BATCH_SIZE = 8; ACCUM_STEPS = 4; EPOCHS = 150; VAL_SPLIT = 0.15
    EARLY_STOP_PATIENCE = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*65}")
    print(f"  Mod-Seg-SE(2) — Dataset: {dataset_key.upper()}")
    print(f"  Epochs: {EPOCHS} | LR: {LEARNING_RATE} | Batch: {BATCH_SIZE} | Device: {device}")
    print(f"{'='*65}\n")

    local_root = prepare_local_data(GDRIVE_DATA, LOCAL_DATA)

    df = pd.read_csv(CSV_REPORT)
    pc = 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'
    df = filter_df_by_dataset(df, dataset_key, pc)
    print(f"  Dataset filter '{dataset_key}': {len(df)} patients found")

    if len(df) == 0:
        print("❌ No patients found for this dataset type. Check folder prefix in CSV."); return

    train_df = df.sample(frac=(1 - VAL_SPLIT), random_state=42)
    val_df   = df.drop(train_df.index)
    print(f"  Train: {len(train_df)} patients | Val: {len(val_df)} patients")

    # CT-specific augmentations: CLAHE boosts local contrast on grey non-contrast images,
    # Sharpen helps the model see blurry hemorrhage edges more clearly.
    # These are intentionally applied ONLY on CT (not CTC) since CTC already has high contrast.
    ct_extra_augs = [
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.4),
    ] if dataset_key == 'ct' else []

    train_transform = A.Compose([
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.06, 0.06), rotate=(-15, 15), p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        A.GridDistortion(p=0.3),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.HorizontalFlip(p=0.5),
        *ct_extra_augs,
    ])

    train_set = CTBrain25DDataset(train_df, local_root, transform=train_transform)
    val_set   = CTBrain25DDataset(val_df,   local_root, transform=None)
    nw = min(os.cpu_count() or 4, 16)
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True,  pin_memory=True, num_workers=nw, persistent_workers=True)
    val_loader   = DataLoader(val_set,   BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=nw, persistent_workers=True)
    print(f"  Train slices: {len(train_set)} | Val slices: {len(val_set)}\n")

    # Class weights [1.0, 10.0]: memberi bobot 10x untuk kelas tumor agar model tidak
    # hanya belajar background (class imbalance ekstrem pada CT non-kontras).
    class_weights = torch.tensor([1.0, 10.0], device=device)

    model     = SE2_CNNET(n_channels=3, n_classes=2, N=8, base_channels=32).to(device)

    criterion = AdvancedCombinedLoss(class_weights=class_weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True, min_lr=1e-7
    )
    scaler    = torch.amp.GradScaler('cuda')

    best_iou  = 0.0
    early_stop_counter = 0
    best_path = os.path.join(MODEL_SAVE_DIR, f'se2_unet_{dataset_key}_best.pth')

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train(); optimizer.zero_grad(); running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS} [Train-{dataset_key.upper()}]", ncols=90)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                loss = criterion(model(images), labels) / ACCUM_STEPS
            scaler.scale(loss).backward()
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            running_loss += loss.item() * ACCUM_STEPS
        if len(train_loader) % ACCUM_STEPS != 0:
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

        # ── Validate ──
        model.eval(); total_tp = total_fp = total_fn = total_tn = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Ep {epoch}/{EPOCHS} [Val-{dataset_key.upper()}]", ncols=90):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
                tp, fp, fn, tn = calculate_metrics_tensors(preds, labels)
                total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn

        eps   = 1e-7
        dice  = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + eps)
        iou   = total_tp / (total_tp + total_fp + total_fn + eps)
        prec  = total_tp / (total_tp + total_fp + eps)
        rec   = total_tp / (total_tp + total_fn + eps)
        avg_loss = running_loss / len(train_loader)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n  Ep {epoch:>3} [{dataset_key.upper()}] Loss {avg_loss:.4f} | Dice {dice:.4f} | IoU {iou:.4f} | Prec {prec:.4f} | Rec {rec:.4f} | LR {current_lr:.2e}")

        # Scheduler step — turunkan LR jika Dice stagnan selama 10 epoch
        scheduler.step(dice)

        if iou > best_iou:
            best_iou = iou
            early_stop_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"  🌟 New Best [{dataset_key.upper()}] Saved → {best_path} (IoU={iou:.4f})")
        else:
            early_stop_counter += 1
            print(f"  ⏳ No improvement. Early stop counter: {early_stop_counter}/{EARLY_STOP_PATIENCE}")
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"  🛑 Early stopping triggered at epoch {epoch}!")
                break

        if epoch % 10 == 0:
            ckpt = os.path.join(MODEL_SAVE_DIR, f'se2_unet_{dataset_key}_epoch_{epoch}.pth')
            torch.save(model.state_dict(), ckpt)
            print(f"  💾 Checkpoint → {ckpt}")

        torch.cuda.empty_cache()

    print(f"\n  ✅ Done! Best IoU [{dataset_key.upper()}]: {best_iou:.4f}")
    print(f"  Best weights: {best_path}")


# ================================================================
# LOGGER
# ================================================================
class Logger:
    def __init__(self, filename, stream):
        self.terminal = stream; self.log = open(filename, "a", encoding="utf-8")
    def write(self, m): self.terminal.write(m); self.log.write(m); self.log.flush()
    def flush(self): self.terminal.flush(); self.log.flush()


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SE2_CNNET on CT or CTC dataset separately")
    parser.add_argument('--dataset', required=True, choices=['ct', 'ctc', 'all'],
                        help="Dataset type to train on: 'ct' (CT_* folders), 'ctc' (CTC_*/CTW_* folders), 'all' (combined)")
    parser.add_argument('--log_dir', type=str, default='.',
                        help="Directory to save the internal training log file")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = os.path.join(args.log_dir, f"training_se2_{args.dataset}_{ts}.txt")
    sys.stdout = Logger(log, sys.stdout)
    sys.stderr = Logger(log, sys.stderr)
    print(f"📝 Logging to {log}")
    train(args.dataset)
