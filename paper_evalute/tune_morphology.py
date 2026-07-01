import os, sys, torch, torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.ndimage as ndi
from skimage.morphology import remove_small_objects

# Import from existing scripts
from evaluate_trained_models import SE2_CNNET, CTBrain25DDatasetNoResize, filter_df_by_dataset, load_se2_weights

def clean_prediction(pred_mask, min_size=50):
    """
    Applies morphological post-processing:
    1. Removes small isolated false positive blobs (noise)
    2. Fills holes inside the hemorrhage (since hemorrhages are solid)
    """
    pred_mask = pred_mask.astype(bool)
    # Remove small objects
    cleaned = remove_small_objects(pred_mask, min_size=min_size)
    # Fill holes
    cleaned = ndi.binary_fill_holes(cleaned)
    return cleaned.astype(np.uint8)

def tune_morphology():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    CSV_REPORT = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
    DATA_PATH = os.path.expanduser("~/Clara/local_ct_workspace_full")
    WEIGHT_PATH = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D/se2_unet_ct_best.pth")

    df = pd.read_csv(CSV_REPORT)
    pc = 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'
    df = filter_df_by_dataset(df, 'ct', pc)
    
    train_df = df.sample(frac=0.85, random_state=42)
    val_df = df.drop(train_df.index)
    
    val_set = CTBrain25DDatasetNoResize(val_df, DATA_PATH)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4) # Batch 1 for easy 2D processing

    model = load_se2_weights(SE2_CNNET, WEIGHT_PATH, device)
    model.eval()

    total_tp = total_fp = total_fn = 0
    total_tp_clean = total_fp_clean = total_fn_clean = 0
    
    THRESHOLD = 0.80

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Testing Morphology"):
            imgs = imgs.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
            probs = F.softmax(logits, dim=1)[:, 1]
            
            preds = (probs > THRESHOLD).long().cpu().numpy()[0]
            masks = masks.cpu().numpy()[0]
            
            # 1. Base prediction metrics
            total_tp += ((preds == 1) & (masks == 1)).sum()
            total_fp += ((preds == 1) & (masks == 0)).sum()
            total_fn += ((preds == 0) & (masks == 1)).sum()
            
            # 2. Clean prediction metrics
            preds_clean = clean_prediction(preds, min_size=100) # Remove blobs smaller than 100 pixels
            total_tp_clean += ((preds_clean == 1) & (masks == 1)).sum()
            total_fp_clean += ((preds_clean == 1) & (masks == 0)).sum()
            total_fn_clean += ((preds_clean == 0) & (masks == 1)).sum()
            
    eps = 1e-7
    iou_base = total_tp / (total_tp + total_fp + total_fn + eps)
    dice_base = (2*total_tp) / (2*total_tp + total_fp + total_fn + eps)
    
    iou_clean = total_tp_clean / (total_tp_clean + total_fp_clean + total_fn_clean + eps)
    dice_clean = (2*total_tp_clean) / (2*total_tp_clean + total_fp_clean + total_fn_clean + eps)
    
    print("\n--- RESULTS ---")
    print(f"Base Model (Threshold 0.80): IoU {iou_base:.4f} | Dice {dice_base:.4f}")
    print(f"Cleaned Model (Morphology) : IoU {iou_clean:.4f} | Dice {dice_clean:.4f}")
    
    if iou_clean > iou_base:
        print(f"\n✅ Peningkatan IoU: +{iou_clean - iou_base:.4f}")
    else:
        print(f"\n❌ Morphology tidak banyak membantu.")

if __name__ == "__main__":
    tune_morphology()
