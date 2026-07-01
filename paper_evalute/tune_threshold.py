import os, sys, torch, torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the model and dataset from existing scripts
from evaluate_trained_models import SE2_CNNET, CTBrain25DDatasetNoResize, filter_df_by_dataset, load_se2_weights

def tune_threshold():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths
    CSV_REPORT = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
    DATA_PATH = os.path.expanduser("~/Clara/local_ct_workspace_full")
    WEIGHT_PATH = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D/se2_unet_ct_best.pth")

    if not os.path.exists(WEIGHT_PATH):
        print("❌ Weight file not found!")
        sys.exit(1)

    # Load Data
    df = pd.read_csv(CSV_REPORT)
    pc = 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'
    df = filter_df_by_dataset(df, 'ct', pc)
    
    # Validation split (exact same as evaluation)
    train_df = df.sample(frac=0.85, random_state=42)
    val_df = df.drop(train_df.index)
    
    val_set = CTBrain25DDatasetNoResize(val_df, DATA_PATH)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4)

    # Load Model
    model = load_se2_weights(SE2_CNNET, WEIGHT_PATH, device)
    model.eval()

    # Collect all predictions and targets
    all_probs = []
    all_masks = []
    
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Collecting Probabilities"):
            imgs = imgs.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
            probs = F.softmax(logits, dim=1)[:, 1] # Class 1 prob
            all_probs.append(probs.cpu())
            all_masks.append(masks.cpu())
            
    all_probs = torch.cat(all_probs).view(-1)
    all_masks = torch.cat(all_masks).view(-1)

    print("\n--- Threshold Tuning Results ---")
    best_iou = 0
    best_th = 0
    
    for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        preds = (all_probs > th).long()
        
        tp = ((preds == 1) & (all_masks == 1)).sum().item()
        fp = ((preds == 1) & (all_masks == 0)).sum().item()
        fn = ((preds == 0) & (all_masks == 1)).sum().item()
        
        eps = 1e-7
        iou = tp / (tp + fp + fn + eps)
        dice = (2*tp) / (2*tp + fp + fn + eps)
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        
        print(f"Threshold: {th:.2f} | IoU: {iou:.4f} | Dice: {dice:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
        
        if iou > best_iou:
            best_iou = iou
            best_th = th
            
    print(f"\n✅ Best Threshold: {best_th:.2f} with IoU: {best_iou:.4f}")

if __name__ == "__main__":
    tune_threshold()
