import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from escnn import gspaces
import escnn.nn as enn
import torch.nn.functional as F

# ==========================================
# 1. MODEL ARCHITECTURE (Copied for standalone execution)
# ==========================================

class DoubleEquivariantConv(nn.Module):
    def __init__(self, in_type, out_type, mid_type=None):
        super().__init__()
        if not mid_type:
            mid_type = out_type
        self.double_conv = enn.SequentialModule(
            enn.R2Conv(in_type, mid_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(mid_type),
            enn.ReLU(mid_type, inplace=True),
            enn.R2Conv(mid_type, out_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True)
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
        self.up = enn.R2Upsampling(in_type, scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleEquivariantConv(in_type + out_type, out_type)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = enn.tensor_directsum([x2, x1])
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_type, n_classes):
        super().__init__()
        gspace = in_type.gspace
        out_type = enn.FieldType(gspace, n_classes * [gspace.trivial_repr])
        self.conv = enn.R2Conv(in_type, out_type, kernel_size=1)
    def forward(self, x): return self.conv(x)

class SE2_CNNET(nn.Module):
    def __init__(self, n_channels, n_classes, N=8, base_channels=24):
        super().__init__()
        self.r2_act = gspaces.rot2dOnR2(N=N)
        c = base_channels

        self.feat_type_in = enn.FieldType(self.r2_act, n_channels * [self.r2_act.trivial_repr])
        self.feat_type_1 = enn.FieldType(self.r2_act, c * [self.r2_act.regular_repr])
        self.feat_type_2 = enn.FieldType(self.r2_act, (c*2) * [self.r2_act.regular_repr])
        self.feat_type_3 = enn.FieldType(self.r2_act, (c*4) * [self.r2_act.regular_repr])
        self.feat_type_4 = enn.FieldType(self.r2_act, (c*8) * [self.r2_act.regular_repr])
        self.feat_type_5 = enn.FieldType(self.r2_act, (c*16) * [self.r2_act.regular_repr])

        self.inc = DoubleEquivariantConv(self.feat_type_in, self.feat_type_1)
        self.down1 = Down(self.feat_type_1, self.feat_type_2)
        self.down2 = Down(self.feat_type_2, self.feat_type_3)
        self.down3 = Down(self.feat_type_3, self.feat_type_4)
        self.down4 = Down(self.feat_type_4, self.feat_type_5)

        self.up1 = Up(self.feat_type_5, self.feat_type_4)
        self.up2 = Up(self.feat_type_4, self.feat_type_3)
        self.up3 = Up(self.feat_type_3, self.feat_type_2)
        self.up4 = Up(self.feat_type_2, self.feat_type_1)
        self.outc = OutConv(self.feat_type_1, n_classes)

    def forward(self, x):
        x_geom = enn.GeometricTensor(x, self.feat_type_in)
        x1 = self.inc(x_geom)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x).tensor

# ==========================================
# 2. METRICS CALCULATION
# ==========================================

def calculate_metrics(pred, target):
    """
    Calculate essential medical image segmentation metrics.
    pred: Flattened binary prediction array
    target: Flattened binary ground truth array
    """
    tp = np.sum((pred == 1) & (target == 1))
    tn = np.sum((pred == 0) & (target == 0))
    fp = np.sum((pred == 1) & (target == 0))
    fn = np.sum((pred == 0) & (target == 1))

    # Add a small epsilon to prevent division by zero
    epsilon = 1e-6
    
    dice = (2.0 * tp) / ((2.0 * tp) + fp + fn + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon) # Also known as Sensitivity

    return dice, iou, accuracy, precision, recall

# ==========================================
# 3. MAIN EVALUATION FUNCTION
# ==========================================

def test_single_slice():
    # --- CONFIGURATION ---
    # Point this to your best saved model weights
    MODEL_WEIGHTS_PATH = "se2_unet_epoch_100.pth" 
    
    # Point this to ONE specific image and its corresponding mask in your local NVMe storage
    # IMPORTANT: Change this path to an actual file that exists in your dataset!
    TEST_IMAGE_PATH = os.path.expanduser("~/Clara/local_ct_workspace/NAMA_FOLDER_PASIEN/NAMA_FILE_img.npy")
    TEST_MASK_PATH = os.path.expanduser("~/Clara/local_ct_workspace/NAMA_FOLDER_PASIEN/NAMA_FILE_mask.npy")
    
    OUTPUT_IMAGE_NAME = "test_result_epoch100.png"

    # --- DEVICE SETUP ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    # --- LOAD MODEL ---
    print(f"Loading weights from {MODEL_WEIGHTS_PATH}...")
    model = SE2_CNNET(n_channels=1, n_classes=2, N=8, base_channels=24).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True))
        print("✅ Model weights loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model weights: {e}")
        return

    model.eval()

    # --- LOAD DATA ---
    print(f"Loading image from {TEST_IMAGE_PATH}...")
    try:
        img_np = np.load(TEST_IMAGE_PATH).astype(np.float32)
        mask_np = np.load(TEST_MASK_PATH).astype(np.uint8)
    except Exception as e:
        print(f"❌ Failed to load data. Please check the paths. Error: {e}")
        return

    # Prepare tensor (add Batch and Channel dimensions: B, C, H, W)
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)

    # --- INFERENCE ---
    print("Running inference...")
    with torch.no_grad():
        logits = model(img_tensor)
        # Apply Softmax to get probabilities, then Argmax to get the predicted class (0 or 1)
        probs = F.softmax(logits, dim=1)
        pred_mask_tensor = torch.argmax(probs, dim=1).squeeze().cpu().numpy()

    # --- CALCULATE METRICS ---
    dice, iou, acc, prec, rec = calculate_metrics(pred_mask_tensor.flatten(), mask_np.flatten())
    
    print("\n" + "="*30)
    print("📊 PERFORMANCE METRICS")
    print("="*30)
    print(f"Dice Score : {dice:.4f}")
    print(f"IoU Score  : {iou:.4f}")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Precision  : {prec:.4f}")
    print(f"Recall     : {rec:.4f}")
    print("="*30 + "\n")

    # --- VISUALIZATION ---
    print(f"Saving visual comparison to {OUTPUT_IMAGE_NAME}...")
    plt.figure(figsize=(15, 5))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.title("Original CT Image")
    plt.imshow(img_np, cmap='gray')
    plt.axis('off')
    
    # Ground Truth Mask
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(mask_np, cmap='gray')
    plt.axis('off')
    
    # Model Prediction
    plt.subplot(1, 3, 3)
    plt.title(f"Model Prediction\nDice: {dice:.4f}")
    plt.imshow(pred_mask_tensor, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_NAME, dpi=150, bbox_inches='tight')
    print(f"✅ Result saved! You can download or view {OUTPUT_IMAGE_NAME} to inspect the segmentation.")

if __name__ == "__main__":
    test_single_slice()