import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Import E2CNN for equivariant network layers
from escnn import gspaces
import escnn.nn as enn

# ─── Model path (training/ folder, relative to project root) ─────────────────
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
MODELS_DIR = os.path.join(_PROJECT_ROOT, "training")


# ==========================================
# MODEL ARCHITECTURE
# ==========================================

class DoubleEquivariantConv(nn.Module):
    """
    Equivariant double convolution block.
    Uses R2Conv to maintain equivariance under rotation and translation.
    """
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
    def forward(self, x): 
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downsampling block for the Equivariant U-Net.
    Applies MaxPool followed by a DoubleEquivariantConv block.
    """
    def __init__(self, in_type, out_type):
        super().__init__()
        self.pool = enn.PointwiseMaxPool(in_type, kernel_size=2)
        self.conv = DoubleEquivariantConv(in_type, out_type)
    def forward(self, x): 
        return self.conv(self.pool(x))

class Up(nn.Module):
    """
    Upsampling block for the Equivariant U-Net.
    Upsamples the spatial dimensions and combines with skip connections
    using tensor direct sum to maintain equivariant representations.
    """
    def __init__(self, in_type, out_type):
        super().__init__()
        self.up = enn.R2Upsampling(in_type, scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleEquivariantConv(in_type + out_type, out_type)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Combine skip connection and upsampled features
        x = enn.tensor_directsum([x2, x1])
        return self.conv(x)

class OutConv(nn.Module):
    """
    Final 1x1 equivariant convolution mapping features to target classes.
    Outputs a trivial representation since semantic segmentation labels 
    are invariant to target rotation.
    """
    def __init__(self, in_type, n_classes):
        super().__init__()
        gspace = in_type.gspace
        out_type = enn.FieldType(gspace, n_classes * [gspace.trivial_repr])
        self.conv = enn.R2Conv(in_type, out_type, kernel_size=1)
    def forward(self, x): 
        return self.conv(x)

class SE2_CNNET(nn.Module):
    """
    SE(2) Equivariant U-Net Architecture for segmentation.
    Ensures feature maps are equivariant to N discrete rotations.
    """
    def __init__(self, n_channels, n_classes, N=8, base_channels=24):
        super().__init__()
        self.r2_act = gspaces.rot2dOnR2(N=N)
        c = base_channels

        # Define field types (representations) for each depth level
        self.feat_type_in = enn.FieldType(self.r2_act, n_channels * [self.r2_act.trivial_repr])
        self.feat_type_1 = enn.FieldType(self.r2_act, c * [self.r2_act.regular_repr])
        self.feat_type_2 = enn.FieldType(self.r2_act, (c*2) * [self.r2_act.regular_repr])
        self.feat_type_3 = enn.FieldType(self.r2_act, (c*4) * [self.r2_act.regular_repr])
        self.feat_type_4 = enn.FieldType(self.r2_act, (c*8) * [self.r2_act.regular_repr])
        self.feat_type_5 = enn.FieldType(self.r2_act, (c*16) * [self.r2_act.regular_repr])

        # Encoder layers
        self.inc = DoubleEquivariantConv(self.feat_type_in, self.feat_type_1)
        self.down1 = Down(self.feat_type_1, self.feat_type_2)
        self.down2 = Down(self.feat_type_2, self.feat_type_3)
        self.down3 = Down(self.feat_type_3, self.feat_type_4)
        self.down4 = Down(self.feat_type_4, self.feat_type_5)

        # Decoder layers 
        self.up1 = Up(self.feat_type_5, self.feat_type_4)
        self.up2 = Up(self.feat_type_4, self.feat_type_3)
        self.up3 = Up(self.feat_type_3, self.feat_type_2)
        self.up4 = Up(self.feat_type_2, self.feat_type_1)
        
        # Output prediction layer
        self.outc = OutConv(self.feat_type_1, n_classes)

    def forward(self, x):
        # Wrap standard PyTorch tensor into a GeometricTensor
        x_geom = enn.GeometricTensor(x, self.feat_type_in)
        
        # Encoder passes
        x1 = self.inc(x_geom)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder passes with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Unwrap back to standard PyTorch tensor
        return self.outc(x).tensor


# ==========================================
# INFERENCE LOGIC & API CONTRACT
# ==========================================

# Internal cache for the newly trained model
_model_cache = None

def _load_model() -> SE2_CNNET:
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    weight_path = os.path.join(MODELS_DIR, "se2_unet_epoch_100.pth")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"State-of-the-art model weights not found at: {weight_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the new SE2_CNNET architecture
    model = SE2_CNNET(n_channels=1, n_classes=2, N=8, base_channels=24)
    
    # Load state dict correctly mapping to device
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()

    _model_cache = model
    return model

def predict_segmentation(image_file, modality: str, **kwargs):
    """
    Run SE2-CNNET inference for brain CTC segmentation.
    This maintains the existing API contract for src/app.py.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Provide requested preprocessing
    # The input image must be converted to grayscale, resized to 256x256
    image = Image.open(image_file).convert("L")
    image_256 = image.resize((256, 256), Image.BILINEAR)

    # Convert to NumPy array
    img_array = np.array(image_256, dtype=np.float32)

    # Normalized (0 to 1)
    if img_array.max() > img_array.min():
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
    else:
        img_array = np.zeros_like(img_array)

    # Reshaped to [1, 1, 256, 256] (Batch, Channel, Height, Width)
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)

    # 2. Load Model
    model = _load_model()

    # 3. Perform Inference
    with torch.no_grad():
        logits = model(img_tensor)           # Shape: [1, 2, 256, 256]
        
        # Apply softmax on dim=1
        probs = F.softmax(logits, dim=1)     # Class probabilities
        
        # Use argmax to extract the final binary mask
        pred = torch.argmax(probs, dim=1)    # Shape: [1, 256, 256]

    # Convert prediction to NumPy and scale to 0-255 for visualization
    pred_np = pred.squeeze(0).cpu().numpy().astype(np.uint8)
    mask_array = (pred_np * 255).astype(np.uint8)

    # Create PIL Image from array
    mask_image = Image.fromarray(mask_array, mode="L")
    
    # Optional Grad-CAM is disabled or simplified since it breaks strict strict no_grad contract
    # But to maintain API exactly, we return None for cam_image if there are 3 expected outputs
    cam_image = None
    
    return image_256, mask_image, cam_image
